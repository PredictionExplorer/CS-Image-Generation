//! Sonification — gallery-grade audio synthesis from three-body simulation data.
//!
//! **Gallery-grade (default):**
//! - **Gravitational Strings**: Karplus-Strong waveguide strings plucked by near-misses,
//!   with mass-derived pitch, chaos-driven damping, sympathetic coupling, HRTF binaural
//!   spatialization, soft saturation, and Freeverb reverb.
//! - **Orbital Terrain**: Wave terrain synthesis where the orbit IS the wavetable — the
//!   three-body trajectory becomes the sound waveform, with FM cross-modulation between
//!   bodies, asynchronous granular cloud, and Freeverb.
//! - **Void Resonance**: Installation-style spectral freeze — captures Karplus-Strong
//!   string tones and loops them as slowly evolving frozen textures, with distant string
//!   events, breath noise, and deep reverb.
//!
//! **Legacy:**
//! - **Crystal Resonance**: FM bell tones triggered by gravitational slingshots
//! - **Orbital Choir**: Formant-filtered choral texture with just intonation

use crate::utils::{angular_momentum_at, plane_normal_at, triangle_area_at};
use nalgebra::Vector3;
use std::io::Write;
use std::process::{Command, Stdio};
use tracing::info;

const SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 2;
const BITS_PER_SAMPLE: u16 = 16;
const SR: f64 = SAMPLE_RATE as f64;
const TAU: f64 = std::f64::consts::TAU;

#[allow(dead_code)]
const PENTATONIC: [f64; 16] = [
    110.00, 130.81, 146.83, 164.81, 196.00, 220.00, 261.63, 293.66,
    329.63, 392.00, 440.00, 523.25, 587.33, 659.26, 783.99, 880.00,
];

const WHOLE_TONE: [f64; 19] = [
    130.81, 146.83, 164.81, 185.00, 207.65, 233.08, 261.63, 293.66,
    329.63, 369.99, 415.30, 466.16, 523.25, 587.33, 659.26, 739.99,
    830.61, 932.33, 1046.50,
];

// ─── Shared Helpers ──────────────────────────────────────────

fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn smoothstep_fade(t: f64, fade_in_s: f64, fade_out_s: f64, duration_s: f64) -> f64 {
    let time_s = t * duration_s;
    smoothstep((time_s / fade_in_s).min(1.0)) * smoothstep(((duration_s - time_s) / fade_out_s).min(1.0))
}

fn quantize_to_scale(freq: f64, scale: &[f64]) -> f64 {
    let mut best = scale[0];
    let mut best_dist = (freq - best).abs();
    for &f in &scale[1..] {
        let d = (freq - f).abs();
        if d < best_dist {
            best = f;
            best_dist = d;
        }
    }
    best
}

fn soft_saturate(x: f64, drive: f64) -> f64 {
    let d = drive.max(0.01);
    (x * d).tanh() / d.tanh()
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }
    fn next_f64(&mut self) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let v = self.state.wrapping_mul(0x2545F4914F6CDD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn body_speed(
    positions: &[Vec<Vector3<f64>>],
    body: usize,
    step: usize,
    frac: f64,
    step_count: usize,
) -> f64 {
    let v0 = if step + 1 < step_count {
        (positions[body][step + 1] - positions[body][step]).norm()
    } else {
        0.0
    };
    let v1 = if step + 2 < step_count {
        (positions[body][step + 2] - positions[body][step + 1]).norm()
    } else {
        v0
    };
    v0 * (1.0 - frac) + v1 * frac
}

fn body_closeness(
    positions: &[Vec<Vector3<f64>>],
    body: usize,
    step: usize,
    num_bodies: usize,
    max_dist: f64,
) -> f64 {
    let mut c = 0.0f64;
    for other in 0..num_bodies {
        if other != body {
            let d = (positions[body][step] - positions[other][step]).norm();
            c += 1.0 - (d / max_dist).clamp(0.0, 1.0);
        }
    }
    c / (num_bodies - 1).max(1) as f64
}

// ─── Freeverb (8 comb + 4 allpass, stereo) ──────────────────

const FREEVERB_COMB_LENGTHS_L: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
const FREEVERB_AP_LENGTHS_L: [usize; 4] = [556, 441, 341, 225];
const FREEVERB_STEREO_SPREAD: usize = 23;

struct CombFilter {
    buffer: Vec<f64>,
    idx: usize,
    filter_store: f64,
}

impl CombFilter {
    fn new(len: usize) -> Self {
        Self { buffer: vec![0.0; len.max(1)], idx: 0, filter_store: 0.0 }
    }
    fn process(&mut self, input: f64, feedback: f64, damp: f64) -> f64 {
        let out = self.buffer[self.idx];
        self.filter_store = out * (1.0 - damp) + self.filter_store * damp;
        self.buffer[self.idx] = input + self.filter_store * feedback;
        self.idx += 1;
        if self.idx >= self.buffer.len() { self.idx = 0; }
        out
    }
}

struct AllpassFilter {
    buffer: Vec<f64>,
    idx: usize,
}

impl AllpassFilter {
    fn new(len: usize) -> Self {
        Self { buffer: vec![0.0; len.max(1)], idx: 0 }
    }
    fn process(&mut self, input: f64) -> f64 {
        let buf_out = self.buffer[self.idx];
        let output = buf_out - input;
        self.buffer[self.idx] = input + buf_out * 0.5;
        self.idx += 1;
        if self.idx >= self.buffer.len() { self.idx = 0; }
        output
    }
}

struct Freeverb {
    combs_l: Vec<CombFilter>,
    combs_r: Vec<CombFilter>,
    aps_l: Vec<AllpassFilter>,
    aps_r: Vec<AllpassFilter>,
    feedback: f64,
    damp: f64,
    wet: f64,
    dry: f64,
    width: f64,
}

impl Freeverb {
    fn new(room_size: f64, damping: f64, wet: f64, width: f64) -> Self {
        let scale = |len: usize| ((len as f64 * SR / 44100.0) as usize).max(1);
        let combs_l: Vec<_> = FREEVERB_COMB_LENGTHS_L.iter()
            .map(|&l| CombFilter::new(scale(l)))
            .collect();
        let combs_r: Vec<_> = FREEVERB_COMB_LENGTHS_L.iter()
            .map(|&l| CombFilter::new(scale(l + FREEVERB_STEREO_SPREAD)))
            .collect();
        let aps_l: Vec<_> = FREEVERB_AP_LENGTHS_L.iter()
            .map(|&l| AllpassFilter::new(scale(l)))
            .collect();
        let aps_r: Vec<_> = FREEVERB_AP_LENGTHS_L.iter()
            .map(|&l| AllpassFilter::new(scale(l + FREEVERB_STEREO_SPREAD)))
            .collect();
        Self {
            combs_l, combs_r, aps_l, aps_r,
            feedback: room_size.clamp(0.0, 1.0) * 0.28 + 0.7,
            damp: damping.clamp(0.0, 1.0),
            wet, dry: 1.0 - wet, width,
        }
    }

    fn process(&mut self, left: f64, right: f64) -> (f64, f64) {
        let input = (left + right) * 0.015;
        let mut out_l = 0.0;
        let mut out_r = 0.0;
        for comb in &mut self.combs_l { out_l += comb.process(input, self.feedback, self.damp); }
        for comb in &mut self.combs_r { out_r += comb.process(input, self.feedback, self.damp); }
        for ap in &mut self.aps_l { out_l = ap.process(out_l); }
        for ap in &mut self.aps_r { out_r = ap.process(out_r); }
        let wet1 = self.wet * (1.0 + self.width) * 0.5;
        let wet2 = self.wet * (1.0 - self.width) * 0.5;
        (out_l * wet1 + out_r * wet2 + left * self.dry,
         out_r * wet1 + out_l * wet2 + right * self.dry)
    }
}

fn apply_freeverb(buf: &mut [f64], n: usize, room_size: f64, damping: f64, wet: f64) {
    let mut verb = Freeverb::new(room_size, damping, wet, 1.0);
    for i in 0..n {
        let (l, r) = verb.process(buf[i * 2], buf[i * 2 + 1]);
        buf[i * 2] = l;
        buf[i * 2 + 1] = r;
    }
}

// ─── HRTF-Lite Binaural Panner ──────────────────────────────

struct HrtfPanner {
    delay_buf_l: Vec<f64>,
    delay_buf_r: Vec<f64>,
    idx: usize,
    shadow_state_l: f64,
    shadow_state_r: f64,
    max_delay: usize,
}

impl HrtfPanner {
    fn new() -> Self {
        let max_delay = (0.0007 * SR) as usize + 2;
        Self {
            delay_buf_l: vec![0.0; max_delay],
            delay_buf_r: vec![0.0; max_delay],
            idx: 0,
            shadow_state_l: 0.0,
            shadow_state_r: 0.0,
            max_delay,
        }
    }

    fn process(&mut self, mono: f64, azimuth: f64, distance: f64) -> (f64, f64) {
        let az = azimuth.clamp(-1.0, 1.0);
        let itd_frac = (1.0 + az) * 0.5;
        let delay_r_samples = itd_frac * (self.max_delay - 1) as f64;
        let delay_l_samples = (1.0 - itd_frac) * (self.max_delay - 1) as f64;

        self.delay_buf_l[self.idx] = mono;
        self.delay_buf_r[self.idx] = mono;

        let read_l = self.read_delay(&self.delay_buf_l, delay_l_samples);
        let read_r = self.read_delay(&self.delay_buf_r, delay_r_samples);

        let shadow_coeff = 0.15 + 0.35 * az.abs();
        let (sig_l, sig_r) = if az > 0.0 {
            self.shadow_state_l = self.shadow_state_l * shadow_coeff + read_l * (1.0 - shadow_coeff);
            (self.shadow_state_l, read_r)
        } else {
            self.shadow_state_r = self.shadow_state_r * shadow_coeff + read_r * (1.0 - shadow_coeff);
            (read_l, self.shadow_state_r)
        };

        let dist_atten = 1.0 / (1.0 + distance * 0.3);

        self.idx = (self.idx + 1) % self.max_delay;
        (sig_l * dist_atten, sig_r * dist_atten)
    }

    fn read_delay(&self, buf: &[f64], delay: f64) -> f64 {
        let d = delay.max(0.0);
        let d_int = d as usize;
        let frac = d - d_int as f64;
        let len = buf.len();
        let i0 = (self.idx + len - d_int) % len;
        let i1 = (self.idx + len - d_int - 1) % len;
        buf[i0] * (1.0 - frac) + buf[i1] * frac
    }
}

// ─── Karplus-Strong Waveguide String ─────────────────────────

struct KarplusStrongString {
    delay: Vec<f64>,
    write_idx: usize,
    len: usize,
    damp_state: f64,
    damping: f64,
    tuning_ap_state: f64,
    tuning_coeff: f64,
}

impl KarplusStrongString {
    fn new(freq: f64, damping: f64) -> Self {
        let period = (SR / freq.max(20.0)).round() as usize;
        let len = period.max(2);
        let d = damping.clamp(0.0, 0.98);
        let frac_delay = SR / freq.max(20.0) - len as f64;
        let tuning_coeff = (1.0 - frac_delay) / (1.0 + frac_delay);
        Self {
            delay: vec![0.0; len],
            write_idx: 0,
            len,
            damp_state: 0.0,
            damping: d,
            tuning_ap_state: 0.0,
            tuning_coeff,
        }
    }

    fn pluck(&mut self, rng: &mut SimpleRng, amplitude: f64, brightness: f64) {
        let bright = brightness.clamp(0.1, 1.0);
        let mut prev = 0.0f64;
        for sample in &mut self.delay {
            let noise = rng.next_f64() * amplitude;
            let filtered = noise * bright + prev * (1.0 - bright);
            *sample += filtered;
            prev = filtered;
        }
    }

    fn tick(&mut self) -> f64 {
        let out_idx = (self.write_idx + 1) % self.len;
        let next_idx = (self.write_idx + 2) % self.len;
        let avg = (self.delay[out_idx] + self.delay[next_idx]) * 0.5;

        self.damp_state = avg * (1.0 - self.damping) + self.damp_state * self.damping;

        let tuned = self.tuning_coeff * self.damp_state
            + self.tuning_ap_state
            - self.tuning_coeff * self.delay[self.write_idx];
        self.tuning_ap_state = self.damp_state;

        self.delay[self.write_idx] = tuned * 0.996;
        let output = self.delay[out_idx];
        self.write_idx = out_idx;
        output
    }

    fn inject(&mut self, sample: f64) {
        self.delay[self.write_idx] += sample;
    }
}

// ─── Wave Terrain ────────────────────────────────────────────

const TERRAIN_SIZE: usize = 256;

struct WaveTerrain {
    terrain: Vec<f64>,
    size: usize,
}

impl WaveTerrain {
    fn from_positions(positions: &[Vec<Vector3<f64>>], step_count: usize) -> Self {
        let size = TERRAIN_SIZE;
        let mut terrain = vec![0.0f64; size * size];
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        for body in positions.iter().take(3) {
            for p in body.iter().take(step_count) {
                min_x = min_x.min(p.x);
                max_x = max_x.max(p.x);
                min_y = min_y.min(p.y);
                max_y = max_y.max(p.y);
            }
        }
        let rx = (max_x - min_x).max(1e-10);
        let ry = (max_y - min_y).max(1e-10);

        let stride = (step_count / 2000).max(1);
        for step in (0..step_count).step_by(stride) {
            for pair in [(0, 1), (1, 2), (2, 0)] {
                let (a, b) = pair;
                if a < positions.len() && b < positions.len() {
                    let mid_x = (positions[a][step].x + positions[b][step].x) * 0.5;
                    let mid_y = (positions[a][step].y + positions[b][step].y) * 0.5;
                    let dist = (positions[a][step] - positions[b][step]).norm();
                    let xi = ((mid_x - min_x) / rx * (size - 1) as f64) as usize;
                    let yi = ((mid_y - min_y) / ry * (size - 1) as f64) as usize;
                    let xi = xi.min(size - 1);
                    let yi = yi.min(size - 1);
                    terrain[yi * size + xi] += 1.0 / (1.0 + dist);
                }
            }
        }

        let peak = terrain.iter().copied().fold(0.0f64, |a, v| a.max(v.abs()));
        if peak > 0.0 {
            for v in &mut terrain { *v /= peak; }
        }
        Self { terrain, size }
    }

    fn sample(&self, x: f64, y: f64) -> f64 {
        let fx = (x.fract() + 1.0).fract() * (self.size - 1) as f64;
        let fy = (y.fract() + 1.0).fract() * (self.size - 1) as f64;
        let ix = fx as usize;
        let iy = fy as usize;
        let fx = fx - ix as f64;
        let fy = fy - iy as f64;
        let ix1 = (ix + 1).min(self.size - 1);
        let iy1 = (iy + 1).min(self.size - 1);
        let s = self.size;
        let v00 = self.terrain[iy * s + ix];
        let v10 = self.terrain[iy * s + ix1];
        let v01 = self.terrain[iy1 * s + ix];
        let v11 = self.terrain[iy1 * s + ix1];
        let top = v00 * (1.0 - fx) + v10 * fx;
        let bot = v01 * (1.0 - fx) + v11 * fx;
        top * (1.0 - fy) + bot * fy
    }
}

// ─── Spectral Freezer ────────────────────────────────────────

const FREEZE_GRAIN_LEN: usize = 2048;

struct SpectralFreezer {
    grains: Vec<Vec<f64>>,
    playback_idx: Vec<f64>,
    amplitudes: Vec<f64>,
    max_grains: usize,
}

impl SpectralFreezer {
    fn new(max_grains: usize) -> Self {
        Self {
            grains: Vec::new(),
            playback_idx: Vec::new(),
            amplitudes: Vec::new(),
            max_grains,
        }
    }

    fn capture(&mut self, source: &[f64], offset: usize, amplitude: f64) {
        if self.grains.len() >= self.max_grains {
            self.grains.remove(0);
            self.playback_idx.remove(0);
            self.amplitudes.remove(0);
        }
        let end = (offset + FREEZE_GRAIN_LEN).min(source.len());
        let start = end.saturating_sub(FREEZE_GRAIN_LEN);
        let mut grain: Vec<f64> = source[start..end].to_vec();
        while grain.len() < FREEZE_GRAIN_LEN { grain.push(0.0); }
        for (i, s) in grain.iter_mut().enumerate() {
            let env = (i as f64 / FREEZE_GRAIN_LEN as f64 * std::f64::consts::PI).sin();
            *s *= env;
        }
        self.grains.push(grain);
        self.playback_idx.push(0.0);
        self.amplitudes.push(amplitude);
    }

    fn tick(&mut self, speed: f64) -> f64 {
        let mut out = 0.0;
        for i in 0..self.grains.len() {
            let idx = self.playback_idx[i];
            let len = self.grains[i].len();
            let i0 = idx as usize % len;
            let i1 = (i0 + 1) % len;
            let frac = idx - idx.floor();
            out += (self.grains[i][i0] * (1.0 - frac) + self.grains[i][i1] * frac) * self.amplitudes[i];
            self.playback_idx[i] += speed;
            if self.playback_idx[i] >= len as f64 { self.playback_idx[i] -= len as f64; }
        }
        for amp in &mut self.amplitudes {
            *amp *= 0.99998;
        }
        out
    }
}

// ─── Gallery Audio Context ───────────────────────────────────

pub struct GalleryAudioContext {
    pub chaos: f64,
    pub equilateralness: f64,
    pub masses: [f64; 3],
}

// ─── Sonic Worlds ────────────────────────────────────────────

/// Sonic world — a curated tuning/material/spatial family selected by orbit character.
///
/// | World | Character | Orbit profile |
/// |-------|-----------|---------------|
/// | **Obsidian** | Dark, metallic, angular | High chaos, low equilateralness |
/// | **Aurora** | Bright, crystalline, shimmering | High chaos, high equilateralness |
/// | **Meridian** | Warm, resonant, vocal | Low chaos, high equilateralness |
/// | **Liminal** | Ethereal, spectral, suspended | Low chaos, low equilateralness |
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SonicWorld {
    Obsidian,
    Aurora,
    Meridian,
    Liminal,
}

impl SonicWorld {
    pub fn name(self) -> &'static str {
        match self {
            Self::Obsidian => "Obsidian",
            Self::Aurora => "Aurora",
            Self::Meridian => "Meridian",
            Self::Liminal => "Liminal",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Obsidian => "Dark waveguide strings with metallic damping",
            Self::Aurora => "Bright crystalline strings with long sustain",
            Self::Meridian => "Warm resonant strings with vocal richness",
            Self::Liminal => "Ethereal strings with ghostly sustain",
        }
    }
}

struct WorldTuning {
    damping: f64,
    brightness: f64,
    reverb_size: f64,
    reverb_damp: f64,
    reverb_wet: f64,
    pluck_threshold: f64,
    terrain_drive: f64,
    freeze_density: f64,
}

pub fn select_sonic_world(chaos: f64, equil: f64) -> SonicWorld {
    let chaos_norm = chaos.clamp(0.0, 1.0);
    let high_chaos = chaos_norm < 0.4;
    let high_equil = equil > 0.5;
    match (high_chaos, high_equil) {
        (true, false) => SonicWorld::Obsidian,
        (true, true) => SonicWorld::Aurora,
        (false, true) => SonicWorld::Meridian,
        (false, false) => SonicWorld::Liminal,
    }
}

fn world_tuning(world: SonicWorld) -> WorldTuning {
    match world {
        SonicWorld::Obsidian => WorldTuning {
            damping: 0.35, brightness: 0.9, reverb_size: 0.65,
            reverb_damp: 0.6, reverb_wet: 0.30, pluck_threshold: 0.55,
            terrain_drive: 2.5, freeze_density: 0.4,
        },
        SonicWorld::Aurora => WorldTuning {
            damping: 0.15, brightness: 0.95, reverb_size: 0.85,
            reverb_damp: 0.3, reverb_wet: 0.40, pluck_threshold: 0.45,
            terrain_drive: 1.8, freeze_density: 0.6,
        },
        SonicWorld::Meridian => WorldTuning {
            damping: 0.45, brightness: 0.6, reverb_size: 0.75,
            reverb_damp: 0.5, reverb_wet: 0.35, pluck_threshold: 0.50,
            terrain_drive: 1.5, freeze_density: 0.5,
        },
        SonicWorld::Liminal => WorldTuning {
            damping: 0.20, brightness: 0.4, reverb_size: 0.95,
            reverb_damp: 0.2, reverb_wet: 0.50, pluck_threshold: 0.40,
            terrain_drive: 1.2, freeze_density: 0.8,
        },
    }
}

// ─── Extended Orbit Analysis ─────────────────────────────────

struct OrbitStats {
    max_speed: f64,
    max_distance: f64,
    max_area: f64,
    max_ang_mom: f64,
    max_tumble_rate: f64,
}

fn compute_orbit_stats(positions: &[Vec<Vector3<f64>>]) -> OrbitStats {
    let nb = 3.min(positions.len());
    let steps = positions[0].len();
    let mut max_speed = 0.0f64;
    let mut max_distance = 0.0f64;
    let mut max_area = 0.0f64;
    let mut max_ang_mom = 0.0f64;
    let mut max_tumble_rate = 0.0f64;
    let mut prev_normal = if nb >= 3 && steps > 0 {
        plane_normal_at(positions, 0)
    } else {
        Vector3::new(0.0, 0.0, 1.0)
    };

    for step in 1..steps {
        for body_pos in positions.iter().take(nb) {
            let vel = (body_pos[step] - body_pos[step - 1]).norm();
            max_speed = max_speed.max(vel);
        }
        for i in 0..nb {
            for j in (i + 1)..nb {
                let d = (positions[i][step] - positions[j][step]).norm();
                max_distance = max_distance.max(d);
            }
        }
        if nb >= 3 {
            let a = (positions[0][step] - positions[1][step]).norm();
            let b_len = (positions[1][step] - positions[2][step]).norm();
            let c = (positions[2][step] - positions[0][step]).norm();
            let s = (a + b_len + c) * 0.5;
            let area_sq = (s * (s - a) * (s - b_len) * (s - c)).max(0.0);
            max_area = max_area.max(area_sq.sqrt());

            let cm = (positions[0][step] + positions[1][step] + positions[2][step]) / 3.0;
            let mut ang = Vector3::zeros();
            for body_pos in positions.iter().take(nb) {
                let r: Vector3<f64> = body_pos[step] - cm;
                let v = body_pos[step] - body_pos[step.saturating_sub(1)];
                ang += r.cross(&v);
            }
            max_ang_mom = max_ang_mom.max(ang.norm());

            let cur_normal = plane_normal_at(positions, step);
            let dot = prev_normal.dot(&cur_normal).clamp(-1.0, 1.0);
            let tumble = dot.acos();
            max_tumble_rate = max_tumble_rate.max(tumble);
            prev_normal = cur_normal;
        }
    }

    OrbitStats {
        max_speed: max_speed.max(1e-10),
        max_distance: max_distance.max(1e-10),
        max_area: max_area.max(1e-10),
        max_ang_mom: max_ang_mom.max(1e-10),
        max_tumble_rate: max_tumble_rate.max(1e-14),
    }
}

// ─── Algorithm F: Gravitational Strings ──────────────────────

fn synthesize_gravitational_strings(
    positions: &[Vec<Vector3<f64>>],
    stats: &OrbitStats,
    ctx: &GalleryAudioContext,
    tuning: &WorldTuning,
    total_samples: usize,
    step_count: usize,
    duration_s: f64,
) -> Vec<f64> {
    let nb = 3.min(positions.len());
    let mut buf = vec![0.0f64; total_samples * 2];

    let freqs: Vec<f64> = (0..nb)
        .map(|b| 55.0 * (2000.0 / ctx.masses[b].max(100.0)))
        .collect();

    let mut strings: Vec<KarplusStrongString> = freqs
        .iter()
        .map(|&f| KarplusStrongString::new(f, tuning.damping))
        .collect();

    let mut panners: Vec<HrtfPanner> = (0..nb).map(|_| HrtfPanner::new()).collect();
    let mut rng = SimpleRng::new(0xD7A1_5471_C5E1_E571_u64.wrapping_mul(
        (ctx.chaos * 1e9) as u64 | 1,
    ));

    let mut sm_close = [[0.0f64; 3]; 3];
    let mut prev_close = [[0.0f64; 3]; 3];
    let close_a = 1.0 - (-1.0 / (0.02 * SR)).exp();
    let sympathy = 0.08;

    let mut sub_ph = 0.0f64;
    let mut prev_normal = plane_normal_at(positions, 0);
    let mut sm_plane_az = 0.0f64;
    let plane_az_a = 1.0 - (-1.0 / (0.3 * SR)).exp();

    for si in 0..total_samples {
        let t = si as f64 / total_samples as f64;
        let sf = t * (step_count.saturating_sub(2)) as f64;
        let step = sf as usize;
        let frac = sf - step as f64;
        let s = step.min(step_count - 1);

        let area = triangle_area_at(positions, s);
        let norm_area = (area / stats.max_area).clamp(0.0, 1.0);

        let cur_normal = plane_normal_at(positions, s);
        let tumble_dot = prev_normal.dot(&cur_normal).clamp(-1.0, 1.0);
        let tumble_rate = tumble_dot.acos();
        let norm_tumble = (tumble_rate / stats.max_tumble_rate).clamp(0.0, 1.0);
        prev_normal = cur_normal;
        let plane_az = cur_normal.x.atan2(cur_normal.y);
        sm_plane_az += plane_az_a * (plane_az - sm_plane_az);

        let mut pluck_bodies = [false; 3];
        let mut pluck_intensities = [0.0f64; 3];

        for b in 0..nb {
            for other in 0..nb {
                if other == b { continue; }
                let d = (positions[b][s] - positions[other][s]).norm();
                let close = 1.0 - (d / stats.max_distance).clamp(0.0, 1.0);
                prev_close[b][other] = sm_close[b][other];
                sm_close[b][other] += close_a * (close - sm_close[b][other]);

                if sm_close[b][other] > tuning.pluck_threshold
                    && prev_close[b][other] <= tuning.pluck_threshold
                {
                    pluck_bodies[b] = true;
                    pluck_intensities[b] =
                        pluck_intensities[b].max(sm_close[b][other]);
                }
            }
        }

        for b in 0..nb {
            if pluck_bodies[b] {
                let intensity = pluck_intensities[b];
                let spd = body_speed(positions, b, step, frac, step_count);
                let ns = (spd / stats.max_speed).clamp(0.0, 1.0);
                let bright = tuning.brightness * (0.5 + 0.5 * ns) * (0.8 + 0.4 * norm_tumble);
                strings[b].pluck(&mut rng, intensity * 0.7, bright);

                for other in 0..nb {
                    if other != b {
                        let coupling = sm_close[b][other].clamp(0.0, 1.0) * sympathy;
                        strings[other].pluck(&mut rng, intensity * coupling, bright * 0.7);
                    }
                }
            }
        }

        let mut bow_noise = [0.0f64; 3];
        for (b, bn) in bow_noise.iter_mut().enumerate().take(nb) {
            let close = body_closeness(positions, b, s, nb, stats.max_distance);
            let spd = body_speed(positions, b, step, frac, step_count);
            let ns = (spd / stats.max_speed).clamp(0.0, 1.0);
            *bn = rng.next_f64() * 0.008 * ns * close;
        }

        let mut left = 0.0f64;
        let mut right = 0.0f64;
        for b in 0..nb {
            strings[b].inject(bow_noise[b]);
            let raw = strings[b].tick();

            let close = body_closeness(positions, b, s, nb, stats.max_distance);
            let drive = 1.0 + close * 2.0;
            let saturated = soft_saturate(raw, drive);

            let pos = positions[b][s];
            let centroid = (positions[0][s] + positions[1][s] + positions[2][s]) / 3.0;
            let rel: Vector3<f64> = pos - centroid;
            let azimuth = rel.x.atan2(rel.y) / std::f64::consts::PI;
            let dist = rel.norm() / stats.max_distance;
            let (l, r) = panners[b].process(saturated, azimuth, dist);
            left += l;
            right += r;
        }

        sub_ph += (freqs[0] * 0.25) / SR;
        let sub = (sub_ph * TAU).sin() * 0.04 * norm_area;
        left += sub;
        right += sub;

        let rot = sm_plane_az * 0.4;
        let cos_r = rot.cos();
        let sin_r = rot.sin();
        let rl = left * cos_r - right * sin_r;
        let rr = left * sin_r + right * cos_r;

        let fade = smoothstep_fade(t, 3.0, 5.0, duration_s);
        buf[si * 2] = rl * fade;
        buf[si * 2 + 1] = rr * fade;
    }

    apply_freeverb(&mut buf, total_samples, tuning.reverb_size, tuning.reverb_damp, tuning.reverb_wet);
    normalize_peak(&mut buf, 0.88);
    buf
}

// ─── Algorithm G: Orbital Terrain ────────────────────────────

fn synthesize_orbital_terrain(
    positions: &[Vec<Vector3<f64>>],
    stats: &OrbitStats,
    ctx: &GalleryAudioContext,
    tuning: &WorldTuning,
    total_samples: usize,
    step_count: usize,
    duration_s: f64,
) -> Vec<f64> {
    let nb = 3.min(positions.len());
    let mut buf = vec![0.0f64; total_samples * 2];

    let terrain = WaveTerrain::from_positions(positions, step_count);

    let mut scan_x = [0.0f64; 3];
    let mut scan_y = [0.0f64; 3];
    let mut panners: Vec<HrtfPanner> = (0..nb).map(|_| HrtfPanner::new()).collect();
    let mut rng = SimpleRng::new(0xA7E1_7A1C_05B1_DE12_u64.wrapping_mul(
        (ctx.equilateralness * 1e9) as u64 | 1,
    ));

    let base_freq = 55.0 * (2000.0 / ctx.masses.iter().sum::<f64>() * 3.0 / ctx.masses[0].max(100.0));

    struct Grain {
        buffer: Vec<f64>,
        pos: f64,
        speed: f64,
        amp: f64,
        pan: f64,
        remaining: usize,
    }

    let mut grains: Vec<Grain> = Vec::new();
    let mut last_grain_si = 0usize;
    let grain_cooldown = (0.05 * SR) as usize;

    let mut sub_ph = 0.0f64;
    let mut prev_normal = plane_normal_at(positions, 0);
    let mut sm_plane_az = 0.0f64;
    let plane_az_a = 1.0 - (-1.0 / (0.3 * SR)).exp();

    let mut pos_range_x = (f64::MAX, f64::MIN);
    let mut pos_range_y = (f64::MAX, f64::MIN);
    for body in positions.iter().take(nb) {
        for p in body.iter().take(step_count) {
            pos_range_x.0 = pos_range_x.0.min(p.x);
            pos_range_x.1 = pos_range_x.1.max(p.x);
            pos_range_y.0 = pos_range_y.0.min(p.y);
            pos_range_y.1 = pos_range_y.1.max(p.y);
        }
    }
    let rx = (pos_range_x.1 - pos_range_x.0).max(1e-10);
    let ry = (pos_range_y.1 - pos_range_y.0).max(1e-10);

    for si in 0..total_samples {
        let t = si as f64 / total_samples as f64;
        let sf = t * (step_count.saturating_sub(2)) as f64;
        let step = sf as usize;
        let frac = sf - step as f64;
        let s = step.min(step_count - 1);

        let cur_normal = plane_normal_at(positions, s);
        let tumble_dot = prev_normal.dot(&cur_normal).clamp(-1.0, 1.0);
        let norm_tumble = (tumble_dot.acos() / stats.max_tumble_rate).clamp(0.0, 1.0);
        prev_normal = cur_normal;
        let plane_az = cur_normal.x.atan2(cur_normal.y);
        sm_plane_az += plane_az_a * (plane_az - sm_plane_az);

        let mut left = 0.0f64;
        let mut right = 0.0f64;

        for b in 0..nb {
            let spd = body_speed(positions, b, step, frac, step_count);
            let ns = (spd / stats.max_speed).clamp(0.0, 1.0);

            let pos = positions[b][s];
            let norm_x = (pos.x - pos_range_x.0) / rx;
            let norm_y = (pos.y - pos_range_y.0) / ry;

            let freq_scale = base_freq * (0.5 + ns * 2.0) * (1.0 + b as f64 * 0.33);
            let scan_speed = freq_scale / SR;
            scan_x[b] += scan_speed;
            scan_y[b] += scan_speed * 0.618;

            let fm_mod_x: f64 = (0..nb)
                .filter(|&o| o != b)
                .map(|o| {
                    let op = positions[o][s];
                    (op.x - pos_range_x.0) / rx * 0.1
                })
                .sum();
            let fm_mod_y: f64 = (0..nb)
                .filter(|&o| o != b)
                .map(|o| {
                    let op = positions[o][s];
                    (op.y - pos_range_y.0) / ry * 0.1
                })
                .sum();

            let tx = scan_x[b] + norm_x * 0.3 + fm_mod_x;
            let ty = scan_y[b] + norm_y * 0.3 + fm_mod_y;
            let raw = terrain.sample(tx, ty);

            let close = body_closeness(positions, b, s, nb, stats.max_distance);
            let amp = 0.2 * (0.3 + 0.7 * close);
            let dynamic_drive = tuning.terrain_drive + norm_tumble * 1.5;
            let driven = soft_saturate(raw * amp, dynamic_drive);

            let centroid = (positions[0][s] + positions[1][s] + positions[2][s]) / 3.0;
            let rel: Vector3<f64> = pos - centroid;
            let azimuth = rel.x.atan2(rel.y) / std::f64::consts::PI;
            let dist = rel.norm() / stats.max_distance;
            let (l, r) = panners[b].process(driven, azimuth, dist);
            left += l;
            right += r;

            if ns > 0.6 && close > 0.5 && si.saturating_sub(last_grain_si) > grain_cooldown {
                last_grain_si = si;
                let grain_len = (0.01 * SR) as usize + (rng.next_f64().abs() * 0.04 * SR) as usize;
                let mut gb = Vec::with_capacity(grain_len);
                let mut gx = scan_x[b];
                let mut gy = scan_y[b];
                let pitch_var = 0.8 + rng.next_f64().abs() * 0.4;
                for gi in 0..grain_len {
                    gx += scan_speed * pitch_var;
                    gy += scan_speed * pitch_var * 0.618;
                    let env = (gi as f64 / grain_len as f64 * std::f64::consts::PI).sin();
                    gb.push(terrain.sample(gx, gy) * env);
                }
                grains.push(Grain {
                    buffer: gb,
                    pos: 0.0,
                    speed: pitch_var,
                    amp: close * 0.15,
                    pan: (azimuth * 0.5 + 0.5).clamp(0.0, 1.0),
                    remaining: grain_len,
                });
            }
        }

        let mut grain_idx = 0;
        while grain_idx < grains.len() {
            let g = &mut grains[grain_idx];
            if g.remaining == 0 {
                grains.swap_remove(grain_idx);
                continue;
            }
            let pi = g.pos as usize;
            if pi < g.buffer.len() {
                let gsamp = g.buffer[pi] * g.amp;
                left += gsamp * (1.0 - g.pan).sqrt();
                right += gsamp * g.pan.sqrt();
            }
            g.pos += g.speed;
            g.remaining -= 1;
            grain_idx += 1;
        }

        let area = triangle_area_at(positions, s);
        let norm_area = (area / stats.max_area).clamp(0.0, 1.0);
        sub_ph += (base_freq * 0.25) / SR;
        let sub = (sub_ph * TAU).sin() * 0.03 * norm_area;
        left += sub;
        right += sub;

        let rot = sm_plane_az * 0.4;
        let cos_r = rot.cos();
        let sin_r = rot.sin();
        let rl = left * cos_r - right * sin_r;
        let rr = left * sin_r + right * cos_r;

        let fade = smoothstep_fade(t, 3.0, 5.0, duration_s);
        buf[si * 2] = rl * fade;
        buf[si * 2 + 1] = rr * fade;
    }

    apply_freeverb(&mut buf, total_samples, tuning.reverb_size, tuning.reverb_damp, tuning.reverb_wet);
    normalize_peak(&mut buf, 0.88);
    buf
}

// ─── Algorithm H: Void Resonance ─────────────────────────────

fn synthesize_void_resonance(
    positions: &[Vec<Vector3<f64>>],
    stats: &OrbitStats,
    ctx: &GalleryAudioContext,
    tuning: &WorldTuning,
    total_samples: usize,
    step_count: usize,
    duration_s: f64,
) -> Vec<f64> {
    let nb = 3.min(positions.len());
    let mut buf = vec![0.0f64; total_samples * 2];

    let freqs: Vec<f64> = (0..nb)
        .map(|b| 55.0 * (2000.0 / ctx.masses[b].max(100.0)))
        .collect();
    let mut strings: Vec<KarplusStrongString> = freqs.iter()
        .map(|&f| KarplusStrongString::new(f * 0.5, tuning.damping * 0.7))
        .collect();

    let mut freezer = SpectralFreezer::new(8);
    let mut ks_mono_buf = vec![0.0f64; total_samples];
    let mut rng = SimpleRng::new(0xF70E_E5E1_C0DE_A1B2_u64.wrapping_mul(
        (ctx.chaos * 1e9 + ctx.equilateralness * 1e6) as u64 | 1,
    ));
    let mut panners: Vec<HrtfPanner> = (0..nb).map(|_| HrtfPanner::new()).collect();

    let close_a = 1.0 - (-1.0 / (0.03 * SR)).exp();
    let mut sm_close = [[0.0f64; 3]; 3];
    let mut prev_close = [[0.0f64; 3]; 3];

    let freeze_interval = (SR * (0.8 + 1.5 * (1.0 - tuning.freeze_density))) as usize;
    let mut last_freeze = 0usize;

    let mut breath_state_l = 0.0f64;
    let mut breath_state_r = 0.0f64;
    let mut prev_normal = plane_normal_at(positions, 0);
    let mut sm_plane_az = 0.0f64;
    let plane_az_a = 1.0 - (-1.0 / (0.3 * SR)).exp();

    for si in 0..total_samples {
        let t = si as f64 / total_samples as f64;
        let sf = t * (step_count.saturating_sub(2)) as f64;
        let step = sf as usize;
        let frac = sf - step as f64;
        let s = step.min(step_count - 1);

        let ang_mom = angular_momentum_at(positions, s);
        let norm_ang = (ang_mom / stats.max_ang_mom).clamp(0.0, 1.0);
        let area = triangle_area_at(positions, s);
        let norm_area = (area / stats.max_area).clamp(0.0, 1.0);

        let cur_normal = plane_normal_at(positions, s);
        let tumble_dot = prev_normal.dot(&cur_normal).clamp(-1.0, 1.0);
        let norm_tumble = (tumble_dot.acos() / stats.max_tumble_rate).clamp(0.0, 1.0);
        prev_normal = cur_normal;
        let plane_az = cur_normal.x.atan2(cur_normal.y);
        sm_plane_az += plane_az_a * (plane_az - sm_plane_az);

        for b in 0..nb {
            for other in 0..nb {
                if other == b { continue; }
                let d = (positions[b][s] - positions[other][s]).norm();
                let close = 1.0 - (d / stats.max_distance).clamp(0.0, 1.0);
                prev_close[b][other] = sm_close[b][other];
                sm_close[b][other] += close_a * (close - sm_close[b][other]);
                if sm_close[b][other] > tuning.pluck_threshold * 1.2
                    && prev_close[b][other] <= tuning.pluck_threshold * 1.2
                {
                    let spd = body_speed(positions, b, step, frac, step_count);
                    let ns = (spd / stats.max_speed).clamp(0.0, 1.0);
                    strings[b].pluck(&mut rng, 0.4 * sm_close[b][other], 0.3 + 0.3 * ns);
                }
            }
        }

        let mut ks_sum = 0.0f64;
        let mut left = 0.0f64;
        let mut right = 0.0f64;

        for b in 0..nb {
            let raw = strings[b].tick() * 0.3;
            ks_sum += raw;

            let pos = positions[b][s];
            let centroid = (positions[0][s] + positions[1][s] + positions[2][s]) / 3.0;
            let rel: Vector3<f64> = pos - centroid;
            let azimuth = rel.x.atan2(rel.y) / std::f64::consts::PI;
            let dist = rel.norm() / stats.max_distance + 1.5;
            let (l, r) = panners[b].process(raw, azimuth, dist);
            left += l;
            right += r;
        }
        ks_mono_buf[si] = ks_sum;

        if si > FREEZE_GRAIN_LEN && si.saturating_sub(last_freeze) > freeze_interval {
            let should_freeze = norm_area > 0.5 || norm_ang > 0.6;
            if should_freeze {
                last_freeze = si;
                freezer.capture(&ks_mono_buf, si, 0.25 + 0.15 * norm_area);
            }
        }

        let freeze_speed = 0.98 + 0.04 * (t * 0.03 * TAU).sin();
        let frozen = freezer.tick(freeze_speed);
        left += frozen * 0.6;
        right += frozen * 0.6;

        let breath_intensity = norm_ang + norm_tumble * 0.5;
        let breath_raw_l = rng.next_f64() * 0.015 * breath_intensity;
        let breath_raw_r = rng.next_f64() * 0.015 * breath_intensity;
        let breath_coeff = 0.05 + 0.15 * norm_area;
        breath_state_l = breath_state_l * (1.0 - breath_coeff) + breath_raw_l * breath_coeff;
        breath_state_r = breath_state_r * (1.0 - breath_coeff) + breath_raw_r * breath_coeff;
        left += breath_state_l;
        right += breath_state_r;

        let rot = sm_plane_az * 0.6;
        let cos_r = rot.cos();
        let sin_r = rot.sin();
        let rl = left * cos_r - right * sin_r;
        let rr = left * sin_r + right * cos_r;

        let fade = smoothstep_fade(t, 5.0, 8.0, duration_s);
        buf[si * 2] = rl * fade;
        buf[si * 2 + 1] = rr * fade;
    }

    apply_freeverb(&mut buf, total_samples, 0.95, 0.15, 0.55);
    normalize_peak(&mut buf, 0.88);
    buf
}

// ─── Legacy: Algorithm B — Crystal Resonance ─────────────────

struct BellVoice {
    carrier_ph: f64,
    mod_ph: f64,
    carrier_freq: f64,
    mod_ratio: f64,
    start: usize,
    amplitude: f64,
    pan: f64,
    decay: f64,
    mod_decay: f64,
    ghost_oct_ph: f64,
    ghost_5th_ph: f64,
}

struct Resonator {
    y1: f64,
    y2: f64,
    a1: f64,
    a2: f64,
    gain: f64,
}

impl Resonator {
    fn new(center: f64, bw: f64) -> Self {
        let r = (-std::f64::consts::PI * bw / SR).exp();
        let theta = TAU * center / SR;
        Self { y1: 0.0, y2: 0.0, a1: -2.0 * r * theta.cos(), a2: r * r, gain: 1.0 - r * r }
    }
    fn tick(&mut self, x: f64) -> f64 {
        let y = self.gain * x - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

// ─── Legacy Infrastructure ───────────────────────────────────

struct VelocityStats {
    max_speed: f64,
    max_distance: f64,
}

fn compute_velocity_stats(positions: &[Vec<Vector3<f64>>]) -> VelocityStats {
    let mut max_speed = 0.0f64;
    let mut max_distance = 0.0f64;
    let steps = positions[0].len();
    for step in 1..steps {
        for body_pos in positions.iter().take(3) {
            let vel = (body_pos[step] - body_pos[step - 1]).norm();
            max_speed = max_speed.max(vel);
        }
        for i in 0..3.min(positions.len()) {
            for j in (i + 1)..3.min(positions.len()) {
                let dist = (positions[i][step] - positions[j][step]).norm();
                max_distance = max_distance.max(dist);
            }
        }
    }
    VelocityStats { max_speed: max_speed.max(1e-10), max_distance: max_distance.max(1e-10) }
}

struct ReverbParams { delay1_ms: f64, delay2_ms: f64, feedback: f64, wet_mix: f64, pre_delay_ms: f64 }

fn apply_stereo_reverb(buf: &mut [f64], n: usize, p: &ReverbParams) {
    let d1 = (p.delay1_ms * SR / 1000.0) as usize;
    let d2 = (p.delay2_ms * SR / 1000.0) as usize;
    let pd = ((p.pre_delay_ms * SR / 1000.0) as usize).max(1);
    let mut bl1 = vec![0.0f64; d1]; let mut br1 = vec![0.0f64; d1];
    let mut bl2 = vec![0.0f64; d2]; let mut br2 = vec![0.0f64; d2];
    let mut pl = vec![0.0f64; pd]; let mut pr = vec![0.0f64; pd];
    let (mut i1, mut i2, mut ip) = (0, 0, 0);
    for i in 0..n {
        let li = i * 2; let ri = i * 2 + 1;
        let dl = pl[ip]; let dr = pr[ip];
        pl[ip] = buf[li]; pr[ip] = buf[ri];
        ip = (ip + 1) % pd;
        let w1l = bl1[i1]; let w1r = br1[i1];
        bl1[i1] = dl + w1r * p.feedback; br1[i1] = dr + w1l * p.feedback;
        i1 = (i1 + 1) % d1;
        let w2l = bl2[i2]; let w2r = br2[i2];
        bl2[i2] = dl * 0.7 + w2r * p.feedback * 0.8; br2[i2] = dr * 0.7 + w2l * p.feedback * 0.8;
        i2 = (i2 + 1) % d2;
        let wl = (w1l + w2l) * 0.5; let wr = (w1r + w2r) * 0.5;
        buf[li] = buf[li] * (1.0 - p.wet_mix) + wl * p.wet_mix;
        buf[ri] = buf[ri] * (1.0 - p.wet_mix) + wr * p.wet_mix;
    }
}

fn normalize_peak(buf: &mut [f64], target: f64) {
    let peak = buf.iter().copied().fold(0.0f64, |a, s| a.max(s.abs()));
    if peak > target {
        let s = target / peak;
        for v in buf.iter_mut() { *v *= s; }
    }
}

fn f64_to_i16(buf: &[f64]) -> Vec<i16> {
    buf.iter().map(|&s| (s.clamp(-1.0, 1.0) * 32000.0) as i16).collect()
}

fn synthesize_crystal_resonance(
    positions: &[Vec<Vector3<f64>>], stats: &VelocityStats,
    total_samples: usize, step_count: usize, duration_s: f64,
) -> Vec<f64> {
    let nb = 3.min(positions.len());
    let mut buf = vec![0.0f64; total_samples * 2];
    let mut bells: Vec<BellVoice> = Vec::new();
    let mut sm_spd = [0.0f64; 3]; let mut prev_spd = [0.0f64; 3];
    let mut last_strike = [0usize; 3]; let mut sub_ph = 0.0f64; let mut trem_ph = 0.0f64;
    let spd_a = 1.0 - (-1.0 / (0.05 * SR)).exp();
    let cooldown = (0.25 * SR) as usize; let threshold = 0.25;
    let mod_ratios = [1.41, 2.76, 3.51];
    for si in 0..total_samples {
        let t = si as f64 / total_samples as f64;
        let sf = t * (step_count.saturating_sub(2)) as f64;
        let step = sf as usize; let frac = sf - step as f64; let s = step.min(step_count - 1);
        let mut avg_close = 0.0f64;
        for b in 0..nb {
            let spd = body_speed(positions, b, step, frac, step_count);
            let ns = (spd / stats.max_speed).clamp(0.0, 1.0);
            prev_spd[b] = sm_spd[b]; sm_spd[b] += spd_a * (ns - sm_spd[b]);
            let close = body_closeness(positions, b, s, nb, stats.max_distance);
            avg_close += close;
            if sm_spd[b] > threshold && prev_spd[b] <= threshold && si.saturating_sub(last_strike[b]) > cooldown {
                last_strike[b] = si;
                let raw = 130.81 + (1046.50 - 130.81) * sm_spd[b];
                let freq = quantize_to_scale(raw, &WHOLE_TONE);
                let min_x = positions[b].iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
                let max_x = positions[b].iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
                let rng = (max_x - min_x).max(1e-10);
                let pan = ((positions[b][s].x - min_x) / rng).clamp(0.0, 1.0);
                bells.push(BellVoice { carrier_ph: 0.0, mod_ph: 0.0, carrier_freq: freq, mod_ratio: mod_ratios[b % 3], start: si, amplitude: 0.12 + close * 0.08, pan, decay: (3.5 + close * 2.0) * SR, mod_decay: 0.8 * SR, ghost_oct_ph: 0.0, ghost_5th_ph: 0.0 });
            }
        }
        avg_close /= nb as f64;
        let mut left = 0.0f64; let mut right = 0.0f64;
        for bell in bells.iter_mut() {
            let age = (si - bell.start) as f64;
            let ae = bell.amplitude * (-age / bell.decay).exp();
            let me = (-age / bell.mod_decay).exp(); let mi = 2.5 * me;
            bell.mod_ph += (bell.carrier_freq * bell.mod_ratio) / SR;
            bell.carrier_ph += bell.carrier_freq / SR;
            let carrier = (bell.carrier_ph * TAU + mi * (bell.mod_ph * TAU).sin()).sin();
            let ge = (-age / (bell.decay * 0.3)).exp() * 0.15;
            bell.ghost_oct_ph += (bell.carrier_freq * 2.0) / SR;
            bell.ghost_5th_ph += (bell.carrier_freq * 1.5) / SR;
            let ghost = (bell.ghost_oct_ph * TAU).sin() * ge + (bell.ghost_5th_ph * TAU).sin() * ge;
            let sample = carrier * ae + ghost;
            left += sample * (1.0 - bell.pan).sqrt(); right += sample * bell.pan.sqrt();
        }
        if si % 4096 == 0 { let now = si; bells.retain(|b| b.amplitude * (-(now as f64 - b.start as f64) / b.decay).exp() >= 1e-4); }
        sub_ph += 55.0 / SR; trem_ph += 0.25 / SR;
        let trem = 0.7 + 0.3 * (trem_ph * TAU).sin();
        let sub = (sub_ph * TAU).sin() * 0.06 * avg_close * trem;
        left += sub; right += sub;
        let fade = smoothstep_fade(t, 3.0, 5.0, duration_s);
        buf[si * 2] = left * fade; buf[si * 2 + 1] = right * fade;
    }
    apply_stereo_reverb(&mut buf, total_samples, &ReverbParams { delay1_ms: 610.0, delay2_ms: 890.0, feedback: 0.60, wet_mix: 0.45, pre_delay_ms: 30.0 });
    normalize_peak(&mut buf, 0.9); buf
}

fn synthesize_orbital_choir(
    positions: &[Vec<Vector3<f64>>], stats: &VelocityStats,
    total_samples: usize, step_count: usize, duration_s: f64,
) -> Vec<f64> {
    let nb = 3.min(positions.len());
    let mut buf = vec![0.0f64; total_samples * 2];
    let ratios = [1.0, 3.0 / 2.0, 5.0 / 4.0];
    let mut filt_ah: Vec<[Resonator; 2]> = (0..nb).map(|_| [Resonator::new(730.0, 90.0), Resonator::new(1090.0, 110.0)]).collect();
    let mut filt_oh: Vec<[Resonator; 2]> = (0..nb).map(|_| [Resonator::new(570.0, 80.0), Resonator::new(840.0, 100.0)]).collect();
    let mut saw_ph = [0.0f64; 3]; let mut vib_ph = [0.0f64; 3]; let mut sm_amp = [0.0f64; 3];
    let mut pedal_ph = 0.0f64; let mut rng = SimpleRng::new(0xCE1E_5714_A1C1_0412);
    let amp_a = 1.0 - (-1.0 / (1.5 * SR)).exp(); let pans: [f64; 3] = [0.3, 0.5, 0.7];
    for si in 0..total_samples {
        let t = si as f64 / total_samples as f64;
        let sf = t * (step_count.saturating_sub(2)) as f64;
        let step = sf as usize; let s = step.min(step_count - 1); let ts = si as f64 / SR;
        let root = 128.5 + 18.5 * (ts * 0.02 * TAU).sin();
        let mut left = 0.0f64; let mut right = 0.0f64;
        for b in 0..nb {
            let base = root * ratios[b]; vib_ph[b] += 4.5 / SR;
            let freq = base + (vib_ph[b] * TAU).sin() * 3.0;
            saw_ph[b] += freq / SR; if saw_ph[b] > 1.0 { saw_ph[b] -= 1.0; }
            let mut saw = 0.0f64;
            for h in 1..=8u32 { let hf = h as f64; if freq * hf > SR * 0.45 { break; } saw += (saw_ph[b] * hf * TAU).sin() / hf; }
            saw *= 0.5;
            let vmix = 0.5 + 0.5 * (ts * 0.08 * TAU).sin();
            let ah = filt_ah[b][0].tick(saw) + filt_ah[b][1].tick(saw);
            let oh = filt_oh[b][0].tick(saw) + filt_oh[b][1].tick(saw);
            let filtered = ah * (1.0 - vmix) + oh * vmix;
            let close = body_closeness(positions, b, s, nb, stats.max_distance);
            let tgt = close.clamp(0.0, 1.0) * 0.20;
            sm_amp[b] += amp_a * (tgt - sm_amp[b]);
            left += filtered * sm_amp[b] * (1.0 - pans[b]).sqrt(); right += filtered * sm_amp[b] * pans[b].sqrt();
        }
        let avg_spd = if step + 1 < step_count { let mut v = 0.0; for pos in positions.iter().take(nb) { v += (pos[(step + 1).min(step_count - 1)] - pos[s]).norm(); } v / nb as f64 } else { 0.0 };
        let ba = (avg_spd / stats.max_speed).clamp(0.0, 1.0) * 0.02;
        left += rng.next_f64() * ba * 0.7; right += rng.next_f64() * ba;
        pedal_ph += (root * 0.5) / SR; left += (pedal_ph * TAU).sin() * 0.04; right += (pedal_ph * TAU).sin() * 0.04;
        let fade = smoothstep_fade(t, 3.0, 5.0, duration_s);
        buf[si * 2] = left * fade; buf[si * 2 + 1] = right * fade;
    }
    apply_stereo_reverb(&mut buf, total_samples, &ReverbParams { delay1_ms: 370.0, delay2_ms: 530.0, feedback: 0.55, wet_mix: 0.40, pre_delay_ms: 80.0 });
    normalize_peak(&mut buf, 0.9); buf
}

// ─── WAV / FFmpeg Infrastructure ─────────────────────────────

fn write_wav(samples: &[i16], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;
    let mut file = std::fs::File::create(path)?;
    file.write_all(b"RIFF")?; file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?; file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&CHANNELS.to_le_bytes())?; file.write_all(&SAMPLE_RATE.to_le_bytes())?;
    let byte_rate = SAMPLE_RATE * CHANNELS as u32 * (BITS_PER_SAMPLE / 8) as u32;
    file.write_all(&byte_rate.to_le_bytes())?;
    let block_align = CHANNELS * (BITS_PER_SAMPLE / 8);
    file.write_all(&block_align.to_le_bytes())?; file.write_all(&BITS_PER_SAMPLE.to_le_bytes())?;
    file.write_all(b"data")?; file.write_all(&data_size.to_le_bytes())?;
    for &sample in samples { file.write_all(&sample.to_le_bytes())?; }
    Ok(())
}

fn mux_audio_video(audio_path: &str, video_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new("ffmpeg").args(["-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", output_path])
        .stdout(Stdio::null()).stderr(Stdio::inherit()).status()?;
    if !status.success() { return Err(format!("ffmpeg mux failed with status {:?}", status).into()); }
    Ok(())
}

// ─── Public API ──────────────────────────────────────────────

#[allow(dead_code)]
pub fn generate_sonification(
    positions: &[Vec<Vector3<f64>>], duration_seconds: f64, video_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating legacy sonification ({:.1}s) for {}...", duration_seconds, video_path);
    let total_samples = (SR * duration_seconds) as usize;
    let step_count = positions[0].len();
    let stats = compute_velocity_stats(positions);
    let sb = synthesize_crystal_resonance(positions, &stats, total_samples, step_count, duration_seconds);
    let sc = synthesize_orbital_choir(positions, &stats, total_samples, step_count, duration_seconds);
    let wb = format!("{}.bells.wav", video_path); let wc = format!("{}.choir.wav", video_path);
    write_wav(&f64_to_i16(&sb), &wb)?; write_wav(&f64_to_i16(&sc), &wc)?;
    let base = video_path.strip_suffix(".mp4").unwrap_or(video_path);
    mux_audio_video(&wb, video_path, &format!("{}_crystal_resonance.mp4", base))?;
    mux_audio_video(&wc, video_path, &format!("{}_orbital_choir.mp4", base))?;
    let tmp = format!("{}.tmp.mp4", video_path);
    mux_audio_video(&wb, video_path, &tmp)?; std::fs::rename(&tmp, video_path)?;
    let _ = std::fs::remove_file(&wb); let _ = std::fs::remove_file(&wc);
    Ok(())
}

pub fn generate_gallery_sonification(
    positions: &[Vec<Vector3<f64>>], duration_seconds: f64, video_path: &str, context: &GalleryAudioContext,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating gallery sonification ({:.1}s) for {}...", duration_seconds, video_path);
    let total_samples = (SR * duration_seconds) as usize;
    let step_count = positions[0].len();
    let orbit_stats = compute_orbit_stats(positions);
    let legacy_stats = compute_velocity_stats(positions);
    let world = select_sonic_world(context.chaos, context.equilateralness);
    let wt = world_tuning(world);
    info!("   Sonic world: {:?}  (chaos={:.4}, equil={:.4})", world, context.chaos, context.equilateralness);

    info!("   Synthesizing: Gravitational Strings (Karplus-Strong waveguide)...");
    let sf = synthesize_gravitational_strings(positions, &orbit_stats, context, &wt, total_samples, step_count, duration_seconds);
    info!("   Synthesizing: Orbital Terrain (wave terrain + granular)...");
    let sg = synthesize_orbital_terrain(positions, &orbit_stats, context, &wt, total_samples, step_count, duration_seconds);
    info!("   Synthesizing: Void Resonance (spectral freeze + breath)...");
    let sh = synthesize_void_resonance(positions, &orbit_stats, context, &wt, total_samples, step_count, duration_seconds);
    info!("   Synthesizing: Crystal Resonance (legacy)...");
    let sb = synthesize_crystal_resonance(positions, &legacy_stats, total_samples, step_count, duration_seconds);
    info!("   Synthesizing: Orbital Choir (legacy)...");
    let sc = synthesize_orbital_choir(positions, &legacy_stats, total_samples, step_count, duration_seconds);

    let wf = format!("{}.strings.wav", video_path);
    let wg = format!("{}.terrain.wav", video_path);
    let wh = format!("{}.void.wav", video_path);
    let wb = format!("{}.bells.wav", video_path);
    let wc = format!("{}.choir.wav", video_path);
    write_wav(&f64_to_i16(&sf), &wf)?;
    write_wav(&f64_to_i16(&sg), &wg)?;
    write_wav(&f64_to_i16(&sh), &wh)?;
    write_wav(&f64_to_i16(&sb), &wb)?;
    write_wav(&f64_to_i16(&sc), &wc)?;

    let base = video_path.strip_suffix(".mp4").unwrap_or(video_path);
    let out_g = format!("{}_orbital_terrain.mp4", base);
    mux_audio_video(&wg, video_path, &out_g)?;
    info!("   Saved => {}", out_g);
    let out_h = format!("{}_void_resonance.mp4", base);
    mux_audio_video(&wh, video_path, &out_h)?;
    info!("   Saved => {}", out_h);
    let out_b = format!("{}_crystal_resonance.mp4", base);
    mux_audio_video(&wb, video_path, &out_b)?;
    info!("   Saved => {}", out_b);
    let out_c = format!("{}_orbital_choir.mp4", base);
    mux_audio_video(&wc, video_path, &out_c)?;
    info!("   Saved => {}", out_c);

    let tmp = format!("{}.tmp.mp4", video_path);
    mux_audio_video(&wf, video_path, &tmp)?;
    std::fs::rename(&tmp, video_path)?;
    info!("   Embedded Gravitational Strings into {}", video_path);

    let _ = std::fs::remove_file(&wf);
    let _ = std::fs::remove_file(&wg);
    let _ = std::fs::remove_file(&wh);
    let _ = std::fs::remove_file(&wb);
    let _ = std::fs::remove_file(&wc);
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_positions() -> Vec<Vec<Vector3<f64>>> {
        vec![
            vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0), Vector3::new(2.0, 1.0, 0.0)],
            vec![Vector3::new(5.0, 0.0, 0.0), Vector3::new(5.0, 1.0, 0.0), Vector3::new(5.0, 2.0, 0.0)],
            vec![Vector3::new(0.0, 5.0, 0.0), Vector3::new(1.0, 5.0, 0.0), Vector3::new(2.0, 5.0, 0.0)],
        ]
    }

    fn gallery_context() -> GalleryAudioContext {
        GalleryAudioContext { chaos: 0.3, equilateralness: 0.6, masses: [200.0, 150.0, 250.0] }
    }

    #[test]
    fn test_smoothstep_fade_shape() {
        assert!(smoothstep_fade(0.0, 3.0, 5.0, 30.0) < 0.01);
        assert!(smoothstep_fade(0.5, 3.0, 5.0, 30.0) > 0.99);
        assert!(smoothstep_fade(1.0, 3.0, 5.0, 30.0) < 0.01);
    }

    #[test]
    fn test_select_sonic_world_quadrants() {
        assert_eq!(select_sonic_world(0.1, 0.3), SonicWorld::Obsidian);
        assert_eq!(select_sonic_world(0.1, 0.7), SonicWorld::Aurora);
        assert_eq!(select_sonic_world(0.6, 0.7), SonicWorld::Meridian);
        assert_eq!(select_sonic_world(0.6, 0.3), SonicWorld::Liminal);
    }

    #[test]
    fn test_soft_saturate_identity_at_low_drive() {
        let v = soft_saturate(0.5, 0.1);
        assert!((v - 0.5).abs() < 0.05, "low drive should be near-linear");
    }

    #[test]
    fn test_soft_saturate_bounds() {
        assert!(soft_saturate(100.0, 5.0) <= 1.01);
        assert!(soft_saturate(-100.0, 5.0) >= -1.01);
    }

    #[test]
    fn test_karplus_strong_produces_sound() {
        let mut rng = SimpleRng::new(42);
        let mut ks = KarplusStrongString::new(220.0, 0.3);
        ks.pluck(&mut rng, 0.8, 0.9);
        let mut peak = 0.0f64;
        for _ in 0..4410 { peak = peak.max(ks.tick().abs()); }
        assert!(peak > 0.01, "KS string should produce audible output after pluck");
    }

    #[test]
    fn test_karplus_strong_decays() {
        let mut rng = SimpleRng::new(42);
        let mut ks = KarplusStrongString::new(220.0, 0.3);
        ks.pluck(&mut rng, 0.8, 0.9);
        let window = 4410;
        let mut prev_peak = f64::MAX;
        for _ in 0..5 {
            let mut peak = 0.0f64;
            for _ in 0..window { peak = peak.max(ks.tick().abs()); }
            assert!(peak < prev_peak + 0.01, "KS string should decay over time");
            prev_peak = peak;
        }
    }

    #[test]
    fn test_karplus_strong_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut ks1 = KarplusStrongString::new(220.0, 0.3);
        ks1.pluck(&mut rng1, 0.8, 0.9);
        let a: Vec<f64> = (0..1000).map(|_| ks1.tick()).collect();

        let mut rng2 = SimpleRng::new(42);
        let mut ks2 = KarplusStrongString::new(220.0, 0.3);
        ks2.pluck(&mut rng2, 0.8, 0.9);
        let b: Vec<f64> = (0..1000).map(|_| ks2.tick()).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn test_wave_terrain_produces_values() {
        let pos = sample_positions();
        let terrain = WaveTerrain::from_positions(&pos, pos[0].len());
        let v = terrain.sample(0.25, 0.75);
        assert!(v.is_finite());
    }

    #[test]
    fn test_wave_terrain_wraps() {
        let pos = sample_positions();
        let terrain = WaveTerrain::from_positions(&pos, pos[0].len());
        let a = terrain.sample(0.1, 0.2);
        let b = terrain.sample(1.1, 1.2);
        assert!((a - b).abs() < 1e-10, "terrain should wrap");
    }

    #[test]
    fn test_freeverb_produces_output() {
        let mut buf = vec![0.0f64; 4410 * 2];
        buf[0] = 1.0; buf[1] = 1.0;
        apply_freeverb(&mut buf, 4410, 0.8, 0.5, 0.4);
        let tail_energy: f64 = buf[4000..].iter().map(|s| s * s).sum();
        assert!(tail_energy > 1e-10, "freeverb should produce reverb tail");
    }

    #[test]
    fn test_hrtf_panner_produces_stereo() {
        let mut panner = HrtfPanner::new();
        let mut sum_l = 0.0f64;
        let mut sum_r = 0.0f64;
        for _ in 0..100 {
            let (l, r) = panner.process(1.0, 0.8, 0.5);
            sum_l += l.abs();
            sum_r += r.abs();
        }
        assert!(sum_l > 0.1, "left channel should have output");
        assert!(sum_r > 0.1, "right channel should have output");
        assert!((sum_l - sum_r).abs() > 0.01, "HRTF should produce L/R difference for off-center source");
    }

    #[test]
    fn test_spectral_freezer_captures_and_plays() {
        let mut freezer = SpectralFreezer::new(4);
        let source: Vec<f64> = (0..4096).map(|i| (i as f64 / 100.0).sin()).collect();
        freezer.capture(&source, 2048, 0.5);
        let mut out = 0.0f64;
        for _ in 0..1000 { out += freezer.tick(1.0).abs(); }
        assert!(out > 0.01, "freezer should produce output after capture");
    }

    #[test]
    fn test_gravitational_strings_length_and_determinism() {
        let pos = sample_positions();
        let stats = compute_orbit_stats(&pos);
        let ctx = gallery_context();
        let wt = world_tuning(SonicWorld::Aurora);
        let a = synthesize_gravitational_strings(&pos, &stats, &ctx, &wt, 441, pos[0].len(), 0.01);
        let b = synthesize_gravitational_strings(&pos, &stats, &ctx, &wt, 441, pos[0].len(), 0.01);
        assert_eq!(a.len(), 441 * 2);
        assert_eq!(a, b, "gravitational strings must be deterministic");
    }

    #[test]
    fn test_orbital_terrain_length_and_determinism() {
        let pos = sample_positions();
        let stats = compute_orbit_stats(&pos);
        let ctx = gallery_context();
        let wt = world_tuning(SonicWorld::Meridian);
        let a = synthesize_orbital_terrain(&pos, &stats, &ctx, &wt, 441, pos[0].len(), 0.01);
        let b = synthesize_orbital_terrain(&pos, &stats, &ctx, &wt, 441, pos[0].len(), 0.01);
        assert_eq!(a.len(), 441 * 2);
        assert_eq!(a, b, "orbital terrain must be deterministic");
    }

    #[test]
    fn test_void_resonance_length_and_determinism() {
        let pos = sample_positions();
        let stats = compute_orbit_stats(&pos);
        let ctx = gallery_context();
        let wt = world_tuning(SonicWorld::Liminal);
        let a = synthesize_void_resonance(&pos, &stats, &ctx, &wt, 441, pos[0].len(), 0.01);
        let b = synthesize_void_resonance(&pos, &stats, &ctx, &wt, 441, pos[0].len(), 0.01);
        assert_eq!(a.len(), 441 * 2);
        assert_eq!(a, b, "void resonance must be deterministic");
    }

    #[test]
    fn test_gravitational_strings_samples_bounded() {
        let pos = sample_positions();
        let stats = compute_orbit_stats(&pos);
        let ctx = gallery_context();
        let wt = world_tuning(SonicWorld::Obsidian);
        let buf = synthesize_gravitational_strings(&pos, &stats, &ctx, &wt, 4410, pos[0].len(), 0.1);
        for (i, &s) in f64_to_i16(&buf).iter().enumerate() {
            assert!((-32000..=32000).contains(&s), "sample {i} out of range: {s}");
        }
    }

    #[test]
    fn test_different_worlds_produce_different_strings() {
        let pos = sample_positions();
        let stats = compute_orbit_stats(&pos);
        let ctx = gallery_context();
        let a = synthesize_gravitational_strings(&pos, &stats, &ctx, &world_tuning(SonicWorld::Aurora), 441, pos[0].len(), 0.01);
        let b = synthesize_gravitational_strings(&pos, &stats, &ctx, &world_tuning(SonicWorld::Obsidian), 441, pos[0].len(), 0.01);
        assert_ne!(a, b, "different worlds must produce different output");
    }

    #[test]
    fn test_compute_orbit_stats_non_degenerate() {
        let pos = sample_positions();
        let stats = compute_orbit_stats(&pos);
        assert!(stats.max_speed > 0.0);
        assert!(stats.max_distance > 0.0);
        assert!(stats.max_area > 0.0);
    }

    #[test]
    fn test_triangle_area_positive() {
        let pos = sample_positions();
        assert!(triangle_area_at(&pos, 0) > 0.0);
    }

    #[test]
    fn test_angular_momentum_nonnegative() {
        let pos = sample_positions();
        assert!(angular_momentum_at(&pos, 1) >= 0.0);
    }

    #[test]
    fn test_crystal_resonance_length() {
        let pos = sample_positions();
        let stats = compute_velocity_stats(&pos);
        let buf = synthesize_crystal_resonance(&pos, &stats, 441, pos[0].len(), 0.01);
        assert_eq!(buf.len(), 441 * 2);
    }

    #[test]
    fn test_orbital_choir_length_and_determinism() {
        let pos = sample_positions();
        let stats = compute_velocity_stats(&pos);
        let a = synthesize_orbital_choir(&pos, &stats, 441, pos[0].len(), 0.01);
        let b = synthesize_orbital_choir(&pos, &stats, 441, pos[0].len(), 0.01);
        assert_eq!(a.len(), 441 * 2);
        assert_eq!(a, b);
    }

    // ─── Plane Normal Tests ──────────────────────────────────

    #[test]
    fn test_plane_normal_unit_length() {
        let pos = sample_positions();
        let n = plane_normal_at(&pos, 0);
        assert!((n.norm() - 1.0).abs() < 1e-10, "plane normal should be unit length");
    }

    #[test]
    fn test_plane_normal_perpendicular_to_edges() {
        let pos = sample_positions();
        let n = plane_normal_at(&pos, 0);
        let e1 = pos[1][0] - pos[0][0];
        let e2 = pos[2][0] - pos[0][0];
        assert!(n.dot(&e1).abs() < 1e-10, "normal should be perpendicular to edge1");
        assert!(n.dot(&e2).abs() < 1e-10, "normal should be perpendicular to edge2");
    }

    #[test]
    fn test_plane_normal_deterministic() {
        let pos = sample_positions();
        let a = plane_normal_at(&pos, 1);
        let b = plane_normal_at(&pos, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn test_orbit_stats_has_tumble_rate() {
        let pos = sample_positions();
        let stats = compute_orbit_stats(&pos);
        assert!(stats.max_tumble_rate > 0.0, "tumble rate should be positive for moving bodies");
    }

    #[test]
    fn test_stereo_rotation_preserves_energy() {
        let left = 0.6f64;
        let right = 0.4f64;
        let energy_before = left * left + right * right;
        let angle = 0.3f64;
        let cos_r = angle.cos();
        let sin_r = angle.sin();
        let rl = left * cos_r - right * sin_r;
        let rr = left * sin_r + right * cos_r;
        let energy_after = rl * rl + rr * rr;
        assert!(
            (energy_before - energy_after).abs() < 1e-10,
            "rotation should preserve total energy"
        );
    }

    #[test]
    fn test_stereo_rotation_zero_angle_is_identity() {
        let left = 0.7f64;
        let right = 0.3f64;
        let angle = 0.0f64;
        let cos_r = angle.cos();
        let sin_r = angle.sin();
        let rl = left * cos_r - right * sin_r;
        let rr = left * sin_r + right * cos_r;
        assert!((rl - left).abs() < 1e-14);
        assert!((rr - right).abs() < 1e-14);
    }
}
