//! Sonification — gallery-grade audio synthesis from three-body simulation data.
//!
//! Five synthesis algorithms transform orbital mechanics into sound:
//!
//! **Gallery-grade (default):**
//! - **Gallery Material**: Spectral-mirror modal synthesis — 16 physical resonators
//!   (mirroring the visual pipeline's 16 wavelength bins) excited by orbital events,
//!   tuned per-seed via four curated Sonic Worlds (Obsidian, Aurora, Meridian, Liminal),
//!   with depth-aware 3D spatialization from body positions.
//! - **Cathedral Orbit**: Mass-ratio-tuned harmonic cloud with angular-momentum bass drone,
//!   triangle-area-driven harmonic density, self-listening spectral feedback, and deep
//!   algorithmic reverb.
//!
//! **Legacy:**
//! - **Celestial Pad**: Ethereal ambient wash with pentatonic harmony and chorus
//! - **Crystal Resonance**: FM bell tones triggered by gravitational slingshots
//! - **Orbital Choir**: Formant-filtered choral texture with just intonation

use nalgebra::Vector3;
use std::io::Write;
use std::process::{Command, Stdio};
use tracing::info;

const SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 2;
const BITS_PER_SAMPLE: u16 = 16;
const SR: f64 = SAMPLE_RATE as f64;
const TAU: f64 = std::f64::consts::TAU;

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

struct ReverbParams {
    delay1_ms: f64,
    delay2_ms: f64,
    feedback: f64,
    wet_mix: f64,
    pre_delay_ms: f64,
}

fn apply_stereo_reverb(buf: &mut [f64], n: usize, p: &ReverbParams) {
    let d1 = (p.delay1_ms * SR / 1000.0) as usize;
    let d2 = (p.delay2_ms * SR / 1000.0) as usize;
    let pd = ((p.pre_delay_ms * SR / 1000.0) as usize).max(1);

    let mut bl1 = vec![0.0f64; d1];
    let mut br1 = vec![0.0f64; d1];
    let mut bl2 = vec![0.0f64; d2];
    let mut br2 = vec![0.0f64; d2];
    let mut pl = vec![0.0f64; pd];
    let mut pr = vec![0.0f64; pd];
    let (mut i1, mut i2, mut ip) = (0, 0, 0);

    for i in 0..n {
        let li = i * 2;
        let ri = i * 2 + 1;
        let dl = pl[ip];
        let dr = pr[ip];
        pl[ip] = buf[li];
        pr[ip] = buf[ri];
        ip = (ip + 1) % pd;

        let w1l = bl1[i1];
        let w1r = br1[i1];
        bl1[i1] = dl + w1r * p.feedback;
        br1[i1] = dr + w1l * p.feedback;
        i1 = (i1 + 1) % d1;

        let w2l = bl2[i2];
        let w2r = br2[i2];
        bl2[i2] = dl * 0.7 + w2r * p.feedback * 0.8;
        br2[i2] = dr * 0.7 + w2l * p.feedback * 0.8;
        i2 = (i2 + 1) % d2;

        let wl = (w1l + w2l) * 0.5;
        let wr = (w1r + w2r) * 0.5;
        buf[li] = buf[li] * (1.0 - p.wet_mix) + wl * p.wet_mix;
        buf[ri] = buf[ri] * (1.0 - p.wet_mix) + wr * p.wet_mix;
    }
}

fn normalize_peak(buf: &mut [f64], target: f64) {
    let peak = buf.iter().copied().fold(0.0f64, |a, s| a.max(s.abs()));
    if peak > target {
        let s = target / peak;
        for v in buf.iter_mut() {
            *v *= s;
        }
    }
}

fn f64_to_i16(buf: &[f64]) -> Vec<i16> {
    buf.iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32000.0) as i16)
        .collect()
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

// ─── Algorithm A: Celestial Pad ──────────────────────────────

fn synthesize_celestial_pad(
    positions: &[Vec<Vector3<f64>>],
    stats: &VelocityStats,
    total_samples: usize,
    step_count: usize,
    duration_s: f64,
) -> Vec<f64> {
    let nb = 3.min(positions.len());
    let mut buf = vec![0.0f64; total_samples * 2];

    let mut phases = [[0.0f64; 3]; 3];
    let mut sm_freq = [220.0f64; 3];
    let mut sm_amp = [0.0f64; 3];
    let mut sub_ph = 0.0f64;
    let mut shimmer_ph = 0.0f64;

    let freq_a = 1.0 - (-1.0 / (0.5 * SR)).exp();
    let att_a = 1.0 - (-1.0 / (1.0 * SR)).exp();
    let rel_a = 1.0 - (-1.0 / (2.0 * SR)).exp();
    let detune = [-0.7, 0.0, 0.7];
    let pan_lfo = [0.05, 0.073, 0.091];

    for si in 0..total_samples {
        let t = si as f64 / total_samples as f64;
        let sf = t * (step_count.saturating_sub(2)) as f64;
        let step = sf as usize;
        let frac = sf - step as f64;
        let s = step.min(step_count - 1);

        let mut left = 0.0f64;
        let mut right = 0.0f64;
        let mut avg_close = 0.0f64;

        for b in 0..nb {
            let spd = body_speed(positions, b, step, frac, step_count);
            let ns = (spd / stats.max_speed).clamp(0.0, 1.0);

            let raw = 110.0 + 770.0 * ns;
            let tgt = quantize_to_scale(raw, &PENTATONIC);
            sm_freq[b] += freq_a * (tgt - sm_freq[b]);

            let close = body_closeness(positions, b, s, nb, stats.max_distance);
            avg_close += close;
            let tgt_amp = close.clamp(0.0, 1.0) * 0.18;
            let alpha = if tgt_amp > sm_amp[b] { att_a } else { rel_a };
            sm_amp[b] += alpha * (tgt_amp - sm_amp[b]);

            let freq = sm_freq[b];
            let mut wave = 0.0f64;
            for (c, &dt) in detune.iter().enumerate() {
                phases[b][c] += (freq + dt) / SR;
                wave += (phases[b][c] * TAU).sin();
            }
            wave /= 3.0;

            let pan = 0.5 + 0.3 * (si as f64 / SR * pan_lfo[b] * TAU).sin();
            left += wave * sm_amp[b] * (1.0 - pan).sqrt();
            right += wave * sm_amp[b] * pan.sqrt();
        }
        avg_close /= nb as f64;

        sub_ph += 55.0 / SR;
        let sub = (sub_ph * TAU).sin() * 0.08 * (0.3 + 0.7 * avg_close);
        left += sub;
        right += sub;

        shimmer_ph += (sm_freq[0] * 2.0) / SR;
        let shmod = (si as f64 / SR * 0.3 * TAU).sin() * 0.5;
        let shim = (shimmer_ph * TAU + shmod).sin() * 0.025 * sm_amp[0];
        left += shim * 0.7;
        right += shim;

        let fade = smoothstep_fade(t, 3.0, 5.0, duration_s);
        buf[si * 2] = left * fade;
        buf[si * 2 + 1] = right * fade;
    }

    apply_stereo_reverb(
        &mut buf,
        total_samples,
        &ReverbParams {
            delay1_ms: 370.0,
            delay2_ms: 530.0,
            feedback: 0.55,
            wet_mix: 0.40,
            pre_delay_ms: 20.0,
        },
    );
    normalize_peak(&mut buf, 0.9);
    buf
}

// ─── Algorithm B: Crystal Resonance ──────────────────────────

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

fn synthesize_crystal_resonance(
    positions: &[Vec<Vector3<f64>>],
    stats: &VelocityStats,
    total_samples: usize,
    step_count: usize,
    duration_s: f64,
) -> Vec<f64> {
    let nb = 3.min(positions.len());
    let mut buf = vec![0.0f64; total_samples * 2];

    let mut bells: Vec<BellVoice> = Vec::new();
    let mut sm_spd = [0.0f64; 3];
    let mut prev_spd = [0.0f64; 3];
    let mut last_strike = [0usize; 3];
    let mut sub_ph = 0.0f64;
    let mut trem_ph = 0.0f64;

    let spd_a = 1.0 - (-1.0 / (0.05 * SR)).exp();
    let cooldown = (0.25 * SR) as usize;
    let threshold = 0.25;
    let mod_ratios = [1.41, 2.76, 3.51];

    for si in 0..total_samples {
        let t = si as f64 / total_samples as f64;
        let sf = t * (step_count.saturating_sub(2)) as f64;
        let step = sf as usize;
        let frac = sf - step as f64;
        let s = step.min(step_count - 1);

        let mut avg_close = 0.0f64;

        for b in 0..nb {
            let spd = body_speed(positions, b, step, frac, step_count);
            let ns = (spd / stats.max_speed).clamp(0.0, 1.0);

            prev_spd[b] = sm_spd[b];
            sm_spd[b] += spd_a * (ns - sm_spd[b]);

            let close = body_closeness(positions, b, s, nb, stats.max_distance);
            avg_close += close;

            if sm_spd[b] > threshold
                && prev_spd[b] <= threshold
                && si.saturating_sub(last_strike[b]) > cooldown
            {
                last_strike[b] = si;
                let raw = 130.81 + (1046.50 - 130.81) * sm_spd[b];
                let freq = quantize_to_scale(raw, &WHOLE_TONE);

                let min_x = positions[b].iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
                let max_x = positions[b]
                    .iter()
                    .map(|p| p.x)
                    .fold(f64::NEG_INFINITY, f64::max);
                let rng = (max_x - min_x).max(1e-10);
                let pan = ((positions[b][s].x - min_x) / rng).clamp(0.0, 1.0);

                bells.push(BellVoice {
                    carrier_ph: 0.0,
                    mod_ph: 0.0,
                    carrier_freq: freq,
                    mod_ratio: mod_ratios[b % 3],
                    start: si,
                    amplitude: 0.12 + close * 0.08,
                    pan,
                    decay: (3.5 + close * 2.0) * SR,
                    mod_decay: 0.8 * SR,
                    ghost_oct_ph: 0.0,
                    ghost_5th_ph: 0.0,
                });
            }
        }
        avg_close /= nb as f64;

        let mut left = 0.0f64;
        let mut right = 0.0f64;

        for bell in bells.iter_mut() {
            let age = (si - bell.start) as f64;
            let ae = bell.amplitude * (-age / bell.decay).exp();
            let me = (-age / bell.mod_decay).exp();
            let mi = 2.5 * me;

            bell.mod_ph += (bell.carrier_freq * bell.mod_ratio) / SR;
            bell.carrier_ph += bell.carrier_freq / SR;
            let carrier = (bell.carrier_ph * TAU + mi * (bell.mod_ph * TAU).sin()).sin();

            let ge = (-age / (bell.decay * 0.3)).exp() * 0.15;
            bell.ghost_oct_ph += (bell.carrier_freq * 2.0) / SR;
            bell.ghost_5th_ph += (bell.carrier_freq * 1.5) / SR;
            let ghost = (bell.ghost_oct_ph * TAU).sin() * ge
                + (bell.ghost_5th_ph * TAU).sin() * ge;

            let sample = carrier * ae + ghost;
            left += sample * (1.0 - bell.pan).sqrt();
            right += sample * bell.pan.sqrt();
        }

        if si % 4096 == 0 {
            let now = si;
            bells.retain(|b| {
                b.amplitude * (-(now as f64 - b.start as f64) / b.decay).exp() >= 1e-4
            });
        }

        sub_ph += 55.0 / SR;
        trem_ph += 0.25 / SR;
        let trem = 0.7 + 0.3 * (trem_ph * TAU).sin();
        let sub = (sub_ph * TAU).sin() * 0.06 * avg_close * trem;
        left += sub;
        right += sub;

        let fade = smoothstep_fade(t, 3.0, 5.0, duration_s);
        buf[si * 2] = left * fade;
        buf[si * 2 + 1] = right * fade;
    }

    apply_stereo_reverb(
        &mut buf,
        total_samples,
        &ReverbParams {
            delay1_ms: 610.0,
            delay2_ms: 890.0,
            feedback: 0.60,
            wet_mix: 0.45,
            pre_delay_ms: 30.0,
        },
    );
    normalize_peak(&mut buf, 0.9);
    buf
}

// ─── Algorithm C: Orbital Choir ──────────────────────────────

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
        Self {
            y1: 0.0,
            y2: 0.0,
            a1: -2.0 * r * theta.cos(),
            a2: r * r,
            gain: 1.0 - r * r,
        }
    }
    fn tick(&mut self, x: f64) -> f64 {
        let y = self.gain * x - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

fn synthesize_orbital_choir(
    positions: &[Vec<Vector3<f64>>],
    stats: &VelocityStats,
    total_samples: usize,
    step_count: usize,
    duration_s: f64,
) -> Vec<f64> {
    let nb = 3.min(positions.len());
    let mut buf = vec![0.0f64; total_samples * 2];

    let ratios = [1.0, 3.0 / 2.0, 5.0 / 4.0];

    let mut filt_ah: Vec<[Resonator; 2]> = (0..nb)
        .map(|_| [Resonator::new(730.0, 90.0), Resonator::new(1090.0, 110.0)])
        .collect();
    let mut filt_oh: Vec<[Resonator; 2]> = (0..nb)
        .map(|_| [Resonator::new(570.0, 80.0), Resonator::new(840.0, 100.0)])
        .collect();

    let mut saw_ph = [0.0f64; 3];
    let mut vib_ph = [0.0f64; 3];
    let mut sm_amp = [0.0f64; 3];
    let mut pedal_ph = 0.0f64;
    let mut rng = SimpleRng::new(0xCE1E_5714_A1C1_0412);

    let amp_a = 1.0 - (-1.0 / (1.5 * SR)).exp();
    let pans: [f64; 3] = [0.3, 0.5, 0.7];

    for si in 0..total_samples {
        let t = si as f64 / total_samples as f64;
        let sf = t * (step_count.saturating_sub(2)) as f64;
        let step = sf as usize;
        let s = step.min(step_count - 1);
        let ts = si as f64 / SR;

        let root = 128.5 + 18.5 * (ts * 0.02 * TAU).sin();

        let mut left = 0.0f64;
        let mut right = 0.0f64;

        for b in 0..nb {
            let base = root * ratios[b];
            vib_ph[b] += 4.5 / SR;
            let freq = base + (vib_ph[b] * TAU).sin() * 3.0;

            saw_ph[b] += freq / SR;
            if saw_ph[b] > 1.0 {
                saw_ph[b] -= 1.0;
            }
            let mut saw = 0.0f64;
            for h in 1..=8u32 {
                let hf = h as f64;
                if freq * hf > SR * 0.45 {
                    break;
                }
                saw += (saw_ph[b] * hf * TAU).sin() / hf;
            }
            saw *= 0.5;

            let vmix = 0.5 + 0.5 * (ts * 0.08 * TAU).sin();
            let ah = filt_ah[b][0].tick(saw) + filt_ah[b][1].tick(saw);
            let oh = filt_oh[b][0].tick(saw) + filt_oh[b][1].tick(saw);
            let filtered = ah * (1.0 - vmix) + oh * vmix;

            let close = body_closeness(positions, b, s, nb, stats.max_distance);
            let tgt = close.clamp(0.0, 1.0) * 0.20;
            sm_amp[b] += amp_a * (tgt - sm_amp[b]);

            left += filtered * sm_amp[b] * (1.0 - pans[b]).sqrt();
            right += filtered * sm_amp[b] * pans[b].sqrt();
        }

        let avg_spd = if step + 1 < step_count {
            let mut v = 0.0;
            for pos in positions.iter().take(nb) {
                v += (pos[(step + 1).min(step_count - 1)] - pos[s]).norm();
            }
            v / nb as f64
        } else {
            0.0
        };
        let ba = (avg_spd / stats.max_speed).clamp(0.0, 1.0) * 0.02;
        left += rng.next_f64() * ba * 0.7;
        right += rng.next_f64() * ba;

        pedal_ph += (root * 0.5) / SR;
        left += (pedal_ph * TAU).sin() * 0.04;
        right += (pedal_ph * TAU).sin() * 0.04;

        let fade = smoothstep_fade(t, 3.0, 5.0, duration_s);
        buf[si * 2] = left * fade;
        buf[si * 2 + 1] = right * fade;
    }

    apply_stereo_reverb(
        &mut buf,
        total_samples,
        &ReverbParams {
            delay1_ms: 370.0,
            delay2_ms: 530.0,
            feedback: 0.55,
            wet_mix: 0.40,
            pre_delay_ms: 80.0,
        },
    );
    normalize_peak(&mut buf, 0.9);
    buf
}

// ─── Gallery Audio Context ───────────────────────────────────

/// Rich context for gallery-grade audio synthesis, carrying orbit character
/// metrics that the original algorithms ignored.
pub struct GalleryAudioContext {
    /// Non-chaoticness score (smaller = more chaotic).
    pub chaos: f64,
    /// Equilateralness score (0..1, higher = more balanced triangle).
    pub equilateralness: f64,
    /// Body masses `[body0, body1, body2]`.
    pub masses: [f64; 3],
}

// ─── Sonic Worlds ────────────────────────────────────────────

/// Sonic world — a curated tuning/material/spatial family selected by orbit character.
///
/// Each world defines a distinct harmonic language so that different seeds
/// are audibly distinguishable and collectible, analogous to visual rarity
/// tiers. The world is deterministic: same chaos + equilateralness always
/// maps to the same world.
///
/// | World | Character | Tuning basis | Orbit profile |
/// |-------|-----------|--------------|---------------|
/// | **Obsidian** | Dark, metallic, angular | Inharmonic partials (sqrt-spaced) | High chaos, low equilateralness |
/// | **Aurora** | Bright, crystalline, shimmering | Near-harmonic series (stretched octaves) | High chaos, high equilateralness |
/// | **Meridian** | Warm, resonant, vocal | Just intonation extended | Low chaos, high equilateralness |
/// | **Liminal** | Ethereal, spectral, suspended | Golden-ratio-based partials | Low chaos, low equilateralness |
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
            Self::Obsidian => "Dark metallic resonances with inharmonic partial series",
            Self::Aurora => "Crystalline shimmer with near-harmonic stretched overtones",
            Self::Meridian => "Warm vocal resonance with just-intonation intervals",
            Self::Liminal => "Ethereal spectral textures with golden-ratio partials",
        }
    }
}

struct WorldParams {
    partial_ratios: [f64; 16],
    fundamental_range: (f64, f64),
    decay_base_s: f64,
    inharmonicity: f64,
    brightness_bias: f64,
    resonator_bw_scale: f64,
    reverb: ReverbParams,
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

fn world_params(world: SonicWorld) -> WorldParams {
    match world {
        SonicWorld::Obsidian => WorldParams {
            partial_ratios: [
                1.0, 1.414, 2.091, 2.828, 3.583, 4.472, 5.236, 6.123, 7.071,
                8.031, 9.138, 10.28, 11.35, 12.49, 13.72, 14.87,
            ],
            fundamental_range: (55.0, 90.0),
            decay_base_s: 2.5,
            inharmonicity: 0.8,
            brightness_bias: 0.3,
            resonator_bw_scale: 1.8,
            reverb: ReverbParams {
                delay1_ms: 430.0,
                delay2_ms: 670.0,
                feedback: 0.48,
                wet_mix: 0.35,
                pre_delay_ms: 40.0,
            },
        },
        SonicWorld::Aurora => WorldParams {
            partial_ratios: [
                1.0, 2.01, 3.0, 4.02, 5.0, 6.01, 7.0, 8.03, 9.0, 10.02,
                11.0, 12.01, 13.0, 14.02, 15.0, 16.01,
            ],
            fundamental_range: (65.0, 110.0),
            decay_base_s: 5.0,
            inharmonicity: 0.1,
            brightness_bias: 0.7,
            resonator_bw_scale: 0.8,
            reverb: ReverbParams {
                delay1_ms: 510.0,
                delay2_ms: 790.0,
                feedback: 0.58,
                wet_mix: 0.42,
                pre_delay_ms: 50.0,
            },
        },
        SonicWorld::Meridian => WorldParams {
            partial_ratios: [
                1.0, 1.125, 1.25, 1.333, 1.5, 1.667, 1.875, 2.0, 2.25, 2.5,
                2.667, 3.0, 3.333, 3.75, 4.0, 4.5,
            ],
            fundamental_range: (70.0, 120.0),
            decay_base_s: 4.0,
            inharmonicity: 0.0,
            brightness_bias: 0.5,
            resonator_bw_scale: 1.0,
            reverb: ReverbParams {
                delay1_ms: 470.0,
                delay2_ms: 730.0,
                feedback: 0.52,
                wet_mix: 0.38,
                pre_delay_ms: 60.0,
            },
        },
        SonicWorld::Liminal => WorldParams {
            partial_ratios: [
                1.0, 1.618, 2.618, 3.236, 4.236, 5.854, 6.854, 8.472, 9.472,
                11.09, 12.09, 13.708, 15.326, 16.944, 18.562, 20.18,
            ],
            fundamental_range: (45.0, 75.0),
            decay_base_s: 6.0,
            inharmonicity: 0.5,
            brightness_bias: 0.2,
            resonator_bw_scale: 0.6,
            reverb: ReverbParams {
                delay1_ms: 610.0,
                delay2_ms: 970.0,
                feedback: 0.62,
                wet_mix: 0.48,
                pre_delay_ms: 90.0,
            },
        },
    }
}

// ─── Extended Orbit Analysis ─────────────────────────────────

#[allow(dead_code)]
struct OrbitStats {
    max_speed: f64,
    max_distance: f64,
    min_distance: f64,
    max_area: f64,
    mean_perimeter: f64,
    max_accel: f64,
    max_ang_mom: f64,
}

fn compute_orbit_stats(positions: &[Vec<Vector3<f64>>]) -> OrbitStats {
    let nb = 3.min(positions.len());
    let steps = positions[0].len();
    let mut max_speed = 0.0f64;
    let mut max_distance = 0.0f64;
    let mut min_distance = f64::MAX;
    let mut max_area = 0.0f64;
    let mut sum_perim = 0.0f64;
    let mut max_accel = 0.0f64;
    let mut max_ang_mom = 0.0f64;
    let mut prev_vel = [Vector3::zeros(); 3];

    for step in 1..steps {
        let mut speeds = [0.0f64; 3];
        for b in 0..nb {
            let vel = positions[b][step] - positions[b][step - 1];
            speeds[b] = vel.norm();
            max_speed = max_speed.max(speeds[b]);

            if step > 1 {
                let accel = (vel - prev_vel[b]).norm();
                max_accel = max_accel.max(accel);
            }
            prev_vel[b] = vel;
        }

        for i in 0..nb {
            for j in (i + 1)..nb {
                let d = (positions[i][step] - positions[j][step]).norm();
                max_distance = max_distance.max(d);
                min_distance = min_distance.min(d);
            }
        }

        if nb >= 3 {
            let a = (positions[0][step] - positions[1][step]).norm();
            let b_len = (positions[1][step] - positions[2][step]).norm();
            let c = (positions[2][step] - positions[0][step]).norm();
            sum_perim += a + b_len + c;
            let s = (a + b_len + c) * 0.5;
            let area_sq = (s * (s - a) * (s - b_len) * (s - c)).max(0.0);
            max_area = max_area.max(area_sq.sqrt());

            let cm = (positions[0][step] + positions[1][step] + positions[2][step])
                / 3.0;
            let mut ang = Vector3::zeros();
            for body_pos in positions.iter().take(nb) {
                let r: Vector3<f64> = body_pos[step] - cm;
                let v = body_pos[step] - body_pos[step.saturating_sub(1)];
                ang += r.cross(&v);
            }
            max_ang_mom = max_ang_mom.max(ang.norm());
        }
    }

    OrbitStats {
        max_speed: max_speed.max(1e-10),
        max_distance: max_distance.max(1e-10),
        min_distance: if min_distance == f64::MAX { 1.0 } else { min_distance },
        max_area: max_area.max(1e-10),
        mean_perimeter: sum_perim / (steps.max(1) as f64),
        max_accel: max_accel.max(1e-10),
        max_ang_mom: max_ang_mom.max(1e-10),
    }
}

fn triangle_area_at(positions: &[Vec<Vector3<f64>>], step: usize) -> f64 {
    let a = (positions[0][step] - positions[1][step]).norm();
    let b = (positions[1][step] - positions[2][step]).norm();
    let c = (positions[2][step] - positions[0][step]).norm();
    let s = (a + b + c) * 0.5;
    (s * (s - a) * (s - b) * (s - c)).max(0.0).sqrt()
}

fn angular_momentum_at(positions: &[Vec<Vector3<f64>>], step: usize) -> f64 {
    let nb = 3.min(positions.len());
    let cm = (positions[0][step] + positions[1][step] + positions[2][step])
        / 3.0;
    let mut ang = Vector3::zeros();
    for body_pos in positions.iter().take(nb) {
        let r: Vector3<f64> = body_pos[step] - cm;
        let v = body_pos[step] - body_pos[step.saturating_sub(1)];
        ang += r.cross(&v);
    }
    ang.norm()
}

// ─── Modal Resonator Bank ────────────────────────────────────

struct ModalResonator {
    y1: f64,
    y2: f64,
    a1: f64,
    a2: f64,
    gain: f64,
}

impl ModalResonator {
    fn new(center_hz: f64, bandwidth_hz: f64) -> Self {
        let bw = bandwidth_hz.max(1.0);
        let freq = center_hz.clamp(20.0, SR * 0.45);
        let r = (-std::f64::consts::PI * bw / SR).exp();
        let theta = TAU * freq / SR;
        Self {
            y1: 0.0,
            y2: 0.0,
            a1: -2.0 * r * theta.cos(),
            a2: r * r,
            gain: 1.0 - r * r,
        }
    }

    fn tick(&mut self, x: f64) -> f64 {
        let y = self.gain * x - self.a1 * self.y1 - self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

struct ModalBank {
    resonators: Vec<ModalResonator>,
}

impl ModalBank {
    fn new(fundamental: f64, ratios: &[f64; 16], bw_scale: f64, decay_s: f64) -> Self {
        let bw_base = 1.0 / (decay_s.max(0.1) * std::f64::consts::PI);
        let resonators = ratios
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                let freq = fundamental * r;
                let bw = bw_base * bw_scale * (1.0 + i as f64 * 0.15);
                ModalResonator::new(freq, bw)
            })
            .collect();
        Self { resonators }
    }

    fn excite(&mut self, input: f64, energy_profile: &[f64; 16]) -> f64 {
        let mut sum = 0.0;
        for (i, res) in self.resonators.iter_mut().enumerate() {
            let amp_scale = 1.0 / (1.0 + i as f64 * 0.25);
            sum += res.tick(input * energy_profile[i]) * amp_scale;
        }
        sum
    }
}

// ─── Algorithm D: Gallery Material ──────────────────────────

fn synthesize_gallery_material(
    positions: &[Vec<Vector3<f64>>],
    stats: &OrbitStats,
    ctx: &GalleryAudioContext,
    world: &WorldParams,
    total_samples: usize,
    step_count: usize,
    duration_s: f64,
) -> Vec<f64> {
    let nb = 3.min(positions.len());
    let mut buf = vec![0.0f64; total_samples * 2];

    let fundamental =
        world.fundamental_range.0 + (world.fundamental_range.1 - world.fundamental_range.0) * ctx.equilateralness;

    let mut banks: Vec<ModalBank> = (0..nb)
        .map(|b| {
            let mass_scale = ctx.masses[b] / ctx.masses.iter().sum::<f64>() * 3.0;
            let body_fund = fundamental * (0.7 + 0.6 * mass_scale);
            ModalBank::new(
                body_fund,
                &world.partial_ratios,
                world.resonator_bw_scale,
                world.decay_base_s,
            )
        })
        .collect();

    let mut sm_spd = [0.0f64; 3];
    let mut prev_spd = [0.0f64; 3];
    let mut last_excite = [0usize; 3];
    let mut sub_ph = 0.0f64;
    let mut rng = SimpleRng::new(0x6A11_E7F0_1A7E_71A1u64.wrapping_mul(
        (ctx.chaos * 1e9) as u64 | 1,
    ));

    let spd_a = 1.0 - (-1.0 / (0.08 * SR)).exp();
    let cooldown = (0.15 * SR) as usize;
    let spectral_spread = 2.0 + world.inharmonicity * 3.0;

    for si in 0..total_samples {
        let t = si as f64 / total_samples as f64;
        let sf = t * (step_count.saturating_sub(2)) as f64;
        let step = sf as usize;
        let frac = sf - step as f64;
        let s = step.min(step_count - 1);

        let area = triangle_area_at(positions, s);
        let norm_area = (area / stats.max_area).clamp(0.0, 1.0);
        let ang_mom = angular_momentum_at(positions, s);
        let norm_ang = (ang_mom / stats.max_ang_mom).clamp(0.0, 1.0);

        let mut left = 0.0f64;
        let mut right = 0.0f64;

        for b in 0..nb {
            let spd = body_speed(positions, b, step, frac, step_count);
            let ns = (spd / stats.max_speed).clamp(0.0, 1.0);

            prev_spd[b] = sm_spd[b];
            sm_spd[b] += spd_a * (ns - sm_spd[b]);

            let close = body_closeness(positions, b, s, nb, stats.max_distance);

            let dominant_bin = sm_spd[b] * 15.0;
            let mut energy_profile = [0.0f64; 16];
            for (i, ep) in energy_profile.iter_mut().enumerate() {
                let dist = (i as f64 - dominant_bin).abs();
                *ep = (-dist * dist / (2.0 * spectral_spread * spectral_spread)).exp()
                    * (close * 0.6 + norm_area * 0.4);
            }

            let mut excitation = 0.0f64;

            let crossed_up = sm_spd[b] > 0.3 && prev_spd[b] <= 0.3;
            let near_miss = close > 0.75 && prev_spd[b] < sm_spd[b];
            if (crossed_up || near_miss) && si.saturating_sub(last_excite[b]) > cooldown {
                last_excite[b] = si;
                excitation += (0.3 + close * 0.5) * (0.5 + sm_spd[b] * 0.5);
            }

            excitation += rng.next_f64().abs() * sm_spd[b] * 0.004 * (0.2 + close * 0.8);

            let modal_out = banks[b].excite(excitation, &energy_profile);

            let pos = positions[b][s];
            let centroid = (positions[0][s] + positions[1][s] + positions[2][s])
                / 3.0;
            let rel: Vector3<f64> = pos - centroid;
            let azimuth = rel.x.atan2(rel.y) / std::f64::consts::PI;
            let depth = (rel.z.abs() / stats.max_distance).clamp(0.0, 1.0);
            let depth_atten = 1.0 - depth * 0.4;
            let depth_dark = 1.0 - depth * 0.3;

            let pan: f64 = (0.5 + azimuth * 0.4).clamp(0.0, 1.0);
            left += modal_out * depth_atten * depth_dark * (1.0 - pan).sqrt();
            right += modal_out * depth_atten * pan.sqrt();
        }

        sub_ph += (fundamental * 0.5) / SR;
        let sub_env = norm_area * 0.5 + norm_ang * 0.3;
        let sub = (sub_ph * TAU).sin() * 0.06 * sub_env;
        left += sub;
        right += sub;

        let fade = smoothstep_fade(t, 4.0, 6.0, duration_s);
        buf[si * 2] = left * fade;
        buf[si * 2 + 1] = right * fade;
    }

    apply_stereo_reverb(&mut buf, total_samples, &world.reverb);
    normalize_peak(&mut buf, 0.88);
    buf
}

// ─── Algorithm E: Cathedral Orbit ───────────────────────────

fn synthesize_cathedral_orbit(
    positions: &[Vec<Vector3<f64>>],
    stats: &OrbitStats,
    ctx: &GalleryAudioContext,
    world: &WorldParams,
    total_samples: usize,
    step_count: usize,
    duration_s: f64,
) -> Vec<f64> {
    let nb = 3.min(positions.len());
    let mut buf = vec![0.0f64; total_samples * 2];

    let mass_ratios = [
        1.0,
        ctx.masses[1] / ctx.masses[0].max(1.0),
        ctx.masses[2] / ctx.masses[0].max(1.0),
    ];

    let base_fund = world.fundamental_range.0
        + (world.fundamental_range.1 - world.fundamental_range.0) * ctx.equilateralness * 0.7;

    let voice_intervals = [
        1.0,
        (mass_ratios[1] * 1.5).fract() + 1.0,
        (mass_ratios[2] * 1.25).fract() + 1.25,
    ];

    let mut voice_ph = [[0.0f64; 20]; 3];
    let mut vib_ph = [0.0f64; 3];
    let mut drone_ph = 0.0f64;
    let mut sm_amp = [0.0f64; 3];
    let mut sm_brightness = [0.5f64; 3];
    let mut sm_density = 4.0f64;

    let amp_a = 1.0 - (-1.0 / (2.0 * SR)).exp();
    let bright_a = 1.0 - (-1.0 / (0.8 * SR)).exp();
    let density_a = 1.0 - (-1.0 / (1.5 * SR)).exp();

    let mut feedback_brightness = 0.5f64;
    let feedback_window = (SR * 0.4) as usize;
    let mut recent_energy = 0.0f64;

    let drift_rate = 0.015 + 0.01 * ctx.chaos;

    let mut rng = SimpleRng::new(0x0CAF_ED7A_107B_17A1_u64.wrapping_mul(
        (ctx.equilateralness * 1e9) as u64 | 1,
    ));

    for si in 0..total_samples {
        let t = si as f64 / total_samples as f64;
        let sf = t * (step_count.saturating_sub(2)) as f64;
        let step = sf as usize;
        let frac = sf - step as f64;
        let s = step.min(step_count - 1);
        let ts = si as f64 / SR;

        let area = triangle_area_at(positions, s);
        let norm_area = (area / stats.max_area).clamp(0.0, 1.0);
        let ang_mom = angular_momentum_at(positions, s);
        let norm_ang = (ang_mom / stats.max_ang_mom).clamp(0.0, 1.0);

        let target_density = 3.0 + norm_area * 13.0;
        sm_density += density_a * (target_density - sm_density);
        let active_harmonics = (sm_density as usize).clamp(2, 18);

        let fund_drift = base_fund * (1.0 + 0.03 * (ts * drift_rate * TAU).sin());

        drone_ph += (fund_drift * 0.25) / SR;
        let drone_env = 0.4 + 0.6 * norm_ang;
        let drone = (drone_ph * TAU).sin() * 0.10 * drone_env
            + (drone_ph * 2.0 * TAU).sin() * 0.03 * drone_env;

        let mut left = 0.0f64;
        let mut right = 0.0f64;

        for b in 0..nb {
            let spd = body_speed(positions, b, step, frac, step_count);
            let ns = (spd / stats.max_speed).clamp(0.0, 1.0);
            let close = body_closeness(positions, b, s, nb, stats.max_distance);

            let tgt_amp = close.clamp(0.0, 1.0) * 0.14;
            sm_amp[b] += amp_a * (tgt_amp - sm_amp[b]);

            let tgt_bright = ns * 0.7 + world.brightness_bias * 0.3 + feedback_brightness * 0.15;
            sm_brightness[b] += bright_a * (tgt_bright.clamp(0.1, 0.95) - sm_brightness[b]);

            let voice_fund = fund_drift * voice_intervals[b];
            vib_ph[b] += (3.5 + b as f64 * 0.7) / SR;
            let vibrato = (vib_ph[b] * TAU).sin() * voice_fund * 0.006;
            let freq = voice_fund + vibrato;

            let spectral_tilt = 1.0 + (1.0 - sm_brightness[b]) * 2.0;
            let mut wave = 0.0f64;
            for (h, ph) in voice_ph[b].iter_mut().enumerate().take(active_harmonics.min(20)) {
                let hf = (h + 1) as f64;
                let harm_freq = freq * hf;
                if harm_freq > SR * 0.45 {
                    break;
                }
                *ph += harm_freq / SR;
                if *ph > 1.0 {
                    *ph -= 1.0;
                }
                let amp = 1.0 / hf.powf(spectral_tilt);
                wave += (*ph * TAU).sin() * amp;
            }
            wave *= 0.35;

            wave += rng.next_f64() * 0.003 * ns;

            let pos = positions[b][s];
            let centroid = (positions[0][s] + positions[1][s] + positions[2][s])
                / 3.0;
            let rel: Vector3<f64> = pos - centroid;
            let azimuth = rel.x.atan2(rel.y) / std::f64::consts::PI;
            let pan_lfo = (ts * (0.04 + b as f64 * 0.012) * TAU).sin() * 0.15;
            let pan = (0.5 + azimuth * 0.35 + pan_lfo).clamp(0.05, 0.95);

            left += wave * sm_amp[b] * (1.0 - pan).sqrt();
            right += wave * sm_amp[b] * pan.sqrt();
        }

        left += drone;
        right += drone;

        let sample_energy = left * left + right * right;
        recent_energy += sample_energy;
        if si > 0 && si % feedback_window == 0 {
            let avg_energy = recent_energy / feedback_window as f64;
            let centroid_est = avg_energy.sqrt().clamp(0.0, 1.0);
            feedback_brightness = feedback_brightness * 0.7 + centroid_est * 0.3;
            recent_energy = 0.0;
        }

        let fade = smoothstep_fade(t, 5.0, 8.0, duration_s);
        buf[si * 2] = left * fade;
        buf[si * 2 + 1] = right * fade;
    }

    let cathedral_reverb = ReverbParams {
        delay1_ms: world.reverb.delay1_ms * 1.3,
        delay2_ms: world.reverb.delay2_ms * 1.3,
        feedback: (world.reverb.feedback * 1.1).min(0.75),
        wet_mix: (world.reverb.wet_mix * 1.15).min(0.55),
        pre_delay_ms: world.reverb.pre_delay_ms * 1.5,
    };
    apply_stereo_reverb(&mut buf, total_samples, &cathedral_reverb);
    normalize_peak(&mut buf, 0.88);
    buf
}

// ─── Infrastructure ──────────────────────────────────────────

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

    VelocityStats {
        max_speed: max_speed.max(1e-10),
        max_distance: max_distance.max(1e-10),
    }
}

fn write_wav(samples: &[i16], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;

    let mut file = std::fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&CHANNELS.to_le_bytes())?;
    file.write_all(&SAMPLE_RATE.to_le_bytes())?;
    let byte_rate = SAMPLE_RATE * CHANNELS as u32 * (BITS_PER_SAMPLE / 8) as u32;
    file.write_all(&byte_rate.to_le_bytes())?;
    let block_align = CHANNELS * (BITS_PER_SAMPLE / 8);
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&BITS_PER_SAMPLE.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for &sample in samples {
        file.write_all(&sample.to_le_bytes())?;
    }
    Ok(())
}

fn mux_audio_video(
    audio_path: &str,
    video_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            output_path,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()?;

    if !status.success() {
        return Err(format!("ffmpeg mux failed with status {:?}", status).into());
    }
    Ok(())
}

// ─── Public API ──────────────────────────────────────────────

pub fn generate_sonification(
    positions: &[Vec<Vector3<f64>>],
    duration_seconds: f64,
    video_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        "Generating sonification ({:.1}s) for {}...",
        duration_seconds, video_path
    );

    let total_samples = (SR * duration_seconds) as usize;
    let step_count = positions[0].len();
    let stats = compute_velocity_stats(positions);

    info!("   Synthesizing variant A: Celestial Pad...");
    let sa = synthesize_celestial_pad(positions, &stats, total_samples, step_count, duration_seconds);
    info!("   Synthesizing variant B: Crystal Resonance...");
    let sb =
        synthesize_crystal_resonance(positions, &stats, total_samples, step_count, duration_seconds);
    info!("   Synthesizing variant C: Orbital Choir...");
    let sc =
        synthesize_orbital_choir(positions, &stats, total_samples, step_count, duration_seconds);

    let wa = format!("{}.pad.wav", video_path);
    let wb = format!("{}.bells.wav", video_path);
    let wc = format!("{}.choir.wav", video_path);
    write_wav(&f64_to_i16(&sa), &wa)?;
    write_wav(&f64_to_i16(&sb), &wb)?;
    write_wav(&f64_to_i16(&sc), &wc)?;

    let base = video_path.strip_suffix(".mp4").unwrap_or(video_path);
    let out_b = format!("{}_crystal_resonance.mp4", base);
    let out_c = format!("{}_orbital_choir.mp4", base);

    mux_audio_video(&wb, video_path, &out_b)?;
    info!("   Saved variant B => {}", out_b);
    mux_audio_video(&wc, video_path, &out_c)?;
    info!("   Saved variant C => {}", out_c);

    let tmp = format!("{}.tmp.mp4", video_path);
    mux_audio_video(&wa, video_path, &tmp)?;
    std::fs::rename(&tmp, video_path)?;
    info!("   Embedded variant A (Celestial Pad) into {}", video_path);

    let _ = std::fs::remove_file(&wa);
    let _ = std::fs::remove_file(&wb);
    let _ = std::fs::remove_file(&wc);
    Ok(())
}

/// Gallery-grade sonification using orbit character, spectral mirroring,
/// physical resonators, and depth-aware spatialization. Produces two new
/// variants (Gallery Material and Cathedral Orbit) alongside the original
/// three, and sets Gallery Material as the default soundtrack.
pub fn generate_gallery_sonification(
    positions: &[Vec<Vector3<f64>>],
    duration_seconds: f64,
    video_path: &str,
    context: &GalleryAudioContext,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        "Generating gallery sonification ({:.1}s) for {}...",
        duration_seconds, video_path
    );

    let total_samples = (SR * duration_seconds) as usize;
    let step_count = positions[0].len();

    let orbit_stats = compute_orbit_stats(positions);
    let legacy_stats = compute_velocity_stats(positions);
    let world = select_sonic_world(context.chaos, context.equilateralness);
    let wp = world_params(world);

    info!("   Sonic world: {:?}  (chaos={:.4}, equil={:.4})", world, context.chaos, context.equilateralness);

    info!("   Synthesizing variant D: Gallery Material...");
    let sd = synthesize_gallery_material(
        positions,
        &orbit_stats,
        context,
        &wp,
        total_samples,
        step_count,
        duration_seconds,
    );

    info!("   Synthesizing variant E: Cathedral Orbit...");
    let se = synthesize_cathedral_orbit(
        positions,
        &orbit_stats,
        context,
        &wp,
        total_samples,
        step_count,
        duration_seconds,
    );

    info!("   Synthesizing variant B: Crystal Resonance...");
    let sb = synthesize_crystal_resonance(
        positions,
        &legacy_stats,
        total_samples,
        step_count,
        duration_seconds,
    );

    info!("   Synthesizing variant C: Orbital Choir...");
    let sc = synthesize_orbital_choir(
        positions,
        &legacy_stats,
        total_samples,
        step_count,
        duration_seconds,
    );

    let wd = format!("{}.gallery.wav", video_path);
    let we = format!("{}.cathedral.wav", video_path);
    let wb = format!("{}.bells.wav", video_path);
    let wc = format!("{}.choir.wav", video_path);
    write_wav(&f64_to_i16(&sd), &wd)?;
    write_wav(&f64_to_i16(&se), &we)?;
    write_wav(&f64_to_i16(&sb), &wb)?;
    write_wav(&f64_to_i16(&sc), &wc)?;

    let base = video_path.strip_suffix(".mp4").unwrap_or(video_path);

    let out_e = format!("{}_cathedral_orbit.mp4", base);
    mux_audio_video(&we, video_path, &out_e)?;
    info!("   Saved variant E => {}", out_e);

    let out_b = format!("{}_crystal_resonance.mp4", base);
    mux_audio_video(&wb, video_path, &out_b)?;
    info!("   Saved variant B => {}", out_b);

    let out_c = format!("{}_orbital_choir.mp4", base);
    mux_audio_video(&wc, video_path, &out_c)?;
    info!("   Saved variant C => {}", out_c);

    let tmp = format!("{}.tmp.mp4", video_path);
    mux_audio_video(&wd, video_path, &tmp)?;
    std::fs::rename(&tmp, video_path)?;
    info!("   Embedded variant D (Gallery Material) into {}", video_path);

    let _ = std::fs::remove_file(&wd);
    let _ = std::fs::remove_file(&we);
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
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(2.0, 1.0, 0.0),
            ],
            vec![
                Vector3::new(5.0, 0.0, 0.0),
                Vector3::new(5.0, 1.0, 0.0),
                Vector3::new(5.0, 2.0, 0.0),
            ],
            vec![
                Vector3::new(0.0, 5.0, 0.0),
                Vector3::new(1.0, 5.0, 0.0),
                Vector3::new(2.0, 5.0, 0.0),
            ],
        ]
    }

    fn fade_envelope(t: f64) -> f64 {
        let fi = (t * 20.0).clamp(0.0, 1.0);
        let fo = ((1.0 - t) * 20.0).clamp(0.0, 1.0);
        fi * fo
    }

    fn synthesize_legacy(
        positions: &[Vec<Vector3<f64>>],
        stats: &VelocityStats,
        total_samples: usize,
        step_count: usize,
    ) -> Vec<i16> {
        let nb = 3.min(positions.len());
        let mut out = vec![0i16; total_samples * CHANNELS as usize];
        let mut phases = vec![0.0f64; nb];
        for si in 0..total_samples {
            let t = si as f64 / total_samples as f64;
            let sf = t * (step_count - 2) as f64;
            let step = sf as usize;
            let frac = sf - step as f64;
            let (mut l, mut r) = (0.0f64, 0.0f64);
            for b in 0..nb {
                let v0 = if step + 1 < step_count {
                    (positions[b][step + 1] - positions[b][step]).norm()
                } else {
                    0.0
                };
                let v1 = if step + 2 < step_count {
                    (positions[b][step + 2] - positions[b][step + 1]).norm()
                } else {
                    v0
                };
                let spd = v0 * (1.0 - frac) + v1 * frac;
                let ns = (spd / stats.max_speed).clamp(0.0, 1.0);
                let freq = 80.0 + 520.0 * ns;
                let mut amp = 0.0f64;
                for o in 0..nb {
                    if o != b {
                        let d = (positions[b][step] - positions[o][step]).norm();
                        amp += 1.0 - (d / stats.max_distance).clamp(0.0, 1.0);
                    }
                }
                amp = (amp / (nb - 1) as f64).clamp(0.0, 1.0) * 0.25;
                phases[b] += freq / SAMPLE_RATE as f64;
                let ph = phases[b] * TAU;
                let w = ph.sin() * 0.6
                    + (ph * 2.0).sin() * 0.2
                    + (ph * 3.0).sin() * 0.1
                    + (ph * 5.0).sin() * 0.05;
                let xn = if step < step_count {
                    let mn = positions[b].iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
                    let mx = positions[b]
                        .iter()
                        .map(|p| p.x)
                        .fold(f64::NEG_INFINITY, f64::max);
                    ((positions[b][step].x - mn) / (mx - mn).max(1e-10)).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                l += w * amp * (1.0 - xn).sqrt();
                r += w * amp * xn.sqrt();
            }
            let f = fade_envelope(t);
            out[si * 2] = (l * f).clamp(-1.0, 1.0).mul_add(32000.0, 0.0) as i16;
            out[si * 2 + 1] = (r * f).clamp(-1.0, 1.0).mul_add(32000.0, 0.0) as i16;
        }
        out
    }

    #[test]
    fn test_fade_envelope_near_zero_at_start() {
        assert!(fade_envelope(0.0) < 0.01);
    }

    #[test]
    fn test_fade_envelope_near_one_at_middle() {
        assert!(fade_envelope(0.5) > 0.99);
    }

    #[test]
    fn test_fade_envelope_near_zero_at_end() {
        assert!(fade_envelope(1.0) < 0.01);
    }

    #[test]
    fn test_fade_envelope_always_non_negative() {
        for i in 0..=1000 {
            let t = i as f64 / 1000.0;
            assert!(fade_envelope(t) >= 0.0, "negative at t={t}");
        }
    }

    #[test]
    fn test_compute_velocity_stats_known_trajectory() {
        let positions = vec![
            vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 4.0, 0.0)],
            vec![
                Vector3::new(10.0, 0.0, 0.0),
                Vector3::new(10.0, 0.0, 0.0),
            ],
            vec![
                Vector3::new(0.0, 10.0, 0.0),
                Vector3::new(0.0, 10.0, 0.0),
            ],
        ];
        let stats = compute_velocity_stats(&positions);
        assert!((stats.max_speed - 5.0).abs() < 1e-10);
        let ed = ((10.0f64).powi(2) + (10.0f64).powi(2)).sqrt();
        assert!((stats.max_distance - ed).abs() < 1e-10);
    }

    #[test]
    fn test_synthesize_determinism() {
        let pos = sample_positions();
        let stats = compute_velocity_stats(&pos);
        let a = synthesize_legacy(&pos, &stats, 100, pos[0].len());
        let b = synthesize_legacy(&pos, &stats, 100, pos[0].len());
        assert_eq!(a, b, "same input must produce identical output");
    }

    #[test]
    fn test_synthesize_output_length() {
        let pos = sample_positions();
        let stats = compute_velocity_stats(&pos);
        let samples = synthesize_legacy(&pos, &stats, 441, pos[0].len());
        assert_eq!(samples.len(), 441 * CHANNELS as usize);
    }

    #[test]
    fn test_synthesize_samples_in_range() {
        let pos = sample_positions();
        let stats = compute_velocity_stats(&pos);
        let samples = synthesize_legacy(&pos, &stats, 4410, pos[0].len());
        for (i, &s) in samples.iter().enumerate() {
            assert!((-32000..=32000).contains(&s), "sample {i} out of range: {s}");
        }
    }

    #[test]
    fn test_smoothstep_fade_shape() {
        assert!(smoothstep_fade(0.0, 3.0, 5.0, 30.0) < 0.01);
        assert!(smoothstep_fade(0.5, 3.0, 5.0, 30.0) > 0.99);
        assert!(smoothstep_fade(1.0, 3.0, 5.0, 30.0) < 0.01);
    }

    #[test]
    fn test_quantize_to_pentatonic() {
        assert!((quantize_to_scale(115.0, &PENTATONIC) - 110.0).abs() < 0.01);
        assert!((quantize_to_scale(440.0, &PENTATONIC) - 440.0).abs() < 0.01);
        assert!((quantize_to_scale(250.0, &PENTATONIC) - 261.63).abs() < 0.01);
    }

    #[test]
    fn test_celestial_pad_length_and_determinism() {
        let pos = sample_positions();
        let stats = compute_velocity_stats(&pos);
        let a = synthesize_celestial_pad(&pos, &stats, 441, pos[0].len(), 0.01);
        let b = synthesize_celestial_pad(&pos, &stats, 441, pos[0].len(), 0.01);
        assert_eq!(a.len(), 441 * 2);
        assert_eq!(a, b);
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

    // ─── Gallery Algorithm Tests ─────────────────────────────

    fn gallery_context() -> GalleryAudioContext {
        GalleryAudioContext {
            chaos: 0.3,
            equilateralness: 0.6,
            masses: [200.0, 150.0, 250.0],
        }
    }

    #[test]
    fn test_select_sonic_world_quadrants() {
        assert_eq!(select_sonic_world(0.1, 0.3), SonicWorld::Obsidian);
        assert_eq!(select_sonic_world(0.1, 0.7), SonicWorld::Aurora);
        assert_eq!(select_sonic_world(0.6, 0.7), SonicWorld::Meridian);
        assert_eq!(select_sonic_world(0.6, 0.3), SonicWorld::Liminal);
    }

    #[test]
    fn test_compute_orbit_stats_non_degenerate() {
        let pos = sample_positions();
        let stats = compute_orbit_stats(&pos);
        assert!(stats.max_speed > 0.0);
        assert!(stats.max_distance > 0.0);
        assert!(stats.max_area > 0.0);
        assert!(stats.mean_perimeter > 0.0);
    }

    #[test]
    fn test_modal_resonator_decays() {
        let mut res = ModalResonator::new(440.0, 5.0);
        let impulse_out = res.tick(1.0);
        assert!(impulse_out.abs() > 0.0);
        let window = 441;
        let mut prev_peak = f64::MAX;
        for w in 0..10 {
            let mut peak = 0.0f64;
            for _ in 0..window {
                peak = peak.max(res.tick(0.0).abs());
            }
            if w > 0 {
                assert!(peak < prev_peak + 1e-6, "resonator envelope should decay over windows");
            }
            prev_peak = peak;
        }
    }

    #[test]
    fn test_gallery_material_length_and_determinism() {
        let pos = sample_positions();
        let ostats = compute_orbit_stats(&pos);
        let ctx = gallery_context();
        let wp = world_params(SonicWorld::Aurora);
        let a = synthesize_gallery_material(&pos, &ostats, &ctx, &wp, 441, pos[0].len(), 0.01);
        let b = synthesize_gallery_material(&pos, &ostats, &ctx, &wp, 441, pos[0].len(), 0.01);
        assert_eq!(a.len(), 441 * 2);
        assert_eq!(a, b, "gallery material must be deterministic");
    }

    #[test]
    fn test_cathedral_orbit_length_and_determinism() {
        let pos = sample_positions();
        let ostats = compute_orbit_stats(&pos);
        let ctx = gallery_context();
        let wp = world_params(SonicWorld::Meridian);
        let a = synthesize_cathedral_orbit(&pos, &ostats, &ctx, &wp, 441, pos[0].len(), 0.01);
        let b = synthesize_cathedral_orbit(&pos, &ostats, &ctx, &wp, 441, pos[0].len(), 0.01);
        assert_eq!(a.len(), 441 * 2);
        assert_eq!(a, b, "cathedral orbit must be deterministic");
    }

    #[test]
    fn test_gallery_material_samples_bounded() {
        let pos = sample_positions();
        let ostats = compute_orbit_stats(&pos);
        let ctx = gallery_context();
        let wp = world_params(SonicWorld::Obsidian);
        let buf = synthesize_gallery_material(&pos, &ostats, &ctx, &wp, 4410, pos[0].len(), 0.1);
        let samples = f64_to_i16(&buf);
        for (i, &s) in samples.iter().enumerate() {
            assert!((-32000..=32000).contains(&s), "gallery sample {i} out of range: {s}");
        }
    }

    #[test]
    fn test_cathedral_orbit_samples_bounded() {
        let pos = sample_positions();
        let ostats = compute_orbit_stats(&pos);
        let ctx = gallery_context();
        let wp = world_params(SonicWorld::Liminal);
        let buf = synthesize_cathedral_orbit(&pos, &ostats, &ctx, &wp, 4410, pos[0].len(), 0.1);
        let samples = f64_to_i16(&buf);
        for (i, &s) in samples.iter().enumerate() {
            assert!((-32000..=32000).contains(&s), "cathedral sample {i} out of range: {s}");
        }
    }

    #[test]
    fn test_different_worlds_produce_different_output() {
        let pos = sample_positions();
        let ostats = compute_orbit_stats(&pos);
        let ctx = gallery_context();
        let wp_a = world_params(SonicWorld::Aurora);
        let wp_b = world_params(SonicWorld::Obsidian);
        let a = synthesize_gallery_material(&pos, &ostats, &ctx, &wp_a, 441, pos[0].len(), 0.01);
        let b = synthesize_gallery_material(&pos, &ostats, &ctx, &wp_b, 441, pos[0].len(), 0.01);
        assert_ne!(a, b, "different sonic worlds must produce different output");
    }

    #[test]
    fn test_triangle_area_positive() {
        let pos = sample_positions();
        let area = triangle_area_at(&pos, 0);
        assert!(area > 0.0, "non-degenerate triangle should have positive area");
    }

    #[test]
    fn test_angular_momentum_nonnegative() {
        let pos = sample_positions();
        let am = angular_momentum_at(&pos, 1);
        assert!(am >= 0.0, "angular momentum magnitude must be non-negative");
    }
}
