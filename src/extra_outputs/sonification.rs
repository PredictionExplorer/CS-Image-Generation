//! Sonification — celestial audio synthesis from three-body simulation data.
//!
//! Three synthesis algorithms transform orbital mechanics into music:
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
    let pans = [0.3, 0.5, 0.7];

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

            left += filtered * sm_amp[b] * (1.0 - pans[b] as f64).sqrt();
            right += filtered * sm_amp[b] * (pans[b] as f64).sqrt();
        }

        let avg_spd = if step + 1 < step_count {
            let mut v = 0.0;
            for b in 0..nb {
                v += (positions[b][(step + 1).min(step_count - 1)] - positions[b][s]).norm();
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
}
