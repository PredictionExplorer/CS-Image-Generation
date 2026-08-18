//! Deterministic audio helpers (master plan II.11): a hand-rolled 24-bit
//! stereo WAV writer, resampling and conditioning utilities, synthesis
//! primitives (polyBLEP saw, ADSR, resonant low-pass, FM bell, equal-power
//! pan, tanh limiter), a BS.1770-approximate loudness normalizer, and the
//! `FFmpeg` mux that marries a WAV onto an existing video.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::process::Command;

/// Standard output sample rate for all viz audio artifacts.
pub const SAMPLE_RATE: u32 = 48_000;

/// Write a 24-bit PCM stereo WAV file.
///
/// Samples are `f64` in `[-1, 1]`; values outside are clamped. Both channels
/// must have equal length.
pub fn write_wav_stereo_24bit(path: &str, left: &[f64], right: &[f64]) -> io::Result<()> {
    assert_eq!(left.len(), right.len(), "stereo channels must match in length");
    let mut out = BufWriter::new(File::create(path)?);

    let num_samples = left.len() as u32;
    let bytes_per_sample = 3u32;
    let channels = 2u32;
    let byte_rate = SAMPLE_RATE * channels * bytes_per_sample;
    let block_align = (channels * bytes_per_sample) as u16;
    let data_len = num_samples * channels * bytes_per_sample;

    out.write_all(b"RIFF")?;
    out.write_all(&(36 + data_len).to_le_bytes())?;
    out.write_all(b"WAVE")?;
    out.write_all(b"fmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&1u16.to_le_bytes())?; // PCM
    out.write_all(&(channels as u16).to_le_bytes())?;
    out.write_all(&SAMPLE_RATE.to_le_bytes())?;
    out.write_all(&byte_rate.to_le_bytes())?;
    out.write_all(&block_align.to_le_bytes())?;
    out.write_all(&24u16.to_le_bytes())?;
    out.write_all(b"data")?;
    out.write_all(&data_len.to_le_bytes())?;

    let mut frame = [0u8; 6];
    for (&l, &r) in left.iter().zip(right.iter()) {
        encode_sample_24(l, &mut frame[0..3]);
        encode_sample_24(r, &mut frame[3..6]);
        out.write_all(&frame)?;
    }
    out.flush()
}

/// Encode one `[-1, 1]` sample as signed 24-bit little-endian PCM.
fn encode_sample_24(value: f64, out: &mut [u8]) {
    let clamped = value.clamp(-1.0, 1.0);
    let scaled = (clamped * 8_388_607.0).round() as i32;
    let bytes = scaled.to_le_bytes();
    out[0] = bytes[0];
    out[1] = bytes[1];
    out[2] = bytes[2];
}

/// Linearly resample a series to a new length (endpoint preserving).
#[must_use]
pub fn resample_linear(series: &[f64], out_len: usize) -> Vec<f64> {
    if series.is_empty() || out_len == 0 {
        return vec![0.0; out_len];
    }
    if series.len() == 1 {
        return vec![series[0]; out_len];
    }
    let last = (series.len() - 1) as f64;
    (0..out_len)
        .map(|index| {
            let t = index as f64 / (out_len.saturating_sub(1).max(1)) as f64 * last;
            let base = t.floor() as usize;
            let next = (base + 1).min(series.len() - 1);
            let frac = t - base as f64;
            series[base] * (1.0 - frac) + series[next] * frac
        })
        .collect()
}

/// One-pole high-pass filter in place (DC / rumble removal).
pub fn high_pass(samples: &mut [f64], cutoff_hz: f64) {
    let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz.max(1e-3));
    let dt = 1.0 / f64::from(SAMPLE_RATE);
    let alpha = rc / (rc + dt);
    let mut previous_in = 0.0;
    let mut previous_out = 0.0;
    for sample in samples.iter_mut() {
        let current = *sample;
        let filtered = alpha * (previous_out + current - previous_in);
        previous_in = current;
        previous_out = filtered;
        *sample = filtered;
    }
}

/// Normalize a stereo pair so the joint peak hits `peak` (e.g. 0.7 for -3 dB).
pub fn normalize_stereo_peak(left: &mut [f64], right: &mut [f64], peak: f64) {
    let max = left
        .iter()
        .chain(right.iter())
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
        .max(1e-12);
    let gain = peak / max;
    for sample in left.iter_mut().chain(right.iter_mut()) {
        *sample *= gain;
    }
}

/// Apply a short raised-cosine fade to both ends of a channel (click guard).
pub fn fade_ends(samples: &mut [f64], fade_len: usize) {
    let n = samples.len();
    let fade = fade_len.min(n / 2);
    for index in 0..fade {
        let gain = 0.5 - 0.5 * (std::f64::consts::PI * index as f64 / fade as f64).cos();
        samples[index] *= gain;
        samples[n - 1 - index] *= gain;
    }
}

// ---------------------------------------------------------------------------
// Synthesis primitives (II.11)
// ---------------------------------------------------------------------------

/// `PolyBLEP` correction for a discontinuity at phase `t` with step `dt`.
#[inline]
fn poly_blep(t: f64, dt: f64) -> f64 {
    if t < dt {
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

/// Band-limited sawtooth sample at `phase` in `[0, 1)` advancing by
/// `dphase` per sample (polyBLEP edge smoothing).
#[inline]
#[must_use]
pub fn saw_bandlimited(phase: f64, dphase: f64) -> f64 {
    let naive = 2.0 * phase - 1.0;
    naive - poly_blep(phase, dphase.max(1e-9))
}

/// Linear attack / exponential-ish release envelope.
#[derive(Clone, Copy, Debug)]
pub struct Adsr {
    /// Attack length in seconds.
    pub attack: f64,
    /// Decay length in seconds.
    pub decay: f64,
    /// Sustain level in `[0, 1]`.
    pub sustain: f64,
    /// Release length in seconds.
    pub release: f64,
}

impl Adsr {
    /// Envelope amplitude at `t` seconds after note-on, with the note held
    /// for `held` seconds.
    #[must_use]
    pub fn amplitude(&self, t: f64, held: f64) -> f64 {
        if t < 0.0 {
            return 0.0;
        }
        let sustained = if t < self.attack {
            t / self.attack.max(1e-9)
        } else if t < self.attack + self.decay {
            1.0 - (1.0 - self.sustain) * (t - self.attack) / self.decay.max(1e-9)
        } else {
            self.sustain
        };
        if t <= held {
            sustained
        } else {
            let release_t = t - held;
            if release_t >= self.release {
                0.0
            } else {
                sustained * (1.0 - release_t / self.release.max(1e-9))
            }
        }
    }
}

/// Two-pole resonant low-pass (RBJ biquad), processed per sample.
#[derive(Clone, Copy, Debug, Default)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    /// Configure as a low-pass at `cutoff_hz` with quality `q`.
    pub fn set_lowpass(&mut self, cutoff_hz: f64, q: f64, sample_rate: f64) {
        let omega = std::f64::consts::TAU * cutoff_hz.clamp(10.0, sample_rate * 0.45) / sample_rate;
        let alpha = omega.sin() / (2.0 * q.max(0.05));
        let cos_omega = omega.cos();
        let a0 = 1.0 + alpha;
        self.b0 = (1.0 - cos_omega) / 2.0 / a0;
        self.b1 = (1.0 - cos_omega) / a0;
        self.b2 = self.b0;
        self.a1 = -2.0 * cos_omega / a0;
        self.a2 = (1.0 - alpha) / a0;
    }

    /// Configure as a high-shelf (BS.1770 pre-filter shape).
    pub fn set_highshelf(&mut self, cutoff_hz: f64, gain_db: f64, sample_rate: f64) {
        let amp = 10.0_f64.powf(gain_db / 40.0);
        let omega = std::f64::consts::TAU * cutoff_hz / sample_rate;
        let (sin_o, cos_o) = omega.sin_cos();
        let alpha = sin_o / 2.0 * std::f64::consts::SQRT_2;
        let a0 = (amp + 1.0) - (amp - 1.0) * cos_o + 2.0 * amp.sqrt() * alpha;
        self.b0 = (amp * ((amp + 1.0) + (amp - 1.0) * cos_o + 2.0 * amp.sqrt() * alpha)) / a0;
        self.b1 = (-2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_o)) / a0;
        self.b2 = (amp * ((amp + 1.0) + (amp - 1.0) * cos_o - 2.0 * amp.sqrt() * alpha)) / a0;
        self.a1 = (2.0 * ((amp - 1.0) - (amp + 1.0) * cos_o)) / a0;
        self.a2 = ((amp + 1.0) - (amp - 1.0) * cos_o - 2.0 * amp.sqrt() * alpha) / a0;
    }

    /// Configure as a high-pass at `cutoff_hz` (Butterworth-ish q).
    pub fn set_highpass(&mut self, cutoff_hz: f64, sample_rate: f64) {
        let omega = std::f64::consts::TAU * cutoff_hz / sample_rate;
        let (sin_o, cos_o) = omega.sin_cos();
        let alpha = sin_o / std::f64::consts::SQRT_2;
        let a0 = 1.0 + alpha;
        self.b0 = f64::midpoint(1.0, cos_o) / a0;
        self.b1 = -(1.0 + cos_o) / a0;
        self.b2 = self.b0;
        self.a1 = -2.0 * cos_o / a0;
        self.a2 = (1.0 - alpha) / a0;
    }

    /// Process one sample.
    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// FM bell sample: carrier plus a 3.5-ratio modulator, both decaying.
#[must_use]
pub fn fm_bell(t: f64, frequency: f64, brightness: f64) -> f64 {
    if t < 0.0 {
        return 0.0;
    }
    let modulator =
        (std::f64::consts::TAU * frequency * 3.5 * t).sin() * brightness * (-t * 5.0).exp();
    (std::f64::consts::TAU * frequency * t + modulator).sin() * (-t * 1.8).exp()
}

/// Equal-power stereo pan: `pan` in `[-1, 1]`.
#[inline]
#[must_use]
pub fn equal_power_pan(sample: f64, pan: f64) -> (f64, f64) {
    let angle = (pan.clamp(-1.0, 1.0) + 1.0) * std::f64::consts::FRAC_PI_4;
    (sample * angle.cos(), sample * angle.sin())
}

/// Soft-clip a pair of channels in place (tanh limiter at `drive`).
pub fn soft_limit(left: &mut [f64], right: &mut [f64], drive: f64) {
    for sample in left.iter_mut().chain(right.iter_mut()) {
        *sample = (*sample * drive).tanh() / drive.max(1e-9).tanh().max(1e-9);
    }
}

/// BS.1770-approximate integrated loudness of a stereo pair, in LUFS.
///
/// K-weighting (high-shelf + high-pass) then 400 ms blocks with the
/// absolute -70 LUFS gate and the relative -10 LU gate.
#[must_use]
pub fn integrated_lufs(left: &[f64], right: &[f64], sample_rate: f64) -> f64 {
    let block = (sample_rate * 0.4) as usize;
    if left.len() < block || block == 0 {
        return -70.0;
    }
    let weight = |samples: &[f64]| -> Vec<f64> {
        let mut shelf = Biquad::default();
        shelf.set_highshelf(1_681.0, 4.0, sample_rate);
        let mut highpass = Biquad::default();
        highpass.set_highpass(38.0, sample_rate);
        samples.iter().map(|&x| highpass.process(shelf.process(x))).collect()
    };
    let wl = weight(left);
    let wr = weight(right);
    // 75% overlapped 400 ms blocks.
    let hop = block / 4;
    let mut blocks: Vec<f64> = Vec::new();
    let mut start = 0usize;
    while start + block <= wl.len() {
        let mean_sq: f64 =
            (start..start + block).map(|i| wl[i] * wl[i] + wr[i] * wr[i]).sum::<f64>()
                / block as f64;
        blocks.push(-0.691 + 10.0 * (mean_sq.max(1e-12)).log10());
        start += hop.max(1);
    }
    let gated: Vec<f64> = blocks.iter().copied().filter(|&l| l > -70.0).collect();
    if gated.is_empty() {
        return -70.0;
    }
    let mean_energy =
        gated.iter().map(|&l| 10.0_f64.powf((l + 0.691) / 10.0)).sum::<f64>() / gated.len() as f64;
    let relative_gate = -0.691 + 10.0 * mean_energy.log10() - 10.0;
    let final_blocks: Vec<f64> = gated.iter().copied().filter(|&l| l > relative_gate).collect();
    if final_blocks.is_empty() {
        return -70.0;
    }
    let energy = final_blocks.iter().map(|&l| 10.0_f64.powf((l + 0.691) / 10.0)).sum::<f64>()
        / final_blocks.len() as f64;
    -0.691 + 10.0 * energy.log10()
}

/// Gain a stereo pair to an integrated loudness target (LUFS).
pub fn normalize_to_lufs(left: &mut [f64], right: &mut [f64], target: f64, sample_rate: f64) {
    let current = integrated_lufs(left, right, sample_rate);
    let gain = 10.0_f64.powf((target - current) / 20.0);
    for sample in left.iter_mut().chain(right.iter_mut()) {
        *sample *= gain;
    }
}

/// Mux a WAV onto an existing video (`-c:v copy -c:a aac -b:a 192k`).
pub fn mux(video_in: &str, wav: &str, video_out: &str) -> io::Result<()> {
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-i",
            video_in,
            "-i",
            wav,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            video_out,
        ])
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("ffmpeg mux failed with {status}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_preserves_endpoints() {
        let series = [0.0, 1.0, 4.0, 9.0];
        let out = resample_linear(&series, 7);
        assert_eq!(out.len(), 7);
        assert!((out[0] - 0.0).abs() < 1e-12);
        assert!((out[6] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn normalize_hits_requested_peak() {
        let mut left = vec![0.1, -0.5, 0.2];
        let mut right = vec![0.05, 0.25, -0.1];
        normalize_stereo_peak(&mut left, &mut right, 0.7);
        let max = left.iter().chain(right.iter()).fold(0.0_f64, |acc, &value| acc.max(value.abs()));
        assert!((max - 0.7).abs() < 1e-12);
    }

    #[test]
    fn saw_stays_bounded_and_crosses_zero() {
        let dphase = 220.0 / f64::from(SAMPLE_RATE);
        let mut phase = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for _ in 0..48_000 {
            let sample = saw_bandlimited(phase, dphase);
            min = min.min(sample);
            max = max.max(sample);
            phase = (phase + dphase) % 1.0;
        }
        assert!(min < -0.8 && max > 0.8, "saw span [{min}, {max}]");
        assert!(min >= -1.6 && max <= 1.6, "polyBLEP overshoot bounded");
    }

    #[test]
    fn adsr_envelope_shape() {
        let envelope = Adsr { attack: 0.1, decay: 0.1, sustain: 0.6, release: 0.2 };
        assert!(envelope.amplitude(0.05, 1.0) < envelope.amplitude(0.1, 1.0));
        assert!((envelope.amplitude(0.5, 1.0) - 0.6).abs() < 1e-9);
        assert!(envelope.amplitude(1.1, 1.0) < 0.6);
        assert!(envelope.amplitude(1.3, 1.0).abs() < 1e-9);
    }

    #[test]
    fn lowpass_attenuates_high_frequencies() {
        let rate = f64::from(SAMPLE_RATE);
        let respond = |freq: f64| -> f64 {
            let mut filter = Biquad::default();
            filter.set_lowpass(800.0, 0.9, rate);
            let mut peak = 0.0f64;
            for n in 0..24_000 {
                let t = f64::from(n) / rate;
                let y = filter.process((std::f64::consts::TAU * freq * t).sin());
                if n > 12_000 {
                    peak = peak.max(y.abs());
                }
            }
            peak
        };
        assert!(respond(100.0) > 0.9);
        assert!(respond(8_000.0) < 0.05);
    }

    #[test]
    fn full_scale_sine_measures_near_reference_lufs() {
        let rate = f64::from(SAMPLE_RATE);
        let samples: Vec<f64> = (0..96_000)
            .map(|n| (std::f64::consts::TAU * 997.0 * f64::from(n) / rate).sin())
            .collect();
        let lufs = integrated_lufs(&samples, &samples, rate);
        // BS.1770 reference: a full-scale 997 Hz stereo sine reads ~ -0.7.
        assert!((lufs + 0.7).abs() < 1.0, "measured {lufs}");
        let mut left = samples.clone();
        let mut right = samples;
        normalize_to_lufs(&mut left, &mut right, -14.0, rate);
        let normalized = integrated_lufs(&left, &right, rate);
        assert!((normalized + 14.0).abs() < 0.5, "normalized {normalized}");
    }

    #[test]
    fn pan_is_equal_power() {
        let (l, r) = equal_power_pan(1.0, 0.0);
        assert!((l * l + r * r - 1.0).abs() < 1e-9);
        let (hard_l, hard_r) = equal_power_pan(1.0, -1.0);
        assert!(hard_l > 0.999 && hard_r.abs() < 1e-9);
    }

    #[test]
    fn wav_header_is_valid() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.wav");
        let path_str = path.to_str().expect("utf8 path");
        let left = vec![0.0, 0.5, -0.5, 1.0];
        let right = vec![0.0, -0.5, 0.5, -1.0];
        write_wav_stereo_24bit(path_str, &left, &right).expect("wav write");
        let bytes = std::fs::read(&path).expect("read back");
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(bytes.len(), 44 + 4 * 6);
    }
}
