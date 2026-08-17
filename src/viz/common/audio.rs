//! Minimal deterministic audio helpers: a hand-rolled 24-bit stereo WAV
//! writer plus resampling and conditioning utilities shared by the audio
//! modes (`gw-chirp`, `oscilloscope`, and future sonifications).

use std::fs::File;
use std::io::{self, BufWriter, Write};

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
