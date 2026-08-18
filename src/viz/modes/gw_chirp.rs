//! V09 `gw-chirp` -- The Sound of Spacetime.
//!
//! Computes the quadrupole-formula gravitational-wave strain of the system
//! (both polarizations from the second time derivative of the mass
//! quadrupole moment), renders it as 48 seconds of stereo audio, and
//! typesets a waveform-plus-spectrogram discovery poster.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::constants::DEFAULT_DT;
use crate::utils::fourier_transform;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::audio;
use crate::viz::common::display::encode_linear_rec2020_png16;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;

/// Stride over simulation steps when sampling the quadrupole moment.
const QUAD_STRIDE: usize = 10;
/// Audio duration in seconds.
const AUDIO_SECONDS: usize = 48;
/// High-pass cutoff removing secular drift (Hz).
const HIGH_PASS_HZ: f64 = 30.0;
/// STFT window length in samples.
const STFT_WINDOW: usize = 4096;
/// STFT hop length in samples.
const STFT_HOP: usize = 1024;
/// Poster base size at final quality.
const POSTER_WIDTH: u32 = 3456;
/// Poster height at final quality.
const POSTER_HEIGHT: u32 = 2234;

/// The gravitational-wave chirp mode.
pub struct GwChirp;

/// Second derivative by 5-point central stencil (interior points only).
fn second_derivative(series: &[f64], h: f64) -> Vec<f64> {
    let n = series.len();
    if n < 5 {
        return vec![0.0; n];
    }
    (0..n)
        .map(|index| {
            if index < 2 || index + 2 >= n {
                return 0.0;
            }
            (-series[index - 2] + 16.0 * series[index - 1] - 30.0 * series[index]
                + 16.0 * series[index + 1]
                - series[index + 2])
                / (12.0 * h * h)
        })
        .collect()
}

/// Quadrupole-formula strain of the run on a [`QUAD_STRIDE`]-strided time
/// grid: `(h_plus, h_cross)` with stencil edges trimmed. Exported for V49
/// `broadcast`, V68 `reliquary`, and V69 `pond` (their scores and drivers).
pub(crate) fn strain_series(
    positions: &[Vec<nalgebra::Vector3<f64>>],
    masses: &[f64; 3],
) -> (Vec<f64>, Vec<f64>) {
    let steps = positions.first().map_or(0, Vec::len);
    let sample_count = steps / QUAD_STRIDE;
    let mut i_xx = Vec::with_capacity(sample_count);
    let mut i_yy = Vec::with_capacity(sample_count);
    let mut i_xy = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        let step = sample * QUAD_STRIDE;
        let mut xx = 0.0;
        let mut yy = 0.0;
        let mut xy = 0.0;
        for (&mass, path) in masses.iter().zip(positions.iter()) {
            let point = path[step];
            xx += mass * point.x * point.x;
            yy += mass * point.y * point.y;
            xy += mass * point.x * point.y;
        }
        i_xx.push(xx);
        i_yy.push(yy);
        i_xy.push(xy);
    }

    let h = QUAD_STRIDE as f64 * DEFAULT_DT;
    let diff: Vec<f64> = i_xx.iter().zip(i_yy.iter()).map(|(&xx, &yy)| xx - yy).collect();
    let mut h_plus = second_derivative(&diff, h);
    let mut h_cross: Vec<f64> =
        second_derivative(&i_xy, h).iter().map(|&value| 2.0 * value).collect();
    // Trim stencil edges.
    for series in [&mut h_plus, &mut h_cross] {
        let n = series.len();
        if n > 4 {
            series[0] = series[2];
            series[1] = series[2];
            series[n - 1] = series[n - 3];
            series[n - 2] = series[n - 3];
        }
    }
    (h_plus, h_cross)
}

impl VizMode for GwChirp {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("gw-chirp").expect("gw-chirp is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        let masses = ctx.kinematics().masses;
        let (h_plus, h_cross) = strain_series(ctx.positions, &masses);

        // --- Audio: resample, condition, write.
        let total_samples = AUDIO_SECONDS * audio::SAMPLE_RATE as usize;
        let mut left = audio::resample_linear(&h_plus, total_samples);
        let mut right = audio::resample_linear(&h_cross, total_samples);
        audio::high_pass(&mut left, HIGH_PASS_HZ);
        audio::high_pass(&mut right, HIGH_PASS_HZ);
        audio::normalize_stereo_peak(&mut left, &mut right, 0.70);
        audio::fade_ends(&mut left, 2400);
        audio::fade_ends(&mut right, 2400);
        audio::write_wav_stereo_24bit(&sink.path("strain.wav"), &left, &right)?;
        sink.record("strain.wav", "audio");

        // --- Poster: waveform strip over a log-frequency spectrogram.
        let width = ctx.quality.scale_dim(POSTER_WIDTH);
        let height = ctx.quality.scale_dim(POSTER_HEIGHT);
        let w = width as usize;
        let hgt = height as usize;
        let mut pixels = vec![(0.0, 0.0, 0.0); w * hgt];

        let (anchor_l, anchor_a, anchor_b) = ctx.mean_color(0);
        let (_, anchor_chroma, anchor_hue) = oklab_to_oklch(anchor_l, anchor_a, anchor_b);
        let ramp = |magnitude: f64| -> (f64, f64, f64) {
            let t = magnitude.clamp(0.0, 1.0);
            let lightness = 0.04 + 0.86 * t.powf(0.8);
            let chroma = (anchor_chroma * 1.1 * (1.0 - (t - 0.75).max(0.0) * 4.0)).max(0.0);
            let (l, a, b) = oklch_to_oklab(lightness, chroma * t.powf(0.35), anchor_hue);
            oklab_to_linear_rec2020(l, a, b)
        };

        // Spectrogram region: rows [waveband..height).
        let waveband = hgt * 30 / 100;
        let spectrum_rows = hgt - waveband;
        let column_count = (total_samples - STFT_WINDOW) / STFT_HOP;
        let hann: Vec<f64> = (0..STFT_WINDOW)
            .map(|index| {
                0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / (STFT_WINDOW - 1) as f64).cos()
            })
            .collect();

        // Magnitude grid (columns x rows), log-frequency remapped.
        let magnitudes: Vec<Vec<f64>> = (0..column_count)
            .into_par_iter()
            .map(|column| {
                let start = column * STFT_HOP;
                let windowed: Vec<f64> =
                    (0..STFT_WINDOW).map(|index| left[start + index] * hann[index]).collect();
                let spectrum = fourier_transform(&windowed);
                let half = STFT_WINDOW / 2;
                let min_bin = 2.0_f64;
                let max_bin = half as f64;
                (0..spectrum_rows)
                    .map(|row| {
                        // Row 0 = top = high frequency; log map.
                        let t = 1.0 - row as f64 / spectrum_rows as f64;
                        let bin = (min_bin * (max_bin / min_bin).powf(t)) as usize;
                        spectrum[bin.min(half)].norm()
                    })
                    .collect()
            })
            .collect();
        let peak =
            magnitudes.iter().flat_map(|column| column.iter().copied()).fold(1e-12_f64, f64::max);

        for x in 0..w {
            let column = x * column_count / w;
            for row in 0..spectrum_rows {
                let magnitude = magnitudes[column.min(column_count - 1)][row] / peak;
                let display = (1.0 + (magnitude.max(1e-7)).log10() / 4.0).clamp(0.0, 1.0);
                pixels[(waveband + row) * w + x] = ramp(display);
            }
        }

        // Waveform strip: per-column min/max envelope of h-plus.
        let wave_color = ramp(0.85);
        let dim_color = ramp(0.45);
        let mid_row = waveband / 2;
        let amplitude_rows = (waveband as f64 * 0.42) as i64;
        for x in 0..w {
            let start = x * total_samples / w;
            let end = ((x + 1) * total_samples / w).min(total_samples);
            let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut low_c, mut high_c) = (f64::INFINITY, f64::NEG_INFINITY);
            for index in start..end.max(start + 1) {
                low = low.min(left[index]);
                high = high.max(left[index]);
                low_c = low_c.min(right[index]);
                high_c = high_c.max(right[index]);
            }
            let mut paint = |lo: f64, hi: f64, color: (f64, f64, f64)| {
                let row_lo = mid_row as i64 + (lo * amplitude_rows as f64) as i64;
                let row_hi = mid_row as i64 + (hi * amplitude_rows as f64) as i64;
                for row in row_lo.min(row_hi)..=row_lo.max(row_hi) {
                    if row >= 0 && (row as usize) < waveband {
                        pixels[row as usize * w + x] = color;
                    }
                }
            };
            paint(low_c, high_c, dim_color);
            paint(low, high, wave_color);
        }

        // Periapsis markers as quiet vertical ticks through the waveband.
        let marker = oklab_to_linear_rec2020(0.40, 0.0, 0.0);
        for approach in &ctx.events().periapses {
            let x = approach.step * w / steps.max(1);
            for row in (waveband * 85 / 100)..waveband {
                pixels[row * w + x.min(w - 1)] = marker;
            }
        }

        let image = encode_linear_rec2020_png16(&pixels, width, height);
        sink.save_png16(&image, "chirp_poster.png")?;

        let meta = serde_json::json!({
            "sample_rate": audio::SAMPLE_RATE,
            "duration_seconds": AUDIO_SECONDS,
            "quad_stride": QUAD_STRIDE,
            "high_pass_hz": HIGH_PASS_HZ,
            "periapsis_steps":
                ctx.events().periapses.iter().map(|event| event.step).collect::<Vec<_>>(),
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("strain.json", &json, "data")?;
        Ok(())
    }
}
