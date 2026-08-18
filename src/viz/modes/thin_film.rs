//! V06 `thin-film` -- Oil-Slick Twin.
//!
//! Pushes every pixel's SPD through thin-film interference: accumulated
//! energy sets a virtual film thickness, each wavelength bin is modulated by
//! its interference reflectance before conversion, and the result is blended
//! with the unmodulated conversion so composition stays anchored. A breathing
//! sweep video oscillates the base thickness (rendered at half resolution).

use crate::error::Result;
use crate::render::context::PixelBuffer;
use crate::render::{VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass};
use crate::spectrum::{NUM_BINS, spd_to_rgba, wavelength_nm_for_bin};
use crate::viz::VizMode;
use crate::viz::VizPhase;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::grade_with_levels;
use crate::viz::common::raster::blur_field;
use crate::viz::common::spd::{energy_field, field_max};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::warn;

/// Base film thickness in nanometers.
const T0_NM: f64 = 380.0;
/// Thickness span driven by log energy (nm).
const T1_NM: f64 = 900.0;
/// Film refractive index.
const FILM_INDEX: f64 = 1.33;
/// Thickness field blur sigma in pixels (films are smooth).
const THICKNESS_BLUR_SIGMA: f64 = 3.0;
/// Blend weight of the filmed conversion over the plain conversion.
const FILM_BLEND: f64 = 0.75;
/// Sweep video length in frames (8 s at 60 fps).
const SWEEP_FRAMES: usize = 480;
/// Sweep thickness oscillation amplitude (fraction of T0).
const SWEEP_AMPLITUDE: f64 = 0.18;

/// The thin-film interference mode.
pub struct ThinFilm;

/// Interference reflectance for one wavelength at one film thickness.
#[inline]
fn reflectance(nm: f64, thickness_nm: f64) -> f64 {
    let phase = 4.0 * std::f64::consts::PI * FILM_INDEX * thickness_nm / nm + std::f64::consts::PI;
    0.5 * (1.0 + phase.cos())
}

/// Build the smoothed per-pixel thickness field from the energy field.
fn thickness_field(energy: &[f32], width: usize, height: usize, t0: f64) -> Vec<f32> {
    let max_energy = f64::from(field_max(energy)).max(1e-9);
    let log_max = (1.0 + max_energy).ln();
    let mut field: Vec<f32> = energy
        .iter()
        .map(|&value| (t0 + T1_NM * (1.0 + f64::from(value)).ln() / log_max) as f32)
        .collect();
    blur_field(&mut field, width, height, THICKNESS_BLUR_SIGMA);
    field
}

/// Convert one SPD pixel through the film alone (no blend).
#[inline]
fn convert_film_only(pixel: &[f64; NUM_BINS], thickness_nm: f64) -> (f64, f64, f64, f64) {
    let mut modulated = [0.0f64; NUM_BINS];
    for (bin, (&energy, slot)) in pixel.iter().zip(modulated.iter_mut()).enumerate() {
        *slot = energy * reflectance(wavelength_nm_for_bin(bin), thickness_nm);
    }
    spd_to_rgba(&modulated)
}

/// Blend the filmed conversion with the cached plain conversion.
#[inline]
fn blend_film(plain: (f64, f64, f64, f64), filmed: (f64, f64, f64, f64)) -> (f64, f64, f64, f64) {
    (
        plain.0 * (1.0 - FILM_BLEND) + filmed.0 * FILM_BLEND,
        plain.1 * (1.0 - FILM_BLEND) + filmed.1 * FILM_BLEND,
        plain.2 * (1.0 - FILM_BLEND) + filmed.2 * FILM_BLEND,
        plain.3 * (1.0 - FILM_BLEND) + filmed.3 * FILM_BLEND,
    )
}

/// Downsample an SPD buffer 2x by bin-wise 2x2 averaging.
fn half_res_spd(
    buffer: &[[f64; NUM_BINS]],
    width: usize,
    height: usize,
) -> (Vec<[f64; NUM_BINS]>, usize, usize) {
    let half_w = (width / 2).max(1);
    let half_h = (height / 2).max(1);
    let mut out = vec![[0.0f64; NUM_BINS]; half_w * half_h];
    out.par_iter_mut().enumerate().for_each(|(index, pixel)| {
        let x = (index % half_w) * 2;
        let y = (index / half_w) * 2;
        for dy in 0..2 {
            for dx in 0..2 {
                let sx = (x + dx).min(width - 1);
                let sy = (y + dy).min(height - 1);
                let source = &buffer[sy * width + sx];
                for (slot, &value) in pixel.iter_mut().zip(source.iter()) {
                    *slot += value * 0.25;
                }
            }
        }
    });
    (out, half_w, half_h)
}

impl VizMode for ThinFilm {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("thin-film").expect("thin-film is in the catalog")
    }

    fn phase(&self) -> VizPhase {
        VizPhase::Spd
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let Some(buffer) = ctx.accum_spd else {
            warn!("thin-film skipped: SPD buffer unavailable");
            return Ok(());
        };
        let width = ctx.width as usize;
        let height = ctx.height as usize;

        // Still at full resolution.
        let energy = ctx.energy_field().expect("SPD present implies energy field");
        let thickness = thickness_field(energy, width, height, T0_NM);
        let rgba: PixelBuffer = buffer
            .par_iter()
            .zip(thickness.par_iter())
            .map(|(pixel, &t)| {
                blend_film(spd_to_rgba(pixel), convert_film_only(pixel, f64::from(t)))
            })
            .collect();
        let image = grade_with_levels(&rgba, ctx.width, ctx.height, ctx.levels);
        sink.save_png16(&image, "thinfilm.png")?;

        // Breathing sweep video at half resolution. The plain conversion is
        // frame-invariant, so it is cached once.
        let (half, half_w, half_h) = half_res_spd(buffer, width, height);
        let half_energy = energy_field(&half);
        let half_plain: PixelBuffer = half.par_iter().map(spd_to_rgba).collect();
        let frame_count = ctx.quality.scale_count(SWEEP_FRAMES);
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("thinfilm_sweep.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("thinfilm_sweep_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        let even_w = (half_w as u32) & !1;
        let even_h = (half_h as u32) & !1;
        create_videos_from_frames_singlepass(
            even_w,
            even_h,
            60,
            |out| {
                let mut frame_u16: Vec<u16> = Vec::new();
                for frame in 0..frame_count {
                    let cycle = (std::f64::consts::TAU * frame as f64 / frame_count as f64).cos();
                    let t0 = T0_NM * (1.0 + SWEEP_AMPLITUDE * cycle);
                    let frame_thickness = thickness_field(&half_energy, half_w, half_h, t0);
                    let frame_rgba: PixelBuffer = half
                        .par_iter()
                        .zip(half_plain.par_iter())
                        .zip(frame_thickness.par_iter())
                        .map(|((pixel, &plain), &t)| {
                            blend_film(plain, convert_film_only(pixel, f64::from(t)))
                        })
                        .collect();
                    let graded =
                        grade_with_levels(&frame_rgba, half_w as u32, half_h as u32, ctx.levels);
                    // Crop to even dimensions for the encoder.
                    frame_u16.clear();
                    for y in 0..even_h {
                        let row_start = (y as usize * half_w) * 3;
                        frame_u16.extend_from_slice(
                            &graded.as_raw()[row_start..row_start + even_w as usize * 3],
                        );
                    }
                    out.write_all(bytemuck::cast_slice(&frame_u16))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                }
                Ok(())
            },
            &outputs,
        )?;
        sink.record("thinfilm_sweep.mp4", "video");
        sink.record("thinfilm_sweep_hq.mp4", "video");

        let meta = serde_json::json!({
            "t0_nm": T0_NM,
            "t1_nm": T1_NM,
            "film_index": FILM_INDEX,
            "blend": FILM_BLEND,
            "sweep_amplitude": SWEEP_AMPLITUDE,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("thinfilm.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflectance_stays_in_unit_range() {
        for nm in [380.0, 500.0, 700.0] {
            for thickness in [0.0, 250.0, 900.0, 1300.0] {
                let value = reflectance(nm, thickness);
                assert!((0.0..=1.0).contains(&value), "R({nm}, {thickness}) = {value}");
            }
        }
    }
}
