//! V56 `tilt` -- The Poster That Plays.
//!
//! Lenticular print masters: column-interlaced sheets where tilting the
//! physical print scrubs time (24 drama-weighted frames from the main
//! render's archive) or rocks parallax (the 8 views of V52's quilt).
//! Strips are exactly one pixel at the sheet's native resolution
//! (`LPI x frames` DPI), so widths are exact by construction; a pitch-test
//! ladder and a spec sheet with the print parameters ship alongside.

use crate::error::Result;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::encode_linear_rec2020_png16;
use crate::viz::common::raster::Rgb64;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Lens pitch (lenses per inch) of the named sheet.
const LPI: f64 = 40.0;
/// Time-flip frame count.
const TIME_FRAMES: usize = 24;
/// Physical master size in millimeters (width, height).
const SHEET_MM: (f64, f64) = (200.0, 133.0);
/// Tone-match clamp in EV around the ensemble mean.
const TONE_CLAMP_EV: f64 = 0.3;

/// The tilt mode.
pub struct Tilt;

/// Interlace `frames` (each `w x h` linear RGB) into a 1-px-strip master.
fn interlace(
    frames: &[Vec<Rgb64>],
    w: usize,
    h: usize,
    lenses: usize,
) -> (Vec<Rgb64>, usize, usize) {
    let count = frames.len();
    let out_w = lenses * count;
    let out_h = out_w * h / w;
    let mut master = vec![(0.0, 0.0, 0.0); out_w * out_h];
    for (column, slot_base) in (0..out_w).map(|c| (c, c)) {
        let lens = column / count;
        let frame_index = column % count;
        // Right-tilt shows later frames: reverse strip order inside a lens.
        let frame = &frames[count - 1 - frame_index];
        // Source column for this lens center.
        let sx = (lens * w / lenses).min(w - 1);
        for y in 0..out_h {
            let sy = (y * h / out_h).min(h - 1);
            master[y * out_w + slot_base] = frame[sy * w + sx];
        }
    }
    (master, out_w, out_h)
}

impl VizMode for Tilt {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("tilt").expect("tilt is in the catalog")
    }

    fn needs_frame_archive(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let Some(archive) = &ctx.frame_archive else {
            warn!("tilt skipped: no frame archive (image-only run)");
            return Ok(());
        };
        if archive.frames.is_empty() {
            warn!("tilt skipped: frame archive is empty");
            return Ok(());
        }
        let aw = archive.width as usize;
        let ah = archive.height as usize;
        let scale = match ctx.quality {
            crate::viz::context::VizQuality::Final => 1.0,
            crate::viz::context::VizQuality::Draft => 0.25,
        };
        let lenses = ((SHEET_MM.0 / 25.4 * LPI) * scale) as usize;
        let dpi_time = LPI * TIME_FRAMES as f64;

        // --- Time flip: 24 evenly picked archive frames, tone-matched.
        let picks: Vec<usize> = (0..TIME_FRAMES)
            .map(|i| i * (archive.frames.len() - 1) / (TIME_FRAMES - 1).max(1))
            .collect();
        let mut frames: Vec<Vec<Rgb64>> = picks
            .iter()
            .map(|&pick| {
                archive.frames[pick]
                    .1
                    .chunks(3)
                    .map(|px| {
                        let decode = |v: u8| (f64::from(v) / 255.0).powf(2.2);
                        (decode(px[0]), decode(px[1]), decode(px[2]))
                    })
                    .collect()
            })
            .collect();
        // Tone match: clamp each frame's mean luminance to +/- 0.3 EV of
        // the ensemble mean (adjacent-frame decorrelation).
        let means: Vec<f64> = frames
            .iter()
            .map(|frame| frame.iter().map(|&(r, g, b)| r + g + b).sum::<f64>() / frame.len() as f64)
            .collect();
        let ensemble = means.iter().sum::<f64>() / means.len() as f64;
        let clamp = 2.0_f64.powf(TONE_CLAMP_EV);
        for (frame, &mean) in frames.iter_mut().zip(means.iter()) {
            let gain = (ensemble / mean.max(1e-9)).clamp(1.0 / clamp, clamp);
            for pixel in frame.iter_mut() {
                pixel.0 *= gain;
                pixel.1 *= gain;
                pixel.2 *= gain;
            }
        }
        let (master, mw, mh) = interlace(&frames, aw, ah, lenses);
        info!(
            "   tilt: time master {mw}x{mh} ({} lenses x {TIME_FRAMES} frames, {dpi_time:.0} DPI)",
            lenses
        );
        let image = encode_linear_rec2020_png16(&master, mw as u32, mh as u32);
        sink.save_png16(&image, "tilt_time_40lpi.png")?;

        // --- Depth flip from V52's quilt, when it exists.
        let mut depth_report = String::from("skipped (viz/depth-pack/quilt_4x2.png not found)");
        let quilt_path = format!("{}/viz/depth-pack/quilt_4x2.png", ctx.seed_dir);
        if let Ok(Ok(decoded)) =
            image::ImageReader::open(&quilt_path).map(image::ImageReader::decode)
        {
            let quilt = decoded.into_rgb16();
            let qw = quilt.width() as usize / 4;
            let qh = quilt.height() as usize / 2;
            let raw = quilt.as_raw();
            let views: Vec<Vec<Rgb64>> = (0..8)
                .map(|view| {
                    let origin_x = (view % 4) * qw;
                    let origin_y = (view / 4) * qh;
                    let mut pixels = Vec::with_capacity(qw * qh);
                    for y in 0..qh {
                        for x in 0..qw {
                            let base = ((origin_y + y) * quilt.width() as usize + origin_x + x) * 3;
                            let decode = |v: u16| (f64::from(v) / 65535.0).powf(2.2);
                            pixels.push((
                                decode(raw[base]),
                                decode(raw[base + 1]),
                                decode(raw[base + 2]),
                            ));
                        }
                    }
                    pixels
                })
                .collect();
            let (depth_master, dw, dh) = interlace(&views, qw, qh, lenses);
            let depth_image = encode_linear_rec2020_png16(&depth_master, dw as u32, dh as u32);
            sink.save_png16(&depth_image, "tilt_depth_40lpi.png")?;
            depth_report = format!("{dw}x{dh} at {:.0} DPI (8 views)", LPI * 8.0);
        } else {
            warn!("tilt: depth flip skipped (run depth-pack first for the quilt)");
        }

        // --- Pitch test: black/white ladders from 39.6 to 40.4 LPI.
        let ladder_w = lenses * TIME_FRAMES;
        let band_h = (ladder_w / 24).max(8);
        let bands = 9usize;
        let mut pitch = vec![(0.92, 0.92, 0.90); ladder_w * band_h * bands];
        for band in 0..bands {
            let lpi = 39.6 + 0.1 * band as f64;
            let strip_px = dpi_time / lpi / 2.0;
            for y in 0..band_h {
                for x in 0..ladder_w {
                    let on = ((x as f64 / strip_px) as usize).is_multiple_of(2);
                    if on {
                        pitch[(band * band_h + y) * ladder_w + x] = (0.01, 0.01, 0.012);
                    }
                }
            }
        }
        let pitch_image =
            encode_linear_rec2020_png16(&pitch, ladder_w as u32, (band_h * bands) as u32);
        sink.save_png16(&pitch_image, "pitch_test.png")?;

        // --- Spec sheet.
        let spec = format!(
            "three-body lenticular masters -- seed 0x{seed}\n\
             \n\
             lens sheet: 40 LPI standard lenticular (e.g. Micro Lens\n\
             Technology 40 LPI 3D/flip sheet, or equivalent)\n\
             \n\
             tilt_time_40lpi.png\n\
             - {mw} x {mh} px, print at {dpi_time:.0} DPI -> {w_mm:.0} x {h_mm:.0} mm\n\
             - {frames} frames, 1 px per strip (exact); tilt scrubs time\n\
             \n\
             tilt_depth_40lpi.png\n\
             - {depth}\n\
             - print at {dpi_depth:.0} DPI for the same lens sheet\n\
             \n\
             pitch_test.png\n\
             - 9 ladders, 39.6-40.4 LPI in 0.1 steps at {dpi_time:.0} DPI;\n\
               print, lay the lens over, pick the band with no moire and\n\
               scale the master by (40.0 / chosen LPI) before the final run\n\
             \n\
             alignment: strips run vertically; laminate with the lens\n\
             lenticules vertical; register the left sheet edge to a strip\n\
             boundary.\n",
            seed = ctx.seed_hex.to_uppercase(),
            mw = mw,
            mh = mh,
            dpi_time = dpi_time,
            w_mm = mw as f64 / dpi_time * 25.4,
            h_mm = mh as f64 / dpi_time * 25.4,
            frames = TIME_FRAMES,
            depth = depth_report,
            dpi_depth = LPI * 8.0,
        );
        sink.write_text("tilt_spec.txt", &spec, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interlace_strip_widths_are_exactly_one_pixel() {
        // Two solid-color frames: the master must alternate columns
        // perfectly (frame order reversed inside each lens).
        let red = vec![(1.0, 0.0, 0.0); 64 * 32];
        let blue = vec![(0.0, 0.0, 1.0); 64 * 32];
        let (master, mw, _) = interlace(&[red, blue], 64, 32, 10);
        assert_eq!(mw, 20);
        for (column, &pixel) in master.iter().enumerate().take(mw) {
            let expect_blue = column % 2 == 0;
            if expect_blue {
                assert!(pixel.2 > 0.9 && pixel.0 < 0.1, "column {column} must be frame 2");
            } else {
                assert!(pixel.0 > 0.9 && pixel.2 < 0.1, "column {column} must be frame 1");
            }
        }
    }
}
