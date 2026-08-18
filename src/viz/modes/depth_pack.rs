//! V52 `depth-pack` -- The Third Dimension, Packaged.
//!
//! Ships the z-data the master render never shows: an energy-weighted depth
//! map, a red/cyan anaglyph, a wiggle-3D loop, and an 8-view Looking-Glass
//! quilt -- all derived by parallax-reprojecting the finished master still
//! against the accumulated depth field.

use crate::error::Result;
use crate::render::context::RenderContext;
use crate::render::{
    ImageBuffer, Rgb, VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass,
};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::raster::{blur_field, splat_line_additive};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::warn;

/// Maximum parallax shift as a fraction of image width (at depth extremes).
const MAX_SHIFT_FRACTION: f64 = 0.012;
/// Number of quilt views.
const VIEW_COUNT: usize = 8;
/// Splat stroke width for the depth channel (pixels at half res).
const DEPTH_STROKE_PX: f64 = 1.6;
/// Step stride for depth splatting.
const DEPTH_STRIDE: usize = 2;
/// Wiggle loop cycles.
const WIGGLE_CYCLES: usize = 8;
/// Frames each wiggle view is held (at 24 fps).
const WIGGLE_HOLD: usize = 3;

/// The depth-pack mode.
pub struct DepthPack;

/// Energy-weighted mean depth field at half resolution, normalized to [0, 1].
fn depth_field(ctx: &VizContext<'_>, half_w: usize, half_h: usize) -> Vec<f32> {
    let render_ctx =
        RenderContext::new(ctx.width, ctx.height, ctx.positions, ctx.settings.aspect_correction);
    let mut weight = vec![0.0f32; half_w * half_h];
    let mut weighted_z = vec![0.0f32; half_w * half_h];
    let steps = ctx.step_count();
    for step in (0..steps.saturating_sub(1)).step_by(DEPTH_STRIDE) {
        for edge in 0..3 {
            let a = ctx.positions[edge][step];
            let b = ctx.positions[(edge + 1) % 3][step];
            let (ax, ay) = render_ctx.to_pixel(a.x, a.y);
            let (bx, by) = render_ctx.to_pixel(b.x, b.y);
            splat_line_additive(
                &mut weight,
                &mut weighted_z,
                half_w,
                half_h,
                (ax * 0.5, ay * 0.5),
                (bx * 0.5, by * 0.5),
                a.z as f32,
                b.z as f32,
                DEPTH_STROKE_PX,
            );
        }
    }

    // Mean depth where inked; percentile-normalize into [0, 1].
    let mut depths: Vec<f32> = weight
        .iter()
        .zip(weighted_z.iter())
        .map(|(&w, &wz)| if w > 1e-4 { wz / w } else { f32::NAN })
        .collect();
    let mut sample: Vec<f32> = depths.iter().copied().filter(|value| value.is_finite()).collect();
    if sample.is_empty() {
        return vec![0.5; half_w * half_h];
    }
    sample.sort_by(f32::total_cmp);
    let low = sample[((sample.len() - 1) as f64 * 0.02) as usize];
    let high = sample[((sample.len() - 1) as f64 * 0.98) as usize];
    let span = (high - low).max(1e-6);
    for value in &mut depths {
        *value = if value.is_finite() { ((*value - low) / span).clamp(0.0, 1.0) } else { 0.5 };
    }
    // Smooth so the reprojection warp is continuous across stroke gaps.
    blur_field(&mut depths, half_w, half_h, 4.0);
    depths
}

/// Sample the half-res depth field at full-res pixel coordinates (bilinear).
fn depth_at(depths: &[f32], half_w: usize, half_h: usize, x: f64, y: f64) -> f64 {
    let sx = (x * 0.5).clamp(0.0, (half_w - 1) as f64);
    let sy = (y * 0.5).clamp(0.0, (half_h - 1) as f64);
    let x0 = sx.floor() as usize;
    let y0 = sy.floor() as usize;
    let x1 = (x0 + 1).min(half_w - 1);
    let y1 = (y0 + 1).min(half_h - 1);
    let fx = sx - x0 as f64;
    let fy = sy - y0 as f64;
    let at = |px: usize, py: usize| f64::from(depths[py * half_w + px]);
    (at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx) * (1.0 - fy)
        + (at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx) * fy
}

/// Reproject the master into one parallax view (`amount` in [-1, 1]).
fn reproject(
    master: &ImageBuffer<Rgb<u16>, Vec<u16>>,
    depths: &[f32],
    half_w: usize,
    half_h: usize,
    amount: f64,
) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    let width = master.width();
    let height = master.height();
    let max_shift = f64::from(width) * MAX_SHIFT_FRACTION;
    let mut out = ImageBuffer::new(width, height);
    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let depth = depth_at(depths, half_w, half_h, f64::from(x), f64::from(y));
        let shift = amount * max_shift * (depth - 0.5) * 2.0;
        let source_x = (f64::from(x) - shift).clamp(0.0, f64::from(width - 1));
        let x0 = source_x.floor() as u32;
        let x1 = (x0 + 1).min(width - 1);
        let fx = source_x - f64::from(x0);
        let a = master.get_pixel(x0, y);
        let b = master.get_pixel(x1, y);
        let mut blended = [0u16; 3];
        for (channel, slot) in blended.iter_mut().enumerate() {
            *slot = (f64::from(a.0[channel]) * (1.0 - fx) + f64::from(b.0[channel]) * fx).round()
                as u16;
        }
        *pixel = Rgb(blended);
    }
    out
}

impl VizMode for DepthPack {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("depth-pack").expect("depth-pack is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        // Load the finished master still (display-space Display P3 u16).
        let master_path = format!("{}/images/source/master.png", ctx.seed_dir);
        let master = match image::ImageReader::open(&master_path) {
            Ok(reader) => match reader.decode() {
                Ok(decoded) => decoded.into_rgb16(),
                Err(error) => {
                    warn!("depth-pack skipped: master decode failed: {error}");
                    return Ok(());
                }
            },
            Err(error) => {
                warn!("depth-pack skipped: master missing ({master_path}): {error}");
                return Ok(());
            }
        };

        let half_w = (ctx.width as usize / 2).max(2);
        let half_h = (ctx.height as usize / 2).max(2);
        let depths = depth_field(ctx, half_w, half_h);

        // Depth map artifact (16-bit grayscale).
        let mut depth_image: ImageBuffer<Rgb<u16>, Vec<u16>> =
            ImageBuffer::new(ctx.width, ctx.height);
        for (x, y, pixel) in depth_image.enumerate_pixels_mut() {
            let depth = depth_at(&depths, half_w, half_h, f64::from(x), f64::from(y));
            let value = (depth * 65535.0).round() as u16;
            *pixel = Rgb([value, value, value]);
        }
        sink.save_png16(&depth_image, "depth.png")?;

        // Anaglyph: grayscale-red left eye against full-color right eye
        // (retinal-rivalry-safe on saturated cores).
        let left = reproject(&master, &depths, half_w, half_h, -0.6);
        let right = reproject(&master, &depths, half_w, half_h, 0.6);
        let mut anaglyph: ImageBuffer<Rgb<u16>, Vec<u16>> =
            ImageBuffer::new(master.width(), master.height());
        for (x, y, pixel) in anaglyph.enumerate_pixels_mut() {
            let l = left.get_pixel(x, y).0;
            let r = right.get_pixel(x, y).0;
            let left_luma =
                (0.2126 * f64::from(l[0]) + 0.7152 * f64::from(l[1]) + 0.0722 * f64::from(l[2]))
                    .round() as u16;
            *pixel = Rgb([left_luma, r[1], r[2]]);
        }
        sink.save_png16(&anaglyph, "anaglyph.png")?;

        // Wiggle loop: three views ping-ponged, each held WIGGLE_HOLD frames.
        let views = [
            reproject(&master, &depths, half_w, half_h, -1.0),
            reproject(&master, &depths, half_w, half_h, 0.0),
            reproject(&master, &depths, half_w, half_h, 1.0),
        ];
        let sequence = [0usize, 1, 2, 1];
        let even_w = master.width() & !1;
        let even_h = master.height() & !1;
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("wiggle.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("wiggle_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        create_videos_from_frames_singlepass(
            even_w,
            even_h,
            24,
            |out| {
                let mut cropped: Vec<u16> = Vec::new();
                for cycle in 0..WIGGLE_CYCLES {
                    let _ = cycle;
                    for &view_index in &sequence {
                        let view = &views[view_index];
                        for _ in 0..WIGGLE_HOLD {
                            cropped.clear();
                            for y in 0..even_h {
                                let row_start = (y * master.width()) as usize * 3;
                                cropped.extend_from_slice(
                                    &view.as_raw()[row_start..row_start + even_w as usize * 3],
                                );
                            }
                            out.write_all(bytemuck::cast_slice(&cropped))
                                .map_err(crate::render::error::RenderError::VideoEncoding)?;
                        }
                    }
                }
                Ok(())
            },
            &outputs,
        )?;
        sink.record("wiggle.mp4", "video");
        sink.record("wiggle_hq.mp4", "video");

        // Looking-Glass-style quilt: 8 views, 4x2 grid at half resolution.
        let quilt_cell_w = master.width() / 2;
        let quilt_cell_h = master.height() / 2;
        let mut quilt: ImageBuffer<Rgb<u16>, Vec<u16>> =
            ImageBuffer::new(quilt_cell_w * 4, quilt_cell_h * 2);
        for view_index in 0..VIEW_COUNT {
            let amount = view_index as f64 / (VIEW_COUNT - 1) as f64 * 2.0 - 1.0;
            let view = reproject(&master, &depths, half_w, half_h, amount);
            let col = (view_index % 4) as u32;
            let row = (view_index / 4) as u32;
            for y in 0..quilt_cell_h {
                for x in 0..quilt_cell_w {
                    let source = view.get_pixel(x * 2, y * 2);
                    quilt.put_pixel(col * quilt_cell_w + x, row * quilt_cell_h + y, *source);
                }
            }
        }
        sink.save_png16(&quilt, "quilt_4x2.png")?;

        let meta = serde_json::json!({
            "max_shift_fraction": MAX_SHIFT_FRACTION,
            "views": VIEW_COUNT,
            "quilt_layout": [4, 2],
            "wiggle_fps": 24,
            "depth_normalization": "p2..p98 asinh-free linear, blurred sigma 4 at half res",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("depth_meta.json", &json, "data")?;
        Ok(())
    }
}
