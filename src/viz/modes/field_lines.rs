//! V12 `field-lines` -- The Gravitational Engraving.
//!
//! At the closest triple approach: equipotential contours plus evenly spaced
//! field streamlines of the instantaneous potential, engraved in hairline
//! ink under a 12% ghost of the finished master. The video re-derives the
//! field every second frame and crossfades line layers while the trail ghost
//! re-accumulates underneath. (Contour value labels are deferred with
//! `text.rs`; the values ship in `field_params.json`.)

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::context::{PixelBuffer, RenderContext};
use crate::render::{
    SpectralRenderSettings, SpectralScene, VideoEncodingOptions, VideoOutputSpec,
    create_videos_from_frames_singlepass, quantize_display_buffer_to_16bit,
    tonemap_to_display_buffer,
};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::{Accumulator, resized_config, scene_levels};
use crate::viz::common::fields::{PointMass, Polyline, PotentialGrid, equipotentials, streamlines};
use crate::viz::common::raster::{Rgb64, draw_line, draw_line_rgba};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::warn;

/// Number of equipotential levels.
const LEVEL_COUNT: usize = 24;
/// Streamline spacing in pixels at the reference short edge (2234).
const STREAM_SPACING_REF: f64 = 22.0;
/// Ghost energy fraction of the master render.
const GHOST_ENERGY: f64 = 0.12;
/// Plummer softening in pixels (converted to world units per grid).
const SOFTENING_PX: f64 = 2.0;
/// Video frames at 60 fps (30 s).
const VIDEO_FRAMES: usize = 1800;
/// Display gamma used to move ink and ghosts between linear and display space.
const DISPLAY_GAMMA: f64 = 2.2;

/// The gravitational engraving mode.
pub struct FieldLines;

/// Ink color as linear Rec.2020 from `OKLCh` coordinates.
fn ink_linear(lightness: f64, chroma: f64, hue: f64) -> Rgb64 {
    let (l, a, b) = oklch_to_oklab(lightness, chroma, hue);
    let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), bl.clamp(0.0, 1.0))
}

/// Gamma-encode a linear ink color into display space.
fn to_display(color: Rgb64) -> Rgb64 {
    (
        color.0.powf(1.0 / DISPLAY_GAMMA),
        color.1.powf(1.0 / DISPLAY_GAMMA),
        color.2.powf(1.0 / DISPLAY_GAMMA),
    )
}

/// Palette hue of one body (`OKLCh` hue of its mean trajectory color).
fn body_hue(ctx: &VizContext<'_>, body: usize) -> f64 {
    let (l, a, b) = ctx.mean_color(body);
    let (_, _, hue) = oklab_to_oklch(l, a, b);
    hue
}

/// Point masses of the three bodies at one simulation step (world xy).
fn masses_at(ctx: &VizContext<'_>, step: usize) -> Vec<PointMass> {
    let masses = ctx.kinematics().masses;
    (0..3)
        .map(|body| PointMass {
            x: ctx.positions[body][step].x,
            y: ctx.positions[body][step].y,
            mass: masses[body],
        })
        .collect()
}

/// Softening length in world units for a context ("2 px in world units").
fn softening_world(render_ctx: &RenderContext) -> f64 {
    SOFTENING_PX * render_ctx.bounds().width / f64::from(render_ctx.width)
}

/// One recomputed field state: contour bundles and streamlines, in the
/// pixel space of the target canvas.
struct FieldState {
    contours: Vec<Vec<Polyline>>,
    streams: Vec<Polyline>,
    levels: Vec<f32>,
}

/// Compute contours + streamlines on a grid context and scale into canvas
/// pixels (`scale` = canvas px per grid px).
fn field_state(
    ctx: &VizContext<'_>,
    step: usize,
    grid_ctx: &RenderContext,
    spacing_grid_px: f64,
    scale: f32,
) -> FieldState {
    let masses = masses_at(ctx, step);
    let grid = PotentialGrid::sample(&masses, grid_ctx, softening_world(grid_ctx));
    let levels = grid.asinh_levels(LEVEL_COUNT);
    let scale_line = |line: Polyline| -> Polyline {
        line.into_iter().map(|(x, y)| (x * scale, y * scale)).collect()
    };
    let contours = equipotentials(&grid, &levels)
        .into_iter()
        .map(|bundle| bundle.into_iter().map(scale_line).collect())
        .collect();
    let streams = streamlines(&grid, spacing_grid_px).into_iter().map(scale_line).collect();
    FieldState { contours, streams, levels }
}

/// Draw one field state as ink into a linear RGB canvas.
fn draw_state_linear(
    canvas: &mut [Rgb64],
    width: usize,
    height: usize,
    state: &FieldState,
    contour_ink: Rgb64,
    stream_ink: Rgb64,
    stroke_scale: f64,
    opacity: f64,
) {
    for bundle in &state.contours {
        for line in bundle {
            for window in line.windows(2) {
                draw_line(
                    canvas,
                    width,
                    height,
                    window[0],
                    window[1],
                    contour_ink,
                    1.0 * stroke_scale,
                    0.5 * opacity,
                );
            }
        }
    }
    for line in &state.streams {
        for window in line.windows(2) {
            draw_line(
                canvas,
                width,
                height,
                window[0],
                window[1],
                stream_ink,
                1.25 * stroke_scale,
                0.85 * opacity,
            );
        }
    }
}

/// Draw one field state as display-space ink into a tonemapped frame.
fn draw_state_display(
    frame: &mut PixelBuffer,
    width: usize,
    height: usize,
    state: &FieldState,
    contour_ink: Rgb64,
    stream_ink: Rgb64,
    opacity: f64,
) {
    for bundle in &state.contours {
        for line in bundle {
            for window in line.windows(2) {
                draw_line_rgba(
                    frame,
                    width,
                    height,
                    window[0],
                    window[1],
                    contour_ink,
                    1.0,
                    0.45 * opacity,
                );
            }
        }
    }
    for line in &state.streams {
        for window in line.windows(2) {
            draw_line_rgba(
                frame,
                width,
                height,
                window[0],
                window[1],
                stream_ink,
                1.1,
                0.8 * opacity,
            );
        }
    }
}

impl VizMode for FieldLines {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("field-lines").expect("field-lines is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps == 0 {
            warn!("field-lines skipped: empty trajectory");
            return Ok(());
        }
        let closest = ctx.events().closest_triple;
        let hue = body_hue(ctx, 0);
        let contour_ink = ink_linear(0.70, 0.045, hue);
        let stream_ink = ink_linear(0.78, 0.06, hue);

        // --- Still: full-resolution engraving over a 12% master ghost.
        let width = ctx.width as usize;
        let height = ctx.height as usize;
        let full_ctx = RenderContext::new(
            ctx.width,
            ctx.height,
            ctx.positions,
            ctx.settings.aspect_correction,
        );
        let rel = f64::from(ctx.width.min(ctx.height)) / 2234.0;
        let spacing_px = (STREAM_SPACING_REF * rel).max(6.0);
        let state = field_state(ctx, closest, &full_ctx, spacing_px, 1.0);

        // Ghost: decode the finished master into linear Display P3 at 12%.
        let mut canvas: Vec<Rgb64> = vec![(0.0, 0.0, 0.0); width * height];
        let master_path = format!("{}/images/source/master.png", ctx.seed_dir);
        match image::ImageReader::open(&master_path).map(image::ImageReader::decode) {
            Ok(Ok(decoded)) => {
                let master = decoded.into_rgb16();
                if master.width() as usize == width && master.height() as usize == height {
                    for (slot, pixel) in canvas.iter_mut().zip(master.pixels()) {
                        let decode = |value: u16| {
                            (f64::from(value) / 65535.0).powf(DISPLAY_GAMMA) * GHOST_ENERGY
                        };
                        *slot = (decode(pixel.0[0]), decode(pixel.0[1]), decode(pixel.0[2]));
                    }
                } else {
                    warn!("field-lines: master.png size mismatch; engraving on black");
                }
            }
            Ok(Err(error)) => warn!("field-lines: master decode failed ({error}); black ghost"),
            Err(error) => warn!("field-lines: master missing ({error}); black ghost"),
        }

        draw_state_linear(
            &mut canvas,
            width,
            height,
            &state,
            contour_ink,
            stream_ink,
            rel.max(0.5),
            1.0,
        );

        // Triple-point sigils: a small circle around each body position.
        let sigil_radius = (10.0 * rel).max(4.0) as f32;
        for body in 0..3 {
            let position = ctx.positions[body][closest];
            let (px, py) = full_ctx.to_pixel(position.x, position.y);
            let sigil_ink = ink_linear(0.75, 0.07, body_hue(ctx, body));
            let segments = 20usize;
            for index in 0..segments {
                let a = index as f64 / segments as f64 * std::f64::consts::TAU;
                let b = (index + 1) as f64 / segments as f64 * std::f64::consts::TAU;
                draw_line(
                    &mut canvas,
                    width,
                    height,
                    (px + sigil_radius * a.cos() as f32, py + sigil_radius * a.sin() as f32),
                    (px + sigil_radius * b.cos() as f32, py + sigil_radius * b.sin() as f32),
                    sigil_ink,
                    1.2 * rel.max(0.5),
                    0.9,
                );
            }
        }

        // Encode linear Display P3 -> gamma -> 16-bit PNG.
        let mut quantized = vec![0u16; width * height * 3];
        for (chunk, &(r, g, b)) in quantized.chunks_mut(3).zip(canvas.iter()) {
            chunk[0] = (r.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA) * 65535.0).round() as u16;
            chunk[1] = (g.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA) * 65535.0).round() as u16;
            chunk[2] = (b.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA) * 65535.0).round() as u16;
        }
        let image = crate::render::ImageBuffer::from_raw(ctx.width, ctx.height, quantized)
            .expect("engraving buffer has width*height*3 samples");
        sink.save_png16(&image, "engraving.png")?;

        // --- Video: half resolution, field recomputed every 2nd frame with a
        // 2-frame crossfade, over the re-accumulating trail ghost at 12%.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);
        let video_rel = f64::from(video_w.min(video_h)) / 2234.0;

        let mut ghost = Accumulator::new(
            ctx.positions.to_vec(),
            ctx.colors.to_vec(),
            ctx.body_alphas.to_vec(),
            video_w,
            video_h,
            ctx.settings.aspect_correction,
            ctx.settings.traits,
            ctx.settings.render_config.hdr_scale,
        );
        let resized = resized_config(ctx.settings.resolved_config, video_w, video_h);
        let render_config = *ctx.settings.render_config;
        let video_settings =
            SpectralRenderSettings::new(&resized, &render_config, ctx.settings.aspect_correction)
                .with_traits(ctx.settings.traits);
        let scene = SpectralScene::new(ctx.positions, ctx.colors, ctx.body_alphas);
        let levels = scene_levels(scene, video_settings, 1.0);

        // Quarter-resolution grid context sharing the ghost's world bounds.
        let grid_ctx = RenderContext::with_bounds(
            (video_w / 4).max(16),
            (video_h / 4).max(16),
            *ghost.render_ctx().bounds(),
        );
        let grid_scale = video_w as f32 / grid_ctx.width as f32;
        let spacing_grid = (STREAM_SPACING_REF * video_rel / f64::from(grid_scale)).max(2.0);

        let contour_ink_display = to_display(contour_ink);
        let stream_ink_display = to_display(stream_ink);

        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("field.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("field_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];

        let mut accumulated = 0usize;
        let mut previous_state: Option<FieldState> = None;
        let mut current_state: Option<FieldState> = None;
        let mut rgba: PixelBuffer = Vec::new();
        let state_count = frame_count.div_ceil(2);
        create_videos_from_frames_singlepass(
            video_w,
            video_h,
            60,
            |out| {
                for frame in 0..frame_count {
                    let cursor = ((frame + 1) * steps / frame_count).min(steps);
                    ghost.accumulate(accumulated..cursor);
                    accumulated = cursor;

                    if frame % 2 == 0 {
                        previous_state = current_state.take();
                        current_state = Some(field_state(
                            ctx,
                            cursor.min(steps - 1),
                            &grid_ctx,
                            spacing_grid,
                            grid_scale,
                        ));
                    }

                    ghost.convert_into(&mut rgba);
                    for pixel in &mut rgba {
                        pixel.0 *= GHOST_ENERGY;
                        pixel.1 *= GHOST_ENERGY;
                        pixel.2 *= GHOST_ENERGY;
                    }
                    let mut display = tonemap_to_display_buffer(&rgba, &levels);

                    // 2-frame crossfade between consecutive field states.
                    let blend = if frame % 2 == 0 { 0.5 } else { 1.0 };
                    if let Some(previous) = previous_state.as_ref()
                        && blend < 1.0
                    {
                        draw_state_display(
                            &mut display,
                            video_w as usize,
                            video_h as usize,
                            previous,
                            contour_ink_display,
                            stream_ink_display,
                            1.0 - blend,
                        );
                    }
                    if let Some(current) = current_state.as_ref() {
                        draw_state_display(
                            &mut display,
                            video_w as usize,
                            video_h as usize,
                            current,
                            contour_ink_display,
                            stream_ink_display,
                            blend,
                        );
                    }

                    let bytes = quantize_display_buffer_to_16bit(&display);
                    out.write_all(bytemuck::cast_slice(&bytes))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                }
                Ok(())
            },
            &outputs,
        )?;
        sink.record("field.mp4", "video");
        sink.record("field_hq.mp4", "video");

        let meta = serde_json::json!({
            "closest_triple_step": closest,
            "levels_phi": state.levels,
            "level_count": LEVEL_COUNT,
            "stream_spacing_px": spacing_px,
            "softening_px": SOFTENING_PX,
            "ghost_energy": GHOST_ENERGY,
            "video_states": state_count,
            "note": "contour labels deferred until text.rs; values listed here",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("field_params.json", &json, "data")?;
        Ok(())
    }
}
