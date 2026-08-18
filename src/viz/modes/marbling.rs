//! V36 `marbling` -- Suminagashi Stirred by Gravity.
//!
//! The three bodies act as moving stirrers in an incompressible fluid; each
//! continuously injects its palette dye. Chaotic advection paints authentic
//! paper-marbling filaments -- the orbit expressed as the wake it leaves in
//! a medium. The still re-runs the final seconds at double resolution from a
//! checkpoint for endpaper-grade filaments.

use crate::error::Result;
use crate::oklab::oklab_to_linear_rec2020;
use crate::render::context::RenderContext;
use crate::render::{
    ImageBuffer, OklabColor, VideoEncodingOptions, VideoOutputSpec,
    create_videos_from_frames_singlepass,
};
use crate::spectrum::linear_rec2020_to_display_p3;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::fluid::Fluid;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::{info, warn};

/// Video frames at 60 fps (30 s).
const VIDEO_FRAMES: usize = 1800;
/// Stirrer radius in cells at the reference half-res short edge (1117).
const STIR_RADIUS_REF: f32 = 14.0;
/// Stirrer impulse per frame (cells/s) at full normalized speed.
const STIR_IMPULSE: f32 = 4.0;
/// Dye injection radius (cells at reference).
const DYE_RADIUS_REF: f32 = 6.0;
/// Dye opacity injected per second.
const DYE_PER_SECOND: f32 = 0.15;
/// Frames between dye contrast (S-curve) passes.
const SHARPEN_INTERVAL: usize = 240;
/// S-curve strength.
const SHARPEN: f32 = 0.08;
/// Tail frames re-run at double resolution for the still.
const TAIL_FRAMES: usize = 120;
/// Paper (background) linear luminance.
const PAPER: f64 = 0.004;
/// Display gamma.
const DISPLAY_GAMMA: f64 = 2.2;

/// The marbling mode.
pub struct Marbling;

/// Mild S-curve on each dye field (normalized to its own maximum).
fn sharpen_dye(fluid: &mut Fluid) {
    for field in 0..3 {
        let peak = fluid.dye[field].iter().copied().fold(0.0f32, f32::max).max(1e-6);
        fluid.dye[field].par_iter_mut().for_each(|value| {
            let normalized = *value / peak;
            *value += SHARPEN * (normalized - 0.5) * normalized * (1.0 - normalized) * peak;
            *value = value.max(0.0);
        });
    }
}

/// Compose the dye fields into a display-encoded u16 frame.
fn compose_frame(
    fluid: &Fluid,
    body_colors: &[OklabColor; 3],
    width: usize,
    height: usize,
    out: &mut Vec<u16>,
) {
    let linear_colors: [(f64, f64, f64); 3] = std::array::from_fn(|body| {
        let (l, a, b) = body_colors[body];
        let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
        (r.max(0.0), g.max(0.0), bl.max(0.0))
    });
    out.resize(width * height * 3, 0);
    let dye = &fluid.dye;
    out.par_chunks_mut(3).enumerate().for_each(|(index, chunk)| {
        let concentrations = [
            f64::from(dye[0][index].clamp(0.0, 2.0)),
            f64::from(dye[1][index].clamp(0.0, 2.0)),
            f64::from(dye[2][index].clamp(0.0, 2.0)),
        ];
        let total = concentrations[0] + concentrations[1] + concentrations[2];
        // Glaze: blend toward paper at low concentration.
        let saturation = total / (total + 0.05);
        let mut linear = (PAPER, PAPER, PAPER);
        if total > 1e-6 {
            let mut mix = (0.0, 0.0, 0.0);
            for body in 0..3 {
                let weight = concentrations[body] / total;
                mix.0 += linear_colors[body].0 * weight;
                mix.1 += linear_colors[body].1 * weight;
                mix.2 += linear_colors[body].2 * weight;
            }
            linear = (
                PAPER + (mix.0 - PAPER) * saturation,
                PAPER + (mix.1 - PAPER) * saturation,
                PAPER + (mix.2 - PAPER) * saturation,
            );
        }
        let (p3_r, p3_g, p3_b) = linear_rec2020_to_display_p3(linear.0, linear.1, linear.2);
        let encode =
            |v: f64| (v.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA) * 65535.0).round() as u16;
        chunk[0] = encode(p3_r);
        chunk[1] = encode(p3_g);
        chunk[2] = encode(p3_b);
    });
}

/// Apply one frame's stirring and dye injection at a given resolution.
#[allow(clippy::too_many_arguments)]
fn stir_and_inject(
    fluid: &mut Fluid,
    ctx: &VizContext<'_>,
    render_ctx: &RenderContext,
    step: usize,
    speed_window: (f64, f64),
    relative: f32,
    dye_amount: f32,
) {
    let kinematics = ctx.kinematics();
    for body in 0..3 {
        let position = ctx.positions[body][step];
        let (px, py) = render_ctx.to_pixel(position.x, position.y);
        let velocity = kinematics.velocities[body][step];
        let speed = kinematics.speeds[body][step];
        let normalized = kinematics.normalized_speed(speed_window, speed).max(0.05);
        let direction = {
            let norm = velocity.x.hypot(velocity.y).max(1e-12);
            ((velocity.x / norm) as f32, (velocity.y / norm) as f32)
        };
        let impulse = STIR_IMPULSE * relative * normalized as f32;
        fluid.add_force(
            px,
            py,
            STIR_RADIUS_REF * relative,
            direction.0 * impulse,
            direction.1 * impulse,
        );
        fluid.inject_dye(body, px, py, DYE_RADIUS_REF * relative, dye_amount);
    }
}

impl VizMode for Marbling {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("marbling").expect("marbling is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps == 0 {
            warn!("marbling skipped: empty trajectory");
            return Ok(());
        }
        let grid_w = (((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16)) as usize;
        let grid_h = (((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16)) as usize;
        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);
        let tail = TAIL_FRAMES.min(frame_count / 4).max(1);
        let relative = grid_w.min(grid_h) as f32 / 1117.0;
        let dt = 1.0 / 60.0;
        // Keep the total injected dye constant across quality levels (draft
        // renders a quarter of the frames).
        let dye_per_frame = DYE_PER_SECOND * (VIDEO_FRAMES as f32 / frame_count as f32) / 60.0;
        let speed_window = ctx.kinematics().speed_window();
        info!("   marbling: {grid_w}x{grid_h} fluid, {frame_count} frames");

        let render_ctx = RenderContext::new(
            grid_w as u32,
            grid_h as u32,
            ctx.positions,
            ctx.settings.aspect_correction,
        );

        let mut fluid = Fluid::new(grid_w, grid_h);
        let mut checkpoint: Option<Fluid> = None;
        let mut frame_bytes: Vec<u16> = Vec::new();
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("marbling.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("marbling_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        create_videos_from_frames_singlepass(
            grid_w as u32,
            grid_h as u32,
            60,
            |out| {
                for frame in 0..frame_count {
                    let step = ((frame + 1) * steps / frame_count).min(steps - 1);
                    stir_and_inject(
                        &mut fluid,
                        ctx,
                        &render_ctx,
                        step,
                        speed_window,
                        relative,
                        dye_per_frame,
                    );
                    fluid.step(dt);
                    if (frame + 1).is_multiple_of(SHARPEN_INTERVAL) {
                        sharpen_dye(&mut fluid);
                    }
                    if frame + tail == frame_count {
                        checkpoint = Some(fluid.upsampled(grid_w, grid_h));
                    }
                    let body_colors: [OklabColor; 3] = std::array::from_fn(|body| {
                        ctx.colors[body][step.min(ctx.colors[body].len() - 1)]
                    });
                    compose_frame(&fluid, &body_colors, grid_w, grid_h, &mut frame_bytes);
                    out.write_all(bytemuck::cast_slice(&frame_bytes))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                }
                Ok(())
            },
            &outputs,
        )?;
        sink.record("marbling.mp4", "video");
        sink.record("marbling_hq.mp4", "video");

        // --- Still: re-run the tail at double resolution from the checkpoint.
        let still_w = (ctx.quality.scale_dim(ctx.width) as usize).max(grid_w);
        let still_h = (ctx.quality.scale_dim(ctx.height) as usize).max(grid_h);
        let mut still_fluid = match checkpoint {
            Some(state) => state.upsampled(still_w, still_h),
            None => fluid.upsampled(still_w, still_h),
        };
        let still_ctx = RenderContext::new(
            still_w as u32,
            still_h as u32,
            ctx.positions,
            ctx.settings.aspect_correction,
        );
        let still_relative = still_w.min(still_h) as f32 / 1117.0;
        for frame in (frame_count - tail)..frame_count {
            let step = ((frame + 1) * steps / frame_count).min(steps - 1);
            stir_and_inject(
                &mut still_fluid,
                ctx,
                &still_ctx,
                step,
                speed_window,
                still_relative,
                dye_per_frame,
            );
            still_fluid.step(dt);
        }
        let final_step = steps - 1;
        let body_colors: [OklabColor; 3] = std::array::from_fn(|body| {
            ctx.colors[body][final_step.min(ctx.colors[body].len() - 1)]
        });
        let mut still_bytes: Vec<u16> = Vec::new();
        compose_frame(&still_fluid, &body_colors, still_w, still_h, &mut still_bytes);
        let image = ImageBuffer::from_raw(still_w as u32, still_h as u32, still_bytes)
            .expect("marbling still buffer has width*height*3 samples");
        sink.save_png16(&image, "marbling.png")?;

        let meta = serde_json::json!({
            "grid": [grid_w, grid_h],
            "frames": frame_count,
            "tail_frames_at_2x": tail,
            "stir_impulse": STIR_IMPULSE,
            "dye_per_second": DYE_PER_SECOND,
            "sharpen_interval": SHARPEN_INTERVAL,
            "note": "collocated stable-fluids solver (GPU-Gems form) with MacCormack dye",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("marbling_params.json", &json, "data")?;
        Ok(())
    }
}
