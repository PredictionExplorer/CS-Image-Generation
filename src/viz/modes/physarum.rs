//! V33 `physarum` -- The Organism Rediscovers the Orbit.
//!
//! Two million Physarum agents live on the artwork's energy field as their
//! food map: they rediscover, reinforce, and embellish the trails with
//! organic vein networks. A time-lapse video shows the colony claiming the
//! artwork; the still is the fully grown network over a faint master ghost.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::{
    ImageBuffer, VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass,
};
use crate::spectrum::linear_rec2020_to_display_p3;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::agents::{PhysarumParams, PhysarumSwarm, diffuse_decay};
use crate::viz::common::raster::{Rgb64, blur_field, splat_line_additive, upsample_bicubic};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Agents at final quality (draft scales to a quarter).
const AGENT_COUNT: usize = 2_000_000;
/// Simulation ticks (a video frame every 2nd tick).
const TICKS: usize = 1800;
/// Sensor distance in pixels at the reference half-res short edge (1117).
const SENSE_DISTANCE_REF: f32 = 9.0;
/// Sensor / turn angle (22.5 degrees).
const SENSE_ANGLE: f32 = std::f32::consts::FRAC_PI_8;
/// Trail deposit per agent per tick.
const DEPOSIT: f32 = 0.06;
/// Trail retention per tick.
const DECAY: f32 = 0.94;
/// Food blend weight while sensing.
const FOOD_WEIGHT: f32 = 0.35;
/// Master ghost energy in the composites.
const GHOST: f64 = 0.10;
/// Display gamma for ghost decode / ink encode.
const DISPLAY_GAMMA: f64 = 2.2;
/// Unsharp radius (pixels) for the upsampled still.
const UNSHARP_SIGMA: f64 = 1.2;

/// The Physarum colony mode.
pub struct Physarum;

/// Normalized (p99 = 1) energy density at grid resolution: the retained
/// energy field when available, else a trajectory splat-density fallback
/// (`--image-only`). Shared with V34 `frost`.
pub(crate) fn energy_density_grid(ctx: &VizContext<'_>, grid_w: usize, grid_h: usize) -> Vec<f32> {
    let mut density = if let Some(energy) = ctx.energy_field() {
        let full_w = ctx.width as usize;
        let full_h = ctx.height as usize;
        let mut down = vec![0.0f32; grid_w * grid_h];
        let scale_x = full_w as f64 / grid_w as f64;
        let scale_y = full_h as f64 / grid_h as f64;
        for (row, line) in down.chunks_mut(grid_w).enumerate() {
            let y0 = (row as f64 * scale_y) as usize;
            let y1 = (((row + 1) as f64 * scale_y) as usize).clamp(y0 + 1, full_h);
            for (col, slot) in line.iter_mut().enumerate() {
                let x0 = (col as f64 * scale_x) as usize;
                let x1 = (((col + 1) as f64 * scale_x) as usize).clamp(x0 + 1, full_w);
                let mut sum = 0.0f32;
                for y in y0..y1 {
                    for x in x0..x1 {
                        sum += energy[y * full_w + x];
                    }
                }
                *slot = sum / ((y1 - y0) * (x1 - x0)) as f32;
            }
        }
        down
    } else {
        warn!("no energy field retained (--image-only); using trajectory density fallback");
        let render_ctx = crate::render::context::RenderContext::new(
            grid_w as u32,
            grid_h as u32,
            ctx.positions,
            ctx.settings.aspect_correction,
        );
        let mut weight = vec![0.0f32; grid_w * grid_h];
        let mut value = vec![0.0f32; grid_w * grid_h];
        let steps = ctx.step_count();
        for step in (0..steps.saturating_sub(1)).step_by(4) {
            for body in 0..3 {
                let a = ctx.positions[body][step];
                let b = ctx.positions[body][step + 1];
                let (ax, ay) = render_ctx.to_pixel(a.x, a.y);
                let (bx, by) = render_ctx.to_pixel(b.x, b.y);
                splat_line_additive(
                    &mut weight,
                    &mut value,
                    grid_w,
                    grid_h,
                    (ax, ay),
                    (bx, by),
                    1.0,
                    1.0,
                    1.6,
                );
            }
        }
        blur_field(&mut weight, grid_w, grid_h, 2.0);
        weight
    };

    // Normalize by the 99th percentile.
    let mut sample: Vec<f32> = density.iter().copied().filter(|v| *v > 0.0).collect();
    if sample.is_empty() {
        return density;
    }
    sample.sort_by(f32::total_cmp);
    let reference = sample[((sample.len() - 1) as f64 * 0.99) as usize].max(1e-9);
    for value in &mut density {
        *value = (*value / reference).clamp(0.0, 1.0);
    }
    density
}

/// Food map: the normalized energy density with compressed dynamic range.
fn food_map(ctx: &VizContext<'_>, grid_w: usize, grid_h: usize) -> Vec<f32> {
    let mut food = energy_density_grid(ctx, grid_w, grid_h);
    for value in &mut food {
        *value = value.sqrt();
    }
    food
}

/// Duotone ink colors (core, halo) in linear Rec.2020 from the palette.
fn vein_inks(ctx: &VizContext<'_>) -> (Rgb64, Rgb64) {
    let (l, a, b) = ctx.mean_color(0);
    let (_, _, hue) = oklab_to_oklch(l, a, b);
    let convert = |lightness: f64, chroma: f64| {
        let (ll, aa, bb) = oklch_to_oklab(lightness, chroma, hue);
        let (r, g, bl) = oklab_to_linear_rec2020(ll, aa, bb);
        (r.max(0.0), g.max(0.0), bl.max(0.0))
    };
    (convert(0.8, 0.09), convert(0.35, 0.05))
}

/// Smoothstep on [lo, hi].
#[inline]
fn smoothstep(lo: f32, hi: f32, value: f32) -> f32 {
    let t = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Compose one frame: ghost + duotone veins, encoded to display u16.
#[allow(clippy::too_many_arguments)]
fn compose_frame(
    trail: &[f32],
    ghost: &[Rgb64],
    width: usize,
    height: usize,
    normalization: f32,
    core: Rgb64,
    halo: Rgb64,
    out: &mut Vec<u16>,
) {
    use rayon::prelude::*;
    out.resize(width * height * 3, 0);
    out.par_chunks_mut(3).enumerate().for_each(|(index, chunk)| {
        let t = trail[index] / normalization;
        let halo_w = f64::from(smoothstep(0.04, 0.5, t)) * 0.8;
        let core_w = f64::from(smoothstep(0.5, 1.6, t));
        let base = ghost[index];
        let linear = (
            base.0 + halo.0 * halo_w + core.0 * core_w,
            base.1 + halo.1 * halo_w + core.1 * core_w,
            base.2 + halo.2 * halo_w + core.2 * core_w,
        );
        let (p3_r, p3_g, p3_b) = linear_rec2020_to_display_p3(linear.0, linear.1, linear.2);
        chunk[0] = (p3_r.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA) * 65535.0).round() as u16;
        chunk[1] = (p3_g.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA) * 65535.0).round() as u16;
        chunk[2] = (p3_b.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA) * 65535.0).round() as u16;
    });
}

/// Load the master ghost at the requested size (linear Rec.2020-ish P3
/// decode scaled to `GHOST`); black when the master is unavailable.
fn master_ghost(ctx: &VizContext<'_>, width: usize, height: usize) -> Vec<Rgb64> {
    let path = format!("{}/images/source/master.png", ctx.seed_dir);
    let mut ghost = vec![(0.0, 0.0, 0.0); width * height];
    match image::ImageReader::open(&path).map(image::ImageReader::decode) {
        Ok(Ok(decoded)) => {
            let master = decoded.into_rgb16();
            let source_w = master.width() as usize;
            let source_h = master.height() as usize;
            let raw = master.as_raw();
            let scale_x = source_w as f64 / width as f64;
            let scale_y = source_h as f64 / height as f64;
            for (row, line) in ghost.chunks_mut(width).enumerate() {
                let sy = ((row as f64 + 0.5) * scale_y) as usize;
                let sy = sy.min(source_h - 1);
                for (col, slot) in line.iter_mut().enumerate() {
                    let sx = (((col as f64 + 0.5) * scale_x) as usize).min(source_w - 1);
                    let base = (sy * source_w + sx) * 3;
                    let decode = |v: u16| (f64::from(v) / 65535.0).powf(DISPLAY_GAMMA) * GHOST;
                    *slot = (decode(raw[base]), decode(raw[base + 1]), decode(raw[base + 2]));
                }
            }
        }
        Ok(Err(error)) => warn!("physarum: master decode failed ({error}); black ghost"),
        Err(error) => warn!("physarum: master missing ({error}); black ghost"),
    }
    ghost
}

impl VizMode for Physarum {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("physarum").expect("physarum is in the catalog")
    }

    fn needs_energy_field(&self) -> bool {
        true
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        if ctx.step_count() == 0 {
            warn!("physarum skipped: empty trajectory");
            return Ok(());
        }
        let grid_w = (((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16)) as usize;
        let grid_h = (((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16)) as usize;
        let agents = ctx.quality.scale_count(AGENT_COUNT);
        let ticks = ctx.quality.scale_count(TICKS);
        let relative = grid_h.min(grid_w) as f32 / 1117.0;

        let food = food_map(ctx, grid_w, grid_h);
        let mut rng = ctx.fork_rng(self.entry().flag);
        let mut swarm = PhysarumSwarm::seed_weighted(agents, grid_w, grid_h, &food, &mut rng);
        info!("   physarum: {agents} agents on a {grid_w}x{grid_h} grid, {ticks} ticks");

        let params = PhysarumParams {
            sense_distance: (SENSE_DISTANCE_REF * relative).max(2.0),
            sense_angle: SENSE_ANGLE,
            turn_angle: SENSE_ANGLE,
            step_length: (relative).max(0.5),
            deposit: DEPOSIT,
            decay: DECAY,
            jitter: 0.12,
            food_weight: FOOD_WEIGHT,
        };
        // Steady-state trail scale: deposit inflow / (cells * dissipation),
        // boosted for vein concentration (agents crowd a fraction of cells).
        let steady = DEPOSIT * agents as f32 / (grid_w * grid_h) as f32 / (1.0 - DECAY);
        let normalization = (steady * 16.0).max(1e-6);

        let (core, halo) = vein_inks(ctx);
        let ghost = master_ghost(ctx, grid_w, grid_h);

        let mut trail = vec![0.0f32; grid_w * grid_h];
        let mut scratch = Vec::new();
        let mut deposits: Vec<(u32, f32)> = Vec::new();
        let mut frame_bytes: Vec<u16> = Vec::new();
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("physarum.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("physarum_hq.mp4"),
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
            30,
            |out| {
                for tick in 0..ticks {
                    swarm.tick(&trail, &food, &params, tick as u64, &mut deposits);
                    PhysarumSwarm::apply_deposits(&mut trail, &deposits);
                    diffuse_decay(&mut trail, &mut scratch, grid_w, grid_h, DECAY);
                    if tick.is_multiple_of(2) {
                        compose_frame(
                            &trail,
                            &ghost,
                            grid_w,
                            grid_h,
                            normalization,
                            core,
                            halo,
                            &mut frame_bytes,
                        );
                        out.write_all(bytemuck::cast_slice(&frame_bytes))
                            .map_err(crate::render::error::RenderError::VideoEncoding)?;
                    }
                }
                Ok(())
            },
            &outputs,
        )?;
        sink.record("physarum.mp4", "video");
        sink.record("physarum_hq.mp4", "video");

        // --- Still: upsample the grown network, re-sharpen, full-res ghost.
        let still_w = ctx.quality.scale_dim(ctx.width) as usize;
        let still_h = ctx.quality.scale_dim(ctx.height) as usize;
        let mut grown = upsample_bicubic(&trail, grid_w, grid_h, still_w, still_h);
        let mut blurred = grown.clone();
        blur_field(&mut blurred, still_w, still_h, UNSHARP_SIGMA);
        for (value, blur) in grown.iter_mut().zip(blurred.iter()) {
            *value = (*value + 1.2 * (*value - *blur)).max(0.0);
        }
        let full_ghost = master_ghost(ctx, still_w, still_h);
        let mut still_bytes: Vec<u16> = Vec::new();
        compose_frame(
            &grown,
            &full_ghost,
            still_w,
            still_h,
            normalization,
            core,
            halo,
            &mut still_bytes,
        );
        let image = ImageBuffer::from_raw(still_w as u32, still_h as u32, still_bytes)
            .expect("physarum still buffer has width*height*3 samples");
        sink.save_png16(&image, "physarum.png")?;

        let meta = serde_json::json!({
            "agents": agents,
            "ticks": ticks,
            "grid": [grid_w, grid_h],
            "sense_distance_px": params.sense_distance,
            "deposit": DEPOSIT,
            "decay": DECAY,
            "food_weight": FOOD_WEIGHT,
            "trail_normalization": normalization,
            "ghost_energy": GHOST,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("physarum_params.json", &json, "data")?;
        Ok(())
    }
}
