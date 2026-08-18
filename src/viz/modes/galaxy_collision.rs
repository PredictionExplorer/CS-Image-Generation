//! V38 `galaxy-collision` -- The Antennae, Choreographed.
//!
//! Each body hosts a rotating exponential disk of test stars; every close
//! passage rips tidal tails and bridges exactly as interacting galaxies do.
//! Stars integrate through the full time-varying three-body field (the
//! restricted approximation), splat with per-star palette jitter, and the
//! final exposure is a galactic wreck portrait unique to the seed.

use crate::error::Result;
use crate::render::constants::DEFAULT_DT;
use crate::render::context::{PixelBuffer, RenderContext};
use crate::render::{LineVertex, OklabColor, SpectralLineSegment};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::{Accumulator, stream_video};
use crate::viz::common::display::{auto_levels, grade_auto_levels};
use crate::viz::common::particles::{BandedSpd, FieldParams, Swarm, seed_disk};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;
use tracing::info;

/// Stars per host disk at final quality (draft scales to a quarter).
const STARS_PER_DISK: usize = 30_000;
/// Simulation steps per star integration step.
const STAR_STRIDE: usize = 4;
/// Star steps per splatted stroke.
const SPLAT_STRIDE: usize = 4;
/// Disk scale radius as a fraction of the initial minimum pair separation.
const SCALE_RADIUS_FRACTION: f64 = 0.12;
/// Maximum disk tilt magnitude (radians, ~18 degrees).
const TILT_MAX: f64 = 0.314;
/// Per-splat alpha.
const SPLAT_ALPHA: f64 = 0.008;
/// Stroke energy.
const ENERGY: f64 = 0.010;
/// `OKLab` a/b jitter for stellar-population texture.
const HUE_JITTER: f64 = 0.03;
/// Star weight in the composite.
const COMPOSITE_STAR_WEIGHT: f64 = 0.8;
/// Master weight in the composite.
const COMPOSITE_MASTER_WEIGHT: f64 = 0.6;
/// Plummer softening (world units) for star integration and seeding.
const SOFTENING: f64 = 0.02;
/// Video frames at 60 fps (30 s).
const VIDEO_FRAMES: usize = 1800;
/// Probe divisor for the fixed-levels pre-pass.
const PROBE_DIVISOR: usize = 4;

/// The galaxy collision mode.
pub struct GalaxyCollision;

/// Initial minimum pair separation.
fn initial_min_separation(ctx: &VizContext<'_>) -> f64 {
    let mut best = f64::INFINITY;
    for a in 0..3 {
        for b in a + 1..3 {
            best = best.min((ctx.positions[a][0] - ctx.positions[b][0]).norm());
        }
    }
    best.max(1e-9)
}

/// Star rendering state advanced in lockstep by the sim loop.
struct StarRun<'a> {
    ctx: &'a VizContext<'a>,
    swarm: Swarm,
    params: FieldParams,
    masses: [f64; 3],
    host: Vec<u8>,
    jitter: Vec<(f64, f64)>,
    last_px: Vec<(f32, f32)>,
    full_ctx: RenderContext,
}

impl<'a> StarRun<'a> {
    fn new(ctx: &'a VizContext<'a>, swarm: Swarm, host: Vec<u8>, jitter: Vec<(f64, f64)>) -> Self {
        let full_ctx = RenderContext::new(
            ctx.width,
            ctx.height,
            ctx.positions,
            ctx.settings.aspect_correction,
        );
        let last_px: Vec<(f32, f32)> =
            swarm.positions.iter().map(|p| full_ctx.to_pixel(p.x, p.y)).collect();
        Self {
            ctx,
            swarm,
            params: FieldParams {
                softening_sq: SOFTENING * SOFTENING,
                freeze_radius_sq: None,
                freeze_center: Vector3::zeros(),
            },
            masses: ctx.kinematics().masses,
            host,
            jitter,
            last_px,
            full_ctx,
        }
    }

    fn advance(
        &mut self,
        range: std::ops::Range<usize>,
        canvases: &mut [(&mut BandedSpd, f32)],
        segments: &mut Vec<SpectralLineSegment>,
    ) {
        let steps = self.ctx.step_count();
        for star_step in range {
            let sim_now = (star_step * STAR_STRIDE).min(steps - 1);
            let sim_next = ((star_step + 1) * STAR_STRIDE).min(steps - 1);
            let bodies_now = std::array::from_fn(|body| self.ctx.positions[body][sim_now]);
            let bodies_next = std::array::from_fn(|body| self.ctx.positions[body][sim_next]);
            self.swarm.step(
                &bodies_now,
                &bodies_next,
                self.masses,
                DEFAULT_DT * STAR_STRIDE as f64,
                &self.params,
            );

            if !(star_step + 1).is_multiple_of(SPLAT_STRIDE) {
                continue;
            }
            let host_colors: [OklabColor; 3] = std::array::from_fn(|body| {
                self.ctx.colors[body][sim_next.min(self.ctx.colors[body].len() - 1)]
            });
            segments.clear();
            for index in 0..self.swarm.len() {
                let position = self.swarm.positions[index];
                let (px, py) = self.full_ctx.to_pixel(position.x, position.y);
                let (last_x, last_y) = self.last_px[index];
                self.last_px[index] = (px, py);
                if !px.is_finite() || !py.is_finite() {
                    continue;
                }
                let base = host_colors[self.host[index] as usize];
                let (ja, jb) = self.jitter[index];
                let color = (base.0, base.1 + ja, base.2 + jb);
                segments.push(SpectralLineSegment {
                    start: LineVertex { x: last_x, y: last_y, z: 0.0, color, alpha: SPLAT_ALPHA },
                    end: LineVertex { x: px, y: py, z: 0.0, color, alpha: SPLAT_ALPHA },
                    hdr_scale: ENERGY,
                    thickness_factor: 0.5,
                });
            }
            for (canvas, scale) in canvases.iter_mut() {
                if (*scale - 1.0).abs() < f32::EPSILON {
                    canvas.splat(segments);
                } else {
                    let scaled: Vec<SpectralLineSegment> = segments
                        .iter()
                        .map(|segment| {
                            let mut scaled = *segment;
                            scaled.start.x *= *scale;
                            scaled.start.y *= *scale;
                            scaled.end.x *= *scale;
                            scaled.end.y *= *scale;
                            scaled
                        })
                        .collect();
                    canvas.splat(&scaled);
                }
            }
        }
    }
}

impl VizMode for GalaxyCollision {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("galaxy-collision").expect("galaxy-collision is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps == 0 {
            tracing::warn!("galaxy-collision skipped: empty trajectory");
            return Ok(());
        }
        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;
        let masses = ctx.kinematics().masses;
        let star_steps = (steps / STAR_STRIDE).max(1);
        let per_disk = ctx.quality.scale_count(STARS_PER_DISK);
        let scale_radius = SCALE_RADIUS_FRACTION * initial_min_separation(ctx);

        // Seed the three disks (tilt/spin/jitter all from the mode RNG).
        let mut rng = ctx.fork_rng(self.entry().flag);
        let kinematics = ctx.kinematics();
        let mut positions = Vec::with_capacity(per_disk * 3);
        let mut velocities = Vec::with_capacity(per_disk * 3);
        let mut host = Vec::with_capacity(per_disk * 3);
        let mut jitter = Vec::with_capacity(per_disk * 3);
        for (body, &mass) in masses.iter().enumerate() {
            let tilt = (rng.next_f64() * 2.0 - 1.0) * TILT_MAX;
            let spin = if rng.next_f64() < 0.5 { 1.0 } else { -1.0 };
            let (disk_positions, disk_velocities) = seed_disk(
                per_disk,
                ctx.positions[body][0],
                kinematics.velocities[body][0],
                mass,
                scale_radius,
                tilt,
                spin,
                SOFTENING,
                &mut rng,
            );
            positions.extend(disk_positions);
            velocities.extend(disk_velocities);
            host.extend(std::iter::repeat_n(body as u8, per_disk));
            for _ in 0..per_disk {
                jitter.push((
                    (rng.next_f64() * 2.0 - 1.0) * HUE_JITTER,
                    (rng.next_f64() * 2.0 - 1.0) * HUE_JITTER,
                ));
            }
        }
        info!("   galaxy-collision: {} stars, scale radius {scale_radius:.3}", positions.len());

        let still_w = ctx.quality.scale_dim(ctx.width);
        let still_h = ctx.quality.scale_dim(ctx.height);
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let video_scale = video_w as f32 / ctx.width as f32;
        let still_scale = still_w as f32 / ctx.width as f32;
        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);

        // --- Levels pre-pass on a quarter of the stars (stratified by disk).
        let levels = {
            let probe_per_disk = (per_disk / PROBE_DIVISOR).max(32);
            let mut probe_positions = Vec::new();
            let mut probe_velocities = Vec::new();
            let mut probe_host = Vec::new();
            let mut probe_jitter = Vec::new();
            for body in 0..3 {
                let start = body * per_disk;
                probe_positions.extend_from_slice(&positions[start..start + probe_per_disk]);
                probe_velocities.extend_from_slice(&velocities[start..start + probe_per_disk]);
                probe_host.extend_from_slice(&host[start..start + probe_per_disk]);
                probe_jitter.extend_from_slice(&jitter[start..start + probe_per_disk]);
            }
            let boost = per_disk as f64 / probe_per_disk as f64;
            let mut probe = StarRun::new(
                ctx,
                Swarm::new(probe_positions, probe_velocities),
                probe_host,
                probe_jitter,
            );
            let mut canvas = BandedSpd::new(video_w, video_h);
            let mut segments = Vec::new();
            probe.advance(0..star_steps, &mut [(&mut canvas, video_scale)], &mut segments);
            let mut rgba = Vec::new();
            canvas.convert_into(&mut rgba);
            for pixel in &mut rgba {
                pixel.0 *= boost;
                pixel.1 *= boost;
                pixel.2 *= boost;
            }
            auto_levels(&rgba, clip_black, clip_white, 1.0)
        };

        // --- Main pass: streaming video + full-res exposure in one sim.
        let mut run = StarRun::new(ctx, Swarm::new(positions, velocities), host, jitter);
        let mut video_canvas = BandedSpd::new(video_w, video_h);
        let mut still_canvas = BandedSpd::new(still_w, still_h);
        let mut segments = Vec::new();
        let mut advanced = 0usize;
        stream_video(
            video_w,
            video_h,
            60,
            &sink.path("galaxies.mp4"),
            &sink.path("galaxies_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let target = ((frame + 1) * star_steps / frame_count).min(star_steps);
                run.advance(
                    advanced..target,
                    &mut [(&mut video_canvas, video_scale), (&mut still_canvas, still_scale)],
                    &mut segments,
                );
                advanced = target;
                video_canvas.convert_into(rgba);
            },
        )?;
        sink.record("galaxies.mp4", "video");
        sink.record("galaxies_hq.mp4", "video");

        // --- Still: the final wreck exposure.
        let mut star_rgba: PixelBuffer = Vec::new();
        still_canvas.convert_into(&mut star_rgba);
        drop(still_canvas);
        sink.save_png16(
            &grade_auto_levels(&star_rgba, still_w, still_h, clip_black, clip_white, 1.0),
            "galaxies.png",
        )?;

        // --- Composite: the wreck with its skeleton.
        let mut master_acc = Accumulator::new(
            ctx.positions.to_vec(),
            ctx.colors.to_vec(),
            ctx.body_alphas.to_vec(),
            still_w,
            still_h,
            ctx.settings.aspect_correction,
            ctx.settings.traits,
            ctx.settings.render_config.hdr_scale,
        );
        master_acc.accumulate(0..steps);
        let master_rgba = master_acc.convert();
        drop(master_acc);

        let mut composite = star_rgba;
        for (pixel, master) in composite.iter_mut().zip(master_rgba.iter()) {
            pixel.0 = pixel.0 * COMPOSITE_STAR_WEIGHT + master.0 * COMPOSITE_MASTER_WEIGHT;
            pixel.1 = pixel.1 * COMPOSITE_STAR_WEIGHT + master.1 * COMPOSITE_MASTER_WEIGHT;
            pixel.2 = pixel.2 * COMPOSITE_STAR_WEIGHT + master.2 * COMPOSITE_MASTER_WEIGHT;
            pixel.3 =
                (pixel.3 * COMPOSITE_STAR_WEIGHT + master.3 * COMPOSITE_MASTER_WEIGHT).min(1.0);
        }
        sink.save_png16(
            &grade_auto_levels(&composite, still_w, still_h, clip_black, clip_white, 1.0),
            "galaxies_composite.png",
        )?;

        let meta = serde_json::json!({
            "stars_per_disk": per_disk,
            "star_steps": star_steps,
            "scale_radius": scale_radius,
            "tilt_max_rad": TILT_MAX,
            "splat_alpha": SPLAT_ALPHA,
            "hue_jitter": HUE_JITTER,
            "composite_weights": [COMPOSITE_STAR_WEIGHT, COMPOSITE_MASTER_WEIGHT],
            "note": "strokes span 4 star steps; levels from a quarter-population probe pass",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("galaxies_params.json", &json, "data")?;
        Ok(())
    }
}
