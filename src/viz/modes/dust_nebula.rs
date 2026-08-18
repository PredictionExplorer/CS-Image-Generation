//! V31 `dust-nebula` -- Gravity's Weather.
//!
//! Fifty thousand massless dust particles advected through the trio's
//! time-varying gravity, their trails accumulated with the production
//! splatter: spiral arms, ejection jets, and temporary captures the body
//! trails alone never show. Dust color follows the instantaneously dominant
//! body's palette at reduced chroma; a composite merges the nebula under
//! the master's trails energy-linearly.

use crate::error::Result;
use crate::oklab::{oklab_to_oklch, oklch_to_oklab};
use crate::render::constants::DEFAULT_DT;
use crate::render::context::{PixelBuffer, RenderContext};
use crate::render::{LineVertex, OklabColor, SpectralLineSegment};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::{Accumulator, stream_video};
use crate::viz::common::display::{auto_levels, grade_auto_levels};
use crate::viz::common::particles::{BandedSpd, FieldParams, Swarm, seed_annulus};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;
use tracing::info;

/// Dust particles at final quality (draft scales to a quarter).
const PARTICLE_COUNT: usize = 50_000;
/// Simulation steps per dust integration step.
const DUST_STRIDE: usize = 4;
/// Dust steps per splatted stroke (polyline-continuous trails).
const SPLAT_STRIDE: usize = 4;
/// Annulus radii as multiples of the initial maximum pair separation.
const ANNULUS_RANGE: (f64, f64) = (1.5, 3.0);
/// Virial fraction for the seeded speeds.
const VIRIAL_FRACTION: f64 = 0.3;
/// Plummer softening (world units).
const SOFTENING: f64 = 0.02;
/// Freeze radius as a multiple of the scene bounding box half-diagonal.
const FREEZE_SCALE: f64 = 12.0;
/// Per-splat alpha.
const SPLAT_ALPHA: f64 = 0.012;
/// Base stroke energy before the speed term.
const ENERGY_BASE: f64 = 0.010;
/// Dust chroma multiplier relative to the body palette.
const CHROMA_SCALE: f64 = 0.6;
/// Dust weight in the energy-linear composite.
const COMPOSITE_DUST_WEIGHT: f64 = 0.65;
/// Video frames at 60 fps (30 s).
const VIDEO_FRAMES: usize = 1800;
/// Probe divisor for the fixed-levels pre-pass.
const PROBE_DIVISOR: usize = 4;

/// The dust nebula mode.
pub struct DustNebula;

/// Dust rendering state advanced in lockstep by the sim loop.
struct DustRun<'a> {
    ctx: &'a VizContext<'a>,
    swarm: Swarm,
    params: FieldParams,
    masses: [f64; 3],
    /// Pixel-space position at the last splat, per particle (full-res px).
    last_px: Vec<(f32, f32)>,
    /// Precomputed chroma-scaled palette per body per dust step.
    dust_colors: Vec<[OklabColor; 3]>,
    full_ctx: RenderContext,
    speed_reference: f64,
}

impl<'a> DustRun<'a> {
    fn new(ctx: &'a VizContext<'a>, swarm: Swarm, dust_steps: usize) -> Self {
        let masses = ctx.kinematics().masses;
        let full_ctx = RenderContext::new(
            ctx.width,
            ctx.height,
            ctx.positions,
            ctx.settings.aspect_correction,
        );
        let bounds = *full_ctx.bounds();
        let half_diagonal = 0.5 * bounds.width.hypot(bounds.height);
        let freeze_center = Vector3::new(
            0.5 * (bounds.min_x + bounds.max_x),
            0.5 * (bounds.min_y + bounds.max_y),
            0.0,
        );
        let params = FieldParams {
            softening_sq: SOFTENING * SOFTENING,
            freeze_radius_sq: Some((FREEZE_SCALE * half_diagonal).powi(2)),
            freeze_center,
        };

        // Chroma-scaled palette per body per dust step.
        let dust_colors: Vec<[OklabColor; 3]> = (0..dust_steps)
            .map(|dust_step| {
                let step = (dust_step * DUST_STRIDE).min(ctx.step_count() - 1);
                std::array::from_fn(|body| {
                    let (l, a, b) = ctx.colors[body][step.min(ctx.colors[body].len() - 1)];
                    let (lightness, chroma, hue) = oklab_to_oklch(l, a, b);
                    oklch_to_oklab(lightness, chroma * CHROMA_SCALE, hue)
                })
            })
            .collect();

        let last_px: Vec<(f32, f32)> =
            swarm.positions.iter().map(|p| full_ctx.to_pixel(p.x, p.y)).collect();
        let total_mass: f64 = masses.iter().sum();
        let mid_radius = {
            let separation = initial_max_separation(ctx);
            0.5 * (ANNULUS_RANGE.0 + ANNULUS_RANGE.1) * separation
        };
        let speed_reference = (crate::sim::G * total_mass / mid_radius.max(1e-9)).sqrt().max(1e-9);

        Self { ctx, swarm, params, masses, last_px, dust_colors, full_ctx, speed_reference }
    }

    /// Advance dust steps `range` (splatting strokes every [`SPLAT_STRIDE`])
    /// into the given canvases (full-res px space, scaled per canvas).
    fn advance(
        &mut self,
        range: std::ops::Range<usize>,
        canvases: &mut [(&mut BandedSpd, f32)],
        segments: &mut Vec<SpectralLineSegment>,
    ) {
        let steps = self.ctx.step_count();
        for dust_step in range {
            let sim_now = (dust_step * DUST_STRIDE).min(steps - 1);
            let sim_next = ((dust_step + 1) * DUST_STRIDE).min(steps - 1);
            let bodies_now = std::array::from_fn(|body| self.ctx.positions[body][sim_now]);
            let bodies_next = std::array::from_fn(|body| self.ctx.positions[body][sim_next]);
            self.swarm.step(
                &bodies_now,
                &bodies_next,
                self.masses,
                DEFAULT_DT * DUST_STRIDE as f64,
                &self.params,
            );

            if !(dust_step + 1).is_multiple_of(SPLAT_STRIDE) {
                continue;
            }
            let colors = self.dust_colors[dust_step.min(self.dust_colors.len() - 1)];
            segments.clear();
            let dt_splat = DEFAULT_DT * (DUST_STRIDE * SPLAT_STRIDE) as f64;
            for index in 0..self.swarm.len() {
                if self.swarm.frozen[index] {
                    continue;
                }
                let position = self.swarm.positions[index];
                let (px, py) = self.full_ctx.to_pixel(position.x, position.y);
                let (last_x, last_y) = self.last_px[index];
                self.last_px[index] = (px, py);
                if !px.is_finite() || !py.is_finite() {
                    continue;
                }
                let speed = self.swarm.velocities[index].norm();
                let energy = ENERGY_BASE * (speed / self.speed_reference).sqrt().clamp(0.35, 2.0);
                let color = colors[self.swarm.dominant_body[index] as usize];
                // Skip pathological jumps (re-projection wraps, huge kicks).
                let jump = (px - last_x).hypot(py - last_y);
                if f64::from(jump) > dt_splat * self.speed_reference * 400.0 {
                    continue;
                }
                segments.push(SpectralLineSegment {
                    start: LineVertex { x: last_x, y: last_y, z: 0.0, color, alpha: SPLAT_ALPHA },
                    end: LineVertex { x: px, y: py, z: 0.0, color, alpha: SPLAT_ALPHA },
                    hdr_scale: energy,
                    thickness_factor: 0.55,
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

/// Maximum pairwise separation at step 0.
fn initial_max_separation(ctx: &VizContext<'_>) -> f64 {
    let mut best = 0.0f64;
    for a in 0..3 {
        for b in a + 1..3 {
            best = best.max((ctx.positions[a][0] - ctx.positions[b][0]).norm());
        }
    }
    best.max(1e-9)
}

impl VizMode for DustNebula {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("dust-nebula").expect("dust-nebula is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps == 0 {
            tracing::warn!("dust-nebula skipped: empty trajectory");
            return Ok(());
        }
        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;
        let masses = ctx.kinematics().masses;
        let total_mass: f64 = masses.iter().sum();
        let dust_steps = (steps / DUST_STRIDE).max(1);
        let count = ctx.quality.scale_count(PARTICLE_COUNT);

        // Seed once; the probe reuses the head of the same population.
        let mut rng = ctx.fork_rng(self.entry().flag);
        let centroid = (ctx.positions[0][0] + ctx.positions[1][0] + ctx.positions[2][0]) / 3.0;
        let separation = initial_max_separation(ctx);
        // Keep the annulus inside the artwork: on wide-drift seeds the
        // initial triangle can be small against the full frame or vice
        // versa, so cap the outer radius at a fraction of the scene bounds.
        let frame_cap = {
            let probe_ctx = RenderContext::new(
                ctx.width,
                ctx.height,
                ctx.positions,
                ctx.settings.aspect_correction,
            );
            let bounds = probe_ctx.bounds();
            0.45 * bounds.width.hypot(bounds.height) * 0.5
        };
        let r_outer = (ANNULUS_RANGE.1 * separation).min(frame_cap).max(1e-6);
        let r_inner = (ANNULUS_RANGE.0 * separation).min(0.6 * r_outer);
        let (positions, velocities) =
            seed_annulus(count, centroid, r_inner, r_outer, total_mass, VIRIAL_FRACTION, &mut rng);
        info!("   dust-nebula: {count} particles, {dust_steps} dust steps");

        let still_w = ctx.quality.scale_dim(ctx.width);
        let still_h = ctx.quality.scale_dim(ctx.height);
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let video_scale = video_w as f32 / ctx.width as f32;
        let still_scale = still_w as f32 / ctx.width as f32;
        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);

        // --- Levels pre-pass: a quarter of the particles approximates the
        // final video exposure at 1/4 energy (boosted back before analysis).
        let levels = {
            let probe_count = (count / PROBE_DIVISOR).max(64).min(count);
            let probe_swarm =
                Swarm::new(positions[..probe_count].to_vec(), velocities[..probe_count].to_vec());
            let mut probe = DustRun::new(ctx, probe_swarm, dust_steps);
            let mut canvas = BandedSpd::new(video_w, video_h);
            let mut segments = Vec::new();
            probe.advance(0..dust_steps, &mut [(&mut canvas, video_scale)], &mut segments);
            let mut rgba = Vec::new();
            canvas.convert_into(&mut rgba);
            for pixel in &mut rgba {
                pixel.0 *= PROBE_DIVISOR as f64;
                pixel.1 *= PROBE_DIVISOR as f64;
                pixel.2 *= PROBE_DIVISOR as f64;
            }
            auto_levels(&rgba, clip_black, clip_white, 1.0)
        };

        // --- Main pass: one simulation drives the streaming video and the
        // full-resolution exposure simultaneously.
        let mut run = DustRun::new(ctx, Swarm::new(positions, velocities), dust_steps);
        let mut video_canvas = BandedSpd::new(video_w, video_h);
        let mut still_canvas = BandedSpd::new(still_w, still_h);
        let mut segments = Vec::new();
        let mut advanced = 0usize;
        stream_video(
            video_w,
            video_h,
            60,
            &sink.path("nebula.mp4"),
            &sink.path("nebula_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let target = ((frame + 1) * dust_steps / frame_count).min(dust_steps);
                run.advance(
                    advanced..target,
                    &mut [(&mut video_canvas, video_scale), (&mut still_canvas, still_scale)],
                    &mut segments,
                );
                advanced = target;
                video_canvas.convert_into(rgba);
            },
        )?;
        sink.record("nebula.mp4", "video");
        sink.record("nebula_hq.mp4", "video");
        let frozen = run.swarm.frozen.iter().filter(|&&f| f).count();

        // --- Still: the full-run dust exposure.
        let mut dust_rgba: PixelBuffer = Vec::new();
        still_canvas.convert_into(&mut dust_rgba);
        drop(still_canvas);
        sink.save_png16(
            &grade_auto_levels(&dust_rgba, still_w, still_h, clip_black, clip_white, 1.0),
            "nebula.png",
        )?;

        // --- Composite: dust under the master's trails, energy-linear.
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

        let luminance =
            |rgba: &PixelBuffer| rgba.iter().map(|&(r, g, b, a)| (r + g + b) * a).sum::<f64>();
        let dust_energy_raw = luminance(&dust_rgba);
        let master_energy = luminance(&master_rgba);
        // The audit is a hard budget: rescale the dust weight so its energy
        // share never exceeds 40% (dust exposures vary wildly with seeds).
        let target_share = 0.35f64;
        let weight = if dust_energy_raw > 1e-12 {
            COMPOSITE_DUST_WEIGHT
                .min(target_share / (1.0 - target_share) * master_energy / dust_energy_raw)
        } else {
            COMPOSITE_DUST_WEIGHT
        };
        let dust_share =
            (dust_energy_raw * weight) / (dust_energy_raw * weight + master_energy).max(1e-12);

        let mut composite = master_rgba;
        for (pixel, dust) in composite.iter_mut().zip(dust_rgba.iter()) {
            pixel.0 += dust.0 * weight;
            pixel.1 += dust.1 * weight;
            pixel.2 += dust.2 * weight;
            pixel.3 = (pixel.3 + dust.3 * weight).min(1.0);
        }
        sink.save_png16(
            &grade_auto_levels(&composite, still_w, still_h, clip_black, clip_white, 1.0),
            "nebula_composite.png",
        )?;

        let meta = serde_json::json!({
            "particles": count,
            "dust_steps": dust_steps,
            "dust_dt_steps": DUST_STRIDE,
            "splat_stride": SPLAT_STRIDE,
            "annulus_range": ANNULUS_RANGE,
            "virial_fraction": VIRIAL_FRACTION,
            "softening": SOFTENING,
            "frozen_particles": frozen,
            "annulus_radii": [r_inner, r_outer],
            "composite_dust_weight": weight,
            "composite_dust_energy_share": dust_share,
            "note": "strokes span 4 dust steps (polyline-continuous); \
                     levels from a quarter-population probe pass",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("nebula_params.json", &json, "data")?;
        Ok(())
    }
}
