//! V37 `roche` -- Lobes That Touch.
//!
//! For the tightest pair: animated Roche equipotential surfaces in the
//! pair's co-rotating frame -- teardrop lobes swelling and shrinking, the
//! critical surface through L1 drawn boldest, the third body wobbling the
//! geometry, and a glowing mass-transfer stream spilling ballistically from
//! L1 during the deepest approaches. Composited over the co-rotating trail
//! ghost (V26's transform) at 8% energy.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::constants::DEFAULT_DT;
use crate::render::context::{BoundingBox, PixelBuffer, RenderContext};
use crate::render::{
    ImageBuffer, VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass,
    quantize_display_buffer_to_16bit, tonemap_to_display_buffer,
};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::Accumulator;
use crate::viz::common::contours::marching_squares;
use crate::viz::common::display::auto_levels;
use crate::viz::common::fields::{PointMass, RochePotential};
use crate::viz::common::kinematics::PAIRS;
use crate::viz::common::raster::{Rgb64, draw_line_rgba};
use crate::viz::context::VizContext;
use crate::viz::modes::corotating::{corotating_positions, tightest_pair};
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Ghost energy fraction for the co-rotating trail underlay.
const GHOST_ENERGY: f64 = 0.08;
/// Number of equipotential levels bracketing the L1 value.
const LEVEL_COUNT: usize = 12;
/// Index of the critical (L1) contour within the level ramp.
const CRITICAL_INDEX: usize = 8;
/// Level spread per step as a fraction of |L1|.
const LEVEL_SPREAD: f64 = 0.045;
/// Lobe-fill fraction that triggers mass transfer.
const FILL_TRIGGER: f64 = 0.98;
/// Interior lobe wash strength.
const WASH: f64 = 0.04;
/// Stream particles emitted per triggering frame.
const EMIT_PER_FRAME: usize = 6;
/// Stream particle lifetime in integration substeps.
const STREAM_LIFE: usize = 400;
/// Integration substeps per video frame.
const SUBSTEPS: usize = 8;
/// Video frames at 60 fps (30 s).
const VIDEO_FRAMES: usize = 1800;
/// Frame window half-width as a multiple of the p95 pair separation.
const WINDOW_SCALE: f64 = 2.2;
/// Plummer softening in pixels of the grid.
const SOFTENING_PX: f64 = 2.0;
/// Display gamma for ink encoding.
const DISPLAY_GAMMA: f64 = 2.2;

/// The Roche lobe mode.
pub struct Roche;

/// One ballistic mass-transfer particle in frame coordinates.
struct StreamParticle {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    age: usize,
}

/// Display-space ink from `OKLCh` coordinates.
fn ink_display(lightness: f64, chroma: f64, hue: f64) -> Rgb64 {
    let (l, a, b) = oklch_to_oklab(lightness, chroma, hue);
    let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
    (
        r.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA),
        g.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA),
        bl.clamp(0.0, 1.0).powf(1.0 / DISPLAY_GAMMA),
    )
}

/// Percentile of a pairwise-separation series (strided, deterministic).
fn separation_percentile(series: &[f64], quantile: f64) -> f64 {
    let mut sample: Vec<f64> = series.iter().step_by(37).copied().collect();
    if sample.is_empty() {
        return 1.0;
    }
    sample.sort_by(f64::total_cmp);
    sample[((sample.len() - 1) as f64 * quantile) as usize]
}

/// Signed pair angular rate (rad per time unit) at a step.
fn pair_omega(ctx: &VizContext<'_>, pair: (usize, usize), step: usize) -> f64 {
    let steps = ctx.step_count();
    let next = (step + 1).min(steps - 1);
    let previous = step.saturating_sub(1);
    let angle_at = |index: usize| {
        let delta = ctx.positions[pair.1][index] - ctx.positions[pair.0][index];
        delta.y.atan2(delta.x)
    };
    let mut diff = angle_at(next) - angle_at(previous);
    while diff > std::f64::consts::PI {
        diff -= std::f64::consts::TAU;
    }
    while diff < -std::f64::consts::PI {
        diff += std::f64::consts::TAU;
    }
    diff / (((next - previous).max(1)) as f64 * DEFAULT_DT)
}

impl VizMode for Roche {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("roche").expect("roche is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps == 0 {
            warn!("roche skipped: empty trajectory");
            return Ok(());
        }
        let masses = ctx.kinematics().masses;
        let pair = tightest_pair(ctx);
        let pair_index =
            PAIRS.iter().position(|&candidate| candidate == pair).expect("canonical pair");
        let third = (0..3).find(|&body| body != pair.0 && body != pair.1).unwrap_or(2);
        let separations = &ctx.kinematics().pairwise[pair_index];
        let window_separation = separation_percentile(separations, 0.95);
        let trigger_radius = separation_percentile(separations, 0.05);
        let touch_step =
            (0..steps).min_by(|&a, &b| separations[a].total_cmp(&separations[b])).unwrap_or(0);
        info!(
            "   roche: pair ({}, {}), trigger radius {trigger_radius:.3}, window {window_separation:.3}",
            pair.0, pair.1
        );

        let corot = corotating_positions(ctx.positions, masses, pair);

        // Frame window: pair-centric, aspect-matched to the video canvas.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);
        let half_w_world = WINDOW_SCALE * window_separation;
        let half_h_world = half_w_world * f64::from(video_h) / f64::from(video_w);
        let window = BoundingBox {
            min_x: -half_w_world,
            max_x: half_w_world,
            min_y: -half_h_world,
            max_y: half_h_world,
            width: half_w_world * 2.0,
            height: half_h_world * 2.0,
        };
        let frame_ctx = RenderContext::with_bounds(video_w, video_h, window);

        // Co-rotating trail ghost with the same framing. Two passes: one
        // full accumulation for fixed levels, then an incremental replay.
        let mut ghost = Accumulator::with_bounds(
            corot.clone(),
            ctx.colors.to_vec(),
            ctx.body_alphas.to_vec(),
            video_w,
            video_h,
            window,
            ctx.settings.traits,
            ctx.settings.render_config.hdr_scale,
        );
        ghost.accumulate(0..steps);
        let levels = {
            let rgba = ghost.convert();
            auto_levels(
                &rgba,
                ctx.settings.resolved_config.clip_black,
                ctx.settings.resolved_config.clip_white,
                1.0,
            )
        };
        ghost.clear();

        // Quarter-resolution grid for the equipotential extraction.
        let grid_ctx =
            RenderContext::with_bounds((video_w / 4).max(16), (video_h / 4).max(16), window);
        let grid_scale = video_w as f32 / grid_ctx.width as f32;
        let softening = SOFTENING_PX * window.width / f64::from(video_w);
        let world_per_px = window.width / f64::from(video_w);

        let donor = if masses[pair.0] <= masses[pair.1] { pair.0 } else { pair.1 };
        let accretor = if donor == pair.0 { pair.1 } else { pair.0 };
        let hue_of = |body: usize| {
            let (l, a, b) = ctx.mean_color(body);
            let (_, _, hue) = oklab_to_oklch(l, a, b);
            hue
        };
        let donor_ink = ink_display(0.72, 0.06, hue_of(donor));
        let accretor_ink = ink_display(0.75, 0.05, hue_of(accretor));
        let critical_ink = ink_display(0.85, 0.04, hue_of(donor));
        let stream_ink = ink_display(0.9, 0.08, hue_of(donor));

        let mut rng = ctx.fork_rng(self.entry().flag);
        let mut particles: Vec<StreamParticle> = Vec::new();
        let dt_frame = DEFAULT_DT * steps as f64 / frame_count as f64;
        let dt_sub = dt_frame / SUBSTEPS as f64;

        let touch_frame = (touch_step * frame_count / steps.max(1)).min(frame_count - 1);
        let mut touch_image: Option<ImageBuffer<crate::render::Rgb<u16>, Vec<u16>>> = None;

        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("roche.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("roche_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];

        let mut cursor = 0usize;
        let mut rgba: PixelBuffer = Vec::new();
        let mut l1_track: Vec<serde_json::Value> = Vec::new();
        let mut stream_events = 0usize;
        create_videos_from_frames_singlepass(
            video_w,
            video_h,
            60,
            |out| {
                for frame in 0..frame_count {
                    let target = ((frame + 1) * steps / frame_count).min(steps);
                    ghost.accumulate(cursor..target);
                    cursor = target;
                    let step = cursor.min(steps - 1);

                    // Roche potential of this frame (third body perturbs).
                    let separation = separations[step].max(1e-9);
                    let total = masses[pair.0] + masses[pair.1];
                    let omega = pair_omega(ctx, pair, step);
                    let potential = RochePotential {
                        bodies: [
                            PointMass {
                                x: -separation * masses[pair.1] / total,
                                y: 0.0,
                                mass: masses[pair.0],
                            },
                            PointMass {
                                x: separation * masses[pair.0] / total,
                                y: 0.0,
                                mass: masses[pair.1],
                            },
                        ],
                        perturber: Some(PointMass {
                            x: corot[third][step].x,
                            y: corot[third][step].y,
                            mass: masses[third],
                        }),
                        omega_sq: omega * omega,
                        softening,
                    };
                    let ((l1_x, l1_y), l1_value) = potential.l1();
                    let grid = potential.sample_grid(&grid_ctx);

                    // Ghost -> tonemapped display frame at 8% energy.
                    ghost.convert_into(&mut rgba);
                    for pixel in &mut rgba {
                        pixel.0 *= GHOST_ENERGY;
                        pixel.1 *= GHOST_ENERGY;
                        pixel.2 *= GHOST_ENERGY;
                    }
                    let mut display = tonemap_to_display_buffer(&rgba, &levels);
                    let width = video_w as usize;
                    let height = video_h as usize;

                    // 4% interior wash inside each lobe (below the critical
                    // surface, split by the L1 x position).
                    let (l1_px, _) = frame_ctx.to_pixel(l1_x, l1_y);
                    let donor_is_negative_x = donor == pair.0;
                    let donor_wash = {
                        let (l, a, b) = oklch_to_oklab(0.6, 0.08, hue_of(donor));
                        let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
                        (r.max(0.0), g.max(0.0), bl.max(0.0))
                    };
                    let accretor_wash = {
                        let (l, a, b) = oklch_to_oklab(0.6, 0.08, hue_of(accretor));
                        let (r, g, bl) = oklab_to_linear_rec2020(l, a, b);
                        (r.max(0.0), g.max(0.0), bl.max(0.0))
                    };
                    for row in 0..height {
                        for col in 0..width {
                            let value = grid.value_at(
                                (col as f64 + 0.5) / f64::from(grid_scale),
                                (row as f64 + 0.5) / f64::from(grid_scale),
                            );
                            if value < l1_value {
                                let on_negative_side = (col as f32) < l1_px;
                                let wash = if on_negative_side == donor_is_negative_x {
                                    donor_wash
                                } else {
                                    accretor_wash
                                };
                                let pixel = &mut display[row * width + col];
                                pixel.0 += wash.0 * WASH;
                                pixel.1 += wash.1 * WASH;
                                pixel.2 += wash.2 * WASH;
                            }
                        }
                    }

                    // Equipotential hairlines: 12 levels bracketing L1.
                    for level_index in 0..LEVEL_COUNT {
                        let offset = (level_index as f64 - CRITICAL_INDEX as f64) * LEVEL_SPREAD;
                        let level = (l1_value + offset * l1_value.abs()) as f32;
                        let is_critical = level_index == CRITICAL_INDEX;
                        let ink = if is_critical {
                            critical_ink
                        } else if level_index < CRITICAL_INDEX {
                            donor_ink
                        } else {
                            accretor_ink
                        };
                        let stroke = if is_critical { 2.1 } else { 1.0 };
                        let opacity = if is_critical { 0.9 } else { 0.42 };
                        for segment in
                            marching_squares(&grid.values, grid.width, grid.height, level)
                        {
                            draw_line_rgba(
                                &mut display,
                                width,
                                height,
                                (segment.0.0 * grid_scale, segment.0.1 * grid_scale),
                                (segment.1.0 * grid_scale, segment.1.1 * grid_scale),
                                ink,
                                stroke,
                                opacity,
                            );
                        }
                    }

                    // Mass transfer: emit from L1 while the donor lobe fills.
                    let fill = (trigger_radius / separation).min(1.2);
                    if fill > FILL_TRIGGER {
                        stream_events += 1;
                        let accretor_x = if donor_is_negative_x {
                            potential.bodies[1].x
                        } else {
                            potential.bodies[0].x
                        };
                        let toward = (accretor_x - l1_x).signum();
                        let speed_scale = 0.12 * (crate::sim::G * total / separation).sqrt();
                        for _ in 0..EMIT_PER_FRAME {
                            let jitter = |rng: &mut crate::sim::Sha3RandomByteStream| {
                                (rng.next_f64() - 0.5) * speed_scale * 0.6
                            };
                            particles.push(StreamParticle {
                                x: l1_x,
                                y: l1_y,
                                vx: toward * speed_scale * (0.6 + rng.next_f64() * 0.4),
                                vy: jitter(&mut rng),
                                age: 0,
                            });
                        }
                    }

                    // Integrate + draw the stream (Coriolis-aware ballistics).
                    let kill_radius = separation * 0.05;
                    let accretor_pos = if donor_is_negative_x {
                        (potential.bodies[1].x, potential.bodies[1].y)
                    } else {
                        (potential.bodies[0].x, potential.bodies[0].y)
                    };
                    particles.retain_mut(|particle| {
                        for _ in 0..SUBSTEPS {
                            let (gx, gy) = potential.gradient(particle.x, particle.y);
                            let ax = -gx + 2.0 * omega * particle.vy;
                            let ay = -gy - 2.0 * omega * particle.vx;
                            let from = frame_ctx.to_pixel(particle.x, particle.y);
                            particle.vx += ax * dt_sub;
                            particle.vy += ay * dt_sub;
                            particle.x += particle.vx * dt_sub;
                            particle.y += particle.vy * dt_sub;
                            particle.age += 1;
                            let to = frame_ctx.to_pixel(particle.x, particle.y);
                            let speed = particle.vx.hypot(particle.vy);
                            let brightness =
                                (speed / (speed_scale_reference(total, separation))).min(1.0);
                            draw_line_rgba(
                                &mut display,
                                width,
                                height,
                                from,
                                to,
                                stream_ink,
                                1.2,
                                0.25 + 0.55 * brightness,
                            );

                            let to_accretor =
                                (particle.x - accretor_pos.0).hypot(particle.y - accretor_pos.1);
                            if to_accretor < kill_radius {
                                // Hot-spot flash on the accretor's limb.
                                for spoke in 0..6 {
                                    let angle = f64::from(spoke) * std::f64::consts::TAU / 6.0;
                                    let tip = (
                                        to.0 + (kill_radius * 2.5 * angle.cos() / world_per_px)
                                            as f32,
                                        to.1 + (kill_radius * 2.5 * angle.sin() / world_per_px)
                                            as f32,
                                    );
                                    draw_line_rgba(
                                        &mut display,
                                        width,
                                        height,
                                        to,
                                        tip,
                                        stream_ink,
                                        1.4,
                                        0.7,
                                    );
                                }
                                return false;
                            }
                            if particle.age >= STREAM_LIFE
                                || particle.x.abs() > window.max_x * 1.2
                                || particle.y.abs() > window.max_y * 1.2
                            {
                                return false;
                            }
                        }
                        true
                    });

                    if frame % 30 == 0 {
                        l1_track.push(serde_json::json!({
                            "frame": frame,
                            "l1": [l1_x, l1_y],
                            "l1_value": l1_value,
                            "separation": separation,
                            "omega": omega,
                            "fill": fill,
                        }));
                    }

                    let bytes = quantize_display_buffer_to_16bit(&display);
                    if frame == touch_frame {
                        touch_image = ImageBuffer::from_raw(video_w, video_h, bytes.clone());
                    }
                    out.write_all(bytemuck::cast_slice(&bytes))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                }
                Ok(())
            },
            &outputs,
        )?;
        sink.record("roche.mp4", "video");
        sink.record("roche_hq.mp4", "video");

        if let Some(image) = touch_image {
            sink.save_png16(&image, "roche_touch.png")?;
        }

        let meta = serde_json::json!({
            "pair": [pair.0, pair.1],
            "donor": donor,
            "accretor": accretor,
            "trigger_radius": trigger_radius,
            "fill_trigger": FILL_TRIGGER,
            "level_count": LEVEL_COUNT,
            "level_spread": LEVEL_SPREAD,
            "touch_step": touch_step,
            "touch_frame": touch_frame,
            "stream_emission_frames": stream_events,
            "l1_track": l1_track,
            "note": "fill = p05 separation / current separation (point-mass proxy); \
                     touch still captured at video resolution",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("roche_params.json", &json, "data")?;
        Ok(())
    }
}

/// Circular-orbit speed scale used to normalize stream brightness.
fn speed_scale_reference(total_mass: f64, separation: f64) -> f64 {
    (crate::sim::G * total_mass / separation.max(1e-9)).sqrt().max(1e-9)
}
