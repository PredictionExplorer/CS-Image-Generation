//! V35 `lightning` -- The Storm Record.
//!
//! Every deep close approach discharges a dielectric-breakdown bolt between
//! the two approaching bodies, grown through the actual potential field at
//! that instant. The long exposure collects a storm record over a dim
//! master ghost; the video strikes the bolts in chronological sequence with
//! return-stroke flashes and a persistent afterglow.

use crate::error::Result;
use crate::render::context::{PixelBuffer, RenderContext};
use crate::render::{DogBloomConfig, apply_dog_bloom};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::agents::grow_streamer;
use crate::viz::common::display::{SpdCanvas, auto_levels};
use crate::viz::common::fields::{PointMass, PotentialGrid};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Maximum number of bolts (deepest approaches win).
const MAX_BOLTS: usize = 24;
/// Dielectric-breakdown exponent.
const ETA: f64 = 2.0;
/// Straight-at-target pick fraction during growth.
const TARGET_BIAS: f64 = 0.15;
/// Core stroke energy scale (multiplied by normalized approach speed).
const CORE_ENERGY: f64 = 0.06;
/// Branch energy relative to the core.
const BRANCH_RATIO: f64 = 0.4;
/// Afterglow energy fraction deposited permanently after a flash.
const AFTERGLOW: f64 = 0.06;
/// Return-stroke brightness multiplier.
const RETURN_STROKE: f64 = 3.0;
/// Decay frames after the return stroke.
const DECAY_FRAMES: usize = 18;
/// Video frames (30 s at 60 fps).
const VIDEO_FRAMES: usize = 1800;
/// Ghost brightness for the still (display-space multiplier).
const STILL_GHOST: f64 = 0.08;
/// Display gamma.
const DISPLAY_GAMMA: f64 = 2.2;

/// The lightning mode.
pub struct Lightning;

/// One drawable bolt segment: (from, to, color, energy, thickness).
type BoltSegment = ((f32, f32), (f32, f32), (f64, f64, f64), f64, f64);

/// One grown bolt, ready to draw.
/// Exported for V67 `vanitas` (its SUMMER act fires these bolts).
pub(crate) struct Bolt {
    /// Segments in full-resolution pixel space.
    pub(crate) segments: Vec<BoltSegment>,
    /// Simulation step of the periapsis (chronology).
    pub(crate) step: usize,
    /// Normalized approach speed (brightness).
    pub(crate) speed: f64,
}

impl Bolt {
    /// Endpoint of the main channel (the arrival point at the struck body),
    /// in full-resolution pixel space. V67's bolt-to-physarum handoff seeds
    /// its scouts here.
    #[must_use]
    pub(crate) fn endpoints(&self) -> Vec<(f32, f32)> {
        // Segment list is (cell -> parent); the last grown core segments end
        // nearest the target. Collect the from-points of the final few core
        // segments plus the very first (root) point.
        let mut points = Vec::new();
        if let Some(&(from, _, _, _, _)) = self.segments.last() {
            points.push(from);
        }
        if let Some(&(_, to, _, _, _)) = self.segments.first() {
            points.push(to);
        }
        points
    }
}

/// Draw a bolt into a canvas with an overall energy multiplier and a
/// pixel-scale factor mapping full-res coordinates to the canvas.
/// Exported for V67 `vanitas`.
pub(crate) fn draw_bolt(canvas: &mut SpdCanvas, bolt: &Bolt, energy_scale: f64, pixel_scale: f32) {
    for &(from, to, color, energy, thickness) in &bolt.segments {
        canvas.draw_stroke(
            (from.0 * pixel_scale, from.1 * pixel_scale),
            (to.0 * pixel_scale, to.1 * pixel_scale),
            color,
            color,
            0.9,
            energy * energy_scale,
            thickness,
        );
    }
}

/// Grow up to `max_bolts` dielectric-breakdown bolts at the run's deepest
/// close approaches (chronological order). Shared by V35 and V67; pass the
/// consuming mode's forked RNG.
pub(crate) fn grow_bolts(
    ctx: &VizContext<'_>,
    rng: &mut crate::sim::Sha3RandomByteStream,
    max_bolts: usize,
) -> Vec<Bolt> {
    let steps = ctx.step_count();
    if steps == 0 {
        return Vec::new();
    }
    let kinematics = ctx.kinematics();
    let masses = kinematics.masses;

    // Deepest approaches, then chronological order for the storm.
    let mut approaches = ctx.events().periapses.clone();
    approaches.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    approaches.truncate(max_bolts);
    if approaches.is_empty() {
        // Short runs may produce no formal periapsis events; fall back
        // to each pair's global minimum-separation step.
        warn!("lightning: no periapsis events; using per-pair minima");
        for (pair_index, &pair) in crate::viz::common::kinematics::PAIRS.iter().enumerate() {
            let series = &kinematics.pairwise[pair_index];
            if let Some(step) = (0..steps).min_by(|&a, &b| series[a].total_cmp(&series[b])) {
                approaches.push(crate::viz::common::events::Approach {
                    step,
                    pair,
                    distance: series[step],
                });
            }
        }
    }
    approaches.sort_by_key(|approach| approach.step);
    if approaches.is_empty() {
        return Vec::new();
    }

    // Approach speed per event: |d separation / dt| around the step.
    let pair_index_of = |pair: (usize, usize)| {
        crate::viz::common::kinematics::PAIRS
            .iter()
            .position(|&candidate| candidate == pair)
            .expect("canonical pair")
    };
    let speeds: Vec<f64> = approaches
        .iter()
        .map(|approach| {
            let series = &kinematics.pairwise[pair_index_of(approach.pair)];
            let window = 200usize;
            let lo = approach.step.saturating_sub(window);
            let hi = (approach.step + window).min(steps - 1);
            ((series[lo] - series[approach.step]).max(0.0)
                + (series[hi] - series[approach.step]).max(0.0))
                / (window as f64 * crate::render::constants::DEFAULT_DT)
        })
        .collect();
    let speed_reference = speeds.iter().copied().fold(1e-12f64, f64::max);

    // Lattice at half of the full resolution for streamer growth.
    let lattice_w = ((ctx.width / 2).max(64)) as usize;
    let lattice_h = ((ctx.height / 2).max(64)) as usize;
    let lattice_ctx = RenderContext::new(
        lattice_w as u32,
        lattice_h as u32,
        ctx.positions,
        ctx.settings.aspect_correction,
    );
    let lattice_to_full = ctx.width as f32 / lattice_w as f32;
    let softening = 2.0 * lattice_ctx.bounds().width / lattice_w as f64;

    let mut bolts: Vec<Bolt> = Vec::new();
    for (event_index, approach) in approaches.iter().enumerate() {
        let step = approach.step;
        let (body_a, body_b) = approach.pair;
        let point_masses: Vec<PointMass> = (0..3)
            .map(|body| PointMass {
                x: ctx.positions[body][step].x,
                y: ctx.positions[body][step].y,
                mass: masses[body],
            })
            .collect();
        let grid = PotentialGrid::sample(&point_masses, &lattice_ctx, softening);

        let cell_of = |body: usize| {
            let position = ctx.positions[body][step];
            let (px, py) = lattice_ctx.to_pixel(position.x, position.y);
            ((px.max(0.0) as usize).min(lattice_w - 1), (py.max(0.0) as usize).min(lattice_h - 1))
        };
        let start = cell_of(body_a);
        let target = cell_of(body_b);
        let streamer = grow_streamer(
            &grid.values,
            lattice_w,
            lattice_h,
            start,
            target,
            ETA,
            TARGET_BIAS,
            60_000,
            rng,
        );
        if !streamer.reached {
            warn!("lightning: bolt {event_index} did not arrive; skipping");
            continue;
        }

        // Color by position between the pair; energy by approach speed.
        let speed = (speeds[event_index] / speed_reference).clamp(0.1, 1.0);
        let color_a = ctx.colors[body_a][step.min(ctx.colors[body_a].len() - 1)];
        let color_b = ctx.colors[body_b][step.min(ctx.colors[body_b].len() - 1)];
        let position_a = ctx.positions[body_a][step];
        let position_b = ctx.positions[body_b][step];
        let blend_at = |col: u16, row: u16| {
            let world_x = lattice_ctx.bounds().min_x
                + (f64::from(col) + 0.5) * lattice_ctx.bounds().width / lattice_w as f64;
            let world_y = lattice_ctx.bounds().min_y
                + (f64::from(row) + 0.5) * lattice_ctx.bounds().height / lattice_h as f64;
            let dist_a = (world_x - position_a.x).hypot(world_y - position_a.y);
            let dist_b = (world_x - position_b.x).hypot(world_y - position_b.y);
            let t = dist_a / (dist_a + dist_b).max(1e-12);
            (
                color_a.0 + (color_b.0 - color_a.0) * t,
                color_a.1 + (color_b.1 - color_a.1) * t,
                color_a.2 + (color_b.2 - color_a.2) * t,
            )
        };

        let mut in_channel = vec![false; streamer.cells.len()];
        for &cell in &streamer.main_channel {
            in_channel[cell as usize] = true;
        }
        let mut segments = Vec::with_capacity(streamer.cells.len());
        for (cell_index, &(col, row)) in streamer.cells.iter().enumerate().skip(1) {
            let parent = streamer.parents[cell_index] as usize;
            let (pcol, prow) = streamer.cells[parent];
            let is_core = in_channel[cell_index] && in_channel[parent];
            let energy = if is_core { CORE_ENERGY } else { CORE_ENERGY * BRANCH_RATIO };
            let thickness = if is_core { 1.1 } else { 0.6 };
            segments.push((
                (
                    (f32::from(col) + 0.5) * lattice_to_full,
                    (f32::from(row) + 0.5) * lattice_to_full,
                ),
                (
                    (f32::from(pcol) + 0.5) * lattice_to_full,
                    (f32::from(prow) + 0.5) * lattice_to_full,
                ),
                blend_at(col, row),
                energy * speed,
                thickness,
            ));
        }
        bolts.push(Bolt { segments, step, speed });
    }
    bolts
}

impl VizMode for Lightning {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("lightning").expect("lightning is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps == 0 {
            warn!("lightning skipped: empty trajectory");
            return Ok(());
        }
        let mut rng = ctx.fork_rng(self.entry().flag);
        let bolts = grow_bolts(ctx, &mut rng, MAX_BOLTS);
        info!("   lightning: {} bolts grown", bolts.len());
        if bolts.is_empty() {
            warn!("lightning skipped: no bolts reached their targets");
            return Ok(());
        }

        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;

        // --- Still: every bolt over a dim master ghost (display-space add).
        let still_w = ctx.quality.scale_dim(ctx.width);
        let still_h = ctx.quality.scale_dim(ctx.height);
        let still_scale = still_w as f32 / ctx.width as f32;
        {
            let mut canvas = SpdCanvas::new(still_w, still_h);
            for bolt in &bolts {
                draw_bolt(&mut canvas, bolt, 1.0, still_scale);
            }
            let mut rgba: PixelBuffer = Vec::new();
            canvas.convert_into(&mut rgba);
            // `apply_dog_bloom` returns the halation layer only; add it.
            let glow = apply_dog_bloom(
                &rgba,
                still_w as usize,
                still_h as usize,
                &DogBloomConfig::default(),
            );
            for (pixel, halo) in rgba.iter_mut().zip(glow.iter()) {
                pixel.0 += halo.0;
                pixel.1 += halo.1;
                pixel.2 += halo.2;
                pixel.3 = (pixel.3 + halo.3).min(1.0);
            }
            let levels = auto_levels(&rgba, clip_black, clip_white, 1.0);
            let display = crate::render::tonemap_to_display_buffer(&rgba, &levels);
            let mut bytes = crate::render::quantize_display_buffer_to_16bit(&display);

            // Master ghost added in display space.
            let master_path = format!("{}/images/source/master.png", ctx.seed_dir);
            match image::ImageReader::open(&master_path).map(image::ImageReader::decode) {
                Ok(Ok(decoded)) => {
                    let master = decoded.into_rgb16();
                    let source_w = master.width() as usize;
                    let source_h = master.height() as usize;
                    let raw = master.as_raw();
                    let ghost_factor = STILL_GHOST.powf(1.0 / DISPLAY_GAMMA);
                    for row in 0..still_h as usize {
                        let sy = (row * source_h / still_h as usize).min(source_h - 1);
                        for col in 0..still_w as usize {
                            let sx = (col * source_w / still_w as usize).min(source_w - 1);
                            let source = (sy * source_w + sx) * 3;
                            let dest = (row * still_w as usize + col) * 3;
                            for channel in 0..3 {
                                let ghost = f64::from(raw[source + channel]) * ghost_factor;
                                let combined = f64::from(bytes[dest + channel]) + ghost;
                                bytes[dest + channel] = combined.min(65535.0) as u16;
                            }
                        }
                    }
                }
                Ok(Err(error)) => warn!("lightning: master decode failed ({error}); no ghost"),
                Err(error) => warn!("lightning: master missing ({error}); no ghost"),
            }
            let image = crate::render::ImageBuffer::from_raw(still_w, still_h, bytes)
                .expect("storm buffer has width*height*3 samples");
            sink.save_png16(&image, "storm.png")?;
        }

        // --- Video: chronological strikes with afterglow.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let video_scale = video_w as f32 / ctx.width as f32;
        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);

        // Fixed levels from the climactic exposure: full afterglow plus the
        // brightest bolt at return-stroke brightness.
        let levels = {
            let mut canvas = SpdCanvas::new(video_w, video_h);
            for bolt in &bolts {
                draw_bolt(&mut canvas, bolt, AFTERGLOW, video_scale);
            }
            if let Some(brightest) = bolts.iter().max_by(|a, b| a.speed.total_cmp(&b.speed)) {
                draw_bolt(&mut canvas, brightest, RETURN_STROKE, video_scale);
            }
            let mut rgba: PixelBuffer = Vec::new();
            canvas.convert_into(&mut rgba);
            auto_levels(&rgba, clip_black, clip_white, 1.0)
        };

        // Strike frames: chronological position within the run.
        let strike_frames: Vec<usize> = bolts
            .iter()
            .map(|bolt| (bolt.step * frame_count / steps).min(frame_count - 1))
            .collect();

        let mut afterglow = SpdCanvas::new(video_w, video_h);
        let mut flash = SpdCanvas::new(video_w, video_h);
        let mut flash_rgba: PixelBuffer = Vec::new();
        let mut deposited = vec![false; bolts.len()];
        stream_video(
            video_w,
            video_h,
            60,
            &sink.path("lightning.mp4"),
            &sink.path("lightning_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                flash.clear();
                let mut any_flash = false;
                for (bolt_index, bolt) in bolts.iter().enumerate() {
                    let strike = strike_frames[bolt_index];
                    if frame < strike {
                        continue;
                    }
                    let age = frame - strike;
                    if age == 0 {
                        // Leader: dim channel probe.
                        draw_bolt(&mut flash, bolt, 0.5, video_scale);
                        any_flash = true;
                    } else if age <= 2 {
                        // Return stroke.
                        draw_bolt(&mut flash, bolt, RETURN_STROKE, video_scale);
                        any_flash = true;
                    } else if age <= 2 + DECAY_FRAMES {
                        let envelope = (-((age - 2) as f64) / (DECAY_FRAMES as f64 / 3.0)).exp();
                        draw_bolt(&mut flash, bolt, RETURN_STROKE * envelope, video_scale);
                        any_flash = true;
                    } else if !deposited[bolt_index] {
                        deposited[bolt_index] = true;
                        draw_bolt(&mut afterglow, bolt, AFTERGLOW, video_scale);
                    }
                }
                afterglow.convert_into(rgba);
                flash.convert_into(&mut flash_rgba);
                for (pixel, extra) in rgba.iter_mut().zip(flash_rgba.iter()) {
                    pixel.0 += extra.0;
                    pixel.1 += extra.1;
                    pixel.2 += extra.2;
                    pixel.3 = (pixel.3 + extra.3).min(1.0);
                }
                if any_flash {
                    let glow = apply_dog_bloom(
                        rgba,
                        video_w as usize,
                        video_h as usize,
                        &DogBloomConfig::default(),
                    );
                    for (pixel, halo) in rgba.iter_mut().zip(glow.iter()) {
                        pixel.0 += halo.0;
                        pixel.1 += halo.1;
                        pixel.2 += halo.2;
                        pixel.3 = (pixel.3 + halo.3).min(1.0);
                    }
                }
            },
        )?;
        sink.record("lightning.mp4", "video");
        sink.record("lightning_hq.mp4", "video");

        let events: Vec<serde_json::Value> = bolts
            .iter()
            .zip(strike_frames.iter())
            .map(|(bolt, &frame)| {
                serde_json::json!({
                    "step": bolt.step,
                    "strike_frame": frame,
                    "speed_norm": bolt.speed,
                    "segments": bolt.segments.len(),
                })
            })
            .collect();
        let meta = serde_json::json!({
            "bolts": bolts.len(),
            "eta": ETA,
            "target_bias": TARGET_BIAS,
            "afterglow": AFTERGLOW,
            "return_stroke": RETURN_STROKE,
            "events": events,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("lightning_params.json", &json, "data")?;
        Ok(())
    }
}
