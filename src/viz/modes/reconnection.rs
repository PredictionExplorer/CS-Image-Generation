//! V39 `reconnection` -- Field Lines That Snap.
//!
//! Elastic field loops strung between body pairs stretch as the pair
//! separates, thin, and snap at critical tension -- recoiling halves retract
//! to their anchors, a flare bursts at the break with chromatic bloom, and a
//! brightness kink travels down both halves. Fresh loops respawn and the
//! ballet repeats, timed by the orbit's separation surges. Rendered as crisp
//! energy strokes over an 8% ghost of the re-accumulating master.

use crate::error::Result;
use crate::post_effects::{ChromaticBloom, ChromaticBloomConfig, PostEffect};
use crate::render::constants::DEFAULT_DT;
use crate::render::context::PixelBuffer;
use crate::render::{
    ImageBuffer, Rgb, quantize_display_buffer_to_16bit, tonemap_to_display_buffer,
};
use crate::sim::Sha3RandomByteStream;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::{Accumulator, stream_video};
use crate::viz::common::display::{SpdCanvas, auto_levels};
use crate::viz::common::kinematics::PAIRS;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;
use tracing::{info, warn};

/// Loops maintained per body pair.
const LOOPS_PER_PAIR: usize = 7;
/// Spring-chain nodes per loop.
const NODES: usize = 48;
/// Rest length as a multiple of the separation at spawn.
const REST_FACTOR: f64 = 1.15;
/// Tension (length / rest) that snaps a loop.
const SNAP_TENSION: f64 = 2.2;
/// Frames a snapped half takes to retract to its anchor.
const RETRACT_FRAMES: usize = 40;
/// Frames after a snap before the slot respawns a fresh loop.
const RESPAWN_DELAY: usize = 60;
/// Frames the traveling kink lives.
const KINK_FRAMES: usize = 15;
/// Frames a snap flare decays over.
const FLARE_FRAMES: usize = 18;
/// Ghost energy fraction of the master underlay.
const GHOST_ENERGY: f64 = 0.08;
/// Base chain stroke energy (hairline, "30% energy" of the flare scale).
const CHAIN_ENERGY: f64 = 0.005;
/// Flare stroke energy scale.
const FLARE_ENERGY: f64 = 0.017;
/// Spring constant (per frame^2, scale-invariant form).
const SPRING_K: f64 = 0.3;
/// Verlet velocity damping per frame.
const DAMPING: f64 = 0.92;
/// Video frames at 60 fps (30 s).
const VIDEO_FRAMES: usize = 1800;

/// The reconnection mode.
pub struct Reconnection;

/// One spring chain (a full loop or a retracting half).
struct Chain {
    pair_index: usize,
    slot: usize,
    nodes: Vec<(f64, f64)>,
    prev: Vec<(f64, f64)>,
    rest_seg: f64,
    /// Body pinning node 0 (None = free end).
    pin_head: Option<usize>,
    /// Body pinning the last node (None = free end).
    pin_tail: Option<usize>,
    /// Frames spent retracting (0 = active loop).
    retract_frames: usize,
    /// Traveling brightness kink: (node position, frames left).
    kink: Option<(f64, usize)>,
    /// Signed transverse bow acceleration factor.
    fan: f64,
}

/// One radial flare burst at a snap point.
struct Flare {
    x: f64,
    y: f64,
    tension: f64,
    separation: f64,
    frames_left: usize,
    pair_index: usize,
}

/// One recorded snap event for the audit sidecar.
struct SnapEvent {
    frame: usize,
    pair_index: usize,
    tension: f64,
    separation_rate: f64,
}

/// Deterministic spring-loop simulation over the whole video timeline.
struct LoopSim<'a> {
    ctx: &'a VizContext<'a>,
    rng: Sha3RandomByteStream,
    chains: Vec<Chain>,
    /// Pending respawns: (pair index, slot, spawn frame).
    respawns: Vec<(usize, usize, usize)>,
    flares: Vec<Flare>,
    events: Vec<SnapEvent>,
    steps: usize,
}

impl<'a> LoopSim<'a> {
    fn new(ctx: &'a VizContext<'a>) -> Self {
        let rng = ctx.fork_rng("reconnection");
        let steps = ctx.step_count();
        let mut sim = Self {
            ctx,
            rng,
            chains: Vec::new(),
            respawns: Vec::new(),
            flares: Vec::new(),
            events: Vec::new(),
            steps,
        };
        // Staggered initial spawns so the fans do not move in lockstep.
        for pair_index in 0..PAIRS.len() {
            for slot in 0..LOOPS_PER_PAIR {
                let delay = (sim.rng.next_f64() * 40.0) as usize;
                sim.respawns.push((pair_index, slot, delay));
            }
        }
        sim
    }

    fn anchor(&self, pair_index: usize, end: usize, step: usize) -> (f64, f64) {
        let (a, b) = PAIRS[pair_index];
        let body = if end == 0 { a } else { b };
        let position: Vector3<f64> = self.ctx.positions[body][step];
        (position.x, position.y)
    }

    fn spawn(&mut self, pair_index: usize, slot: usize, step: usize) {
        let head = self.anchor(pair_index, 0, step);
        let tail = self.anchor(pair_index, 1, step);
        let separation = (tail.0 - head.0).hypot(tail.1 - head.1).max(1e-9);
        let rest_seg = separation * REST_FACTOR / (NODES - 1) as f64;
        // Fan of loops: alternating sides, seeded bulge amplitudes.
        let side = if slot.is_multiple_of(2) { 1.0 } else { -1.0 };
        let bulge = separation * (0.05 + 0.12 * self.rng.next_f64()) * side;
        let normal = {
            let dx = tail.0 - head.0;
            let dy = tail.1 - head.1;
            (-dy / separation, dx / separation)
        };
        let nodes: Vec<(f64, f64)> = (0..NODES)
            .map(|index| {
                let t = index as f64 / (NODES - 1) as f64;
                let arc = (t * std::f64::consts::PI).sin() * bulge;
                (
                    head.0 + (tail.0 - head.0) * t + normal.0 * arc,
                    head.1 + (tail.1 - head.1) * t + normal.1 * arc,
                )
            })
            .collect();
        let (a, b) = PAIRS[pair_index];
        self.chains.push(Chain {
            pair_index,
            slot,
            prev: nodes.clone(),
            nodes,
            rest_seg,
            pin_head: Some(a),
            pin_tail: Some(b),
            retract_frames: 0,
            kink: None,
            fan: bulge * 0.0025,
        });
    }

    /// Advance one video frame; returns whether any flare is active.
    fn step_frame(&mut self, frame: usize, step: usize) -> bool {
        // Spawn due loops.
        let due: Vec<(usize, usize)> = self
            .respawns
            .iter()
            .filter(|&&(_, _, when)| when <= frame)
            .map(|&(pair_index, slot, _)| (pair_index, slot))
            .collect();
        self.respawns.retain(|&(_, _, when)| when > frame);
        for (pair_index, slot) in due {
            self.spawn(pair_index, slot, step);
        }

        let mut snapped: Vec<Chain> = Vec::new();
        let mut removed: Vec<usize> = Vec::new();
        for (chain_index, chain) in self.chains.iter_mut().enumerate() {
            // Pin anchored ends to the bodies' current positions.
            let last = chain.nodes.len() - 1;
            if let Some(body) = chain.pin_head {
                let position = self.ctx.positions[body][step];
                chain.nodes[0] = (position.x, position.y);
                chain.prev[0] = chain.nodes[0];
            }
            if let Some(body) = chain.pin_tail {
                let position = self.ctx.positions[body][step];
                chain.nodes[last] = (position.x, position.y);
                chain.prev[last] = chain.nodes[last];
            }

            // Damped Verlet with neighbor springs and the fan bow.
            let count = chain.nodes.len();
            let axis = {
                let head = chain.nodes[0];
                let tail = chain.nodes[count - 1];
                let dx = tail.0 - head.0;
                let dy = tail.1 - head.1;
                let norm = dx.hypot(dy).max(1e-12);
                (-dy / norm, dx / norm)
            };
            let old = chain.nodes.clone();
            for index in 0..count {
                let pinned = (index == 0 && chain.pin_head.is_some())
                    || (index == count - 1 && chain.pin_tail.is_some());
                if pinned {
                    continue;
                }
                let mut accel = (chain.fan * axis.0, chain.fan * axis.1);
                for neighbor in [index.wrapping_sub(1), index + 1] {
                    if neighbor >= count {
                        continue;
                    }
                    let dx = old[neighbor].0 - old[index].0;
                    let dy = old[neighbor].1 - old[index].1;
                    let length = dx.hypot(dy).max(1e-12);
                    let stretch = length - chain.rest_seg;
                    accel.0 += SPRING_K * stretch * dx / length;
                    accel.1 += SPRING_K * stretch * dy / length;
                }
                let velocity = (
                    (old[index].0 - chain.prev[index].0) * DAMPING,
                    (old[index].1 - chain.prev[index].1) * DAMPING,
                );
                chain.prev[index] = old[index];
                chain.nodes[index] =
                    (old[index].0 + velocity.0 + accel.0, old[index].1 + velocity.1 + accel.1);
            }

            // Advance the kink.
            if let Some((position, frames_left)) = chain.kink {
                chain.kink = (frames_left > 1)
                    .then(|| (position + (count - 1) as f64 / KINK_FRAMES as f64, frames_left - 1));
            }

            if chain.retract_frames > 0 {
                // Retracting half: contract toward the pinned anchor.
                chain.rest_seg *= 0.88;
                chain.retract_frames += 1;
                let length: f64 = chain
                    .nodes
                    .windows(2)
                    .map(|pair| (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1))
                    .sum();
                let anchor_body = chain.pin_head.or(chain.pin_tail).unwrap_or(0);
                let anchor = self.ctx.positions[anchor_body][step];
                let span = chain
                    .nodes
                    .iter()
                    .map(|node| (node.0 - anchor.x).hypot(node.1 - anchor.y))
                    .fold(0.0f64, f64::max);
                if chain.retract_frames > RETRACT_FRAMES || length < span * 0.1 || span < 1e-9 {
                    removed.push(chain_index);
                }
                continue;
            }

            // Tension check for active loops.
            let length: f64 = chain
                .nodes
                .windows(2)
                .map(|pair| (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1))
                .sum();
            let rest_total = chain.rest_seg * (count - 1) as f64;
            let tension = length / rest_total.max(1e-12);
            if tension > SNAP_TENSION {
                removed.push(chain_index);
                snapped.push(Chain {
                    pair_index: chain.pair_index,
                    slot: chain.slot,
                    nodes: chain.nodes.clone(),
                    prev: chain.prev.clone(),
                    rest_seg: chain.rest_seg,
                    pin_head: chain.pin_head,
                    pin_tail: chain.pin_tail,
                    retract_frames: 0,
                    kink: None,
                    fan: chain.fan,
                });
                // Separation rate around the snap (audit).
                let window = 50usize;
                let series = &self.ctx.kinematics().pairwise[chain.pair_index];
                let lo = step.saturating_sub(window);
                let hi = (step + window).min(self.steps - 1);
                let rate = (series[hi] - series[lo]) / (((hi - lo).max(1)) as f64 * DEFAULT_DT);
                self.events.push(SnapEvent {
                    frame,
                    pair_index: chain.pair_index,
                    tension,
                    separation_rate: rate,
                });
            }
        }

        // Split snapped chains into retracting halves + flares + respawns.
        for chain in snapped {
            let count = chain.nodes.len();
            // Max-curvature interior node.
            let mut snap_node = count / 2;
            let mut best_turn = -1.0f64;
            for index in 1..count - 1 {
                let previous = chain.nodes[index - 1];
                let current = chain.nodes[index];
                let next = chain.nodes[index + 1];
                let angle_in = (current.1 - previous.1).atan2(current.0 - previous.0);
                let angle_out = (next.1 - current.1).atan2(next.0 - current.0);
                let mut turn = (angle_out - angle_in).abs();
                if turn > std::f64::consts::PI {
                    turn = std::f64::consts::TAU - turn;
                }
                if turn > best_turn {
                    best_turn = turn;
                    snap_node = index;
                }
            }
            let snap_point = chain.nodes[snap_node];
            let rest_total = chain.rest_seg * (count - 1) as f64;
            let length: f64 = chain
                .nodes
                .windows(2)
                .map(|pair| (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1))
                .sum();
            let tension = length / rest_total.max(1e-12);
            let kick = chain.rest_seg * (tension - 1.0) * 0.8;

            let mut make_half = |range: std::ops::Range<usize>,
                                 pin_head: Option<usize>,
                                 pin_tail: Option<usize>,
                                 kink_at_head: bool| {
                if range.len() < 3 {
                    return;
                }
                let nodes: Vec<(f64, f64)> = chain.nodes[range.clone()].to_vec();
                let mut prev: Vec<(f64, f64)> = chain.prev[range].to_vec();
                // Elastic recoil: kick the free end away from the break.
                let free_index = if kink_at_head { 0 } else { nodes.len() - 1 };
                let inner_index = if kink_at_head { 1 } else { nodes.len() - 2 };
                let dx = nodes[free_index].0 - nodes[inner_index].0;
                let dy = nodes[free_index].1 - nodes[inner_index].1;
                let norm = dx.hypot(dy).max(1e-12);
                prev[free_index] = (
                    nodes[free_index].0 + dx / norm * kick,
                    nodes[free_index].1 + dy / norm * kick,
                );
                let kink_position = if kink_at_head { 0.0 } else { (nodes.len() - 1) as f64 };
                self.chains.push(Chain {
                    pair_index: chain.pair_index,
                    slot: chain.slot,
                    nodes,
                    prev,
                    rest_seg: chain.rest_seg,
                    pin_head,
                    pin_tail,
                    retract_frames: 1,
                    kink: Some((kink_position, KINK_FRAMES)),
                    fan: 0.0,
                });
            };
            make_half(0..snap_node + 1, chain.pin_head, None, false);
            make_half(snap_node..count, None, chain.pin_tail, true);

            let (a, b) = PAIRS[chain.pair_index];
            let separation = {
                let pa = self.ctx.positions[a][step];
                let pb = self.ctx.positions[b][step];
                (pb.x - pa.x).hypot(pb.y - pa.y)
            };
            self.flares.push(Flare {
                x: snap_point.0,
                y: snap_point.1,
                tension,
                separation,
                frames_left: FLARE_FRAMES,
                pair_index: chain.pair_index,
            });
            self.respawns.push((chain.pair_index, chain.slot, frame + RESPAWN_DELAY));
        }

        // Remove dead chains (descending order keeps indices valid).
        for index in removed.into_iter().rev() {
            self.chains.swap_remove(index);
        }

        // Age flares.
        for flare in &mut self.flares {
            flare.frames_left = flare.frames_left.saturating_sub(1);
        }
        self.flares.retain(|flare| flare.frames_left > 0);
        !self.flares.is_empty()
    }

    /// Total flare brightness this frame (for the max-flare still pick).
    fn flare_intensity(&self) -> f64 {
        self.flares
            .iter()
            .map(|flare| flare.tension * f64::from(flare.frames_left as u32) / FLARE_FRAMES as f64)
            .sum()
    }

    /// Draw chains and flares as energy strokes into the canvas.
    fn draw(
        &self,
        canvas: &mut SpdCanvas,
        render_ctx: &crate::render::context::RenderContext,
        step: usize,
        scale: f32,
    ) {
        for chain in &self.chains {
            let (a, b) = PAIRS[chain.pair_index];
            let color_a = self.ctx.colors[a][step.min(self.ctx.colors[a].len() - 1)];
            let color_b = self.ctx.colors[b][step.min(self.ctx.colors[b].len() - 1)];
            let count = chain.nodes.len();
            for index in 0..count - 1 {
                let t0 = index as f64 / (count - 1) as f64;
                let t1 = (index + 1) as f64 / (count - 1) as f64;
                let lerp = |t: f64| {
                    (
                        color_a.0 + (color_b.0 - color_a.0) * t,
                        color_a.1 + (color_b.1 - color_a.1) * t,
                        color_a.2 + (color_b.2 - color_a.2) * t,
                    )
                };
                let mut energy = CHAIN_ENERGY;
                if let Some((kink_position, frames_left)) = chain.kink
                    && (index as f64 - kink_position).abs() < 3.0
                {
                    energy *= 1.0 + 2.5 * frames_left as f64 / KINK_FRAMES as f64;
                }
                let from = render_ctx.to_pixel(chain.nodes[index].0, chain.nodes[index].1);
                let to = render_ctx.to_pixel(chain.nodes[index + 1].0, chain.nodes[index + 1].1);
                canvas.draw_stroke(
                    (from.0 * scale, from.1 * scale),
                    (to.0 * scale, to.1 * scale),
                    lerp(t0),
                    lerp(t1),
                    0.9,
                    energy,
                    0.7,
                );
            }
        }

        for flare in &self.flares {
            let envelope = flare.frames_left as f64 / FLARE_FRAMES as f64;
            let energy = FLARE_ENERGY * flare.tension * envelope;
            let radius = flare.separation * 0.22 * (1.2 - envelope * 0.4);
            let (a, b) = PAIRS[flare.pair_index];
            let color_a = self.ctx.colors[a][step.min(self.ctx.colors[a].len() - 1)];
            let color_b = self.ctx.colors[b][step.min(self.ctx.colors[b].len() - 1)];
            let blend = (
                (color_a.0 + color_b.0) * 0.5,
                (color_a.1 + color_b.1) * 0.5,
                (color_a.2 + color_b.2) * 0.5,
            );
            let center = render_ctx.to_pixel(flare.x, flare.y);
            for spoke in 0..10 {
                let angle = f64::from(spoke) / 10.0 * std::f64::consts::TAU
                    + f64::from(flare.frames_left as u32) * 0.05;
                let tip = render_ctx
                    .to_pixel(flare.x + radius * angle.cos(), flare.y + radius * angle.sin());
                canvas.draw_stroke(
                    (center.0 * scale, center.1 * scale),
                    (tip.0 * scale, tip.1 * scale),
                    blend,
                    blend,
                    0.95,
                    energy,
                    1.4,
                );
            }
        }
    }
}

impl VizMode for Reconnection {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("reconnection").expect("reconnection is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps == 0 {
            warn!("reconnection skipped: empty trajectory");
            return Ok(());
        }
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(VIDEO_FRAMES);

        // Pass 1: simulate only, to find the max-flare frame.
        let mut probe = LoopSim::new(ctx);
        let mut max_flare = (0usize, -1.0f64);
        for frame in 0..frame_count {
            let step = (((frame + 1) * steps / frame_count).min(steps)).saturating_sub(1);
            probe.step_frame(frame, step);
            let intensity = probe.flare_intensity();
            if intensity > max_flare.1 {
                max_flare = (frame, intensity);
            }
        }
        let snap_count = probe.events.len();
        info!(
            "   reconnection: {snap_count} snap(s), max flare at frame {} (intensity {:.2})",
            max_flare.0, max_flare.1
        );

        // Ghost of the master scene; fixed levels from its full exposure.
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
        ghost.accumulate(0..steps);
        let world_to_px = crate::render::context::RenderContext::with_bounds(
            video_w,
            video_h,
            *ghost.render_ctx().bounds(),
        );

        // Fixed levels from a representative combined exposure: the 8% ghost
        // plus the chains re-simulated to the max-flare frame (levels from
        // the ghost alone would leave every chain stroke clipped white).
        let levels = {
            let mut rgba = ghost.convert();
            for pixel in &mut rgba {
                pixel.0 *= GHOST_ENERGY;
                pixel.1 *= GHOST_ENERGY;
                pixel.2 *= GHOST_ENERGY;
            }
            let mut level_sim = LoopSim::new(ctx);
            let mut level_step = 0usize;
            for frame in 0..=max_flare.0 {
                level_step = (((frame + 1) * steps / frame_count).min(steps)).saturating_sub(1);
                level_sim.step_frame(frame, level_step);
            }
            let mut level_canvas = SpdCanvas::new(video_w, video_h);
            level_sim.draw(&mut level_canvas, &world_to_px, level_step, 1.0);
            let mut level_rgba: PixelBuffer = Vec::new();
            level_canvas.convert_into(&mut level_rgba);
            for (pixel, chain) in rgba.iter_mut().zip(level_rgba.iter()) {
                pixel.0 += chain.0;
                pixel.1 += chain.1;
                pixel.2 += chain.2;
                pixel.3 = (pixel.3 + chain.3).min(1.0);
            }
            auto_levels(
                &rgba,
                ctx.settings.resolved_config.clip_black,
                ctx.settings.resolved_config.clip_white,
                1.0,
            )
        };
        ghost.clear();

        let bloom = ChromaticBloom::new(ChromaticBloomConfig::from_resolution(
            video_w as usize,
            video_h as usize,
        ));

        // Pass 2: identical re-simulation (same fork), rendered.
        let mut sim = LoopSim::new(ctx);
        let mut canvas = SpdCanvas::new(video_w, video_h);
        let mut chain_rgba: PixelBuffer = Vec::new();
        let mut still: Option<ImageBuffer<Rgb<u16>, Vec<u16>>> = None;
        let mut cursor = 0usize;
        stream_video(
            video_w,
            video_h,
            60,
            &sink.path("reconnection.mp4"),
            &sink.path("reconnection_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let target = ((frame + 1) * steps / frame_count).min(steps);
                ghost.accumulate(cursor..target);
                cursor = target;
                let step = target.saturating_sub(1);

                let flare_active = sim.step_frame(frame, step);
                canvas.clear();
                sim.draw(&mut canvas, &world_to_px, step, 1.0);

                ghost.convert_into(rgba);
                canvas.convert_into(&mut chain_rgba);
                for (pixel, chain) in rgba.iter_mut().zip(chain_rgba.iter()) {
                    pixel.0 = pixel.0 * GHOST_ENERGY + chain.0;
                    pixel.1 = pixel.1 * GHOST_ENERGY + chain.1;
                    pixel.2 = pixel.2 * GHOST_ENERGY + chain.2;
                    pixel.3 = (pixel.3 * GHOST_ENERGY + chain.3).min(1.0);
                }
                if flare_active
                    && let Ok(bloomed) = bloom.process(rgba, video_w as usize, video_h as usize)
                {
                    rgba.copy_from_slice(&bloomed);
                }

                if frame == max_flare.0 {
                    let display = tonemap_to_display_buffer(rgba, &levels);
                    let bytes = quantize_display_buffer_to_16bit(&display);
                    still = ImageBuffer::from_raw(video_w, video_h, bytes);
                }
            },
        )?;
        sink.record("reconnection.mp4", "video");
        sink.record("reconnection_hq.mp4", "video");

        if let Some(image) = still {
            sink.save_png16(&image, "reconnection_still.png")?;
        }

        let events: Vec<serde_json::Value> = sim
            .events
            .iter()
            .map(|event| {
                serde_json::json!({
                    "frame": event.frame,
                    "pair": [PAIRS[event.pair_index].0, PAIRS[event.pair_index].1],
                    "tension": event.tension,
                    "separation_rate": event.separation_rate,
                })
            })
            .collect();
        let separating = sim.events.iter().filter(|event| event.separation_rate > 0.0).count();
        let meta = serde_json::json!({
            "loops_per_pair": LOOPS_PER_PAIR,
            "nodes": NODES,
            "snap_tension": SNAP_TENSION,
            "respawn_delay_frames": RESPAWN_DELAY,
            "snap_count": events.len(),
            "snaps_while_separating": separating,
            "max_flare_frame": max_flare.0,
            "events": events,
            "note": "still captured at video resolution; bloom applied on flare frames only",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("reconnection_events.json", &json, "data")?;
        Ok(())
    }
}
