//! Orchestration of the ember edition: fluid, ink, shading and the frame stream.
//!
//! One frame is rendered per entry of the frame schedule (the same orbit checkpoints as the main
//! video). Between two frames the fluid is advanced through a few velocity snapshots, the ink
//! fields are remapped along exact backward characteristics through those snapshots, and every
//! pixel is shaded from its `q×q` ink nodes. The last frame is the still.

use std::time::Instant;

use nalgebra::Vector3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;

use super::config::{EmberConfig, FluidConfig};
use super::error::{EmberError, EmberResult};
use super::fluid::{FluidGrid, FluidStats, Snapshot, WakeSolver};
use super::ink::{InkDecay, InkFields, NodeGrid, remap};
use super::look::Look;
use super::math;
use super::optics::Optics;
use super::orbit::{BodyMotion, BodyTrack};
use super::paper::{KozoSheet, check_fibre_count};
use super::trace::{ContactRules, FlowWindow};
use crate::sim::Sha3RandomByteStream;

/// Largest supported output side, matching the main renderer's limit.
const MAX_SIDE: u32 = 16_384;

/// Most threads the fluid solver uses.
///
/// Its transforms work on the rows and columns of a grid of about 1440×1024 nodes: too little
/// work per task to feed very many threads. On a 64-core Threadripper PRO 9985WX one step took
/// 23.6 ms with 16 or 32 threads, 26.4 ms with 64 and 32.2 ms with 128. The solver therefore runs
/// in its own pool of at most this many threads, while the ink remap and the shading, which keep
/// scaling, use the caller's pool. Results never depend on the thread count.
const FLUID_MAX_THREADS: usize = 32;

/// A dedicated pool for the fluid solver when the caller's pool has more than
/// [`FLUID_MAX_THREADS`] threads; `None` (smaller pools, or if the pool cannot be built) runs the
/// solver in the caller's pool.
fn fluid_pool() -> Option<rayon::ThreadPool> {
    if rayon::current_num_threads() <= FLUID_MAX_THREADS {
        return None;
    }
    rayon::ThreadPoolBuilder::new().num_threads(FLUID_MAX_THREADS).build().ok()
}

/// Runs `work` in `pool` if there is one, else in the caller's pool.
fn in_pool<T: Send>(pool: Option<&rayon::ThreadPool>, work: impl FnOnce() -> T + Send) -> T {
    match pool {
        Some(pool) => pool.install(work),
        None => work(),
    }
}

/// What [`render_ember`] renders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmberMode {
    /// Every scheduled frame is shaded and handed to the sink; the last one is the still.
    Video,
    /// Only the final frame is shaded (the fluid and ink still run through the whole schedule,
    /// so the still is identical to the video's last frame). The sink is not called.
    StillOnly,
}

/// Inputs of one ember render.
#[derive(Clone, Copy, Debug)]
pub struct EmberRequest<'a> {
    /// Raw recorded orbit, `positions[body][knot]`: exactly three bodies, at least two knots.
    pub positions: &'a [Vec<Vector3<f64>>],
    /// Recorded knots to render as frames: strictly increasing, the last one must be the final
    /// knot (the still). Use the main video's checkpoints so both videos stay in step.
    pub frame_steps: &'a [usize],
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Seed of the kozo sheet's texture.
    pub paper_seed: &'a [u8],
    /// Look and simulation parameters.
    pub config: &'a EmberConfig,
    /// Whether to shade every frame or only the still.
    pub mode: EmberMode,
}

/// One rendered frame, handed to the sink in schedule order.
#[derive(Clone, Copy, Debug)]
pub struct EmberFrame<'a> {
    /// Frame number (0-based).
    pub index: usize,
    /// Number of frames in the schedule.
    pub count: usize,
    /// Recorded orbit knot shown by this frame.
    pub orbit_step: usize,
    /// Fluid time of this frame.
    pub time: f64,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// sRGB-encoded 16-bit RGB samples, row-major from the top-left pixel.
    pub rgb: &'a [u16],
    /// The same samples as little-endian bytes (`rgb48le`), ready for an encoder pipe.
    pub rgb48le: &'a [u8],
}

/// Projection of the orbit onto the canvas (recorded as the certificate's `derived.projection`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmberProjection {
    /// Centre of the orbit's bounding box (original units).
    pub origin: [f64; 3],
    /// Largest side of the orbit's bounding box (original units).
    pub extent: f64,
    /// The two principal axes spanning the canvas (normalised 3-D space).
    pub axes: [[f64; 3]; 2],
    /// World units per normalised projected unit.
    pub scale: f64,
    /// Principal variances, descending.
    pub variances: [f64; 3],
}

/// Deterministic statistics of a render (part of the determinism contract; recorded as the
/// certificate's `stats`).
///
/// Node fractions count the visible ink nodes, `width·height·q²` (`q` =
/// [`RasterConfig::supersample`](super::config::RasterConfig::supersample)), the margin
/// excluded. A node "carries cinnabar" when the look gives it a positive cinnabar load (a fresh
/// meeting or a glowing ember). Every count is an integer summed or maximised exactly, and every
/// fraction is one such count divided by the node count, so the statistics are identical on
/// every architecture and for every thread count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmberStats {
    /// Navier–Stokes time steps.
    pub fluid_steps: u64,
    /// Smallest fluid time step.
    pub min_dt: f64,
    /// Largest fluid time step.
    pub max_dt: f64,
    /// Largest penalised flow speed.
    pub max_flow_speed: f64,
    /// Velocity snapshots used for tracing.
    pub snapshots: u64,
    /// Sum over frames of ink nodes that touched a soak zone during the frame interval.
    pub contact_events: u64,
    /// Fraction of the still's ink nodes carrying any ink.
    pub still_ink_fraction: f64,
    /// Fraction of the still's ink nodes carrying cinnabar.
    pub still_cinnabar_fraction: f64,
    /// Pixels of the still that needed gamut mapping.
    pub still_gamut_mapped_pixels: u64,
    /// Shaded frames in which at least one visible ink node carries cinnabar: every frame in
    /// [`EmberMode::Video`], only the still (so 0 or 1) in [`EmberMode::StillOnly`]. With
    /// `still_cinnabar_fraction` it tells, without opening the media, whether the video shows
    /// vermilion even when the still has none.
    pub frames_with_cinnabar: u64,
    /// Largest fraction, over the shaded frames, of the visible ink nodes carrying cinnabar
    /// (equal to `still_cinnabar_fraction` in [`EmberMode::StillOnly`]).
    pub peak_frame_cinnabar_fraction: f64,
}

/// Wall-clock timings of a render in seconds (informational, not deterministic).
///
/// Serialised without the `_seconds` suffix (`{"fluid": …, "ink": …, …}`), as the certificate's
/// `timings_seconds` section.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmberTimings {
    /// Fluid solver.
    #[serde(rename = "fluid")]
    pub fluid_seconds: f64,
    /// Ink tracing and remapping.
    #[serde(rename = "ink")]
    pub ink_seconds: f64,
    /// Shading and encoding.
    #[serde(rename = "shade")]
    pub shade_seconds: f64,
    /// Hashing and the frame sink (including video encoder back-pressure).
    #[serde(rename = "sink")]
    pub sink_seconds: f64,
    /// Whole render.
    #[serde(rename = "total")]
    pub total_seconds: f64,
}

/// Result of a render.
#[derive(Clone, Debug)]
pub struct EmberSummary {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The still (the final frame): sRGB-encoded 16-bit RGB, row-major from the top-left pixel.
    pub still: Vec<u16>,
    /// SHA-256 (hex) of the still as `rgb48le` bytes.
    pub still_sha256: String,
    /// Frames handed to the sink (`0` in [`EmberMode::StillOnly`]).
    pub frames_emitted: usize,
    /// SHA-256 (hex) of the concatenated `rgb48le` frame stream, in [`EmberMode::Video`].
    pub frames_sha256: Option<String>,
    /// Orbit duration in fluid time units.
    pub duration: f64,
    /// Fluid time at which the bodies stopped inking.
    pub valve_time: f64,
    /// Fluid grid `[nx, ny]`.
    pub fluid_grid: [usize; 2],
    /// Fluid grid spacing.
    pub fluid_dx: f64,
    /// Ink node grid `[cols, rows]` including the margin.
    pub ink_grid: [usize; 2],
    /// Orbit projection.
    pub projection: EmberProjection,
    /// Deterministic statistics.
    pub stats: EmberStats,
    /// Wall-clock timings.
    pub timings: EmberTimings,
}

impl EmberSummary {
    /// The still as an image buffer.
    pub fn still_image(&self) -> image::ImageBuffer<image::Rgb<u16>, Vec<u16>> {
        image::ImageBuffer::from_raw(self.width, self.height, self.still.clone())
            .expect("the still holds width × height × 3 samples by construction")
    }
}

/// Everything [`render_ember`] checks and derives before the fluid starts.
///
/// Planning costs one projection of the orbit (about 50 ms for a million recorded steps) and
/// allocates nothing proportional to the output, so callers can run it long before rendering to
/// reject an orbit the edition cannot draw. [`render_ember`] plans the request itself; see
/// [`plan_ember`] for what a successful plan does and does not guarantee.
#[derive(Debug)]
pub struct EmberPlan {
    track: BodyTrack,
    grid: FluidGrid,
    nodes: NodeGrid,
    rules: ContactRules,
    aspect: f64,
    frames: usize,
}

impl EmberPlan {
    /// Orbit duration `T` in fluid time units (the median body speed is the reference speed).
    pub fn duration(&self) -> f64 {
        self.track.duration()
    }

    /// Fluid time at which the bodies start inking (the pre-roll).
    pub fn ink_start(&self) -> f64 {
        self.rules.t_on
    }

    /// Fluid time at which the bodies stop inking (`T - valve_lead`).
    pub fn valve_time(&self) -> f64 {
        self.rules.t_valve
    }

    /// Scheduled frames; the last one is the still.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Fluid grid `[nx, ny]`.
    pub fn fluid_grid(&self) -> [usize; 2] {
        [self.grid.nx, self.grid.ny]
    }

    /// Ink node grid `[cols, rows]`, margin included.
    pub fn ink_grid(&self) -> [usize; 2] {
        [self.nodes.cols, self.nodes.rows]
    }
}

/// Validates a request and derives its plan: the configuration, the output size (at most
/// 16,384 pixels per side), the kozo sheet's fibre count at that size (at most 16,777,216), the
/// frame schedule (non-empty, strictly increasing, ending on the final knot), the orbit's
/// projection (it must span a plane), its duration against the pre-roll and the valve
/// (`T - valve_lead > pre_roll`), and the fluid and ink grids.
///
/// Barring resource exhaustion, a request that plans successfully can fail later only if the
/// simulated flow becomes non-finite or the sink fails. Planning checks ranges; it does not
/// budget memory or time, which grow with the output size, the fluid grid and the snapshot
/// cadence.
pub fn plan_ember(request: &EmberRequest<'_>) -> EmberResult<EmberPlan> {
    let config = request.config;
    config.validate()?;
    validate_size(request.width, request.height)?;
    // The paper's memory guard depends on the density and the output's aspect together, which
    // `validate` cannot see; checked here so that `render_ember` never fails on it later.
    check_fibre_count(request.width, request.height, &config.paper)?;
    let knots = request.positions.first().map_or(0, Vec::len);
    validate_schedule(request.frame_steps, knots)?;

    let aspect = f64::from(request.width) / f64::from(request.height);
    let track = BodyTrack::new(request.positions, aspect, config)?;
    let t_valve = valve_time(track.duration(), config)?;
    let grid = FluidGrid::for_canvas(aspect, config.fluid.rows, config.fluid.box_margin)?;
    let nodes = NodeGrid::new(
        request.width,
        request.height,
        config.raster.supersample,
        config.raster.ink_margin,
    )?;
    let rules = ContactRules {
        reach: config.fluid.body_radius + config.contact.soak_depth,
        vorticity_gate: config.contact.vorticity_gate,
        t_on: config.contact.pre_roll,
        t_valve,
    };
    Ok(EmberPlan { track, grid, nodes, rules, aspect, frames: request.frame_steps.len() })
}

/// `t_valve = T - valve_lead` for an orbit of duration `T`; the orbit is too short unless
/// `t_valve > pre_roll`, i.e. unless ink can be laid down for some time after the start settles
/// (a NaN duration is rejected too).
fn valve_time(duration: f64, config: &EmberConfig) -> EmberResult<f64> {
    let contact = &config.contact;
    let t_valve = duration - contact.valve_lead;
    if t_valve.partial_cmp(&contact.pre_roll) == Some(std::cmp::Ordering::Greater) {
        Ok(t_valve)
    } else {
        Err(EmberError::OrbitTooShort { duration, required: contact.pre_roll + contact.valve_lead })
    }
}

/// Renders the ember edition of an orbit.
///
/// In [`EmberMode::Video`] every scheduled frame is passed to `sink` in order; the returned
/// summary's `still` equals the last frame. The result is bit-identical on every CPU architecture
/// and for every rayon thread count.
pub fn render_ember(
    request: &EmberRequest<'_>,
    sink: &mut dyn FnMut(&EmberFrame<'_>) -> EmberResult<()>,
) -> EmberResult<EmberSummary> {
    let started = Instant::now();
    let config = request.config;
    let EmberPlan { track, grid, nodes, rules, aspect, .. } = plan_ember(request)?;
    let duration = track.duration();
    let (t_on, t_valve) = (rules.t_on, rules.t_valve);
    let fluid_pool = fluid_pool();
    let mut solver = in_pool(fluid_pool.as_ref(), || WakeSolver::new(grid, &config.fluid, aspect))?;
    info!(
        "Ember edition: orbit lasts {duration:.3} fluid units; fluid {}x{} (dx {:.5}); ink nodes \
         {}x{}; {} frames",
        grid.nx,
        grid.ny,
        grid.dx,
        nodes.cols,
        nodes.rows,
        request.frame_steps.len()
    );

    let look = Look::new(&config.look);
    let optics = Optics::new();
    let mut paper_rng = Sha3RandomByteStream::new(request.paper_seed, 0.0, 1.0, 1.0, 1.0);
    let sheet = KozoSheet::generate(request.width, request.height, &config.paper, &mut paper_rng)?;
    let paper = PaperCache::new(&sheet, &optics);
    let shader =
        Shader { nodes: &nodes, look: &look, optics: &optics, sheet: &sheet, paper: &paper };

    let mut fields = InkFields::zeros(&nodes);
    let mut next_fields = InkFields::zeros(&nodes);
    let mut window = SnapshotWindow::new(&grid);
    in_pool(fluid_pool.as_ref(), || solver.snapshot_into(window.first_mut()));
    window.first_bodies = positions_of(&track, 0.0);

    let pixels = request.width as usize * request.height as usize;
    let mut rgb = vec![0u16; pixels * 3];
    let mut rgb48le = vec![0u8; pixels * 6];
    let mut stream_hasher = Sha256::new();
    let mut stats = EmberStats::default();
    let mut timings = EmberTimings::default();
    let mut frames_emitted = 0;
    // Visible ink nodes (the margin excluded): the denominator of the node fractions.
    let view_nodes = (nodes.width * nodes.height * nodes.supersample * nodes.supersample) as f64;
    // Most cinnabar nodes in one shaded frame (an exact integer maximum).
    let mut peak_cinnabar_nodes = 0u64;

    let count = request.frame_steps.len();
    let mut previous_step = 0;
    let mut t_prev = 0.0;
    let progress_every = (count / 20).max(1);
    for (index, &step) in request.frame_steps.iter().enumerate() {
        let t_frame = track.knot_time(step);
        let intervals =
            snapshot_intervals(&track, |k| track.knot_time(k), previous_step, step, &config.fluid);

        let clock = Instant::now();
        window.begin_frame();
        in_pool(fluid_pool.as_ref(), || -> EmberResult<()> {
            for s in 1..=intervals {
                let target = if s == intervals {
                    t_frame
                } else {
                    t_prev + (t_frame - t_prev) * (s as f64) / (intervals as f64)
                };
                solver.advance_to(target, &track)?;
                debug_assert_eq!(solver.time(), target, "the solver lands on snapshot times");
                solver.snapshot_into(window.push());
                window.push_bodies(positions_of(&track, target));
            }
            Ok(())
        })?;
        stats.snapshots += intervals as u64;
        timings.fluid_seconds += clock.elapsed().as_secs_f64();

        let clock = Instant::now();
        if intervals > 0 && t_frame >= t_on {
            let flow = FlowWindow { grid, snapshots: window.snapshots(), bodies: window.bodies() };
            let dt = t_frame - t_prev;
            let decay = InkDecay {
                frame_time: t_frame,
                fresh_tau: config.look.fresh_tau,
                fresh_fade: math::exp(-dt / config.look.fresh_tau),
                floor_fade: config.look.floor_tau.map_or(1.0, |tau| math::exp(-dt / tau)),
                ember_fade: config.look.ember_tau.map_or(0.0, |tau| math::exp(-dt / tau)),
            };
            let remap_stats =
                remap(&fields, &mut next_fields, &nodes, &flow, &rules, &decay, &look);
            if remap_stats.non_finite_origins > 0 {
                return Err(EmberError::NonFinite { stage: "ink trace", time: t_frame });
            }
            std::mem::swap(&mut fields, &mut next_fields);
            stats.contact_events += remap_stats.contacted_nodes;
        }
        timings.ink_seconds += clock.elapsed().as_secs_f64();

        let is_last = index + 1 == count;
        if request.mode == EmberMode::Video || is_last {
            let clock = Instant::now();
            let shade_stats = shader.shade(&fields, &mut rgb);
            if !shade_stats.finite {
                return Err(EmberError::NonFinite { stage: "shading", time: t_frame });
            }
            encode_le(&rgb, &mut rgb48le);
            timings.shade_seconds += clock.elapsed().as_secs_f64();
            stats.frames_with_cinnabar += u64::from(shade_stats.cinnabar_nodes > 0);
            peak_cinnabar_nodes = peak_cinnabar_nodes.max(shade_stats.cinnabar_nodes);
            if is_last {
                stats.still_ink_fraction = shade_stats.inked_nodes as f64 / view_nodes;
                stats.still_cinnabar_fraction = shade_stats.cinnabar_nodes as f64 / view_nodes;
                stats.still_gamut_mapped_pixels = shade_stats.gamut_mapped;
            }
            if request.mode == EmberMode::Video {
                let clock = Instant::now();
                stream_hasher.update(&rgb48le);
                sink(&EmberFrame {
                    index,
                    count,
                    orbit_step: step,
                    time: t_frame,
                    width: request.width,
                    height: request.height,
                    rgb: &rgb,
                    rgb48le: &rgb48le,
                })?;
                frames_emitted += 1;
                timings.sink_seconds += clock.elapsed().as_secs_f64();
            }
        }

        window.end_frame();
        previous_step = step;
        t_prev = t_frame;
        if (index + 1) % progress_every == 0 || is_last {
            let elapsed = started.elapsed().as_secs_f64();
            let eta = elapsed / (index + 1) as f64 * (count - index - 1) as f64;
            info!(
                "   Ember frame {}/{count} (t = {t_frame:.3}/{duration:.3}) — {elapsed:.0}s \
                 elapsed, ~{eta:.0}s left",
                index + 1
            );
        }
    }

    stats.peak_frame_cinnabar_fraction = peak_cinnabar_nodes as f64 / view_nodes;
    let fluid: FluidStats = solver.stats();
    stats.fluid_steps = fluid.steps;
    stats.min_dt = fluid.min_dt;
    stats.max_dt = fluid.max_dt;
    stats.max_flow_speed = fluid.max_speed;
    timings.total_seconds = started.elapsed().as_secs_f64();

    let projection = track.projection();
    Ok(EmberSummary {
        width: request.width,
        height: request.height,
        still_sha256: hex::encode(Sha256::digest(&rgb48le)),
        still: rgb,
        frames_emitted,
        frames_sha256: (request.mode == EmberMode::Video)
            .then(|| hex::encode(stream_hasher.finalize())),
        duration,
        valve_time: t_valve,
        fluid_grid: [grid.nx, grid.ny],
        fluid_dx: grid.dx,
        ink_grid: [nodes.cols, nodes.rows],
        projection: EmberProjection {
            origin: projection.origin,
            extent: projection.extent,
            axes: projection.axes,
            scale: projection.scale,
            variances: projection.variances,
        },
        stats,
        timings,
    })
}

/// Rejects empty or oversized outputs.
fn validate_size(width: u32, height: u32) -> EmberResult<()> {
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return Err(EmberError::InvalidConfig {
            parameter: "resolution".into(),
            reason: format!("{width}x{height} must be non-empty and at most {MAX_SIDE} per side"),
        });
    }
    Ok(())
}

/// The schedule must be non-empty, strictly increasing, and end on the final knot.
fn validate_schedule(steps: &[usize], knots: usize) -> EmberResult<()> {
    let invalid = |reason: String| Err(EmberError::InvalidSchedule { reason });
    let Some(&last) = steps.last() else {
        return invalid("no frames scheduled".into());
    };
    if knots < 2 {
        return invalid(format!("the orbit has {knots} recorded knots; at least 2 are needed"));
    }
    if last != knots - 1 {
        let final_knot = knots - 1;
        return invalid(format!(
            "the last frame shows knot {last}, not the final knot {final_knot}"
        ));
    }
    if let Some(&[earlier, later]) = steps.windows(2).find(|pair| pair[1] <= pair[0]) {
        return invalid(format!("frames must be strictly increasing ({earlier} then {later})"));
    }
    Ok(())
}

/// Number of snapshot intervals `S` between the frames at recorded knots `from ≤ to`:
///
/// ```text
/// S = max(⌈Δt / max_snapshot_interval⌉, ⌈max_b path_b / (max_snapshot_travel·body_radius)⌉, 1)
/// ```
///
/// with `Δt = knot_time(to) - knot_time(from)` and `path_b` body `b`'s path length along its
/// recorded polyline, i.e. the sum of `|pos_b(knot_time(k)) - pos_b(knot_time(k - 1))|` over
/// `k ∈ (from, to]` (not the displacement between the frames, which misses a body that turns
/// back). No interval is then longer than `max_snapshot_interval` or lets a body move more than
/// `max_snapshot_travel` radii. `from == to` returns 0: the frame shows the previous frame's
/// time and nothing advances.
///
/// `knot_time(k)` is the fluid time of recorded knot `k` (`BodyTrack::knot_time` in the
/// pipeline; strictly increasing), and `motion` gives the body positions at those times.
fn snapshot_intervals<M: BodyMotion + ?Sized>(
    motion: &M,
    knot_time: impl Fn(usize) -> f64,
    from: usize,
    to: usize,
    fluid: &FluidConfig,
) -> usize {
    if to == from {
        return 0;
    }
    let dt = knot_time(to) - knot_time(from);
    let by_time = (dt / fluid.max_snapshot_interval).ceil();
    let mut path = [0.0f64; 3];
    let mut previous = positions_of(motion, knot_time(from));
    for k in from + 1..=to {
        let current = positions_of(motion, knot_time(k));
        for b in 0..3 {
            let (dx, dy) = (current[b][0] - previous[b][0], current[b][1] - previous[b][1]);
            path[b] += (dx * dx + dy * dy).sqrt();
        }
        previous = current;
    }
    let longest = path.iter().copied().fold(0.0, f64::max);
    let by_travel = (longest / (fluid.max_snapshot_travel * fluid.body_radius)).ceil();
    (by_time.max(by_travel).max(1.0)) as usize
}

/// World positions of the three bodies at fluid time `t`.
fn positions_of<M: BodyMotion + ?Sized>(motion: &M, t: f64) -> [[f64; 2]; 3] {
    motion.bodies_at(t).map(|body| body.position)
}

/// Little-endian bytes of 16-bit samples (explicit, so the stream is identical on any host).
fn encode_le(samples: &[u16], out: &mut [u8]) {
    out.par_chunks_mut(1 << 16).zip(samples.par_chunks(1 << 15)).for_each(|(bytes, words)| {
        for (pair, word) in bytes.chunks_exact_mut(2).zip(words) {
            pair.copy_from_slice(&word.to_le_bytes());
        }
    });
}

/// The snapshots of the current frame interval, recycled between frames.
struct SnapshotWindow {
    /// `snapshots[0]` is the previous frame's last snapshot; `len` are in use.
    snapshots: Vec<Snapshot>,
    /// Body positions at each snapshot in use.
    bodies: Vec<[[f64; 2]; 3]>,
    /// Snapshots in use.
    len: usize,
    /// Body positions of `snapshots[0]` (set before the first frame).
    first_bodies: [[f64; 2]; 3],
    grid: FluidGrid,
}

impl SnapshotWindow {
    fn new(grid: &FluidGrid) -> Self {
        Self {
            snapshots: vec![Snapshot::zeros(grid)],
            bodies: Vec::new(),
            len: 1,
            first_bodies: [[0.0; 2]; 3],
            grid: *grid,
        }
    }

    fn first_mut(&mut self) -> &mut Snapshot {
        &mut self.snapshots[0]
    }

    /// Starts a frame interval whose first snapshot is the previous frame's last.
    fn begin_frame(&mut self) {
        self.len = 1;
        self.bodies.clear();
        self.bodies.push(self.first_bodies);
    }

    /// A slot for the next snapshot.
    fn push(&mut self) -> &mut Snapshot {
        if self.len == self.snapshots.len() {
            self.snapshots.push(Snapshot::zeros(&self.grid));
        }
        self.len += 1;
        &mut self.snapshots[self.len - 1]
    }

    fn push_bodies(&mut self, bodies: [[f64; 2]; 3]) {
        self.bodies.push(bodies);
    }

    fn snapshots(&self) -> &[Snapshot] {
        &self.snapshots[..self.len]
    }

    fn bodies(&self) -> &[[[f64; 2]; 3]] {
        &self.bodies
    }

    /// Makes this interval's last snapshot the next interval's first.
    fn end_frame(&mut self) {
        if self.len > 1 {
            self.snapshots.swap(0, self.len - 1);
            self.first_bodies = self.bodies[self.len - 1];
        }
    }
}

/// Bare-paper XYZ and code values per pixel, computed once.
struct PaperCache {
    xyz: Vec<[f64; 3]>,
    code: Vec<[u16; 3]>,
}

impl PaperCache {
    fn new(sheet: &KozoSheet, optics: &Optics) -> Self {
        let pixels = sheet.width * sheet.height;
        let xyz: Vec<[f64; 3]> =
            (0..pixels).into_par_iter().map(|i| optics.paper_xyz(sheet.sample(i))).collect();
        let code = xyz.par_iter().map(|&xyz| optics.encode_srgb16(xyz).0).collect();
        Self { xyz, code }
    }
}

/// Shading counters of one frame (integers, summed in row order).
#[derive(Clone, Copy, Debug)]
struct ShadeStats {
    inked_nodes: u64,
    cinnabar_nodes: u64,
    gamut_mapped: u64,
    finite: bool,
}

/// Everything needed to shade a frame from the ink fields.
struct Shader<'a> {
    nodes: &'a NodeGrid,
    look: &'a Look,
    optics: &'a Optics,
    sheet: &'a KozoSheet,
    paper: &'a PaperCache,
}

impl Shader<'_> {
    /// Shades every pixel: the mean XYZ of its `q×q` ink nodes (fixed row-major order), encoded
    /// to 16-bit sRGB. A pixel whose nodes are all bare is, by definition, the cached bare-paper
    /// code value.
    fn shade(&self, fields: &InkFields, rgb: &mut [u16]) -> ShadeStats {
        let nodes = self.nodes;
        let q = nodes.supersample;
        let inverse_samples = 1.0 / (q * q) as f64;
        let rows: Vec<ShadeStats> = rgb
            .par_chunks_mut(nodes.width * 3)
            .enumerate()
            .map(|(row, out)| {
                let mut stats =
                    ShadeStats { inked_nodes: 0, cinnabar_nodes: 0, gamut_mapped: 0, finite: true };
                for col in 0..nodes.width {
                    let pixel = row * nodes.width + col;
                    let paper = self.sheet.sample(pixel);
                    let mut xyz = [0.0f64; 3];
                    let mut bare = true;
                    for i in 0..q {
                        let base =
                            (nodes.margin + q * row + i) * nodes.cols + nodes.margin + q * col;
                        for n in base..base + q {
                            let loads = self.look.loads(
                                fields.presence[n],
                                [
                                    fields.freshness[0][n],
                                    fields.freshness[1][n],
                                    fields.freshness[2][n],
                                ],
                                fields.ember[n],
                            );
                            let sample = if loads.is_bare() {
                                self.paper.xyz[pixel]
                            } else {
                                bare = false;
                                stats.inked_nodes += 1;
                                stats.cinnabar_nodes += u64::from(loads.cinnabar > 0.0);
                                self.optics.reflect_xyz(loads, paper)
                            };
                            for (acc, value) in xyz.iter_mut().zip(sample) {
                                *acc += value;
                            }
                        }
                    }
                    let code = if bare {
                        self.paper.code[pixel]
                    } else {
                        let mean = xyz.map(|value| value * inverse_samples);
                        stats.finite &= mean.iter().all(|value| value.is_finite());
                        let (code, mapped) = self.optics.encode_srgb16(mean);
                        stats.gamut_mapped += u64::from(mapped);
                        code
                    };
                    out[col * 3..col * 3 + 3].copy_from_slice(&code);
                }
                stats
            })
            .collect();
        rows.into_iter().fold(
            ShadeStats { inked_nodes: 0, cinnabar_nodes: 0, gamut_mapped: 0, finite: true },
            |acc, row| ShadeStats {
                inked_nodes: acc.inked_nodes + row.inked_nodes,
                cinnabar_nodes: acc.cinnabar_nodes + row.cinnabar_nodes,
                gamut_mapped: acc.gamut_mapped + row.gamut_mapped,
                finite: acc.finite && row.finite,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ember::orbit::BodyState;

    /// Fluid time between consecutive recorded knots of the synthetic motions.
    const KNOT_DT: f64 = 1e-3;

    /// Fluid time of synthetic knot `k`.
    fn knot_time(k: usize) -> f64 {
        k as f64 * KNOT_DT
    }

    /// Three bodies moving from the origin at constant velocities, except that body 0 turns
    /// back at `turn` and retraces its line: its path keeps growing at its full speed while its
    /// displacement shrinks.
    struct Shuttle {
        velocity: [[f64; 2]; 3],
        turn: f64,
    }

    impl BodyMotion for Shuttle {
        fn bodies_at(&self, t: f64) -> [BodyState; 3] {
            std::array::from_fn(|b| {
                let (s, sign) =
                    if b == 0 && t > self.turn { (2.0 * self.turn - t, -1.0) } else { (t, 1.0) };
                let [vx, vy] = self.velocity[b];
                BodyState { position: [vx * s, vy * s], velocity: [sign * vx, sign * vy] }
            })
        }

        fn speed_bound(&self, _t: f64) -> f64 {
            self.velocity.iter().map(|[vx, vy]| (vx * vx + vy * vy).sqrt()).fold(0.0, f64::max)
        }
    }

    /// Body speeds `[21.3, 1, 1]` (world units per fluid time), body 0 turning back at knot 70.
    fn fast_shuttle() -> Shuttle {
        // |(12.78, 17.04)| = 21.3: one body far faster than the median body.
        Shuttle { velocity: [[12.78, 17.04], [0.6, -0.8], [-1.0, 0.0]], turn: knot_time(70) }
    }

    #[test]
    fn snapshot_travel_bound_binds_for_a_fast_body_along_its_path() {
        let fluid = EmberConfig::default().fluid;
        // Knots 0..=137: Δt = 0.137, so the time bound is ⌈0.137/0.0025⌉ = ⌈54.8⌉ = 55. Body 0
        // travels 21.3·0.137 = 2.9181 along its path, so the travel bound (0.5 radii of 0.05) is
        // ⌈2.9181/0.025⌉ = ⌈116.72⌉ = 117. Its displacement, 21.3·(0.14 - 0.137) = 0.0639, would
        // need only ⌈2.56⌉ = 3.
        assert_eq!((fluid.max_snapshot_interval, fluid.max_snapshot_travel), (2.5e-3, 0.5));
        assert_eq!(fluid.body_radius, 0.05);
        let intervals = snapshot_intervals(&fast_shuttle(), knot_time, 0, 137, &fluid);
        assert_eq!(intervals, 117);
        // A later frame counts only its own knots: 21.3·0.05/0.025 = 42.6 → 43 (time bound ≈ 20).
        assert_eq!(snapshot_intervals(&fast_shuttle(), knot_time, 40, 90, &fluid), 43);
    }

    #[test]
    fn snapshot_time_bound_binds_for_slow_bodies() {
        let fluid = EmberConfig::default().fluid;
        let slow = Shuttle { velocity: [[0.3, 0.4], [0.4, 0.0], [0.0, -0.3]], turn: 1.0 };
        // Δt = 0.137: ⌈54.8⌉ = 55 by time; the fastest path, 0.5·0.137 = 0.0685, needs ⌈2.74⌉ = 3.
        assert_eq!(snapshot_intervals(&slow, knot_time, 0, 137, &fluid), 55);
        // A single short knot interval still gets one snapshot interval.
        assert_eq!(snapshot_intervals(&slow, knot_time, 5, 6, &fluid), 1);
    }

    #[test]
    fn snapshot_intervals_of_a_repeated_knot_are_zero() {
        let fluid = EmberConfig::default().fluid;
        assert_eq!(snapshot_intervals(&fast_shuttle(), knot_time, 0, 0, &fluid), 0);
        assert_eq!(snapshot_intervals(&fast_shuttle(), knot_time, 90, 90, &fluid), 0);
    }

    #[test]
    fn schedule_must_end_on_the_final_knot() {
        assert!(validate_schedule(&[1, 2, 9], 10).is_ok());
        assert!(validate_schedule(&[1, 2, 8], 10).is_err());
        assert!(validate_schedule(&[], 10).is_err());
        assert!(validate_schedule(&[3, 3, 9], 10).is_err());
        assert!(validate_schedule(&[5, 2, 9], 10).is_err());
        assert!(validate_schedule(&[0], 1).is_err());
    }

    #[test]
    fn the_orbit_must_outlast_pre_roll_and_valve() {
        let config = EmberConfig::default();
        let contact = &config.contact;
        let required = contact.pre_roll + contact.valve_lead;
        assert_eq!(valve_time(10.0, &config).expect("long orbit"), 10.0 - contact.valve_lead);
        for duration in [required, required - 1e-9, 0.0, f64::NAN] {
            match valve_time(duration, &config) {
                Err(EmberError::OrbitTooShort { required: needed, .. }) => {
                    assert_eq!(needed, required);
                }
                other => panic!("duration {duration}: expected OrbitTooShort, got {other:?}"),
            }
        }
        assert!(valve_time(required + 1e-9, &config).expect("just enough") > contact.pre_roll);
    }

    /// The raw recorded orbit of the figure-eight choreography (Chenciner–Montgomery, unit
    /// masses, velocities rescaled from `G = 1` to the simulator's `G`), `steps` knots.
    fn figure_eight_positions(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        use crate::sim::{Body, G, get_positions};
        let speed = G.sqrt();
        let (x, y) = (0.970_004_36, -0.243_087_53);
        let (vx, vy) = (-0.932_407_37 * speed, -0.864_731_46 * speed);
        let bodies = vec![
            Body::new(1.0, Vector3::new(x, y, 0.0), Vector3::new(-vx / 2.0, -vy / 2.0, 0.0)),
            Body::new(1.0, Vector3::new(-x, -y, 0.0), Vector3::new(-vx / 2.0, -vy / 2.0, 0.0)),
            Body::new(1.0, Vector3::zeros(), Vector3::new(vx, vy, 0.0)),
        ];
        get_positions(bodies, steps).positions
    }

    /// The paper's fibre-count guard fails a request in planning, not after the fluid ran: a
    /// density that passes `EmberConfig::validate` (a range check) but would put more than
    /// 16,777,216 fibres on this sheet is rejected by `plan_ember`, naming the density.
    #[test]
    fn a_fibre_count_the_paper_rejects_fails_in_planning() {
        let steps = 20_000;
        let positions = figure_eight_positions(steps);
        let frame_steps = crate::render::main_video_checkpoints(steps);
        let request = |config| EmberRequest {
            positions: &positions,
            frame_steps: &frame_steps,
            width: 96,
            height: 64,
            paper_seed: b"",
            config,
            mode: EmberMode::StillOnly,
        };
        let production = EmberConfig::default();
        plan_ember(&request(&production)).expect("the production paper plans");

        let mut dense = production.clone();
        dense.paper.fibres.density_per_cm2 = 1e4;
        dense.validate().expect("the density is in range");
        match plan_ember(&request(&dense)) {
            Err(EmberError::InvalidConfig { parameter, reason }) => {
                assert_eq!(parameter, "paper.fibres.density_per_cm2");
                assert!(reason.contains("fibres (at most 16777216)"), "{reason}");
            }
            other => panic!("expected the fibre-count rejection, got {other:?}"),
        }
    }

    #[test]
    fn size_limits() {
        assert!(validate_size(1, 1).is_ok());
        assert!(validate_size(0, 10).is_err());
        assert!(validate_size(10, MAX_SIDE + 1).is_err());
    }

    #[test]
    fn little_endian_encoding_is_explicit() {
        let samples: Vec<u16> = (0..70_000u32).map(|i| (i * 7919) as u16).collect();
        let mut bytes = vec![0u8; samples.len() * 2];
        encode_le(&samples, &mut bytes);
        for (i, word) in samples.iter().enumerate() {
            assert_eq!([bytes[2 * i], bytes[2 * i + 1]], word.to_le_bytes());
        }
    }
}
