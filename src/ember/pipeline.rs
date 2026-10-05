//! Orchestration of the ember edition: fluid, ink, shading and the frame stream.
//!
//! One frame is rendered per entry of the frame schedule (the same orbit checkpoints as the main
//! video). Between two frames the fluid is advanced through a few velocity snapshots, the ink
//! fields are remapped along exact backward characteristics through those snapshots, and every
//! pixel is shaded from its `q×q` ink nodes. The last frame is the still.
//!
//! # The slow films
//!
//! [`EmberMode::VideoAndSlow`] also renders the slow films: the same film `factor` times slower
//! for each of [`EmberRequest::slow_factors`], with `factor - 1` real in-between frames inside
//! every interval between two scheduled frames. The snapshots of an interval lie on a uniform
//! lattice whose size is a multiple of every factor in every mode (of their least common
//! multiple), so each in-between time is a snapshot time. An in-between frame is a *side*
//! remap: the previous scheduled frame's fields remapped through the window's prefix that ends
//! at its snapshot, shaded, and discarded. A moment that two films show (the middle of an
//! interval is one for factors 2 and 4) is rendered once and handed to both. The scheduled
//! chain never sees a side remap, so the normal film and the still do not depend on whether the
//! slow films are rendered, and every frame of every film is one remap away from a scheduled
//! frame's fields through the same fine steps (docs/ember-design.md §8.3).

use std::fmt::Write as _;
use std::time::Instant;

use nalgebra::Vector3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;

use super::config::{EmberConfig, FluidConfig};
use super::error::{EmberError, EmberResult};
use super::fluid::{FluidGrid, FluidStats, Snapshot, WakeSolver};
use super::ink::{InkDecay, InkFields, NodeGrid, remap, remap_visible};
use super::look::Look;
use super::math;
use super::optics::Optics;
use super::orbit::{BodyMotion, BodyState, BodyTrack, TrackSurvey};
use super::paper::{KozoSheet, check_fibre_count};
use super::trace::{ContactRules, FlowWindow};
use super::view::View;
use crate::sim::Sha3RandomByteStream;

/// Largest supported output side, matching the main renderer's limit.
const MAX_SIDE: u32 = 16_384;

/// Largest supported snapshot lattice: the least common multiple of
/// [`EmberRequest::slow_factors`], and so each factor, must not exceed it.
const MAX_SLOW_LATTICE: usize = 240;

/// Scheduled frames of bare paper that a slow film shows before the frame interval in which the
/// ink first appears: at 60 frames per second, one second for the factor 10, 0.4 s for 4.
const SLOW_FILM_LEAD_FRAMES: usize = 6;

/// Most threads the fluid solver uses.
///
/// Its transforms work on the rows and columns of a grid of 2160×1536 nodes: too little work per
/// task to feed very many threads. On a 64-core Threadripper PRO 9985WX one step of that grid
/// took 55.8 ms with 16 threads, 48.9 ms with 32, 47.9 ms with 48, 50.2 ms with 64 and 63.6 ms
/// with 96 (measured on the loaded production host; on the earlier 1440×1024 grid, idle: 23.6 ms
/// with 16 or 32 threads, 26.4 ms with 64, 32.2 ms with 128). The solver therefore runs in its
/// own pool of at most this many threads, while the ink remap and the shading, which keep
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

/// What [`render_ember`] renders. The fluid and the scheduled ink remaps are the same in every
/// mode, so the still and the scheduled frames do not depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmberMode {
    /// Only the final frame is shaded (the fluid and ink still run through the whole schedule,
    /// so the still is identical to the films' last frame). The sink is not called.
    StillOnly,
    /// The normal film: every scheduled frame is shaded and handed to the sink; the last one is
    /// the still.
    Video,
    /// The normal film and the slow films: the sink also receives the in-between frames (see the
    /// module documentation).
    VideoAndSlow,
}

/// Inputs of one ember render.
#[derive(Clone, Copy, Debug)]
pub struct EmberRequest<'a> {
    /// Raw recorded orbit, `positions[body][knot]`: exactly three bodies, at least two knots.
    pub positions: &'a [Vec<Vector3<f64>>],
    /// Time step of the recording (the phase-space projections difference positions over it).
    pub dt: f64,
    /// The bodies' masses (finite and positive), in the order of `positions`: they set the tidal
    /// field that stretches the bodies.
    pub masses: [f64; 3],
    /// The main edition's view of the orbit, which the bodies follow.
    pub view: &'a View,
    /// Recorded knots to render as frames: strictly increasing, the last one must be the final
    /// knot (the still). Use the main video's checkpoints so both videos stay in step.
    pub frame_steps: &'a [usize],
    /// How many times slower each slow film is, strictly increasing (at least 1 each; their
    /// least common multiple at most 240). They also fix the snapshot lattice, in every mode, so
    /// they are an input of the still and the normal film too.
    pub slow_factors: &'a [u32],
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

/// One rendered frame, handed to the sink in time order. A scheduled frame belongs to the normal
/// film and to every slow film that has started; an in-between frame belongs to the slow films
/// whose moments it shows.
#[derive(Clone, Copy, Debug)]
pub struct EmberFrame<'a> {
    /// Position in the normal film (0-based); `None` for an in-between frame.
    pub index: Option<usize>,
    /// Frames of the normal film (the schedule's length).
    pub count: usize,
    /// Position in each slow film (0-based), in the order of [`EmberRequest::slow_factors`];
    /// `None` for a film this frame is not part of. Empty unless the mode is
    /// [`EmberMode::VideoAndSlow`].
    pub slow_indices: &'a [Option<usize>],
    /// Recorded orbit knot shown by a scheduled frame.
    pub orbit_step: Option<usize>,
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

/// Deterministic statistics of a render (part of the determinism contract; recorded as the
/// certificate's `stats`).
///
/// Node fractions count the visible ink nodes, `width·height·q²` (`q` =
/// [`RasterConfig::supersample`](super::config::RasterConfig::supersample)), the margin
/// excluded. Every count is an integer summed exactly, and every fraction is one such count
/// divided by the node count, so the statistics are identical on every architecture and for
/// every thread count.
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
    /// Pixels of the still that needed gamut mapping.
    pub still_gamut_mapped_pixels: u64,
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
    /// Frames of the normal film handed to the sink (`0` in [`EmberMode::StillOnly`]).
    pub frames_emitted: usize,
    /// SHA-256 (hex) of the normal film's concatenated `rgb48le` frames, unless
    /// [`EmberMode::StillOnly`].
    pub frames_sha256: Option<String>,
    /// The slow films, in the order of [`EmberRequest::slow_factors`] (planned in every mode,
    /// rendered in [`EmberMode::VideoAndSlow`]).
    pub slow_films: Vec<SlowFilmSummary>,
    /// Orbit duration in fluid time units.
    pub duration: f64,
    /// Fluid time at which the bodies stopped inking.
    pub valve_time: f64,
    /// Time for which fresh ink stays black, in fluid units (`look.hold_fraction` of the orbit).
    pub hold_time: f64,
    /// E-folding time of the fade, in fluid units (`look.fade_fraction` of the orbit).
    pub fade_time: f64,
    /// The orbit's reference tidal anisotropy (see the tidal model in `orbit`).
    pub tidal_reference: f64,
    /// Fluid grid `[nx, ny]`.
    pub fluid_grid: [usize; 2],
    /// Fluid grid spacing.
    pub fluid_dx: f64,
    /// Ink node grid `[cols, rows]` including the margin.
    pub ink_grid: [usize; 2],
    /// Deterministic statistics.
    pub stats: EmberStats,
    /// Wall-clock timings.
    pub timings: EmberTimings,
}

/// What a render did for one slow film.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlowFilmSummary {
    /// How many times slower than the normal film it is.
    pub factor: u32,
    /// The scheduled frame at which it starts.
    pub first_frame: usize,
    /// Its frames handed to the sink (`0` unless [`EmberMode::VideoAndSlow`]).
    pub frames_emitted: usize,
    /// SHA-256 (hex) of its concatenated `rgb48le` frames, in [`EmberMode::VideoAndSlow`].
    pub frames_sha256: Option<String>,
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
/// Planning costs one pass of the view over the orbit and the track's tables (a fraction of a
/// second for a million recorded steps) and allocates nothing proportional to the output, so callers can run it long before
/// rendering to reject an orbit the edition cannot draw. [`render_ember`] plans the request itself; see
/// [`plan_ember`] for what a successful plan does and does not guarantee.
#[derive(Debug)]
pub struct EmberPlan {
    track: BodyTrack,
    grid: FluidGrid,
    nodes: NodeGrid,
    rules: ContactRules,
    aspect: f64,
    frames: usize,
    lattice: usize,
    slow: Vec<SlowFilm>,
    survey: TrackSurvey,
}

/// A slow film of a request: the normal film `factor` times slower, starting a little before
/// the ink first appears.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlowFilm {
    /// How many times slower than the normal film it is: `factor - 1` in-between frames per
    /// scheduled interval.
    pub factor: u32,
    /// The scheduled frame that is its first frame.
    pub first_frame: usize,
    /// Its frames: `(scheduled frames after the first)·factor + 1`.
    pub frames: usize,
}

impl SlowFilm {
    /// The slow film of a schedule whose frames are at the fluid times `frame_time(i)`
    /// (`count ≥ 1` of them, the last one after `t_on`): it starts [`SLOW_FILM_LEAD_FRAMES`]
    /// scheduled frames before the interval in which the inking starts at `t_on`. `None` if its
    /// frames would overflow a count.
    fn plan(
        count: usize,
        frame_time: impl Fn(usize) -> f64,
        t_on: f64,
        factor: u32,
    ) -> Option<Self> {
        let inked = (0..count).find(|&index| frame_time(index) >= t_on).unwrap_or(count - 1);
        let first_frame = inked.saturating_sub(SLOW_FILM_LEAD_FRAMES + 1);
        let frames = (count - 1 - first_frame).checked_mul(factor as usize)?.checked_add(1)?;
        Some(Self { factor, first_frame, frames })
    }
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

    /// The slow films ([`EmberMode::VideoAndSlow`]), in the order of
    /// [`EmberRequest::slow_factors`].
    pub fn slow_films(&self) -> &[SlowFilm] {
        &self.slow
    }

    /// Snapshot intervals per scheduled interval are a multiple of this: the least common
    /// multiple of the slow factors (1 without any).
    pub fn snapshot_lattice(&self) -> usize {
        self.lattice
    }

    /// Fluid steps the orbit needs at least: the bodies' own speeds through the solver's step
    /// rule. The stirred water is faster in places, so a render takes somewhat more.
    pub fn estimated_fluid_steps(&self) -> f64 {
        self.survey.fluid_steps
    }

    /// Fastest body speed on the canvas, in units of the reference (median) speed.
    pub fn peak_speed(&self) -> f64 {
        self.survey.peak_speed
    }

    /// Smallest distance of a body's centre from the canvas edge over the orbit, in canvas units
    /// (the canvas is 2 high).
    pub fn edge_clearance(&self) -> f64 {
        self.survey.edge_clearance
    }

    /// Fraction of the orbit during which two bodies' discs overlap on the canvas.
    pub fn overlap_fraction(&self) -> f64 {
        self.survey.overlap_fraction
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
/// frame schedule (non-empty, strictly increasing, ending on the final knot) and the slow
/// factors (see [`EmberRequest::slow_factors`]), the view and the canvas track it gives the
/// orbit, the orbit's duration against the pre-roll and the valve (`T - valve_lead >
/// pre_roll`), and the fluid and ink grids.
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
    let lattice = slow_lattice(request.slow_factors)
        .map_err(|reason| EmberError::InvalidSchedule { reason })?;

    let aspect = f64::from(request.width) / f64::from(request.height);
    let canvas = request.view.canvas_track(request.positions, request.dt, aspect)?;
    let track = BodyTrack::new(canvas, request.masses, config)?;
    let t_valve = valve_time(track.duration(), config)?;
    let grid = FluidGrid::for_canvas(aspect, config.fluid.rows, config.fluid.box_margin)?;
    let nodes = NodeGrid::new(
        request.width,
        request.height,
        config.raster.supersample,
        config.raster.ink_margin,
    )?;
    let rules = ContactRules {
        soak_depth: config.contact.soak_depth,
        vorticity_gate: config.contact.vorticity_gate,
        t_on: config.contact.pre_roll,
        t_valve,
    };
    let frames = request.frame_steps.len();
    let frame_time = |index: usize| track.knot_time(request.frame_steps[index]);
    let slow = request
        .slow_factors
        .iter()
        .map(|&factor| {
            SlowFilm::plan(frames, frame_time, rules.t_on, factor).ok_or_else(|| {
                EmberError::InvalidSchedule {
                    reason: format!("the {factor}x film of {frames} frames is too long to count"),
                }
            })
        })
        .collect::<EmberResult<_>>()?;
    let survey = track.survey(aspect, grid.dx, config);
    Ok(EmberPlan { track, grid, nodes, rules, aspect, frames, lattice, slow, survey })
}

/// The snapshot lattice of the slow `factors` (their least common multiple, 1 without any).
/// The certificate reader applies the same rule to a recorded list.
///
/// # Errors
///
/// Why the factors are invalid, unless they are strictly increasing, each at least 1, and their
/// least common multiple is at most [`MAX_SLOW_LATTICE`].
pub(crate) fn slow_lattice(factors: &[u32]) -> Result<usize, String> {
    if !factors.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(format!("the slow factors {factors:?} must be strictly increasing"));
    }
    let too_fine =
        || format!("the slow factors {factors:?} need a snapshot lattice above {MAX_SLOW_LATTICE}");
    let mut lattice = 1_usize;
    for &factor in factors {
        if factor == 0 {
            return Err(format!("the slow factors {factors:?} must be at least 1"));
        }
        // A factor divides the lattice, so one above the limit fails before the product, which
        // then stays below the limit squared.
        let factor = usize::try_from(factor).map_err(|_| too_fine())?;
        if factor > MAX_SLOW_LATTICE {
            return Err(too_fine());
        }
        lattice = lattice / greatest_common_divisor(lattice, factor) * factor;
        if lattice > MAX_SLOW_LATTICE {
            return Err(too_fine());
        }
    }
    Ok(lattice)
}

/// Greatest common divisor (Euclid).
fn greatest_common_divisor(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
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
/// Unless the mode is [`EmberMode::StillOnly`], every scheduled frame is passed to `sink` in
/// order, and in [`EmberMode::VideoAndSlow`] the slow films' in-between frames are passed
/// between them, in time order; the returned summary's `still` equals the last frame. The result is
/// bit-identical on every CPU architecture and for every rayon thread count.
pub fn render_ember(
    request: &EmberRequest<'_>,
    sink: &mut dyn FnMut(&EmberFrame<'_>) -> EmberResult<()>,
) -> EmberResult<EmberSummary> {
    let started = Instant::now();
    let config = request.config;
    let EmberPlan { track, grid, nodes, rules, aspect, lattice, slow, .. } = plan_ember(request)?;
    let duration = track.duration();
    let (t_on, t_valve) = (rules.t_on, rules.t_valve);
    let films = request.mode != EmberMode::StillOnly;
    // The slow films this render emits (none unless the mode asks for them).
    let emitted_slow: &[SlowFilm] =
        if request.mode == EmberMode::VideoAndSlow { &slow } else { &[] };
    let fluid_pool = fluid_pool();
    let mut solver = in_pool(fluid_pool.as_ref(), || WakeSolver::new(grid, &config.fluid, aspect))?;
    info!(
        "Ember edition: orbit lasts {duration:.3} fluid units; fluid {}x{} (dx {:.5}); ink nodes \
         {}x{}; {} frames{}",
        grid.nx,
        grid.ny,
        grid.dx,
        nodes.cols,
        nodes.rows,
        request.frame_steps.len(),
        emitted_slow.iter().fold(String::new(), |mut text, film| {
            let _ = write!(text, ", {} in the {}x film", film.frames, film.factor);
            text
        })
    );

    let look = Look::new(&config.look, duration);
    let optics = Optics::new();
    let mut paper_rng = Sha3RandomByteStream::new(request.paper_seed, 0.0, 1.0, 1.0, 1.0);
    let sheet = KozoSheet::generate(request.width, request.height, &config.paper, &mut paper_rng)?;
    let paper = PaperCache::new(&sheet, &optics);
    let shader =
        Shader { nodes: &nodes, look: &look, optics: &optics, sheet: &sheet, paper: &paper };
    // The fields' decay over the fluid time `dt` that ends at `frame_time`.
    let decay_over = |dt: f64, frame_time: f64| InkDecay {
        frame_time,
        fade_tau: look.fade_tau(),
        fresh_fade: math::exp(-dt / look.fade_tau()),
        floor_fade: config.look.floor_tau.map_or(1.0, |tau| math::exp(-dt / tau)),
    };

    let mut fields = InkFields::zeros(&nodes);
    let mut next_fields = InkFields::zeros(&nodes);
    let mut window = SnapshotWindow::new(&grid);
    in_pool(fluid_pool.as_ref(), || solver.snapshot_into(window.first_mut()));
    window.first_bodies = track.bodies_at(0.0);

    let pixels = request.width as usize * request.height as usize;
    let mut rgb = vec![0u16; pixels * 3];
    let mut rgb48le = vec![0u8; pixels * 6];
    let mut stream_hasher = Sha256::new();
    let mut slow_hashers: Vec<Sha256> = emitted_slow.iter().map(|_| Sha256::new()).collect();
    let mut stats = EmberStats::default();
    let mut timings = EmberTimings::default();
    let mut frames_emitted = 0;
    let mut slow_emitted = vec![0_usize; emitted_slow.len()];
    // Each frame's position in every emitted slow film (see `EmberFrame::slow_indices`).
    let mut slow_indices: Vec<Option<usize>> = vec![None; emitted_slow.len()];
    // Visible ink nodes (the margin excluded): the denominator of the node fractions.
    let view_nodes = (nodes.width * nodes.height * nodes.supersample * nodes.supersample) as f64;

    let count = request.frame_steps.len();
    let mut previous_step = 0;
    let mut t_prev = 0.0;
    let progress_every = (count / 20).max(1);
    for (index, &step) in request.frame_steps.iter().enumerate() {
        let t_frame = track.knot_time(step);
        let intervals = lattice_intervals(
            snapshot_intervals(&track, |k| track.knot_time(k), previous_step, step, &config.fluid),
            lattice,
        );

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
                window.push_bodies(track.bodies_at(target));
            }
            Ok(())
        })?;
        stats.snapshots += intervals as u64;
        timings.fluid_seconds += clock.elapsed().as_secs_f64();

        // The slow films' in-between frames of this interval, in time order: side remaps of the
        // previous scheduled frame's fields into `next_fields`, which the scheduled remap
        // overwrites. They are shaded and dropped, so only the visible nodes are remapped. The
        // moment at snapshot `last` belongs to every started film whose grid it lies on, and is
        // rendered once for all of them.
        for last in 1..intervals {
            let mut shown_by_any = false;
            for (slot, (film, emitted)) in
                slow_indices.iter_mut().zip(emitted_slow.iter().zip(&slow_emitted))
            {
                let shows =
                    is_slow_moment(last, intervals, film.factor) && index > film.first_frame;
                *slot = shows.then_some(*emitted);
                shown_by_any |= shows;
            }
            if !shown_by_any {
                continue;
            }
            let time = window.snapshots()[last].time;
            let clock = Instant::now();
            let inked = time >= t_on;
            if inked {
                let flow = FlowWindow {
                    grid,
                    snapshots: &window.snapshots()[..=last],
                    bodies: &window.bodies()[..=last],
                };
                let decay = decay_over(time - t_prev, time);
                let side = remap_visible(&fields, &mut next_fields, &nodes, &flow, &rules, &decay);
                if side.non_finite_origins > 0 {
                    return Err(EmberError::NonFinite { stage: "ink trace", time });
                }
            }
            timings.ink_seconds += clock.elapsed().as_secs_f64();

            let clock = Instant::now();
            let shown = if inked { &next_fields } else { &fields };
            if !shader.shade(shown, &mut rgb).finite {
                return Err(EmberError::NonFinite { stage: "shading", time });
            }
            encode_le(&rgb, &mut rgb48le);
            timings.shade_seconds += clock.elapsed().as_secs_f64();

            let clock = Instant::now();
            count_into_slow_films(&slow_indices, &mut slow_hashers, &mut slow_emitted, &rgb48le);
            sink(&EmberFrame {
                index: None,
                count,
                slow_indices: &slow_indices,
                orbit_step: None,
                time,
                width: request.width,
                height: request.height,
                rgb: &rgb,
                rgb48le: &rgb48le,
            })?;
            timings.sink_seconds += clock.elapsed().as_secs_f64();
        }

        let clock = Instant::now();
        if intervals > 0 && t_frame >= t_on {
            let flow = FlowWindow { grid, snapshots: window.snapshots(), bodies: window.bodies() };
            let decay = decay_over(t_frame - t_prev, t_frame);
            let remap_stats = remap(&fields, &mut next_fields, &nodes, &flow, &rules, &decay);
            if remap_stats.non_finite_origins > 0 {
                return Err(EmberError::NonFinite { stage: "ink trace", time: t_frame });
            }
            std::mem::swap(&mut fields, &mut next_fields);
            stats.contact_events += remap_stats.contacted_nodes;
        }
        timings.ink_seconds += clock.elapsed().as_secs_f64();

        let is_last = index + 1 == count;
        if films || is_last {
            let clock = Instant::now();
            let shade_stats = shader.shade(&fields, &mut rgb);
            if !shade_stats.finite {
                return Err(EmberError::NonFinite { stage: "shading", time: t_frame });
            }
            encode_le(&rgb, &mut rgb48le);
            timings.shade_seconds += clock.elapsed().as_secs_f64();
            if is_last {
                stats.still_ink_fraction = shade_stats.inked_nodes as f64 / view_nodes;
                stats.still_gamut_mapped_pixels = shade_stats.gamut_mapped;
            }
            if films {
                let clock = Instant::now();
                for (slot, (film, emitted)) in
                    slow_indices.iter_mut().zip(emitted_slow.iter().zip(&slow_emitted))
                {
                    *slot = (index >= film.first_frame).then_some(*emitted);
                }
                stream_hasher.update(&rgb48le);
                count_into_slow_films(
                    &slow_indices,
                    &mut slow_hashers,
                    &mut slow_emitted,
                    &rgb48le,
                );
                sink(&EmberFrame {
                    index: Some(index),
                    count,
                    slow_indices: &slow_indices,
                    orbit_step: Some(step),
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
    for (film, emitted) in emitted_slow.iter().zip(&slow_emitted) {
        debug_assert_eq!(*emitted, film.frames, "the {}x film has its planned length", film.factor);
    }

    let fluid: FluidStats = solver.stats();
    stats.fluid_steps = fluid.steps;
    stats.min_dt = fluid.min_dt;
    stats.max_dt = fluid.max_dt;
    stats.max_flow_speed = fluid.max_speed;
    timings.total_seconds = started.elapsed().as_secs_f64();

    // Every planned film, with what this mode emitted of it.
    let mut emitted = slow_hashers.into_iter().zip(slow_emitted);
    let slow_films = slow
        .iter()
        .map(|film| {
            let (frames_emitted, frames_sha256) =
                emitted.next().map_or((0, None), |(hasher, frames)| {
                    (frames, Some(hex::encode(hasher.finalize())))
                });
            SlowFilmSummary {
                factor: film.factor,
                first_frame: film.first_frame,
                frames_emitted,
                frames_sha256,
            }
        })
        .collect();

    Ok(EmberSummary {
        width: request.width,
        height: request.height,
        still_sha256: hex::encode(Sha256::digest(&rgb48le)),
        still: rgb,
        frames_emitted,
        frames_sha256: films.then(|| hex::encode(stream_hasher.finalize())),
        slow_films,
        duration,
        valve_time: t_valve,
        hold_time: look.hold(),
        fade_time: look.fade_tau(),
        tidal_reference: track.tidal_reference(),
        fluid_grid: [grid.nx, grid.ny],
        fluid_dx: grid.dx,
        ink_grid: [nodes.cols, nodes.rows],
        stats,
        timings,
    })
}

/// The snapshot intervals of a scheduled frame interval that needs `intervals` of them
/// ([`snapshot_intervals`]): the next multiple of the snapshot `lattice` (the slow factors' least
/// common multiple), so that every in-between frame of every slow film ends on a snapshot. It is
/// the same in every mode, so the fluid's steps, the scheduled frames and the still do not
/// depend on whether the slow films are rendered.
fn lattice_intervals(intervals: usize, lattice: usize) -> usize {
    intervals.div_ceil(lattice) * lattice
}

/// Whether the moment at snapshot `last` (`0 < last < intervals`) of a frame interval simulated
/// on `intervals` snapshot intervals (a multiple of `factor`, [`lattice_intervals`]) is one of
/// the in-between frames of the film `factor` times slower.
fn is_slow_moment(last: usize, intervals: usize, factor: u32) -> bool {
    last.is_multiple_of(intervals / factor as usize)
}

/// Hashes a frame into, and counts it for, every slow film it belongs to (`slow_indices`).
fn count_into_slow_films(
    slow_indices: &[Option<usize>],
    hashers: &mut [Sha256],
    emitted: &mut [usize],
    rgb48le: &[u8],
) {
    for ((index, hasher), frames) in slow_indices.iter().zip(hashers).zip(emitted) {
        if index.is_some() {
            hasher.update(rgb48le);
            *frames += 1;
        }
    }
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
/// with `Δt = knot_time(to) - knot_time(from)` and `path_b` the farthest any material of body
/// `b` can travel: the sum over `k ∈ (from, to]` of its centre's step
/// `|pos_b(t_k) - pos_b(t_{k-1})|` along the recorded polyline (not the displacement between the
/// frames, which misses a body that turns back) plus its outline's deformation speed at `t_{k-1}`
/// times `t_k - t_{k-1}` (a tidally turning ellipse sweeps water even where its centre rests). No
/// interval is then longer than `max_snapshot_interval` or lets a body move more than
/// `max_snapshot_travel` radii. `from == to` returns 0: the frame shows the previous frame's
/// time and nothing advances.
///
/// `knot_time(k)` is the fluid time `t_k` of recorded knot `k` (`BodyTrack::knot_time` in the
/// pipeline; strictly increasing), and `motion` gives the bodies at those times.
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
    let mut previous = motion.bodies_at(knot_time(from));
    for k in from + 1..=to {
        let current = motion.bodies_at(knot_time(k));
        let step = knot_time(k) - knot_time(k - 1);
        for (length, (now, before)) in path.iter_mut().zip(current.iter().zip(&previous)) {
            let (dx, dy) =
                (now.position[0] - before.position[0], now.position[1] - before.position[1]);
            *length += (dx * dx + dy * dy).sqrt() + before.shape.deformation_speed() * step;
        }
        previous = current;
    }
    let longest = path.iter().copied().fold(0.0, f64::max);
    let by_travel = (longest / (fluid.max_snapshot_travel * fluid.body_radius)).ceil();
    (by_time.max(by_travel).max(1.0)) as usize
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
    /// The bodies at each snapshot in use.
    bodies: Vec<[BodyState; 3]>,
    /// Snapshots in use.
    len: usize,
    /// The bodies at `snapshots[0]` (set before the first frame).
    first_bodies: [BodyState; 3],
    grid: FluidGrid,
}

impl SnapshotWindow {
    fn new(grid: &FluidGrid) -> Self {
        Self {
            snapshots: vec![Snapshot::zeros(grid)],
            bodies: Vec::new(),
            len: 1,
            first_bodies: [BodyState::default(); 3],
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

    fn push_bodies(&mut self, bodies: [BodyState; 3]) {
        self.bodies.push(bodies);
    }

    fn snapshots(&self) -> &[Snapshot] {
        &self.snapshots[..self.len]
    }

    fn bodies(&self) -> &[[BodyState; 3]] {
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
                let mut stats = ShadeStats { inked_nodes: 0, gamut_mapped: 0, finite: true };
                for col in 0..nodes.width {
                    let pixel = row * nodes.width + col;
                    let paper = self.sheet.sample(pixel);
                    let mut xyz = [0.0f64; 3];
                    let mut bare = true;
                    for i in 0..q {
                        let base =
                            (nodes.margin + q * row + i) * nodes.cols + nodes.margin + q * col;
                        for n in base..base + q {
                            let carbon = self.look.carbon(
                                fields.presence[n],
                                [
                                    fields.freshness[0][n],
                                    fields.freshness[1][n],
                                    fields.freshness[2][n],
                                ],
                            );
                            let sample = if carbon == 0.0 {
                                self.paper.xyz[pixel]
                            } else {
                                bare = false;
                                stats.inked_nodes += 1;
                                self.optics.reflect_xyz(carbon, paper)
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
            ShadeStats { inked_nodes: 0, gamut_mapped: 0, finite: true },
            |acc, row| ShadeStats {
                inked_nodes: acc.inked_nodes + row.inked_nodes,
                gamut_mapped: acc.gamut_mapped + row.gamut_mapped,
                finite: acc.finite && row.finite,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ember::orbit::{BodyState, Shape};

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
                BodyState {
                    position: [vx * s, vy * s],
                    velocity: [sign * vx, sign * vy],
                    shape: Shape::disc(0.05),
                }
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
        // travels 21.3·0.137 = 2.9181 along its path, so the travel bound (0.28 radii of 0.05) is
        // ⌈2.9181/0.014⌉ = ⌈208.4⌉ = 209. Its displacement, 21.3·(0.14 - 0.137) = 0.0639, would
        // need only ⌈4.56⌉ = 5.
        assert_eq!((fluid.max_snapshot_interval, fluid.max_snapshot_travel), (2.5e-3, 0.28));
        assert_eq!(fluid.body_radius, 0.05);
        let intervals = snapshot_intervals(&fast_shuttle(), knot_time, 0, 137, &fluid);
        assert_eq!(intervals, 209);
        // A later frame counts only its own knots: 21.3·0.05/0.014 = 76.1 → 77 (time bound ≈ 20).
        assert_eq!(snapshot_intervals(&fast_shuttle(), knot_time, 40, 90, &fluid), 77);
    }

    /// A body that turns in place still sweeps water: its outline's deformation speed counts.
    #[test]
    fn snapshot_travel_bound_counts_a_turning_outline() {
        struct Spinner(Shape);
        impl BodyMotion for Spinner {
            fn bodies_at(&self, _t: f64) -> [BodyState; 3] {
                [BodyState { position: [0.0; 2], velocity: [0.0; 2], shape: self.0 }; 3]
            }

            fn speed_bound(&self, _t: f64) -> f64 {
                self.0.deformation_speed()
            }
        }
        let fluid = EmberConfig::default().fluid;
        // Aspect 2.25 at the disc's area, turning at 200 rad per fluid unit:
        // k = 200·(a² - b²)/(a² + b²) ≈ 134.0, deformation speed k·a ≈ 10.05. Over Δt = 0.137
        // that is ≈ 1.377 of travel, ⌈98.4⌉ = 99 intervals, while the centres never move.
        let (a, b) = (0.075, 0.05 * 0.05 / 0.075);
        let shape = Shape { semi: [a, b], axis: [1.0, 0.0], spin: 200.0, strain: 0.0 };
        assert!((shape.deformation_speed() - 10.0515).abs() < 1e-4);
        assert_eq!(snapshot_intervals(&Spinner(shape), knot_time, 0, 137, &fluid), 99);
        assert_eq!(snapshot_intervals(&Spinner(Shape::disc(0.05)), knot_time, 0, 137, &fluid), 55);
    }

    #[test]
    fn snapshot_time_bound_binds_for_slow_bodies() {
        let fluid = EmberConfig::default().fluid;
        let slow = Shuttle { velocity: [[0.3, 0.4], [0.4, 0.0], [0.0, -0.3]], turn: 1.0 };
        // Δt = 0.137: ⌈54.8⌉ = 55 by time; the fastest path, 0.5·0.137 = 0.0685, needs ⌈4.89⌉ = 5.
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

    /// Every scheduled interval gets a multiple of the slow factors' least common multiple of
    /// snapshot intervals, so every slow film's in-between times are snapshot times.
    #[test]
    fn the_snapshot_lattice_is_a_multiple_of_every_slow_factor() {
        for (intervals, lattice, rounded) in
            [(0, 20, 0), (1, 20, 20), (20, 20, 20), (21, 20, 40), (55, 10, 60), (7, 1, 7)]
        {
            assert_eq!(lattice_intervals(intervals, lattice), rounded, "{intervals} by {lattice}");
        }
        // The lattice is the factors' least common multiple (1 without slow films).
        for (factors, lattice) in [
            (&[][..], 1),
            (&[10][..], 10),
            (&[4, 10][..], 20),
            (&[2, 4][..], 4),
            (&[1, 3][..], 3),
            (&[16, 240][..], 240),
        ] {
            assert_eq!(slow_lattice(factors).expect("valid factors"), lattice, "{factors:?}");
        }
        // Rejected: unordered or repeated factors, a zero, a lattice above the limit (also one
        // whose product would overflow a 32-bit count).
        for factors in
            [&[10, 4][..], &[4, 4], &[0, 4], &[241], &[7, 11, 13], &[16, 268_435_457], &[u32::MAX]]
        {
            let reason = slow_lattice(factors).expect_err("invalid factors");
            assert!(reason.contains("slow factors"), "{reason}");
        }
    }

    /// The moments of the product's 4x and 10x films interleave: per frame interval of 20 (or
    /// 40) snapshot intervals, 3 of the 4x film and 9 of the 10x film, the middle one shared, so
    /// 11 moments are rendered.
    #[test]
    fn the_slow_films_share_their_common_moments() {
        for intervals in [20, 40] {
            let moments = |factor| {
                (1..intervals).filter(|&last| is_slow_moment(last, intervals, factor)).collect()
            };
            let (medium, slow): (Vec<usize>, Vec<usize>) = (moments(4), moments(10));
            let step = intervals / 20;
            assert_eq!(medium, [5, 10, 15].map(|moment| moment * step));
            assert_eq!(slow, (1..10).map(|moment| 2 * moment * step).collect::<Vec<_>>());
            let shared: Vec<_> = medium.iter().filter(|moment| slow.contains(moment)).collect();
            assert_eq!(shared, [&(intervals / 2)]);
            assert_eq!(medium.len() + slow.len() - shared.len(), 11);
        }
        // A film of factor 1 has no in-between frames.
        assert!((1..20).all(|last| !is_slow_moment(last, 20, 1)));
    }

    /// The slow film starts [`SLOW_FILM_LEAD_FRAMES`] scheduled frames before the interval in
    /// which the bodies start inking, and has `factor` frames per scheduled interval from there.
    #[test]
    fn a_slow_film_starts_a_little_before_the_ink() {
        let time = |index: usize| index as f64;
        let plan = |count, t_on, factor| {
            let film =
                SlowFilm::plan(count, time, t_on, factor).expect("a film of countable frames");
            (film.first_frame, film.frames)
        };
        // Inking starts inside the interval (20, 21]: that interval starts at frame 20, and the
        // film 6 frames earlier.
        assert_eq!(SLOW_FILM_LEAD_FRAMES, 6);
        assert_eq!(plan(100, 20.5, 10), (14, 85 * 10 + 1));
        // Exactly on a frame, the interval is the one that ends there.
        assert_eq!(plan(100, 20.0, 10), (13, 86 * 10 + 1));
        // Ink from the start: the whole film, slowed.
        assert_eq!(plan(100, 0.0, 10), (0, 99 * 10 + 1));
        assert_eq!(plan(100, 3.5, 10), (0, 99 * 10 + 1));
        // Factor 1 is the normal film from the first frame on.
        assert_eq!(plan(100, 20.5, 1), (14, 86));
        // A single frame is the still.
        assert_eq!(plan(1, 0.5, 10), (0, 1));
        // The start does not depend on the factor; the length does.
        assert_eq!(plan(100, 20.5, 4), (14, 85 * 4 + 1));
        // A length beyond any count is refused.
        assert_eq!(SlowFilm::plan(usize::MAX, time, 0.0, 2), None);
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
        let view = crate::ember::view::tests::frontal_view(&positions, 1.5);
        let request = |config| EmberRequest {
            positions: &positions,
            dt: crate::render::constants::DEFAULT_DT,
            masses: [1.0; 3],
            view: &view,
            slow_factors: &[4, 10],
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
