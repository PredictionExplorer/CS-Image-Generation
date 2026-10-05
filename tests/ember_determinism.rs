//! Cross-architecture golden tests of the ember edition.
//!
//! A small but complete ember render (the view, tidal bodies, fluid, ink, paper, shading, every
//! frame stream) of a fixed orbit is hashed and compared with digests recorded on another
//! machine. CI runs this file on `x86_64` Linux and `aarch64` macOS, so any
//! architecture-dependent arithmetic in the ember path fails here.
//!
//! Every test renders exactly once, in its own small rayon pool, and compares its own render
//! with the golden digests: the tests stay cheap when a runner such as `cargo nextest` gives each
//! one its own process, and parallel test processes do not oversubscribe the CPU. Agreement of
//! the renders (3-thread with the slow films, 2-thread still-only, 2-thread video without them,
//! 1-thread with them) with the same digests also proves that neither the mode nor the thread
//! count changes a bit: the still and the normal film do not depend on whether the slow films
//! are rendered.
//!
//! To re-bless after an intentional change of the rendering algorithm, run
//!
//! ```text
//! EMBER_BLESS=1 cargo test --release --test ember_determinism -- --include-ignored --nocapture
//! ```
//!
//! check that all renders print the same values (with `--include-ignored` that includes the
//! 40-thread render) and that the ignored `the_golden_render_stretches_its_bodies` passes (the
//! golden render must keep exercising the tidal shapes), copy the printed `GOLDEN_FRAMES_SHA256`,
//! `GOLDEN_SLOW_FRAMES_SHA256`, `GOLDEN_STILL_SHA256`, `GOLDEN_CONTACT_EVENTS` and
//! `GOLDEN_STILL_INK_NODES` into the constants below, bump
//! `ember::certificate::ALGORITHM_VERSION` if published editions change, and verify the new
//! values on both architectures before committing. A change that alters rendered bits usually
//! moves the unit goldens in `src/ember/*.rs` too (`cargo test --release --lib ember` prints each
//! new value in its assertion message); re-bless them the same way.

use nalgebra::Vector3;
use three_body_problem::ember::{
    EmberConfig, EmberMode, EmberRequest, EmberStats, EmberSummary, View, ViewDrift, ViewFrame,
    ViewProjection, plan_ember, render_ember,
};
use three_body_problem::render::constants::DEFAULT_DT;
use three_body_problem::sim::{Body, get_positions};

/// SHA-256 of the golden render's `rgb48le` frame stream.
const GOLDEN_FRAMES_SHA256: &str =
    "b76659708de75e7d36e7a5c82c3ed5875f27c1e93ba32fa3473e7c3205732ef7";
/// SHA-256 of the golden render's slow films as `rgb48le`, in the order of [`SLOW_FACTORS`].
const GOLDEN_SLOW_FRAMES_SHA256: [&str; 2] = [
    "1dbb8c32ad6dd26cf8a7f5901a653d597b3113d9e2386c61cccb1af998045c2c",
    "951181709982cff22126e822dd1074476e70bd74c836dee28f604e6c7b2747d5",
];
/// SHA-256 of the golden render's still as `rgb48le`.
const GOLDEN_STILL_SHA256: &str =
    "28afdf5b8ddc114dfcd97b85d5886494ab627de921c3300a5e3b745909685f9f";
/// `stats.contact_events` of the golden render (the same in video and still-only mode: every
/// frame interval is remapped either way).
const GOLDEN_CONTACT_EVENTS: u64 = 39_225;
/// Inked nodes of the golden still: `stats.still_ink_fraction` is this over the [`view_nodes`]
/// visible ink nodes.
const GOLDEN_STILL_INK_NODES: u32 = 17_436;

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;
const STEPS: usize = 3_000;
const FRAME_INTERVAL: usize = 150;
/// How many times slower the golden render's slow films are: the moment half-way through each
/// interval belongs to both, so the render shares it. Their least common multiple, 4, is the
/// snapshot lattice of `ember-v3`'s single 4x golden film, so the still, the normal film and the
/// 4x film keep that look's digests.
const SLOW_FACTORS: [u32; 2] = [2, 4];

/// The figure-eight choreography (Chenciner–Montgomery), slightly tilted out of its plane, with
/// unit masses and velocities rescaled from `G = 1` to the simulator's `G` (period ≈ 2.02 time
/// units). Exact decimal literals and exactly rounded arithmetic only: no RNG, no libm.
fn golden_bodies() -> Vec<Body> {
    let speed = three_body_problem::sim::G.sqrt();
    let (x, y, z) = (0.970_004_36, -0.243_087_53, 0.05);
    let (vx, vy) = (-0.932_407_37 * speed, -0.864_731_46 * speed);
    vec![
        Body::new(1.0, Vector3::new(x, y, z), Vector3::new(-vx / 2.0, -vy / 2.0, 0.0)),
        Body::new(1.0, Vector3::new(-x, -y, -z), Vector3::new(-vx / 2.0, -vy / 2.0, 0.0)),
        Body::new(1.0, Vector3::zeros(), Vector3::new(vx, vy, 0.0)),
    ]
}

/// The view of the golden orbit, as the main edition would record one: seen through a tilted
/// rotation, carried along an arc of a drift ellipse, in a frame of the canvas's aspect that is
/// shrunk a little about its centre (as the symmetric seeds do). Exact decimal literals only; the
/// two rotations are exactly orthonormal 3-4-5 matrices.
fn golden_view() -> View {
    View {
        projection: ViewProjection::Position,
        rotation: [[0.8, -0.36, 0.48], [0.6, 0.48, -0.64], [0.0, 0.8, 0.6]],
        drift: ViewDrift::Elliptical {
            rotation: [[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]],
            mean_anomaly: 2.5,
            mean_motion: 0.5,
            eccentricity: 0.3,
            semi_major: 0.6,
            semi_minor: 0.45,
        },
        frame: ViewFrame { min_x: -2.2, min_y: -1.8, width: 3.6, height: 2.4, scale: 0.9 },
    }
}

/// A coarse but complete configuration in which every stage runs (the tidal shapes, the gate,
/// the hold and the fade, the solid bodies, the valve). One render costs a few seconds of CPU
/// time in release builds; more threads barely help a 96 × 64 fluid grid.
///
/// On a 64-row grid the (enlarged) bodies cannot spin their boundary layers up to the production
/// gate of |ω| = 40, so the gate is lowered. The golden orbit is short (about six fluid units):
/// the bodies start inking half-way through it, so the slow films skip the first frames of bare
/// paper, and the hold and the fade are a quarter of it each, so fresh black, fading grey and
/// the floor wash all appear in the frames.
fn golden_config() -> EmberConfig {
    let mut config = EmberConfig::default();
    config.fluid.rows = 64;
    config.fluid.body_radius = 0.08;
    config.fluid.mask_width = 0.02;
    config.fluid.max_dt = 0.01;
    config.fluid.max_snapshot_interval = 0.01;
    config.contact.vorticity_gate = 1.0;
    config.contact.soak_depth = 0.12;
    config.contact.pre_roll = 3.0;
    config.contact.valve_lead = 0.2;
    config.look.hold_fraction = 0.25;
    config.look.fade_fraction = 0.25;
    config.paper.formation_modes = 64;
    config
}

/// The main video's frame schedule rule applied to the golden orbit.
fn golden_schedule() -> Vec<usize> {
    let mut steps: Vec<usize> = (FRAME_INTERVAL..STEPS).step_by(FRAME_INTERVAL).collect();
    if steps.last() != Some(&(STEPS - 1)) {
        steps.push(STEPS - 1);
    }
    steps
}

/// A render and the frames its sink received.
struct GoldenRun {
    summary: EmberSummary,
    /// The scheduled frames (the normal film).
    frames: Vec<Vec<u16>>,
    /// Each slow film, in the order of [`SLOW_FACTORS`]: scheduled frames from its first one on,
    /// and the frames between them.
    slow_frames: [Vec<Vec<u16>>; 2],
    /// Frames the sink received: every moment once, however many films show it.
    sink_calls: usize,
}

/// Renders the golden orbit with `config` in a rayon pool of `threads` threads.
fn render(threads: usize, mode: EmberMode, config: &EmberConfig) -> GoldenRun {
    let bodies = golden_bodies();
    let masses = std::array::from_fn(|b| bodies[b].mass);
    let positions = get_positions(bodies, STEPS).positions;
    let schedule = golden_schedule();
    let view = golden_view();
    let request = EmberRequest {
        positions: &positions,
        dt: DEFAULT_DT,
        masses,
        view: &view,
        frame_steps: &schedule,
        slow_factors: &SLOW_FACTORS,
        width: WIDTH,
        height: HEIGHT,
        paper_seed: b"ember-golden-paper",
        config,
        mode,
    };
    if blessing() {
        let plan = plan_ember(&request).expect("the golden render plans");
        println!(
            "plan: duration {}, ink from {}, edge clearance {}, overlap {}, peak speed {}",
            plan.duration(),
            plan.ink_start(),
            plan.edge_clearance(),
            plan.overlap_fraction(),
            plan.peak_speed()
        );
    }
    let (mut frames, mut slow_frames) = (Vec::new(), [Vec::new(), Vec::new()]);
    let (mut sink_calls, mut last_time) = (0, f64::NEG_INFINITY);
    let summary = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool builds")
        .install(|| {
            render_ember(&request, &mut |frame| {
                // The sink sees the frames of every film in time order, each moment once.
                assert!(frame.index.is_some() || frame.slow_indices.iter().any(Option::is_some));
                assert!(frame.time > last_time, "{} after {last_time}", frame.time);
                (sink_calls, last_time) = (sink_calls + 1, frame.time);
                if let Some(index) = frame.index {
                    assert_eq!(index, frames.len());
                    assert_eq!(frame.orbit_step, Some(schedule[index]));
                    frames.push(frame.rgb.to_vec());
                }
                for (film, index) in slow_frames.iter_mut().zip(frame.slow_indices) {
                    if let Some(index) = *index {
                        assert_eq!(index, film.len());
                        film.push(frame.rgb.to_vec());
                    }
                }
                Ok(())
            })
        })
        .expect("the golden render succeeds");
    GoldenRun { summary, frames, slow_frames, sink_calls }
}

/// Whether the digests are being re-blessed (`EMBER_BLESS` set).
fn blessing() -> bool {
    std::env::var_os("EMBER_BLESS").is_some()
}

/// Visible ink nodes of the golden render (`WIDTH·HEIGHT·supersample²`, the ink margin
/// excluded): the denominator of the node fractions in `EmberStats`.
fn view_nodes() -> u32 {
    let q = golden_config().raster.supersample;
    WIDTH * HEIGHT * q * q
}

/// The inked node count behind `stats.still_ink_fraction`, recovered exactly: the render divides
/// an integer node count by [`view_nodes`] once, so the nearest integer to the product is that
/// count, and dividing it again must give back the recorded fraction bit for bit.
fn still_ink_nodes(stats: &EmberStats) -> u32 {
    let view = f64::from(view_nodes());
    let fraction = stats.still_ink_fraction;
    let nodes = (fraction * view).round();
    assert!((0.0..=view).contains(&nodes), "{fraction} of {view} nodes");
    assert_eq!(
        (nodes / view).to_bits(),
        fraction.to_bits(),
        "still_ink_fraction {fraction} is not a node count over {view} nodes"
    );
    nodes as u32
}

/// Asserts that a render reproduces the golden digests and statistics (the digest of each film
/// that was rendered); or prints them all when blessing.
fn assert_golden(label: &str, summary: &EmberSummary) {
    let stats = &summary.stats;
    if blessing() {
        println!("{label}: GOLDEN_FRAMES_SHA256 = {:?}", summary.frames_sha256);
        let slow: Vec<_> = summary.slow_films.iter().map(|film| &film.frames_sha256).collect();
        println!("{label}: GOLDEN_SLOW_FRAMES_SHA256 = {slow:?}");
        println!("{label}: GOLDEN_STILL_SHA256 = {}", summary.still_sha256);
        println!("{label}: GOLDEN_CONTACT_EVENTS = {}", stats.contact_events);
        println!("{label}: GOLDEN_STILL_INK_NODES = {}", still_ink_nodes(stats));
        println!("{label}: stats = {stats:?}");
        return;
    }
    if let Some(frames) = summary.frames_sha256.as_deref() {
        assert_eq!(frames, GOLDEN_FRAMES_SHA256, "{label}: frame stream digest changed");
    }
    for (film, golden) in summary.slow_films.iter().zip(GOLDEN_SLOW_FRAMES_SHA256) {
        if let Some(frames) = film.frames_sha256.as_deref() {
            assert_eq!(frames, golden, "{label}: the {}x film's digest changed", film.factor);
        }
    }
    assert_eq!(summary.still_sha256, GOLDEN_STILL_SHA256, "{label}: still digest changed");
    // The statistics are part of the certificate's deterministic contract too.
    assert_eq!(stats.contact_events, GOLDEN_CONTACT_EVENTS, "{label}: {stats:?}");
    assert_eq!(still_ink_nodes(stats), GOLDEN_STILL_INK_NODES, "{label}: {stats:?}");
}

/// Sum of the absolute sample differences of two frames.
fn distance(a: &[u16], b: &[u16]) -> u64 {
    a.iter().zip(b).map(|(&x, &y)| u64::from(x.abs_diff(y))).sum()
}

#[test]
fn the_golden_films_match_on_every_architecture() {
    let run = render(3, EmberMode::VideoAndSlow, &golden_config());
    assert_golden("all films, 3 threads", &run.summary);
    let summary = &run.summary;

    // Every scheduled frame reaches the sink, and the last one is the still.
    let count = golden_schedule().len();
    assert_eq!(run.frames.len(), count);
    assert_eq!(summary.frames_emitted, count);
    assert_eq!(run.frames.last().expect("frames"), &summary.still);

    // Each slow film starts at one of the scheduled frames, shows every later one, and
    // `factor - 1` frames between each two of them.
    assert_eq!(summary.slow_films.len(), SLOW_FACTORS.len());
    for (film, frames) in summary.slow_films.iter().zip(&run.slow_frames) {
        let (first, factor) = (film.first_frame, film.factor as usize);
        assert_eq!(frames.len(), (count - 1 - first) * factor + 1, "{factor}x");
        assert_eq!(film.frames_emitted, frames.len(), "{factor}x");
        for (offset, scheduled) in run.frames[first..].iter().enumerate() {
            assert!(frames[offset * factor] == *scheduled, "{factor}x: scheduled frame {offset}");
        }
        assert_eq!(frames.last().expect("slow frames"), &summary.still);
        // It starts on bare paper, a little before the ink.
        assert!(first > 0, "the golden {factor}x film must skip some of the pre-roll");
        assert_eq!(frames[0], run.frames[0]);
    }
    // Both films start together, and the 2x film's frames are every other frame of the 4x film:
    // the same moments, rendered once (the sink gets the scheduled frames and the 4x film's
    // three moments per interval, nothing more).
    let [half, quarter] = &run.slow_frames;
    let first = summary.slow_films[0].first_frame;
    assert_eq!(summary.slow_films[1].first_frame, first);
    assert_eq!(quarter.len(), 2 * half.len() - 1);
    for (index, frame) in half.iter().enumerate() {
        assert!(quarter[2 * index] == *frame, "2x frame {index}");
    }
    assert_eq!(run.sink_calls, count + (count - 1 - first) * 3);

    // The golden orbit actually draws: fluid, contacts and sumi, with tidally stretched bodies.
    let stats = summary.stats;
    assert!(stats.fluid_steps > 100, "{stats:?}");
    assert!(stats.contact_events > 0, "{stats:?}");
    assert!(stats.still_ink_fraction > 0.01, "{stats:?}");
    assert!(summary.tidal_reference > 0.0, "{}", summary.tidal_reference);
    // The look's timing is a fraction of the orbit.
    let duration = summary.duration;
    assert_eq!(summary.hold_time, 0.25 * duration);
    assert_eq!(summary.fade_time, 0.25 * duration);
    // The first frames precede the pre-roll: bare paper, identical to each other.
    assert_eq!(run.frames[0], run.frames[1]);
    assert_ne!(run.frames[0], summary.still);
}

/// The slow films are not choppy: their in-between frames are real, evenly spaced moments of the
/// same flow. Each opens with a short lead of bare paper; once the ink is on the sheet, the
/// picture changes from every frame of the film to the next by about the same amount, whether
/// the next frame is an in-between frame or a scheduled one (no frame repeats, none jumps), and
/// it keeps going the same way (two frames apart, the picture has changed more than in either
/// step between them: nothing flickers back).
#[test]
fn the_slow_films_move_evenly_between_scheduled_frames() {
    let run = render(2, EmberMode::VideoAndSlow, &golden_config());
    let bare = &run.frames[0];
    // The ink first shows in the scheduled frame `inked`, so it starts in the interval before it.
    let inked = run.frames.iter().position(|frame| frame != bare).expect("ink in the film");
    for (film, frames) in run.summary.slow_films.iter().zip(&run.slow_frames) {
        let (first, factor) = (film.first_frame, film.factor as usize);
        let steps: Vec<u64> = frames.windows(2).map(|pair| distance(&pair[0], &pair[1])).collect();
        if blessing() {
            for (interval, steps) in steps.chunks(factor).enumerate() {
                println!("{factor}x interval {}: steps {steps:?}", first + interval);
            }
        }

        // The film starts 6 scheduled frames before the interval in which the ink starts (the
        // pipeline's lead), and its first inked frame is one of that interval's.
        assert_eq!(inked - 1 - first, 6, "{factor}x: the lead of bare paper, first {first}");
        let inked_from = frames.iter().position(|frame| frame != bare).expect("ink");
        let interval = (inked - 1 - first) * factor;
        assert!(
            interval < inked_from && inked_from <= interval + factor,
            "{factor}x: the first inked frame {inked_from} is outside the interval from {interval}"
        );

        // From the second inked frame on (the first is the ink's sudden onset), every step of
        // the film is within a factor of its neighbour, across the scheduled frames too.
        let moving = &steps[inked_from..];
        assert!(moving.len() >= 5 * factor, "{factor}x: only {} inked steps", moving.len());
        for (index, pair) in moving.windows(2).enumerate() {
            let frame = inked_from + index;
            let (small, large) = (pair[0].min(pair[1]), pair[0].max(pair[1]));
            assert!(small > 0, "{factor}x frame {frame}: a repeated frame");
            assert!(2 * large <= 3 * small, "{factor}x frame {frame}: uneven steps {pair:?}");
            let across = distance(&frames[frame], &frames[frame + 2]);
            assert!(across > large, "{factor}x frame {frame}: steps {pair:?} lead back");
        }
    }
}

#[test]
fn still_only_mode_reproduces_the_golden_still() {
    let run = render(2, EmberMode::StillOnly, &golden_config());
    assert!(run.frames.is_empty(), "still-only renders never call the sink");
    assert!(run.slow_frames.iter().all(Vec::is_empty));
    assert_eq!(run.summary.frames_emitted, 0);
    assert_eq!(run.summary.frames_sha256, None);
    for film in &run.summary.slow_films {
        assert_eq!((film.frames_emitted, &film.frames_sha256), (0, &None));
    }
    assert_golden("still only, 2 threads", &run.summary);
}

/// The normal film and the still do not depend on whether the slow films are rendered.
#[test]
fn the_video_without_the_slow_films_reproduces_the_golden_frames() {
    let run = render(2, EmberMode::Video, &golden_config());
    assert!(run.slow_frames.iter().all(Vec::is_empty), "no slow frame reaches the sink");
    for film in &run.summary.slow_films {
        assert_eq!((film.frames_emitted, &film.frames_sha256), (0, &None));
    }
    assert!(run.summary.frames_sha256.is_some());
    assert_golden("video, 2 threads", &run.summary);
}

#[test]
fn a_single_thread_reproduces_every_bit() {
    let run = render(1, EmberMode::VideoAndSlow, &golden_config());
    assert_golden("all films, 1 thread", &run.summary);
}

/// Pools larger than the fluid solver's thread cap run the solver in a dedicated, smaller pool;
/// that must not change a bit either. Run on request: it oversubscribes small CI runners.
#[test]
#[ignore = "40-thread pool; run on many-core machines and when re-blessing"]
fn a_large_pool_with_a_capped_fluid_pool_reproduces_every_bit() {
    let run = render(40, EmberMode::VideoAndSlow, &golden_config());
    assert_golden("all films, 40 threads", &run.summary);
}

/// The golden render must exercise the tidal shapes: with rigid discs (`max_aspect = 1`) the
/// same orbit draws a different still, so the golden digests cover the shape arithmetic. Two
/// renders, so it runs on request (`--include-ignored`, as the re-blessing instructions say).
#[test]
#[ignore = "two extra renders; run when re-blessing the golden digests"]
fn the_golden_render_stretches_its_bodies() {
    let stretched = render(2, EmberMode::StillOnly, &golden_config());
    let mut config = golden_config();
    config.tidal.max_aspect = 1.0;
    let discs = render(2, EmberMode::StillOnly, &config);
    println!(
        "ink fraction of the still: {} stretched, {} as discs",
        stretched.summary.stats.still_ink_fraction, discs.summary.stats.still_ink_fraction
    );
    assert_ne!(stretched.summary.still, discs.summary.still, "the shapes change nothing");
}
