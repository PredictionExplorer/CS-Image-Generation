//! Cross-architecture golden tests of the ember edition.
//!
//! A small but complete ember render (tidal bodies, fluid, ink, paper, shading, frame stream) of
//! a fixed orbit is hashed and compared with digests recorded on another machine. CI runs this file
//! on `x86_64` Linux and `aarch64` macOS, so any architecture-dependent arithmetic in the ember
//! path fails here.
//!
//! Every test renders exactly once, in its own small rayon pool, and compares its own render
//! with the golden digests: the tests stay cheap when a runner such as `cargo nextest` gives each
//! one its own process, and parallel test processes do not oversubscribe the CPU. Agreement of
//! the three renders (3-thread video, 2-thread still-only, 1-thread video) with the same digests
//! also proves that the still-only mode and the thread count do not change a bit.
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
//! `GOLDEN_STILL_SHA256`, `GOLDEN_CONTACT_EVENTS` and `GOLDEN_STILL_INK_NODES` into the constants
//! below, bump `ember::certificate::ALGORITHM_VERSION` if published editions change, and verify
//! the new values on both architectures before committing. A change that alters rendered bits usually moves the unit goldens in
//! `src/ember/*.rs` too (`cargo test --release --lib ember` prints each new value in its assertion
//! message); re-bless them the same way.

use nalgebra::Vector3;
use three_body_problem::ember::{
    EmberConfig, EmberMode, EmberRequest, EmberStats, EmberSummary, render_ember,
};
use three_body_problem::sim::{Body, get_positions};

/// SHA-256 of the golden render's `rgb48le` frame stream.
const GOLDEN_FRAMES_SHA256: &str =
    "7fbc90645520c4fc98d6ba9ab04d3c6f7163ff430dde0feeb397871ed2ecec55";
/// SHA-256 of the golden render's still as `rgb48le`.
const GOLDEN_STILL_SHA256: &str =
    "cc7427d5e0ed1092116c7f98a14b2f890e8f742b8b7080540394ab1f0488e3ef";
/// `stats.contact_events` of the golden render (the same in video and still-only mode: every
/// frame interval is remapped either way).
const GOLDEN_CONTACT_EVENTS: u64 = 310_043;
/// Inked nodes of the golden still: `stats.still_ink_fraction` is this over the [`view_nodes`]
/// visible ink nodes.
const GOLDEN_STILL_INK_NODES: u32 = 37_473;

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;
const STEPS: usize = 3_000;
const FRAME_INTERVAL: usize = 150;

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

/// A coarse but complete configuration in which every stage runs (the tidal shapes, the gate,
/// the hold and the fade, the solid bodies, the valve). One render costs a few seconds of CPU
/// time in release builds; more threads barely help a 96 × 64 fluid grid.
///
/// On a 64-row grid the (enlarged) bodies cannot spin their boundary layers up to the production
/// gate of |ω| = 40, so the gate is lowered. The golden orbit is short (about two fluid units), so
/// the hold and the fade are a quarter of it each: fresh black, fading grey and the floor wash
/// all appear in the frames.
fn golden_config() -> EmberConfig {
    let mut config = EmberConfig::default();
    config.fluid.rows = 64;
    config.fluid.body_radius = 0.08;
    config.fluid.mask_width = 0.02;
    config.fluid.max_dt = 0.01;
    config.fluid.max_snapshot_interval = 0.01;
    config.contact.vorticity_gate = 1.0;
    config.contact.soak_depth = 0.12;
    config.contact.pre_roll = 1.0;
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
    frames: Vec<Vec<u16>>,
}

/// Renders the golden orbit with `config` in a rayon pool of `threads` threads.
fn render(threads: usize, mode: EmberMode, config: &EmberConfig) -> GoldenRun {
    let bodies = golden_bodies();
    let masses = std::array::from_fn(|b| bodies[b].mass);
    let positions = get_positions(bodies, STEPS).positions;
    let schedule = golden_schedule();
    let request = EmberRequest {
        positions: &positions,
        masses,
        frame_steps: &schedule,
        width: WIDTH,
        height: HEIGHT,
        paper_seed: b"ember-golden-paper",
        config,
        mode,
    };
    let mut frames = Vec::new();
    let summary = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool builds")
        .install(|| {
            render_ember(&request, &mut |frame| {
                frames.push(frame.rgb.to_vec());
                Ok(())
            })
        })
        .expect("the golden render succeeds");
    GoldenRun { summary, frames }
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

/// Asserts that a render reproduces the golden digests and statistics (the frames digest for a
/// video render); or prints them all when blessing.
fn assert_golden(label: &str, summary: &EmberSummary) {
    let stats = &summary.stats;
    if blessing() {
        println!("{label}: GOLDEN_FRAMES_SHA256 = {:?}", summary.frames_sha256);
        println!("{label}: GOLDEN_STILL_SHA256 = {}", summary.still_sha256);
        println!("{label}: GOLDEN_CONTACT_EVENTS = {}", stats.contact_events);
        println!("{label}: GOLDEN_STILL_INK_NODES = {}", still_ink_nodes(stats));
        println!("{label}: stats = {stats:?}");
        return;
    }
    if let Some(frames) = summary.frames_sha256.as_deref() {
        assert_eq!(frames, GOLDEN_FRAMES_SHA256, "{label}: frame stream digest changed");
    }
    assert_eq!(summary.still_sha256, GOLDEN_STILL_SHA256, "{label}: still digest changed");
    // The statistics are part of the certificate's deterministic contract too.
    assert_eq!(stats.contact_events, GOLDEN_CONTACT_EVENTS, "{label}: {stats:?}");
    assert_eq!(still_ink_nodes(stats), GOLDEN_STILL_INK_NODES, "{label}: {stats:?}");
}

#[test]
fn the_golden_video_matches_on_every_architecture() {
    let run = render(3, EmberMode::Video, &golden_config());
    assert_golden("video, 3 threads", &run.summary);

    // Every scheduled frame reaches the sink, and the last one is the still.
    assert_eq!(run.frames.len(), golden_schedule().len());
    assert_eq!(run.summary.frames_emitted, run.frames.len());
    assert_eq!(run.frames.last().expect("frames"), &run.summary.still);

    // The golden orbit actually draws: fluid, contacts and sumi, with tidally stretched bodies.
    let stats = run.summary.stats;
    assert!(stats.fluid_steps > 100, "{stats:?}");
    assert!(stats.contact_events > 0, "{stats:?}");
    assert!(stats.still_ink_fraction > 0.01, "{stats:?}");
    assert!(run.summary.tidal_reference > 0.0, "{}", run.summary.tidal_reference);
    // The look's timing is a fraction of the orbit.
    let duration = run.summary.duration;
    assert_eq!(run.summary.hold_time, 0.25 * duration);
    assert_eq!(run.summary.fade_time, 0.25 * duration);
    // The first frames precede the pre-roll: bare paper, identical to each other.
    assert_eq!(run.frames[0], run.frames[1]);
    assert_ne!(run.frames[0], run.summary.still);
}

#[test]
fn still_only_mode_reproduces_the_golden_still() {
    let run = render(2, EmberMode::StillOnly, &golden_config());
    assert!(run.frames.is_empty(), "still-only renders never call the sink");
    assert_eq!(run.summary.frames_emitted, 0);
    assert_eq!(run.summary.frames_sha256, None);
    assert_golden("still only, 2 threads", &run.summary);
}

#[test]
fn a_single_thread_reproduces_every_bit() {
    let run = render(1, EmberMode::Video, &golden_config());
    assert_golden("video, 1 thread", &run.summary);
}

/// Pools larger than the fluid solver's thread cap run the solver in a dedicated, smaller pool;
/// that must not change a bit either. Run on request: it oversubscribes small CI runners.
#[test]
#[ignore = "40-thread pool; run on many-core machines and when re-blessing"]
fn a_large_pool_with_a_capped_fluid_pool_reproduces_every_bit() {
    let run = render(40, EmberMode::Video, &golden_config());
    assert_golden("video, 40 threads", &run.summary);
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
