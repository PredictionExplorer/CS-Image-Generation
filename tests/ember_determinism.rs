//! Cross-architecture golden tests of the ember edition.
//!
//! A small but complete ember render (fluid, ink, embers, paper, shading, frame stream) of a
//! fixed orbit is hashed and compared with digests recorded on another machine. CI runs this file
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
//! 40-thread render) and that the ignored `the_golden_still_shows_embers` passes (the golden
//! render must keep exercising ember memory), copy the printed `GOLDEN_FRAMES_SHA256`,
//! `GOLDEN_STILL_SHA256`, `GOLDEN_FRAMES_WITH_CINNABAR` and `GOLDEN_PEAK_CINNABAR_NODES` (the
//! last two are printed by the video renders) into the constants below, bump
//! `ember::certificate::ALGORITHM_VERSION`, and verify the new values on both architectures
//! before committing. A change that alters rendered bits usually moves the unit goldens in
//! `src/ember/*.rs` too (`cargo test --release --lib ember` prints each new value in its assertion
//! message); re-bless them the same way.

use nalgebra::Vector3;
use three_body_problem::ember::{
    EmberConfig, EmberMode, EmberRequest, EmberStats, EmberSummary, render_ember,
};
use three_body_problem::sim::{Body, get_positions};

/// SHA-256 of the golden render's `rgb48le` frame stream.
const GOLDEN_FRAMES_SHA256: &str =
    "41369c65797a5520aab7cc834ab12d04196d135fb08446d2ca98283b88b6800c";
/// SHA-256 of the golden render's still as `rgb48le`.
const GOLDEN_STILL_SHA256: &str =
    "5f3cf8dede3609c845291c2f34c26d5a8ca467e720c6e8871dc992a4a87e1ee6";
/// `stats.frames_with_cinnabar` of the golden video.
const GOLDEN_FRAMES_WITH_CINNABAR: u64 = 15;
/// Cinnabar nodes of the golden video's most vermilion frame:
/// `stats.peak_frame_cinnabar_fraction` is this over the [`view_nodes`] visible ink nodes.
const GOLDEN_PEAK_CINNABAR_NODES: u32 = 845;

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

/// A coarse but complete configuration in which every stage runs (the gate, both pigments,
/// ember memory, the valve). One render costs about 1.5 s of CPU time in release builds (Apple
/// M4 Max: 1.5 s wall on one thread, 1 s on three; more threads barely help a 90 × 64 fluid
/// grid, and on a loaded machine they make it slower).
///
/// On a 64-row grid the (enlarged) discs cannot spin their boundary layers up to the production
/// gate of |ω| = 40, so the gate is lowered; the three figure-eight bodies follow each other
/// about two fluid time units apart, so the hold is lengthened (3.0) until their waters meet in
/// vermilion. `fresh_tau` is lengthened with it (0.25, so `hold / fresh_tau = 12 ≤ ln 10⁶`): the
/// ink fields flush freshness below 10⁻¹² to 0, and the hold gain `e^{hold/fresh_tau}` must not
/// lift that cut above 10⁻⁶ of full strength, which the configuration's validation enforces.
/// Ember memory is the production default.
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
    config.look.hold = 3.0;
    config.look.fresh_tau = 0.25;
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
    let positions = get_positions(golden_bodies(), STEPS).positions;
    let schedule = golden_schedule();
    let request = EmberRequest {
        positions: &positions,
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

/// The cinnabar node count behind `stats.peak_frame_cinnabar_fraction`, recovered exactly: the
/// render divides an integer node count by [`view_nodes`] once, so the nearest integer to the
/// product is that count, and dividing it again must give back the recorded fraction bit for
/// bit.
fn peak_cinnabar_nodes(stats: &EmberStats) -> u32 {
    let view = f64::from(view_nodes());
    let fraction = stats.peak_frame_cinnabar_fraction;
    let nodes = (fraction * view).round();
    assert!((0.0..=view).contains(&nodes), "{fraction} of {view} nodes");
    assert_eq!(
        (nodes / view).to_bits(),
        fraction.to_bits(),
        "peak_frame_cinnabar_fraction {fraction} is not a node count over {view} nodes"
    );
    nodes as u32
}

/// Asserts that a render reproduces the golden digests and, for a video render, the golden
/// frame-level statistics; or prints them all when blessing.
fn assert_golden(label: &str, summary: &EmberSummary) {
    let stats = &summary.stats;
    if blessing() {
        println!("{label}: GOLDEN_FRAMES_SHA256 = {:?}", summary.frames_sha256);
        println!("{label}: GOLDEN_STILL_SHA256 = {}", summary.still_sha256);
        if summary.frames_sha256.is_some() {
            println!("{label}: GOLDEN_FRAMES_WITH_CINNABAR = {}", stats.frames_with_cinnabar);
            println!("{label}: GOLDEN_PEAK_CINNABAR_NODES = {}", peak_cinnabar_nodes(stats));
        }
        println!("{label}: stats = {stats:?}");
        return;
    }
    if let Some(frames) = summary.frames_sha256.as_deref() {
        assert_eq!(frames, GOLDEN_FRAMES_SHA256, "{label}: frame stream digest changed");
        // The frame-level statistics are part of the certificate's deterministic contract too.
        assert_eq!(stats.frames_with_cinnabar, GOLDEN_FRAMES_WITH_CINNABAR, "{label}: {stats:?}");
        assert_eq!(peak_cinnabar_nodes(stats), GOLDEN_PEAK_CINNABAR_NODES, "{label}: {stats:?}");
    }
    assert_eq!(summary.still_sha256, GOLDEN_STILL_SHA256, "{label}: still digest changed");
}

#[test]
fn the_golden_video_matches_on_every_architecture() {
    let run = render(3, EmberMode::Video, &golden_config());
    assert_golden("video, 3 threads", &run.summary);

    // Every scheduled frame reaches the sink, and the last one is the still.
    assert_eq!(run.frames.len(), golden_schedule().len());
    assert_eq!(run.summary.frames_emitted, run.frames.len());
    assert_eq!(run.frames.last().expect("frames"), &run.summary.still);

    // The golden orbit actually draws: fluid, contacts, sumi and vermilion.
    let stats = run.summary.stats;
    assert!(stats.fluid_steps > 100, "{stats:?}");
    assert!(stats.contact_events > 0, "{stats:?}");
    assert!(stats.still_ink_fraction > 0.01, "{stats:?}");
    assert!(stats.still_cinnabar_fraction > 0.0, "{stats:?}");
    // Frame-level cinnabar coverage: the still is one of the frames with vermilion, and no
    // frame can show less than none.
    assert!(stats.frames_with_cinnabar >= 1, "{stats:?}");
    assert!(stats.frames_with_cinnabar <= run.frames.len() as u64, "{stats:?}");
    assert!(stats.peak_frame_cinnabar_fraction >= stats.still_cinnabar_fraction, "{stats:?}");
    assert!(stats.peak_frame_cinnabar_fraction <= 1.0, "{stats:?}");
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
    // Only the still is shaded, so only the still is counted.
    let stats = run.summary.stats;
    assert_eq!(stats.frames_with_cinnabar, u64::from(stats.still_cinnabar_fraction > 0.0));
    assert_eq!(stats.peak_frame_cinnabar_fraction, stats.still_cinnabar_fraction);
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

/// The golden still must show embers: vermilion that ember memory keeps after the fresh
/// meetings have gone, so that the golden digests cover the ember arithmetic. Two renders, so it
/// runs on request (`--include-ignored`, as the re-blessing instructions say).
#[test]
#[ignore = "two extra renders; run when re-blessing the golden digests"]
fn the_golden_still_shows_embers() {
    let with_memory = render(2, EmberMode::StillOnly, &golden_config());
    let mut config = golden_config();
    config.look.ember_tau = None;
    let without = render(2, EmberMode::StillOnly, &config);
    let (glowing, fresh) = (with_memory.summary.stats, without.summary.stats);
    println!(
        "cinnabar fraction of the still: {} with ember memory, {} without",
        glowing.still_cinnabar_fraction, fresh.still_cinnabar_fraction
    );
    assert!(
        glowing.still_cinnabar_fraction > fresh.still_cinnabar_fraction,
        "ember memory adds no vermilion to the golden still: {glowing:?} vs {fresh:?}"
    );
    assert_ne!(with_memory.summary.still, without.summary.still);
}
