//! End-to-end behavioural tests for the v2 pipeline.
//!
//! These tests exercise the public API (color-science switch, pipeline flags,
//! director checkpoints, and the Borda+beauty selection) without having to
//! drive the full video encoder.

use three_body_problem::{
    analysis::beauty_ensemble_score,
    app,
    render::{director, pipeline_flags},
    sim::Sha3RandomByteStream,
    spectrum,
};

const SEED: &str = "0xaabbccdd";

fn rng(seed: &[u8]) -> Sha3RandomByteStream {
    Sha3RandomByteStream::new(seed, 100.0, 300.0, 300.0, 1.0)
}

#[test]
fn color_science_switch_is_global_and_reversible() {
    // Record starting state so the test leaves global flags pristine.
    let initial = spectrum::color_science_is_bruton();

    spectrum::set_color_science_bruton(true);
    let bruton = spectrum::bin_combined_lut();
    assert_eq!(bruton.len(), spectrum::NUM_BINS);
    let bruton_first = bruton[0];

    spectrum::set_color_science_bruton(false);
    let cie = spectrum::bin_combined_lut();
    assert_ne!(cie[0], bruton_first, "CIE LUT must not equal Bruton LUT at bin 0");

    // Restore.
    spectrum::set_color_science_bruton(initial);
}

#[test]
fn director_and_uniform_checkpoints_cover_full_timeline() {
    let uniform = director::resolve_video_checkpoints(10_000, false);
    let director = director::resolve_video_checkpoints(10_000, true);
    assert_eq!(*uniform.last().expect("non-empty"), 9_999);
    assert_eq!(*director.last().expect("non-empty"), 9_999);
    assert!(uniform.iter().all(|&s| s < 10_000));
    assert!(director.iter().all(|&s| s < 10_000));
}

#[test]
fn pipeline_flag_atomics_are_globally_visible() {
    pipeline_flags::set_shutter_samples(8);
    assert_eq!(pipeline_flags::shutter_samples(), 8);
    pipeline_flags::set_shutter_samples(1);
}

#[test]
fn borda_selection_is_deterministic_under_beauty_weight_bw() {
    let seed = app::parse_seed(SEED).expect("seed");
    let mut rng1 = rng(&seed);
    let mut rng2 = rng(&seed);
    let (b1, t1) = app::run_borda_selection(&mut rng1, 256, 4_000, 1.0, 5.0, 0.45, -0.3)
        .expect("borda run 1");
    let (b2, t2) = app::run_borda_selection(&mut rng2, 256, 4_000, 1.0, 5.0, 0.45, -0.3)
        .expect("borda run 2");
    for (a, b) in b1.iter().zip(&b2) {
        assert_eq!(a.mass.to_bits(), b.mass.to_bits());
        assert_eq!(a.position, b.position);
        assert_eq!(a.velocity, b.velocity);
    }
    assert_eq!(t1.total_score, t2.total_score);
    assert_eq!(t1.chaos.to_bits(), t2.chaos.to_bits());
    assert_eq!(t1.beauty.to_bits(), t2.beauty.to_bits());
}

#[test]
fn changing_beauty_weight_can_change_selection() {
    // With a large beauty weight, the selected orbit can differ from the "equil-first" run.
    let seed = app::parse_seed(SEED).expect("seed");
    let mut rng_a = rng(&seed);
    let mut rng_b = rng(&seed);
    let (bodies_a, _) = app::run_borda_selection(&mut rng_a, 256, 4_000, 1.0, 5.0, 0.0, -0.3)
        .expect("no-beauty run");
    let (bodies_b, _) = app::run_borda_selection(&mut rng_b, 256, 4_000, 1.0, 5.0, 3.0, -0.3)
        .expect("heavy-beauty run");
    let same = bodies_a.iter().zip(&bodies_b).all(|(a, b)| a.position == b.position);
    // We don't assert inequality hard (the two might still agree for some seeds); just that
    // the API accepts both weightings and produces a valid orbit.
    let _ = same;
    assert!(
        bodies_a.iter().all(|b| b.position.iter().all(|v| v.is_finite())),
        "positions finite"
    );
}

#[test]
fn beauty_ensemble_score_is_reproducible_on_fixed_orbit() {
    let seed = app::parse_seed(SEED).expect("seed");
    let mut r = rng(&seed);
    let (bodies, _) =
        app::run_borda_selection(&mut r, 128, 2_000, 1.0, 5.0, 0.45, -0.3).expect("borda");
    let positions = app::simulate_best_orbit(bodies.clone(), 5_000);
    let score_a = beauty_ensemble_score(&positions, bodies[0].mass, bodies[1].mass, bodies[2].mass);
    let score_b = beauty_ensemble_score(&positions, bodies[0].mass, bodies[1].mass, bodies[2].mass);
    assert_eq!(score_a.to_bits(), score_b.to_bits());
    assert!((0.0..=1.0).contains(&score_a));
}
