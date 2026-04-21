//! End-to-end integration checks for the curated rendering pipeline.

mod common;

use std::fs;

use common::{HEIGHT, WIDTH, compute_metrics, hash_u16_buffer, render_default, save_and_reload_pixels};

#[test]
fn pinned_seeds_are_bit_deterministic() {
    for seed in [
        "0x3cbc65976065",
        "0x8f1d87d78ba2",
        "0x870a93725533",
    ] {
        let run_a = render_default(seed);
        let run_b = render_default(seed);
        assert_eq!(
            hash_u16_buffer(&run_a.pixels),
            hash_u16_buffer(&run_b.pixels),
            "{seed} should render deterministically",
        );
    }
}

#[test]
fn regression_zoo_avoids_black_and_white_failures() {
    for seed in [
        "0x3cbc65976065",
        "0x2be2c9848e715e620be2e07fbc78ec7a926fe76337097f94206e59bebb052ea3",
        "0x8f1d87d78ba2",
        "0x870a93725533",
    ] {
        let sample = render_default(seed);
        let metrics = compute_metrics(&sample.pixels, WIDTH, HEIGHT);
        assert!(
            metrics.mean_luma > 0.010,
            "{seed} regressed toward black: mean_luma={:.4}",
            metrics.mean_luma,
        );
        assert!(
            metrics.mean_luma < 0.45,
            "{seed} regressed toward haze/white: mean_luma={:.4}",
            metrics.mean_luma,
        );
        assert!(
            metrics.nonzero_fraction > 0.01,
            "{seed} should light more than a trace of the frame: nonzero_fraction={:.4}",
            metrics.nonzero_fraction,
        );
        assert!(
            metrics.near_white_fraction < 0.10,
            "{seed} has too many near-white pixels: near_white_fraction={:.4}",
            metrics.near_white_fraction,
        );
    }
}

#[test]
fn png_round_trip_preserves_signal() {
    let sample = render_default("0x870a93725533");
    let path = std::env::temp_dir().join(format!(
        "three-body-roundtrip-{}.png",
        std::process::id()
    ));
    let round_tripped = save_and_reload_pixels(&sample.image, &path);
    let _ = fs::remove_file(&path);

    assert_eq!(
        hash_u16_buffer(&sample.pixels),
        hash_u16_buffer(&round_tripped),
        "16-bit PNG round-trip should preserve pixel data",
    );
}
