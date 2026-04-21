//! Image-space metric checks for a curated regression zoo.

mod common;

use common::{HEIGHT, WIDTH, compute_metrics, render_default};

#[test]
fn regression_zoo_meets_quality_metric_floors() {
    for seed in [
        "0x3cbc65976065",
        "0x2be2c9848e715e620be2e07fbc78ec7a926fe76337097f94206e59bebb052ea3",
        "0x8f1d87d78ba2",
        "0x870a93725533",
        "0x33c27a6f9b701c53",
        "0xba60619bd98335f7",
    ] {
        let sample = render_default(seed);
        let metrics = compute_metrics(&sample.pixels, WIDTH, HEIGHT);
        assert!(
            metrics.mean_luma > 0.010,
            "{seed} fell too dark: mean_luma={:.4}",
            metrics.mean_luma,
        );
        assert!(
            metrics.mean_luma < 0.45,
            "{seed} became too bright on average: mean_luma={:.4}",
            metrics.mean_luma,
        );
        assert!(
            metrics.near_white_fraction < 0.10,
            "{seed} has too many near-white pixels: {:.4}",
            metrics.near_white_fraction,
        );
        assert!(
            metrics.contrast_span > 0.015,
            "{seed} contrast span is too flat: {:.4}",
            metrics.contrast_span,
        );
        assert!(
            metrics.detail_energy > 0.004,
            "{seed} detail energy is too low: {:.4}",
            metrics.detail_energy,
        );
    }
}
