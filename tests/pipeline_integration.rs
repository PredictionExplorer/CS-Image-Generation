//! End-to-end pipeline integration tests.
//!
//! These tests run the public spectral rendering pipeline for a set of pinned
//! seeds at a small resolution, and guard the following invariants:
//!
//! * `run_full_render` is deterministic: identical seed produces bit-identical
//!   output over multiple runs (SHA-256 of the raw 16-bit buffer).
//! * The art-style meta-system produces diverse `ArtStyle` / `NebulaPalette`
//!   selections over 64 seeds (Shannon-entropy-style coverage).
//! * The generation log now exposes the new style fields.
//!
//! These tests deliberately avoid encoding exact hashes in source, because
//! refactors that change RNG ordering are expected during development. Instead
//! they assert *self-consistency* (same seed → same hash) and *coverage* (many
//! distinct styles appear across seeds).

use std::collections::HashMap;
use std::fs;

use image::{ImageBuffer, Rgb};
use three_body_problem::app::{
    Enhancements, apply_drift_transformation, generate_colors_with_mode, simulate_best_orbit,
};
use three_body_problem::render::art_style::ArtStyle;
use three_body_problem::render::nebula_presets::NebulaPalette;
use three_body_problem::render::randomizable_config::RandomizableEffectConfig;
use three_body_problem::render::{
    ChannelLevels, RenderConfig, SpectralRenderSettings, SpectralScene, ToneMappingControls,
    histogram, pass_1_build_histogram_spectral, render_final_frame_spectral,
    save_image_as_png_16bit,
};
use three_body_problem::sim::{Sha3RandomByteStream, select_best_trajectory};

const MIN_MASS: f64 = 100.0;
const MAX_MASS: f64 = 300.0;
const LOCATION: f64 = 300.0;
const VELOCITY: f64 = 1.0;
// Bumped from (2_000, 12) to (20_000, 200) so the dwell-entropy filter
// in `select_best_trajectory` can find at least one orbit whose bodies
// cover enough of the canvas. At small step counts the bounded chaotic
// dance hasn't had time to fill the frame, so the visual-quality gate
// rejects every candidate.
const SIM_STEPS: usize = 20_000;
const NUM_SIMS: usize = 200;
const WIDTH: u32 = 192;
const HEIGHT: u32 = 108;

/// Fowler–Noll–Vo 64-bit hash. We avoid a full SHA-256 dep just to keep the
/// test surface compact; FNV is more than sufficient to detect pixel-level
/// drift between runs.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn hash_u16_buffer(buf: &[u16]) -> u64 {
    let mut bytes = Vec::with_capacity(buf.len() * 2);
    for &v in buf {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fnv1a64(&bytes)
}

struct PipelineOutput {
    pixels: Vec<u16>,
    art_style: ArtStyle,
    nebula_palette: NebulaPalette,
}

fn run_full_render(seed_bytes: &[u8]) -> PipelineOutput {
    let mut rng = Sha3RandomByteStream::new(seed_bytes, MIN_MASS, MAX_MASS, LOCATION, VELOCITY);

    // Use a minimal config where downstream post-effects are disabled so the
    // integration test is insulated from their randomised parameter ranges.
    // ArtStyle / NebulaPalette still flow through the resolver so we can
    // verify that they are selected deterministically.
    let randomizable = RandomizableEffectConfig {
        enable_bloom: Some(false),
        enable_glow: Some(false),
        enable_chromatic_bloom: Some(false),
        enable_perceptual_blur: Some(false),
        enable_micro_contrast: Some(false),
        enable_gradient_map: Some(false),
        enable_color_grade: Some(false),
        enable_champleve: Some(false),
        enable_aether: Some(false),
        enable_opalescence: Some(false),
        enable_edge_luminance: Some(false),
        enable_atmospheric_depth: Some(false),
        enable_fine_texture: Some(false),
        nebula_strength: Some(0.0),
        ..Default::default()
    };
    let (resolved, _log) = randomizable.resolve(&mut rng, WIDTH, HEIGHT);

    let (best_bodies, _) = select_best_trajectory(&mut rng, NUM_SIMS, SIM_STEPS, 0.75, 11.0, -0.3)
        .expect("Borda search should succeed for test seed");

    let mut positions = simulate_best_orbit(best_bodies, SIM_STEPS);
    apply_drift_transformation(&mut positions, "elliptical", None, None, None, &mut rng)
        .expect("drift resolution should succeed");

    let enhancements = Enhancements::default();
    let (colors, body_alphas) = generate_colors_with_mode(
        &mut rng,
        SIM_STEPS,
        15_000_000,
        &enhancements,
        Some(resolved.hue_palette_mode),
    );

    let render_config =
        RenderConfig { hdr_scale: resolved.hdr_scale, bloom_mode: resolved.bloom_mode_choice };

    let scene = SpectralScene::new(&positions, &colors, &body_alphas);
    let settings = SpectralRenderSettings::new(&resolved, &render_config, 0x1234_5678, false);

    let frame_interval = (scene.step_count() / 6).max(1);
    let histogram_data = pass_1_build_histogram_spectral(scene, frame_interval, settings);
    let analysis = histogram::analyze_tonemapping(
        histogram_data.data(),
        resolved.clip_black,
        resolved.clip_white,
    );
    let levels = ChannelLevels::with_tone_mapping(
        analysis.black_r,
        analysis.white_r,
        analysis.black_g,
        analysis.white_g,
        analysis.black_b,
        analysis.white_b,
        ToneMappingControls {
            exposure_scale: analysis.exposure_scale,
            paper_white: 0.92,
            highlight_rolloff: 2.25,
        },
    );

    let image = render_final_frame_spectral(scene, &levels, settings)
        .expect("final frame render should succeed");

    PipelineOutput {
        pixels: image.into_raw(),
        art_style: resolved.art_style,
        nebula_palette: resolved.nebula_palette,
    }
}

fn seed_from_u64(value: u64) -> [u8; 8] {
    value.to_le_bytes()
}

// ---------------------------------------------------------------------------
// Determinism tests
// ---------------------------------------------------------------------------

#[test]
fn pinned_seeds_produce_deterministic_output() {
    let seeds: [u64; 4] = [
        0x0000_0000_DEAD_BEEF,
        0x1234_5678_90AB_CDEF,
        0xCAFE_F00D_0000_0001,
        0x2026_0420_1337_C0DE,
    ];
    for seed in seeds {
        let seed_bytes = seed_from_u64(seed);
        let run_a = run_full_render(&seed_bytes);
        let run_b = run_full_render(&seed_bytes);
        assert_eq!(
            hash_u16_buffer(&run_a.pixels),
            hash_u16_buffer(&run_b.pixels),
            "seed 0x{seed:016x} should be bit-deterministic across runs"
        );
        assert_eq!(
            run_a.art_style, run_b.art_style,
            "seed 0x{seed:016x} should pick the same ArtStyle across runs",
        );
        assert_eq!(
            run_a.nebula_palette, run_b.nebula_palette,
            "seed 0x{seed:016x} should pick the same NebulaPalette across runs",
        );
        // Image must contain at least some non-trivial pixels (the pipeline is
        // not returning a fully-clipped black canvas). We allow zero pixels for
        // very rare seeds where the trajectory collapses to sub-pixel extent.
        let energy: u64 = run_a.pixels.iter().map(|&v| u64::from(v)).sum();
        assert!(energy > 0, "seed 0x{seed:016x} produced an all-zero canvas (total energy 0)");
    }
}

#[test]
fn pinned_final_frame_round_trips_through_png_without_going_black() {
    let seed = 0x0000_0000_DEAD_BEEFu64;
    let run = run_full_render(&seed_from_u64(seed));

    let image = ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(WIDTH, HEIGHT, run.pixels.clone())
        .expect("raw render pixels should form a valid 16-bit RGB image");
    let temp_dir = std::env::temp_dir().join(format!(
        "three_body_png_roundtrip_{}_{}",
        std::process::id(),
        seed
    ));
    fs::create_dir_all(&temp_dir).expect("temp output dir should be creatable");
    let png_path = temp_dir.join("image.png");
    let png_path_str = png_path.to_string_lossy().into_owned();

    save_image_as_png_16bit(&image, &png_path_str).expect("PNG save should succeed");

    let loaded = image::open(&png_path).expect("saved PNG should be readable").into_rgb16();
    let roundtrip_energy: u64 = loaded.into_raw().into_iter().map(u64::from).sum();
    assert!(
        roundtrip_energy > 0,
        "round-tripped PNG for seed 0x{seed:016x} should retain non-zero image energy"
    );

    let _ = fs::remove_file(&png_path);
    let _ = fs::remove_dir_all(&temp_dir);
}

// ---------------------------------------------------------------------------
// Art-style / nebula coverage over 64 seeds
// ---------------------------------------------------------------------------

#[test]
fn art_style_distribution_is_diverse_over_64_seeds() {
    let mut style_counts: HashMap<ArtStyle, usize> = HashMap::new();
    let mut nebula_counts: HashMap<NebulaPalette, usize> = HashMap::new();

    for seed in 0u64..64 {
        let seed_bytes = seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut rng =
            Sha3RandomByteStream::new(&seed_bytes, MIN_MASS, MAX_MASS, LOCATION, VELOCITY);
        let randomizable = RandomizableEffectConfig::default();
        let (resolved, _) = randomizable.resolve(&mut rng, WIDTH, HEIGHT);
        *style_counts.entry(resolved.art_style).or_default() += 1;
        *nebula_counts.entry(resolved.nebula_palette).or_default() += 1;
    }

    // We should see at least half of the 18 ArtStyle variants across 64 seeds.
    assert!(
        style_counts.len() >= 9,
        "only {} distinct ArtStyle variants seen in 64 seeds (want >= 9): {style_counts:?}",
        style_counts.len(),
    );

    // Nebula palettes are style-driven but still must show meaningful spread.
    assert!(
        nebula_counts.len() >= 6,
        "only {} distinct NebulaPalette variants seen in 64 seeds (want >= 6): {nebula_counts:?}",
        nebula_counts.len(),
    );

    // No single art style may dominate more than 50% of 64 seeds.
    let max_count = style_counts.values().copied().max().unwrap_or(0);
    assert!(
        max_count <= 32,
        "one ArtStyle appears {max_count}/64 times; expected <= 32 for healthy variety",
    );
}

// ---------------------------------------------------------------------------
// Generation log schema exposure
// ---------------------------------------------------------------------------

#[test]
fn generation_log_exposes_style_fields_when_populated() {
    use three_body_problem::generation_log::LoggedRenderConfig;

    let cfg = LoggedRenderConfig {
        art_style: Some("DeepCosmos".to_string()),
        nebula_palette: Some("DeepSpace".to_string()),
        grade_preset: Some("CinematicTeal".to_string()),
        hue_palette_mode: Some("Triadic".to_string()),
        bloom_mode_choice: Some("dog".to_string()),
        drift_character: Some("linear".to_string()),
        framing_zoom: Some(1.10),
        starfield_enabled: Some(true),
        lens_flare_enabled: Some(false),
        ..LoggedRenderConfig::default()
    };

    let json = serde_json::to_value(&cfg).expect("LoggedRenderConfig should serialize");
    let obj = json.as_object().expect("LoggedRenderConfig should be a JSON object");
    for key in [
        "art_style",
        "nebula_palette",
        "grade_preset",
        "hue_palette_mode",
        "bloom_mode_choice",
        "drift_character",
        "framing_zoom",
        "starfield_enabled",
        "lens_flare_enabled",
    ] {
        assert!(obj.contains_key(key), "missing key '{key}' in serialized LoggedRenderConfig");
    }
    assert_eq!(obj["art_style"].as_str(), Some("DeepCosmos"));
    assert_eq!(obj["framing_zoom"].as_f64(), Some(1.10));
}

#[test]
fn generation_log_default_omits_new_style_fields() {
    use three_body_problem::generation_log::LoggedRenderConfig;

    let cfg = LoggedRenderConfig::default();
    let json = serde_json::to_value(&cfg).expect("default should serialize");
    let obj = json.as_object().expect("JSON object");
    // skip_serializing_if should drop None fields from the default record.
    for key in [
        "art_style",
        "nebula_palette",
        "grade_preset",
        "hue_palette_mode",
        "bloom_mode_choice",
        "drift_character",
        "framing_zoom",
        "starfield_enabled",
        "lens_flare_enabled",
    ] {
        assert!(
            !obj.contains_key(key),
            "default LoggedRenderConfig should NOT include '{key}' (expected skip_serializing_if)"
        );
    }
}
