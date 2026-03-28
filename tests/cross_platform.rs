//! Integration tests for cross-platform compatibility.
//!
//! These tests verify that the codebase works correctly on any supported
//! platform: x86_64 (with or without AVX2/AVX-512), aarch64 (Apple Silicon,
//! ARM servers), and potential future targets.

use three_body_problem::spectrum::NUM_BINS;
use three_body_problem::spectrum_simd;

// ── SIMD dispatch tests ─────────────────────────────────────────────────────

#[test]
fn test_simd_dispatch_produces_valid_rgba_on_any_platform() {
    let spd = [0.3, 0.1, 0.7, 0.2, 0.0, 0.5, 0.9, 0.4, 0.0, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.0];
    let (r, g, b, a) = spectrum_simd::spd_to_rgba_simd(&spd);
    assert!((0.0..=1.0).contains(&r), "R out of range: {r}");
    assert!((0.0..=1.0).contains(&g), "G out of range: {g}");
    assert!((0.0..=1.0).contains(&b), "B out of range: {b}");
    assert!((0.0..=1.0).contains(&a), "A out of range: {a}");
}

#[test]
fn test_simd_path_name_is_valid_known_value() {
    let name = spectrum_simd::detected_simd_path_name();
    let valid = ["AVX-512", "AVX2+FMA", "NEON", "Scalar"];
    assert!(valid.contains(&name), "unknown SIMD path: {name}");
}

#[test]
fn test_simd_dispatch_is_deterministic_across_calls() {
    use std::sync::atomic::Ordering;
    // Pin the sat-boost flag so parallel tests toggling it don't interfere.
    spectrum_simd::SAT_BOOST_ENABLED.store(true, Ordering::SeqCst);

    let spd = [0.0, 0.2, 0.5, 0.7, 1.0, 0.9, 0.4, 0.1, 0.0, 0.3, 0.6, 0.8, 0.5, 0.2, 0.0, 0.0];
    let reference = spectrum_simd::spd_to_rgba_simd(&spd);
    for _ in 0..500 {
        // Re-pin in case a parallel test toggled it.
        spectrum_simd::SAT_BOOST_ENABLED.store(true, Ordering::SeqCst);
        let result = spectrum_simd::spd_to_rgba_simd(&spd);
        assert_eq!(result.0.to_bits(), reference.0.to_bits(), "R non-deterministic");
        assert_eq!(result.1.to_bits(), reference.1.to_bits(), "G non-deterministic");
        assert_eq!(result.2.to_bits(), reference.2.to_bits(), "B non-deterministic");
        assert_eq!(result.3.to_bits(), reference.3.to_bits(), "A non-deterministic");
    }
}

#[test]
fn test_simd_dispatch_zero_input_is_black() {
    let spd = [0.0; NUM_BINS];
    let result = spectrum_simd::spd_to_rgba_simd(&spd);
    assert_eq!(result, (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn test_simd_dispatch_uniform_high_energy_is_clamped() {
    let spd = [1000.0; NUM_BINS];
    let (r, g, b, a) = spectrum_simd::spd_to_rgba_simd(&spd);
    assert!(r <= 1.0 && g <= 1.0 && b <= 1.0 && a <= 1.0);
    assert!(a > 0.99, "high energy should saturate alpha");
}

#[test]
fn test_simd_every_bin_produces_nonzero_output() {
    for bin in 0..NUM_BINS {
        let mut spd = [0.0; NUM_BINS];
        spd[bin] = 1.0;
        let result = spectrum_simd::spd_to_rgba_simd(&spd);
        assert!(result.3 > 0.0, "bin {bin} should produce nonzero alpha");
    }
}

#[test]
fn test_simd_negative_energy_is_black() {
    let spd = [-5.0; NUM_BINS];
    let result = spectrum_simd::spd_to_rgba_simd(&spd);
    assert_eq!(result, (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn test_simd_mixed_positive_negative_valid() {
    let spd = [-1.0, 0.5, -0.3, 0.7, 1.0, -0.1, 0.0, 0.3, -0.5, 0.2, 0.4, -0.2, 0.6, 0.1, -0.4, 0.8];
    let (r, g, b, a) = spectrum_simd::spd_to_rgba_simd(&spd);
    assert!((0.0..=1.0).contains(&r));
    assert!((0.0..=1.0).contains(&g));
    assert!((0.0..=1.0).contains(&b));
    assert!((0.0..=1.0).contains(&a));
}

#[test]
fn test_simd_sat_boost_toggle_is_observable() {
    use std::sync::atomic::Ordering;
    let spd = [0.0, 0.3, 0.6, 0.9, 1.2, 0.8, 0.4, 0.1, 0.0, 0.5, 0.7, 0.9, 0.6, 0.3, 0.1, 0.0];

    spectrum_simd::SAT_BOOST_ENABLED.store(true, Ordering::Relaxed);
    let boosted = spectrum_simd::spd_to_rgba_simd(&spd);

    spectrum_simd::SAT_BOOST_ENABLED.store(false, Ordering::Relaxed);
    let unboosted = spectrum_simd::spd_to_rgba_simd(&spd);

    spectrum_simd::SAT_BOOST_ENABLED.store(true, Ordering::Relaxed);

    assert_ne!(boosted, unboosted, "sat boost toggle must produce different output");
}

#[test]
fn test_simd_stress_1000_random_spectra() {
    let mut seed = 0xDEAD_BEEF_u64;
    for _ in 0..1000 {
        let mut spd = [0.0; NUM_BINS];
        for v in spd.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *v = (seed % 50_000) as f64 / 100.0;
        }
        let (r, g, b, a) = spectrum_simd::spd_to_rgba_simd(&spd);
        assert!(r >= 0.0 && r <= 1.0, "R={r}");
        assert!(g >= 0.0 && g <= 1.0, "G={g}");
        assert!(b >= 0.0 && b <= 1.0, "B={b}");
        assert!(a >= 0.0 && a <= 1.0, "A={a}");
    }
}

// ── u16_slice_as_bytes tests ────────────────────────────────────────────────

#[test]
fn test_u16_slice_as_bytes_empty() {
    let empty: &[u16] = &[];
    let bytes = three_body_problem::utils::u16_slice_as_bytes(empty);
    assert!(bytes.is_empty());
}

#[test]
fn test_u16_slice_as_bytes_single_value() {
    let data: &[u16] = &[0x0102];
    let bytes = three_body_problem::utils::u16_slice_as_bytes(data);
    assert_eq!(bytes.len(), 2);
    assert_eq!(bytes[0], 0x02, "little-endian: low byte first");
    assert_eq!(bytes[1], 0x01, "little-endian: high byte second");
}

#[test]
fn test_u16_slice_as_bytes_roundtrip() {
    let original: Vec<u16> = (0..256).collect();
    let bytes = three_body_problem::utils::u16_slice_as_bytes(&original);
    assert_eq!(bytes.len(), original.len() * 2);

    for (i, &val) in original.iter().enumerate() {
        let lo = bytes[i * 2];
        let hi = bytes[i * 2 + 1];
        let reconstructed = u16::from_le_bytes([lo, hi]);
        assert_eq!(reconstructed, val, "mismatch at index {i}");
    }
}

#[test]
fn test_u16_slice_as_bytes_large_values() {
    let data: &[u16] = &[0, 0xFFFF, 0x8000, 1, 0x7FFF];
    let bytes = three_body_problem::utils::u16_slice_as_bytes(data);
    assert_eq!(bytes.len(), 10);
    assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 0);
    assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 0xFFFF);
    assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 0x8000);
}

#[test]
fn test_u16_slice_as_bytes_rgb48_frame() {
    let width = 64u32;
    let height = 48u32;
    let pixel_count = (width * height) as usize;
    let frame: Vec<u16> = (0..pixel_count * 3).map(|i| (i % 65536) as u16).collect();
    let bytes = three_body_problem::utils::u16_slice_as_bytes(&frame);
    assert_eq!(bytes.len(), pixel_count * 6, "rgb48le: 6 bytes per pixel");
}

// ── endianness verification ─────────────────────────────────────────────────

#[test]
fn test_native_endianness_is_little() {
    let val: u16 = 0x0102;
    let bytes = val.to_ne_bytes();
    assert_eq!(bytes, [0x02, 0x01], "platform must be little-endian");
}

#[test]
fn test_u32_native_byte_order_matches_le() {
    let val: u32 = 0x04030201;
    assert_eq!(val.to_ne_bytes(), val.to_le_bytes());
}

// ── video encoding options ──────────────────────────────────────────────────

#[test]
fn test_default_video_options_valid() {
    let opts = three_body_problem::render::VideoEncodingOptions::default();
    assert!(!opts.codec.is_empty());
    assert!(!opts.pixel_format.is_empty());
    assert!(!opts.input_pixel_format.is_empty());
    assert!(opts.pixel_format.contains("10"), "should use 10-bit");
}

#[test]
fn test_fast_encode_options_valid() {
    let opts = three_body_problem::render::VideoEncodingOptions::fast_encode();
    assert!(!opts.codec.is_empty());
    assert!(!opts.pixel_format.is_empty());
    assert_eq!(opts.input_pixel_format, "rgb48le");
}

#[test]
fn test_default_and_fast_encode_differ() {
    let default = three_body_problem::render::VideoEncodingOptions::default();
    let fast = three_body_problem::render::VideoEncodingOptions::fast_encode();
    assert!(
        default.codec != fast.codec || default.preset != fast.preset || default.crf != fast.crf,
        "fast_encode should differ from default"
    );
}

// ── RSS measurement ─────────────────────────────────────────────────────────

#[test]
fn test_perf_profile_creation_does_not_panic() {
    let profile = three_body_problem::perf::PerformanceProfile::new();
    assert!(profile.num_logical_cpus > 0);
    assert!(profile.rayon_threads > 0);
}

#[test]
fn test_stage_timer_measures_nonzero_time() {
    let timer = three_body_problem::perf::StageTimer::start("cross_platform_test");
    std::thread::sleep(std::time::Duration::from_millis(5));
    let timing = timer.finish();
    assert!(timing.wall_clock_secs > 0.0);
}

// ── spectrum constants ──────────────────────────────────────────────────────

#[test]
fn test_spd_to_rgba_public_api_matches_simd() {
    let spd = [0.1, 0.4, 0.7, 0.3, 0.9, 0.5, 0.2, 0.8, 0.6, 0.1, 0.3, 0.7, 0.4, 0.2, 0.0, 0.5];
    let via_public = three_body_problem::spectrum::spd_to_rgba(&spd);
    let via_simd = spectrum_simd::spd_to_rgba_simd(&spd);
    assert_eq!(via_public.0.to_bits(), via_simd.0.to_bits());
    assert_eq!(via_public.1.to_bits(), via_simd.1.to_bits());
    assert_eq!(via_public.2.to_bits(), via_simd.2.to_bits());
    assert_eq!(via_public.3.to_bits(), via_simd.3.to_bits());
}

#[test]
fn test_wavelength_bins_span_visible_range() {
    let first = three_body_problem::spectrum::wavelength_nm_for_bin(0);
    let last = three_body_problem::spectrum::wavelength_nm_for_bin(NUM_BINS - 1);
    assert!(first >= 380.0 && first < 420.0, "first bin should be near 380nm");
    assert!(last > 660.0 && last <= 700.0, "last bin should be near 700nm");
}

// ── simulation RNG portability ──────────────────────────────────────────────

#[test]
fn test_rng_same_seed_same_output() {
    use three_body_problem::sim::Sha3RandomByteStream;
    let seed = b"cross_platform_test_seed";
    let mut rng1 = Sha3RandomByteStream::new(seed, 100.0, 300.0, 300.0, 1.0);
    let mut rng2 = Sha3RandomByteStream::new(seed, 100.0, 300.0, 300.0, 1.0);

    for i in 0..1000 {
        let a = rng1.next_f64();
        let b = rng2.next_f64();
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "RNG diverged at step {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_rng_different_seeds_different_output() {
    use three_body_problem::sim::Sha3RandomByteStream;
    let mut rng1 = Sha3RandomByteStream::new(b"seed_A", 100.0, 300.0, 300.0, 1.0);
    let mut rng2 = Sha3RandomByteStream::new(b"seed_B", 100.0, 300.0, 300.0, 1.0);

    let mut same_count = 0;
    for _ in 0..100 {
        if rng1.next_f64().to_bits() == rng2.next_f64().to_bits() {
            same_count += 1;
        }
    }
    assert!(same_count < 5, "different seeds should produce different outputs");
}

#[test]
fn test_rng_output_in_unit_range() {
    use three_body_problem::sim::Sha3RandomByteStream;
    let mut rng = Sha3RandomByteStream::new(b"range_test", 100.0, 300.0, 300.0, 1.0);
    for _ in 0..10_000 {
        let val = rng.next_f64();
        assert!(val >= 0.0 && val <= 1.0, "f64 out of [0,1]: {val}");
    }
}
