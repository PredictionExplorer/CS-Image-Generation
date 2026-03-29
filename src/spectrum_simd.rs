//! SIMD-accelerated spectral conversion with runtime CPU dispatch.
//!
//! Uses **runtime** feature detection on x86_64 so a single binary runs on any
//! x86_64 processor while selecting the widest available SIMD path:
//!
//! - x86_64 AVX-512F: 8 f64 lanes, ~5-6x speedup (Skylake-X, Ice Lake, Sapphire Rapids)
//! - x86_64 AVX2+FMA: 4 f64 lanes, ~3-4x speedup (Haswell and later)
//! - aarch64 NEON:    2 f64 lanes, ~2x speedup (all AArch64 including Apple Silicon)
//! - Scalar fallback: portable, any architecture
//!
//! Detection is performed once and cached in an atomic for zero overhead on
//! subsequent calls.

use crate::spectrum::{BIN_COMBINED_LUT, NUM_BINS};
use std::sync::atomic::{AtomicBool, Ordering};

pub static SAT_BOOST_ENABLED: AtomicBool = AtomicBool::new(true);

// ── runtime SIMD path detection (x86_64) ────────────────────────────────────

#[cfg(all(target_arch = "x86_64", not(miri)))]
mod x86_detect {
    use std::sync::atomic::{AtomicU8, Ordering};

    const UNINIT: u8 = 255;
    pub const SCALAR: u8 = 0;
    pub const AVX2: u8 = 1;
    pub const AVX512: u8 = 2;

    static DETECTED: AtomicU8 = AtomicU8::new(UNINIT);

    fn detect() -> u8 {
        if is_x86_feature_detected!("avx512f") {
            AVX512
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            AVX2
        } else {
            SCALAR
        }
    }

    #[inline]
    pub fn path() -> u8 {
        let cached = DETECTED.load(Ordering::Relaxed);
        if cached != UNINIT {
            return cached;
        }
        let p = detect();
        DETECTED.store(p, Ordering::Relaxed);
        p
    }
}

/// Returns a human-readable name for the SIMD path selected at runtime.
pub fn detected_simd_path_name() -> &'static str {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        return match x86_detect::path() {
            x86_detect::AVX512 => "AVX-512",
            x86_detect::AVX2 => "AVX2+FMA",
            _ => "Scalar",
        };
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        return "NEON";
    }
    #[allow(unreachable_code)]
    "Scalar"
}

// ── SIMD-friendly LUT (x86_64: contiguous arrays for vector loads) ──────────

#[cfg(all(target_arch = "x86_64", not(miri)))]
pub(crate) struct SimdLut {
    pub r: [f64; NUM_BINS],
    pub g: [f64; NUM_BINS],
    pub b: [f64; NUM_BINS],
    pub k: [f64; NUM_BINS],
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
pub(crate) static SIMD_LUT: once_cell::sync::Lazy<SimdLut> = once_cell::sync::Lazy::new(|| {
    let mut lut = SimdLut { r: [0.0; 16], g: [0.0; 16], b: [0.0; 16], k: [0.0; 16] };
    for i in 0..NUM_BINS {
        let (r, g, b, k) = BIN_COMBINED_LUT[i];
        lut.r[i] = r;
        lut.g[i] = g;
        lut.b[i] = b;
        lut.k[i] = k;
    }
    lut
});

// ── public dispatch entry point ─────────────────────────────────────────────

/// Convert SPD to RGBA using the best SIMD path available at runtime.
///
/// Automatically selects AVX-512, AVX2, NEON, or scalar based on runtime
/// CPU detection (x86_64) or compile-time architecture (aarch64).
///
/// # Accuracy
/// AVX2/AVX-512 paths use a vectorized exp() approximation (< 2e-11 relative
/// error). Results may differ from the scalar path by a few ULPs but are
/// deterministic within the same architecture and detected path.
#[inline]
pub fn spd_to_rgba_simd(spd: &[f64; NUM_BINS]) -> (f64, f64, f64, f64) {
    let boosted = SAT_BOOST_ENABLED.load(Ordering::Relaxed);
    spd_to_rgba_simd_with_sat_boost(spd, boosted)
}

#[inline]
fn spd_to_rgba_simd_with_sat_boost(spd: &[f64; NUM_BINS], boosted: bool) -> (f64, f64, f64, f64) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        return match x86_detect::path() {
            x86_detect::AVX512 => unsafe { spd_to_rgba_avx512(spd, boosted) },
            x86_detect::AVX2 => unsafe { spd_to_rgba_avx2(spd, boosted) },
            _ => spd_to_rgba_scalar_with_sat_boost(spd, boosted),
        };
    }

    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        return unsafe { spd_to_rgba_neon(spd, boosted) };
    }

    #[allow(unreachable_code)]
    spd_to_rgba_scalar_with_sat_boost(spd, boosted)
}

// ── scalar fallback (always compiled) ───────────────────────────────────────

#[cfg(test)]
#[inline]
pub(crate) fn spd_to_rgba_scalar(spd: &[f64; NUM_BINS]) -> (f64, f64, f64, f64) {
    let boosted = SAT_BOOST_ENABLED.load(Ordering::Relaxed);
    spd_to_rgba_scalar_with_sat_boost(spd, boosted)
}

#[inline]
fn spd_to_rgba_scalar_with_sat_boost(spd: &[f64; NUM_BINS], boosted: bool) -> (f64, f64, f64, f64) {
    let mut r = 0.0;
    let mut g = 0.0;
    let mut b = 0.0;
    let mut total = 0.0;

    for i in 0..NUM_BINS {
        let e = spd[i];
        if e <= 1e-10 {
            continue;
        }
        let (lr, lg, lb, k) = BIN_COMBINED_LUT[i];
        let e_mapped = 1.0 - (-k * e).exp();
        total += e_mapped;
        r += e_mapped * lr;
        g += e_mapped * lg;
        b += e_mapped * lb;
    }

    finalize_rgba(r, g, b, total, boosted)
}

// ── shared helpers ──────────────────────────────────────────────────────────

#[inline]
fn sat_boost_factor(color_range: f64, boosted: bool) -> f64 {
    if color_range < 0.1 {
        if boosted { 3.0 } else { 2.5 }
    } else if color_range < 0.3 {
        if boosted { 2.6 } else { 2.2 }
    } else if boosted {
        2.2
    } else {
        1.8
    }
}

#[inline]
fn finalize_rgba(
    mut r: f64,
    mut g: f64,
    mut b: f64,
    total: f64,
    boosted: bool,
) -> (f64, f64, f64, f64) {
    if total < 1e-10 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    r /= total;
    g /= total;
    b /= total;

    let mean = (r + g + b) / 3.0;
    let max_channel = r.max(g).max(b);
    let min_channel = r.min(g).min(b);
    let color_range = max_channel - min_channel;

    let sat_boost = sat_boost_factor(color_range, boosted);

    r = mean + (r - mean) * sat_boost;
    g = mean + (g - mean) * sat_boost;
    b = mean + (b - mean) * sat_boost;

    let max_value = r.max(g).max(b);
    if max_value > 1.0 {
        let scale = 1.0 / max_value;
        r *= scale;
        g *= scale;
        b *= scale;
    }

    r = r.clamp(0.0, 1.0);
    g = g.clamp(0.0, 1.0);
    b = b.clamp(0.0, 1.0);

    let brightness = 1.0 - (-total).exp();
    (r * brightness, g * brightness, b * brightness, brightness)
}

// ── x86_64 AVX2+FMA implementation ─────────────────────────────────────────

/// Fully vectorized 1 - exp(-x) for 4 f64 lanes using AVX2+FMA.
///
/// Uses Cody-Waite range reduction with a degree-7 Taylor polynomial,
/// keeping the entire computation in SIMD registers (no scalar fallback).
/// Input x must be non-negative; relative error < 2e-11 for x in [0, 700].
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn one_minus_exp_neg_avx2(x: std::arch::x86_64::__m256d) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::*;

    let x_safe = _mm256_min_pd(x, _mm256_set1_pd(700.0));
    let zero = _mm256_setzero_pd();
    let neg_x = _mm256_sub_pd(zero, x_safe);

    let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E);
    let ln2_hi = _mm256_set1_pd(6.931_471_803_691_238e-1);
    let ln2_lo = _mm256_set1_pd(1.908_214_929_270_585e-10);

    let t = _mm256_mul_pd(neg_x, log2_e);
    let n = _mm256_round_pd::<0x08>(t);
    let neg_n = _mm256_sub_pd(zero, n);
    let r = _mm256_fmadd_pd(neg_n, ln2_hi, neg_x);
    let r = _mm256_fmadd_pd(neg_n, ln2_lo, r);

    let c7 = _mm256_set1_pd(1.984_126_984_126_984e-4);
    let c6 = _mm256_set1_pd(1.388_888_888_888_889e-3);
    let c5 = _mm256_set1_pd(8.333_333_333_333_333e-3);
    let c4 = _mm256_set1_pd(4.166_666_666_666_666_4e-2);
    let c3 = _mm256_set1_pd(1.666_666_666_666_666_6e-1);
    let c2 = _mm256_set1_pd(0.5);
    let one = _mm256_set1_pd(1.0);

    let p = _mm256_fmadd_pd(c7, r, c6);
    let p = _mm256_fmadd_pd(p, r, c5);
    let p = _mm256_fmadd_pd(p, r, c4);
    let p = _mm256_fmadd_pd(p, r, c3);
    let p = _mm256_fmadd_pd(p, r, c2);
    let p = _mm256_fmadd_pd(p, r, one);
    let exp_r = _mm256_fmadd_pd(p, r, one);

    let n_i32 = _mm256_cvtpd_epi32(n);
    let n_i64 = _mm256_cvtepi32_epi64(n_i32);
    let pow2n = _mm256_castsi256_pd(_mm256_slli_epi64(
        _mm256_add_epi64(n_i64, _mm256_set1_epi64x(1023)),
        52,
    ));

    let exp_neg = _mm256_mul_pd(exp_r, pow2n);
    _mm256_sub_pd(one, exp_neg)
}

/// AVX2 SIMD implementation — fully vectorized inner loop (no scalar exp).
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn spd_to_rgba_avx2(spd: &[f64; NUM_BINS], boosted: bool) -> (f64, f64, f64, f64) {
    use std::arch::x86_64::*;

    let lut = &*SIMD_LUT;

    unsafe {
        let mut r_accum = _mm256_setzero_pd();
        let mut g_accum = _mm256_setzero_pd();
        let mut b_accum = _mm256_setzero_pd();
        let mut total_accum = _mm256_setzero_pd();
        let threshold = _mm256_set1_pd(1e-10);

        for chunk_start in (0..NUM_BINS).step_by(4) {
            let energy = _mm256_loadu_pd(&spd[chunk_start]);

            let r_lut = _mm256_loadu_pd(&lut.r[chunk_start]);
            let g_lut = _mm256_loadu_pd(&lut.g[chunk_start]);
            let b_lut = _mm256_loadu_pd(&lut.b[chunk_start]);
            let k_lut = _mm256_loadu_pd(&lut.k[chunk_start]);

            let kx = _mm256_mul_pd(k_lut, energy);
            let e_mapped_raw = one_minus_exp_neg_avx2(kx);
            let mask = _mm256_cmp_pd::<_CMP_GT_OQ>(energy, threshold);
            let e_mapped = _mm256_and_pd(e_mapped_raw, mask);

            r_accum = _mm256_fmadd_pd(e_mapped, r_lut, r_accum);
            g_accum = _mm256_fmadd_pd(e_mapped, g_lut, g_accum);
            b_accum = _mm256_fmadd_pd(e_mapped, b_lut, b_accum);
            total_accum = _mm256_add_pd(total_accum, e_mapped);
        }

        let mut r_vals = [0.0; 4];
        let mut g_vals = [0.0; 4];
        let mut b_vals = [0.0; 4];
        let mut total_vals = [0.0; 4];

        _mm256_storeu_pd(r_vals.as_mut_ptr(), r_accum);
        _mm256_storeu_pd(g_vals.as_mut_ptr(), g_accum);
        _mm256_storeu_pd(b_vals.as_mut_ptr(), b_accum);
        _mm256_storeu_pd(total_vals.as_mut_ptr(), total_accum);

        let r: f64 = r_vals.iter().sum();
        let g: f64 = g_vals.iter().sum();
        let b: f64 = b_vals.iter().sum();
        let total: f64 = total_vals.iter().sum();

        finalize_rgba(r, g, b, total, boosted)
    }
}

// ── x86_64 AVX-512 implementation ──────────────────────────────────────────

/// AVX-512 SIMD implementation — processes 8 bins per iteration (2 iterations for 16 bins).
///
/// On 64+ core x86 servers with AVX-512 (Skylake-X, Ice Lake, Sapphire Rapids),
/// this provides ~2x speedup over the AVX2 path by using 512-bit vectors.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn spd_to_rgba_avx512(spd: &[f64; NUM_BINS], boosted: bool) -> (f64, f64, f64, f64) {
    use std::arch::x86_64::*;

    let lut = &*SIMD_LUT;

    unsafe {
        let mut r_accum = _mm512_setzero_pd();
        let mut g_accum = _mm512_setzero_pd();
        let mut b_accum = _mm512_setzero_pd();
        let mut total_accum = _mm512_setzero_pd();
        let threshold = _mm512_set1_pd(1e-10);

        for chunk_start in (0..NUM_BINS).step_by(8) {
            let energy = _mm512_loadu_pd(&spd[chunk_start]);

            let r_lut = _mm512_loadu_pd(&lut.r[chunk_start]);
            let g_lut = _mm512_loadu_pd(&lut.g[chunk_start]);
            let b_lut = _mm512_loadu_pd(&lut.b[chunk_start]);
            let k_lut = _mm512_loadu_pd(&lut.k[chunk_start]);

            let kx = _mm512_mul_pd(k_lut, energy);
            let e_mapped_raw = one_minus_exp_neg_avx512(kx);
            let mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(energy, threshold);
            let e_mapped = _mm512_maskz_mov_pd(mask, e_mapped_raw);

            r_accum = _mm512_fmadd_pd(e_mapped, r_lut, r_accum);
            g_accum = _mm512_fmadd_pd(e_mapped, g_lut, g_accum);
            b_accum = _mm512_fmadd_pd(e_mapped, b_lut, b_accum);
            total_accum = _mm512_add_pd(total_accum, e_mapped);
        }

        let r = _mm512_reduce_add_pd(r_accum);
        let g = _mm512_reduce_add_pd(g_accum);
        let b = _mm512_reduce_add_pd(b_accum);
        let total = _mm512_reduce_add_pd(total_accum);

        finalize_rgba(r, g, b, total, boosted)
    }
}

/// Fully vectorized 1 - exp(-x) for 8 f64 lanes using AVX-512.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn one_minus_exp_neg_avx512(x: std::arch::x86_64::__m512d) -> std::arch::x86_64::__m512d {
    use std::arch::x86_64::*;

    let x_safe = _mm512_min_pd(x, _mm512_set1_pd(700.0));
    let zero = _mm512_setzero_pd();
    let neg_x = _mm512_sub_pd(zero, x_safe);

    let log2_e = _mm512_set1_pd(std::f64::consts::LOG2_E);
    let ln2_hi = _mm512_set1_pd(6.931_471_803_691_238e-1);
    let ln2_lo = _mm512_set1_pd(1.908_214_929_270_585e-10);

    let t = _mm512_mul_pd(neg_x, log2_e);
    let n = _mm512_roundscale_pd::<0x08>(t);
    let neg_n = _mm512_sub_pd(zero, n);
    let r = _mm512_fmadd_pd(neg_n, ln2_hi, neg_x);
    let r = _mm512_fmadd_pd(neg_n, ln2_lo, r);

    let c7 = _mm512_set1_pd(1.984_126_984_126_984e-4);
    let c6 = _mm512_set1_pd(1.388_888_888_888_889e-3);
    let c5 = _mm512_set1_pd(8.333_333_333_333_333e-3);
    let c4 = _mm512_set1_pd(4.166_666_666_666_666_4e-2);
    let c3 = _mm512_set1_pd(1.666_666_666_666_666_6e-1);
    let c2 = _mm512_set1_pd(0.5);
    let one = _mm512_set1_pd(1.0);

    let p = _mm512_fmadd_pd(c7, r, c6);
    let p = _mm512_fmadd_pd(p, r, c5);
    let p = _mm512_fmadd_pd(p, r, c4);
    let p = _mm512_fmadd_pd(p, r, c3);
    let p = _mm512_fmadd_pd(p, r, c2);
    let p = _mm512_fmadd_pd(p, r, one);
    let exp_r = _mm512_fmadd_pd(p, r, one);

    let n_i32 = _mm512_cvtpd_epi32(n);
    let n_i64 = _mm512_cvtepi32_epi64(n_i32);
    let bias = _mm512_set1_epi64(1023);
    let pow2n = _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_add_epi64(n_i64, bias), 52));

    let exp_neg = _mm512_mul_pd(exp_r, pow2n);
    _mm512_sub_pd(one, exp_neg)
}

// ── aarch64 NEON implementation ─────────────────────────────────────────────

/// aarch64 NEON SIMD implementation (~2x faster than scalar).
///
/// Processes 2 bins at a time using 128-bit NEON FMA. The exp() call remains
/// scalar (no hardware transcendental on NEON), but accumulation is fully vectorized.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[inline]
unsafe fn spd_to_rgba_neon(spd: &[f64; NUM_BINS], boosted: bool) -> (f64, f64, f64, f64) {
    use std::arch::aarch64::*;

    unsafe {
        let mut r_accum = vdupq_n_f64(0.0);
        let mut g_accum = vdupq_n_f64(0.0);
        let mut b_accum = vdupq_n_f64(0.0);
        let mut total_accum = vdupq_n_f64(0.0);

        for chunk_start in (0..NUM_BINS).step_by(2) {
            let lut0 = BIN_COMBINED_LUT[chunk_start];
            let lut1 = BIN_COMBINED_LUT[chunk_start + 1];

            let e0 = spd[chunk_start];
            let e1 = spd[chunk_start + 1];
            let em0 = if e0 > 1e-10 { 1.0 - (-lut0.3 * e0).exp() } else { 0.0 };
            let em1 = if e1 > 1e-10 { 1.0 - (-lut1.3 * e1).exp() } else { 0.0 };

            let e_data = [em0, em1];
            let r_data = [lut0.0, lut1.0];
            let g_data = [lut0.1, lut1.1];
            let b_data = [lut0.2, lut1.2];

            let e_mapped = vld1q_f64(e_data.as_ptr());
            let r_lut = vld1q_f64(r_data.as_ptr());
            let g_lut = vld1q_f64(g_data.as_ptr());
            let b_lut = vld1q_f64(b_data.as_ptr());

            r_accum = vfmaq_f64(r_accum, e_mapped, r_lut);
            g_accum = vfmaq_f64(g_accum, e_mapped, g_lut);
            b_accum = vfmaq_f64(b_accum, e_mapped, b_lut);
            total_accum = vaddq_f64(total_accum, e_mapped);
        }

        let r = vgetq_lane_f64::<0>(r_accum) + vgetq_lane_f64::<1>(r_accum);
        let g = vgetq_lane_f64::<0>(g_accum) + vgetq_lane_f64::<1>(g_accum);
        let b = vgetq_lane_f64::<0>(b_accum) + vgetq_lane_f64::<1>(b_accum);
        let total = vgetq_lane_f64::<0>(total_accum) + vgetq_lane_f64::<1>(total_accum);

        finalize_rgba(r, g, b, total, boosted)
    }
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────

    fn assert_simd_scalar_match(spd: &[f64; NUM_BINS], label: &str) {
        assert_simd_scalar_close(spd, 1e-10, label);
    }

    fn assert_simd_scalar_close(spd: &[f64; NUM_BINS], tol: f64, label: &str) {
        let scalar = spd_to_rgba_scalar(spd);
        let simd = spd_to_rgba_simd(spd);
        assert!(
            (scalar.0 - simd.0).abs() < tol,
            "{label}: R mismatch scalar={} simd={} (tol={tol})",
            scalar.0,
            simd.0
        );
        assert!(
            (scalar.1 - simd.1).abs() < tol,
            "{label}: G mismatch scalar={} simd={} (tol={tol})",
            scalar.1,
            simd.1
        );
        assert!(
            (scalar.2 - simd.2).abs() < tol,
            "{label}: B mismatch scalar={} simd={} (tol={tol})",
            scalar.2,
            simd.2
        );
        assert!(
            (scalar.3 - simd.3).abs() < tol,
            "{label}: A mismatch scalar={} simd={} (tol={tol})",
            scalar.3,
            simd.3
        );
    }

    fn assert_in_unit_range(val: f64, name: &str) {
        assert!((0.0..=1.0).contains(&val), "{name} out of [0,1]: {val}");
    }

    // ── runtime detection tests ─────────────────────────────────────

    #[test]
    fn test_detected_simd_path_name_is_non_empty() {
        let name = detected_simd_path_name();
        assert!(!name.is_empty());
        assert!(
            ["AVX-512", "AVX2+FMA", "NEON", "Scalar"].contains(&name),
            "unexpected SIMD path name: {name}"
        );
    }

    #[test]
    fn test_simd_dispatch_returns_valid_output() {
        let spd = [0.3, 0.0, 0.7, 0.1, 0.0, 0.5, 0.9, 0.2, 0.0, 0.4, 0.6, 0.0, 0.8, 0.1, 0.3, 0.0];
        let result = spd_to_rgba_simd(&spd);
        assert_in_unit_range(result.0, "dispatch.R");
        assert_in_unit_range(result.1, "dispatch.G");
        assert_in_unit_range(result.2, "dispatch.B");
        assert_in_unit_range(result.3, "dispatch.A");
    }

    // ── original tests (preserved) ──────────────────────────────────

    #[test]
    fn test_simd_matches_scalar() {
        let test_cases = vec![
            [1.0, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1; 16],
        ];
        for spd in test_cases {
            assert_simd_scalar_match(&spd, "basic");
        }
    }

    #[test]
    fn test_simd_zero_input() {
        let spd = [0.0; NUM_BINS];
        let result = spd_to_rgba_simd(&spd);
        assert_eq!(result, (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_simd_single_peak() {
        let mut spd = [0.0; NUM_BINS];
        spd[8] = 1.0;
        let result = spd_to_rgba_simd(&spd);
        assert!(result.3 > 0.0, "Should have non-zero alpha");
    }

    #[test]
    fn test_sat_boost_toggle_changes_output() {
        let mut spd = [0.0; NUM_BINS];
        spd[4] = 0.5;
        spd[10] = 0.3;

        let boosted = spd_to_rgba_scalar_with_sat_boost(&spd, true);
        let original = spd_to_rgba_scalar_with_sat_boost(&spd, false);

        let sat = |r: f64, g: f64, b: f64| {
            let mx = r.max(g).max(b);
            let mn = r.min(g).min(b);
            if mx > 0.0 { (mx - mn) / mx } else { 0.0 }
        };
        assert!(sat(boosted.0, boosted.1, boosted.2) >= sat(original.0, original.1, original.2));
    }

    #[test]
    fn test_output_values_clamped() {
        let spd = [10.0; NUM_BINS];
        let result = spd_to_rgba_simd(&spd);
        assert_in_unit_range(result.0, "R");
        assert_in_unit_range(result.1, "G");
        assert_in_unit_range(result.2, "B");
        assert_in_unit_range(result.3, "A");
    }

    #[test]
    fn test_brightness_increases_with_energy() {
        let mut low = [0.0; NUM_BINS];
        low[8] = 0.1;
        let mut high = [0.0; NUM_BINS];
        high[8] = 5.0;
        assert!(spd_to_rgba_simd(&high).3 > spd_to_rgba_simd(&low).3);
    }

    // ── cross-platform parity tests ─────────────────────────────────

    #[test]
    fn test_simd_scalar_parity_exhaustive() {
        let cases: Vec<[f64; NUM_BINS]> = vec![
            [0.0; NUM_BINS],
            [1.0; NUM_BINS],
            [0.001; NUM_BINS],
            [100.0; NUM_BINS],
            {
                let mut s = [0.0; 16];
                s[0] = 1.0;
                s
            },
            {
                let mut s = [0.0; 16];
                s[7] = 1.0;
                s
            },
            {
                let mut s = [0.0; 16];
                s[15] = 1.0;
                s
            },
            {
                let mut s = [0.0; 16];
                s[0] = 0.8;
                s[14] = 0.8;
                s
            },
            {
                let mut s = [0.0; 16];
                for (i, value) in s.iter_mut().enumerate() {
                    *value = (16 - i) as f64 / 16.0;
                }
                s
            },
            {
                let mut s = [0.0; 16];
                for (i, value) in s.iter_mut().enumerate() {
                    *value = (i + 1) as f64 / 16.0;
                }
                s
            },
            {
                let mut s = [0.0; 16];
                for i in (0..16).step_by(2) {
                    s[i] = 0.5;
                }
                s
            },
            {
                let mut s = [0.01; 16];
                s[5] = 500.0;
                s
            },
            [0.0, 0.0, 0.1, 0.4, 0.9, 1.0, 0.7, 0.3, 0.1, 0.2, 0.6, 0.8, 0.5, 0.1, 0.0, 0.0],
        ];

        for (i, spd) in cases.iter().enumerate() {
            assert_simd_scalar_match(spd, &format!("case[{i}]"));
        }
    }

    #[test]
    fn test_per_bin_isolation() {
        for bin in 0..NUM_BINS {
            let mut spd = [0.0; NUM_BINS];
            spd[bin] = 1.0;

            let result = spd_to_rgba_simd(&spd);
            assert!(result.3 > 0.0, "bin {bin} should produce non-zero alpha");
            assert_in_unit_range(result.0, &format!("bin{bin}.R"));
            assert_in_unit_range(result.1, &format!("bin{bin}.G"));
            assert_in_unit_range(result.2, &format!("bin{bin}.B"));
            assert_in_unit_range(result.3, &format!("bin{bin}.A"));
        }
    }

    #[test]
    fn test_simd_deterministic() {
        let spd = [0.3, 0.0, 0.7, 0.1, 0.0, 0.5, 0.9, 0.2, 0.0, 0.4, 0.6, 0.0, 0.8, 0.1, 0.3, 0.0];
        let reference = spd_to_rgba_simd_with_sat_boost(&spd, true);
        for _ in 0..200 {
            let r = spd_to_rgba_simd_with_sat_boost(&spd, true);
            assert_eq!(r.0.to_bits(), reference.0.to_bits(), "R non-deterministic");
            assert_eq!(r.1.to_bits(), reference.1.to_bits(), "G non-deterministic");
            assert_eq!(r.2.to_bits(), reference.2.to_bits(), "B non-deterministic");
            assert_eq!(r.3.to_bits(), reference.3.to_bits(), "A non-deterministic");
        }
    }

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    #[test]
    fn test_avx2_kernel_is_bitwise_deterministic() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let spd = [0.3, 0.0, 0.7, 0.1, 0.0, 0.5, 0.9, 0.2, 0.0, 0.4, 0.6, 0.0, 0.8, 0.1, 0.3, 0.0];
        let reference = unsafe { spd_to_rgba_avx2(&spd, true) };
        for _ in 0..200 {
            let r = unsafe { spd_to_rgba_avx2(&spd, true) };
            assert_eq!(r.0.to_bits(), reference.0.to_bits(), "AVX2 R non-deterministic");
            assert_eq!(r.1.to_bits(), reference.1.to_bits(), "AVX2 G non-deterministic");
            assert_eq!(r.2.to_bits(), reference.2.to_bits(), "AVX2 B non-deterministic");
            assert_eq!(r.3.to_bits(), reference.3.to_bits(), "AVX2 A non-deterministic");
        }
    }

    #[cfg(all(target_arch = "x86_64", not(miri)))]
    #[test]
    fn test_avx2_vectorized_exp_accuracy() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use std::arch::x86_64::*;
        let test_inputs: &[f64] = &[
            0.0, 1e-15, 1e-10, 1e-5, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
            500.0, 700.0,
        ];
        for &x in test_inputs {
            let expected = 1.0 - (-x).exp();
            let result = unsafe {
                let xv = _mm256_set1_pd(x);
                let rv = one_minus_exp_neg_avx2(xv);
                let mut out = [0.0; 4];
                _mm256_storeu_pd(out.as_mut_ptr(), rv);
                out[0]
            };
            let abs_err = (result - expected).abs();
            let rel_err = if expected.abs() > 1e-15 { abs_err / expected.abs() } else { abs_err };
            assert!(
                rel_err < 1e-10 || abs_err < 1e-15,
                "vectorized exp error for x={x}: got={result} expected={expected} rel={rel_err:.2e}"
            );
        }
    }

    #[cfg(all(target_arch = "aarch64", not(miri)))]
    #[test]
    fn test_neon_kernel_is_bitwise_deterministic() {
        let spd = [0.3, 0.0, 0.7, 0.1, 0.0, 0.5, 0.9, 0.2, 0.0, 0.4, 0.6, 0.0, 0.8, 0.1, 0.3, 0.0];
        let reference = unsafe { spd_to_rgba_neon(&spd, true) };
        for _ in 0..200 {
            let r = unsafe { spd_to_rgba_neon(&spd, true) };
            assert_eq!(r.0.to_bits(), reference.0.to_bits(), "NEON R non-deterministic");
            assert_eq!(r.1.to_bits(), reference.1.to_bits(), "NEON G non-deterministic");
            assert_eq!(r.2.to_bits(), reference.2.to_bits(), "NEON B non-deterministic");
            assert_eq!(r.3.to_bits(), reference.3.to_bits(), "NEON A non-deterministic");
        }
    }

    #[test]
    fn test_extreme_energy_values() {
        let tiny_cases: &[f64] = &[1e-15, 1e-11, 1e-10, 5e-11];
        for &val in tiny_cases {
            let spd = [val; NUM_BINS];
            let result = spd_to_rgba_simd(&spd);
            assert_in_unit_range(result.0, "tiny.R");
            assert_in_unit_range(result.3, "tiny.A");
            assert_simd_scalar_match(&spd, &format!("tiny({val})"));
        }

        let huge_cases: &[f64] = &[50.0, 500.0, 5000.0, 1e6];
        for &val in huge_cases {
            let spd = [val; NUM_BINS];
            let result = spd_to_rgba_simd(&spd);
            assert_in_unit_range(result.0, "huge.R");
            assert_in_unit_range(result.3, "huge.A");
            assert_simd_scalar_match(&spd, &format!("huge({val})"));
        }
    }

    #[test]
    fn test_near_zero_threshold() {
        let below = [9e-11; NUM_BINS];
        let above = [2e-10; NUM_BINS];

        let rb = spd_to_rgba_simd(&below);
        let ra = spd_to_rgba_simd(&above);

        assert_simd_scalar_match(&below, "below_threshold");
        assert_simd_scalar_match(&above, "above_threshold");

        assert!(ra.3 >= rb.3, "above threshold should be >= below threshold in brightness");
    }

    #[test]
    fn test_uniform_spectrum_is_near_neutral() {
        let spd = [1.0; NUM_BINS];
        let result = spd_to_rgba_simd(&spd);

        let max_c = result.0.max(result.1).max(result.2);
        let min_c = result.0.min(result.1).min(result.2);
        let chroma = if max_c > 0.0 { (max_c - min_c) / max_c } else { 0.0 };

        assert!(chroma < 0.65, "uniform SPD should be relatively neutral, got chroma={chroma:.4}");
    }

    #[test]
    fn test_energy_scaling_preserves_hue() {
        let mut spd_low = [0.0; NUM_BINS];
        let mut spd_high = [0.0; NUM_BINS];
        spd_low[3] = 0.5;
        spd_low[4] = 1.0;
        spd_high[3] = 2.5;
        spd_high[4] = 5.0;

        let lo = spd_to_rgba_simd(&spd_low);
        let hi = spd_to_rgba_simd(&spd_high);

        let hue = |r: f64, g: f64, b: f64| -> f64 {
            let mx = r.max(g).max(b);
            if mx < 1e-12 {
                return 0.0;
            }
            (r / mx, g / mx, b / mx).0
        };

        let hue_lo = hue(lo.0, lo.1, lo.2);
        let hue_hi = hue(hi.0, hi.1, hi.2);
        assert!(
            (hue_lo - hue_hi).abs() < 0.15,
            "scaling energy should roughly preserve hue: lo={hue_lo:.4} hi={hue_hi:.4}"
        );
    }

    #[test]
    fn test_sat_boost_affects_output() {
        let spd = [0.0, 0.2, 0.5, 0.8, 1.0, 0.6, 0.3, 0.0, 0.0, 0.1, 0.4, 0.7, 0.9, 0.5, 0.2, 0.0];

        let boosted = spd_to_rgba_simd_with_sat_boost(&spd, true);
        let unboosted = spd_to_rgba_simd_with_sat_boost(&spd, false);

        assert!(boosted != unboosted, "sat_boost toggle should produce different output");
        assert_in_unit_range(boosted.0, "boosted.R");
        assert_in_unit_range(unboosted.0, "unboosted.R");
    }

    #[test]
    fn test_monotonic_brightness_gradient() {
        let energies = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        let mut prev_alpha = 0.0_f64;
        for &e in &energies {
            let mut spd = [0.0; NUM_BINS];
            spd[8] = e;
            let result = spd_to_rgba_simd(&spd);
            assert!(
                result.3 >= prev_alpha,
                "brightness should be monotonic: e={e} alpha={} prev={prev_alpha}",
                result.3
            );
            prev_alpha = result.3;
        }
    }

    #[test]
    fn test_output_always_valid_for_random_like_inputs() {
        let mut seed = 12345u64;
        for _ in 0..50 {
            let mut spd = [0.0; NUM_BINS];
            for v in spd.iter_mut() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *v = (seed % 10000) as f64 / 1000.0;
            }
            let r = spd_to_rgba_simd(&spd);
            assert_in_unit_range(r.0, "rand.R");
            assert_in_unit_range(r.1, "rand.G");
            assert_in_unit_range(r.2, "rand.B");
            assert_in_unit_range(r.3, "rand.A");
        }
    }

    // ── stress tests ────────────────────────────────────────────────

    #[test]
    fn test_stress_10k_random_inputs_all_valid() {
        let mut seed = 99887766u64;
        for _ in 0..10_000 {
            let mut spd = [0.0; NUM_BINS];
            for v in spd.iter_mut() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *v = (seed % 100_000) as f64 / 100.0;
            }
            let r = spd_to_rgba_simd(&spd);
            assert!(r.0 >= 0.0 && r.0 <= 1.0, "R={} out of range", r.0);
            assert!(r.1 >= 0.0 && r.1 <= 1.0, "G={} out of range", r.1);
            assert!(r.2 >= 0.0 && r.2 <= 1.0, "B={} out of range", r.2);
            assert!(r.3 >= 0.0 && r.3 <= 1.0, "A={} out of range", r.3);
        }
    }

    #[test]
    fn test_sat_boost_boundary_straddling() {
        for &cr in &[0.099, 0.100, 0.101, 0.299, 0.300, 0.301] {
            let factor_on = sat_boost_factor(cr, true);
            let factor_off = sat_boost_factor(cr, false);
            assert!(factor_on > 0.0, "boost factor should be positive for cr={cr}");
            assert!(factor_off > 0.0, "boost factor should be positive for cr={cr}");
            assert!(factor_on >= factor_off, "boosted should be >= unboosted for cr={cr}");
        }
    }

    #[test]
    fn test_finalize_rgba_returns_zero_for_tiny_total() {
        let result = finalize_rgba(0.5, 0.3, 0.2, 1e-11, true);
        assert_eq!(result, (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_negative_energy_treated_as_zero() {
        let spd = [-1.0; NUM_BINS];
        let result = spd_to_rgba_simd(&spd);
        assert_eq!(result, (0.0, 0.0, 0.0, 0.0), "negative energy should produce black");
    }
}
