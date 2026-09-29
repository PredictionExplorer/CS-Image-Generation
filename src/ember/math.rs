//! Portable transcendental functions and comparison primitives.
//!
//! # Transcendental functions
//!
//! The standard library's `f64::exp`, `f64::sin`, ... call the platform's C math library, whose
//! results differ in the last bits between glibc, Apple's libm and MSVC, and between their
//! FMA/non-FMA code paths. The chaotic fluid of the ember edition would turn such a one-ULP
//! difference into a visibly different picture, so every transcendental function in
//! `crate::ember` goes through this facade over the pure-Rust [`libm`] crate (a port of musl).
//! Its only architecture-specific code paths are `sqrt`, `fma` and `rint`, which IEEE-754 defines
//! exactly, so these functions return the same bits on every CPU.
//!
//! The dependency is pinned (`libm = "=0.2.16"`): a version bump may change results in the last
//! bit and must go through the golden-hash tests. A unit test pins the bits of every wrapper and
//! scans the ember sources for calls that bypass the facade (docs/ember-design.md §0).
//!
//! # Comparison primitives
//!
//! [`min`], [`max`] and [`clamp_unit`] are single comparisons with a fixed operand order.
//! `f64::min` and `f64::max` leave the choice between `+0` and `-0` unspecified, and
//! `f64::clamp` panics on reversed bounds; these helpers return the same bits on every CPU for
//! every input (NaN included), so the ember modules use them wherever a result is stored or
//! hashed.

/// `e^x`.
#[inline]
pub(crate) fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// Natural logarithm.
#[inline]
pub(crate) fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// `x^y`.
#[inline]
pub(crate) fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// Real cube root.
#[inline]
pub(crate) fn cbrt(x: f64) -> f64 {
    libm::cbrt(x)
}

/// Hyperbolic tangent.
#[inline]
pub(crate) fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}

/// Sine (radians).
#[inline]
pub(crate) fn sin(x: f64) -> f64 {
    libm::sin(x)
}

/// Cosine (radians). Production code uses [`sin_cos`]; tests use this as an independent oracle.
#[cfg(test)]
#[inline]
pub(crate) fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// `(sin x, cos x)` (radians), computed together.
#[inline]
pub(crate) fn sin_cos(x: f64) -> (f64, f64) {
    libm::sincos(x)
}

/// The smaller of `a` and `b`: `b` if `b < a`, otherwise `a`.
///
/// Ties (including `+0` against `-0`) return `a`, and so does a NaN `b`; a NaN `a` is returned
/// unchanged. Callers pass non-NaN values.
#[inline(always)]
pub(crate) fn min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// The larger of `a` and `b`: `b` if `b > a`, otherwise `a`.
///
/// Ties (including `+0` against `-0`) return `a`, and so does a NaN `b`; a NaN `a` is returned
/// unchanged. Callers pass non-NaN values.
#[inline(always)]
pub(crate) fn max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// `x` clamped to `[0, 1]`: `x` inside the open interval, `1` from `1` up, and `+0` otherwise.
/// NaN and `-0` both map to `+0`, so the result is never NaN and never a negative zero.
#[inline(always)]
pub(crate) fn clamp_unit(x: f64) -> f64 {
    if x > 0.0 { if x < 1.0 { x } else { 1.0 } } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Code fragments that reach the platform math library (or fuse operations, or dispatch on the
    /// CPU) and therefore differ across architectures. None of them may appear in ember code.
    /// Path forms `f32::…`/`f64::…` are checked against [`ALLOWED_FLOAT_ITEMS`] instead.
    const FORBIDDEN: &[&str] = &[
        ".exp(",
        ".exp2(",
        ".exp_m1(",
        ".ln(",
        ".ln_1p(",
        ".log(",
        ".log2(",
        ".log10(",
        ".powf(",
        ".powi(",
        ".sin(",
        ".cos(",
        ".tan(",
        ".sin_cos(",
        ".sinh(",
        ".cosh(",
        ".tanh(",
        ".asin(",
        ".acos(",
        ".atan(",
        ".atan2(",
        ".asinh(",
        ".acosh(",
        ".atanh(",
        ".cbrt(",
        ".hypot(",
        ".mul_add(",
        ".gamma(",
        ".ln_gamma(",
        // Trait routes to the same functions (num-traits `Float`/`Real`, nalgebra/simba
        // `RealField`/`ComplexField`/`simd_*`), plain or fully qualified (`<f64 as Float>::`).
        "Float::",
        "Float>::",
        "Real::",
        "Real>::",
        "Field::",
        "Field>::",
        "simd_",
        // Architecture-specific code.
        "core::arch",
        "std::arch",
        "target_feature",
        "rustfft",
    ];

    /// Code fragments allowed only in this file: everything else reaches `libm` through the
    /// facade, so the pinned-bit test below covers every function the ember edition uses.
    const FACADE_ONLY: &[&str] = &["libm::"];

    /// The associated items of `f32`/`f64` that ember code may name: constants, conversions,
    /// ordering and exactly rounded operations. Any other `f32::name`/`f64::name` (`f64::atan2`,
    /// `f32::tanh`, `f64::mul_add`, ...) fails the guard, so a new transcendental cannot slip
    /// past a list of known names. (`max`/`min` are exact but leave the sign of a zero tie
    /// unspecified; production code uses [`max`]/[`min`], tests may fold with them.)
    const ALLOWED_FLOAT_ITEMS: &[&str] = &[
        "abs",
        "ceil",
        "consts",
        "floor",
        "from",
        "from_bits",
        "is_finite",
        "is_nan",
        "max",
        "min",
        "round",
        "sqrt",
        "to_bits",
        "total_cmp",
        "trunc",
        "EPSILON",
        "INFINITY",
        "MAX",
        "MIN",
        "MIN_POSITIVE",
        "NAN",
        "NEG_INFINITY",
    ];

    /// Every other ember source file (path relative to `src/ember`), embedded at compile time so
    /// the guard cannot skip one.
    const SOURCES: &[(&str, &str)] = &[
        ("certificate.rs", include_str!("certificate.rs")),
        ("config.rs", include_str!("config.rs")),
        ("error.rs", include_str!("error.rs")),
        ("fft.rs", include_str!("fft.rs")),
        ("fluid.rs", include_str!("fluid.rs")),
        ("ink.rs", include_str!("ink.rs")),
        ("look.rs", include_str!("look.rs")),
        ("mod.rs", include_str!("mod.rs")),
        ("optics.rs", include_str!("optics.rs")),
        ("orbit.rs", include_str!("orbit.rs")),
        ("paper.rs", include_str!("paper.rs")),
        ("pipeline.rs", include_str!("pipeline.rs")),
        ("trace.rs", include_str!("trace.rs")),
    ];

    /// This file.
    const OWN: &str = include_str!("math.rs");

    /// Strips `//` comments (outside string literals is good enough for this codebase's style).
    fn code_part(line: &str) -> &str {
        line.split("//").next().unwrap_or("")
    }

    /// The lines of this file before its test module: the whole facade, including the test-only
    /// `cos` oracle (whose `#[cfg(test)]` must not end the scan early). Nothing but the test
    /// module follows it (clippy's `items_after_test_module`).
    ///
    /// # Panics
    ///
    /// Unless the file has exactly one `#[cfg(test)] mod tests {` (so a renamed test module
    /// cannot silently switch the scan off).
    fn own_non_test_lines() -> Vec<&'static str> {
        let lines: Vec<&str> = OWN.lines().collect();
        let starts: Vec<usize> = (1..lines.len())
            .filter(|&i| {
                lines[i].trim_end() == "mod tests {" && lines[i - 1].trim() == "#[cfg(test)]"
            })
            .collect();
        assert_eq!(starts.len(), 1, "math.rs must have exactly one `#[cfg(test)] mod tests`");
        lines[..starts[0] - 1].to_vec()
    }

    /// The `f32::`/`f64::` associated items named on `code` that are not allow-listed.
    fn disallowed_float_items(code: &str) -> Vec<String> {
        let mut found = Vec::new();
        for prefix in ["f32::", "f64::"] {
            for (at, _) in code.match_indices(prefix) {
                let rest = &code[at + prefix.len()..];
                let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'));
                let item = &rest[..end.unwrap_or(rest.len())];
                if !ALLOWED_FLOAT_ITEMS.contains(&item) {
                    found.push(format!("{prefix}{item}"));
                }
            }
        }
        found
    }

    /// Asserts that one line of ember code uses only portable math (`facade` = the line belongs to
    /// this file's non-test part, which may call `libm` directly).
    fn assert_portable(name: &str, line_no: usize, line: &str, facade: bool) {
        let code = code_part(line);
        let forbidden = FORBIDDEN.iter().chain(if facade { &[][..] } else { FACADE_ONLY });
        for pattern in forbidden {
            assert!(!code.contains(pattern), "src/ember/{name}:{line_no} uses `{pattern}`: {line}");
        }
        let items = disallowed_float_items(code);
        assert!(
            items.is_empty(),
            "src/ember/{name}:{line_no} names {items:?}, which is not in ALLOWED_FLOAT_ITEMS \
             (add it there only if it is exactly rounded on every CPU): {line}"
        );
    }

    #[test]
    fn ember_sources_use_only_portable_math() {
        for (name, source) in SOURCES {
            for (line_no, line) in source.lines().enumerate() {
                assert_portable(name, line_no + 1, line, false);
            }
        }
        let own = own_non_test_lines();
        for (line_no, line) in own.iter().enumerate() {
            assert_portable("math.rs", line_no + 1, line, true);
        }
    }

    /// The scan of this file reaches every production wrapper (it once stopped at the test-only
    /// `cos` and never saw `sin_cos`), and the guard catches the forms it is meant to catch.
    #[test]
    fn the_guard_scans_the_whole_facade_and_catches_bypasses() {
        let own = own_non_test_lines().join("\n");
        for wrapper in [
            "fn exp(",
            "fn ln(",
            "fn pow(",
            "fn cbrt(",
            "fn tanh(",
            "fn sin(",
            "fn cos(",
            "fn sin_cos(",
            "fn min(",
            "fn max(",
            "fn clamp_unit(",
        ] {
            assert!(own.contains(wrapper), "the math.rs scan misses `{wrapper}`");
        }
        assert!(!own.contains("mod tests"), "the math.rs scan includes the test module");
        let caught = |line: &str| {
            let code = code_part(line);
            FORBIDDEN.iter().chain(FACADE_ONLY).any(|p| code.contains(p))
                || !disallowed_float_items(code).is_empty()
        };
        for bypass in [
            "    x.sin_cos()",
            "    let a = f64::atan2(y, x);",
            "    let t = f32::tanh(x);",
            "    let (s, c) = libm::sincos(x);",
            "    let e = x.simd_exp();",
            "    let r = <f64 as Float>::sin(x);",
            "    let r = num_traits::Float::exp(x);",
            "    let r = <f64 as RealField>::atan2(y, x);",
            "    let r = ComplexField::cos(x);",
            "    let y = f64::mul_add(a, b, c);",
            "    #[cfg(target_feature = \"fma\")]",
        ] {
            assert!(caught(bypass), "the guard misses `{bypass}`");
        }
        for fine in [
            "    let x = f64::from(v) * f64::consts::PI + f64::EPSILON;",
            "    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);",
            "    let s = math::sin(x); // not x.sin()",
            "    let y = f32::from_bits(bits).to_bits();",
        ] {
            assert!(!caught(fine), "the guard rejects portable code `{fine}`");
        }
    }

    #[test]
    fn every_ember_source_file_is_guarded() {
        /// Every `.rs` file below `dir`, as a `/`-separated path relative to `root`.
        fn rust_files(root: &std::path::Path, dir: &std::path::Path, out: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).expect("readable source directory") {
                let path = entry.expect("readable directory entry").path();
                if path.is_dir() {
                    rust_files(root, &path, out);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    let relative = path.strip_prefix(root).expect("below the root");
                    let parts: Vec<_> =
                        relative.components().map(|c| c.as_os_str().to_string_lossy()).collect();
                    out.push(parts.join("/"));
                }
            }
        }
        let root = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ember"));
        let mut on_disk = Vec::new();
        rust_files(root, root, &mut on_disk);
        on_disk.sort();
        let mut guarded: Vec<String> =
            SOURCES.iter().map(|(name, _)| (*name).to_string()).collect();
        guarded.push("math.rs".to_string());
        guarded.sort();
        assert_eq!(on_disk, guarded, "add every new src/ember/**/*.rs file to SOURCES");
    }

    #[test]
    fn libm_reproduces_exact_reference_values() {
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(ln(1.0), 0.0);
        assert_eq!(pow(2.0, 10.0), 1024.0);
        assert_eq!(cbrt(27.0), 3.0);
        assert_eq!(tanh(0.0), 0.0);
        assert_eq!(sin_cos(0.0), (0.0, 1.0));
        assert!((exp(1.0) - std::f64::consts::E).abs() <= 2.0 * f64::EPSILON);
        assert!((sin(std::f64::consts::FRAC_PI_6) - 0.5).abs() <= 2.0 * f64::EPSILON);
        assert!((cos(std::f64::consts::FRAC_PI_3) - 0.5).abs() <= 2.0 * f64::EPSILON);
    }

    /// Bit patterns of libm 0.2.16 results. They are the same on every architecture; a change
    /// here means the pinned math library changed and every golden hash must be re-blessed.
    /// The `sin_cos` probes cover its three argument-reduction paths: none (`|x| < π/4`), the
    /// medium Cody–Waite reduction (`-7.25`, `10⁶`) and the large Payne–Hanek one (`10²²`).
    #[test]
    fn libm_bits_are_pinned() {
        let trig = [0.5, -7.25, 1.0e6, 1.0e22].map(sin_cos);
        let probes = [
            exp(-0.123_456_789).to_bits(),
            ln(std::f64::consts::PI).to_bits(),
            pow(0.731_058_578_630_004_9, 1.0 / 2.4).to_bits(),
            cbrt(0.008_856_451_679_035_631).to_bits(),
            tanh(0.577_215_664_901_532_9).to_bits(),
            sin(1.0e6).to_bits(),
            cos(-7.25).to_bits(),
            trig[0].0.to_bits(),
            trig[0].1.to_bits(),
            trig[1].0.to_bits(),
            trig[1].1.to_bits(),
            trig[2].0.to_bits(),
            trig[2].1.to_bits(),
            trig[3].0.to_bits(),
            trig[3].1.to_bits(),
        ];
        assert_eq!(probes, PINNED_BITS, "got {probes:#018x?}");
    }

    /// See [`libm_bits_are_pinned`].
    const PINNED_BITS: [u64; 15] = [
        0x3fec_4894_6a8f_cf96,
        0x3ff2_50d0_48e7_a1bd,
        0x3fec_1593_c2ef_1412,
        0x3fca_7b96_11a7_b961,
        0x3fe0_a912_a4c3_bfd9,
        0xbfd6_664b_2568_d867,
        0x3fe2_2c6f_50dc_3fbe,
        // sin_cos(0.5)
        0x3fde_aee8_744b_05f0,
        0x3fec_1528_065b_7d50,
        // sin_cos(-7.25)
        0xbfea_56ad_b62a_27b9,
        0x3fe2_2c6f_50dc_3fbe,
        // sin_cos(1e6)
        0xbfd6_664b_2568_d867,
        0x3fed_f9df_9906_d32c,
        // sin_cos(1e22)
        0xbfeb_453a_b76b_f397,
        0x3fe0_be2c_ef01_c8f4,
    ];

    /// musl's `sincos` shares the argument reduction and the kernels of `sin` and `cos`, so the
    /// production `sin_cos` agrees bit for bit with the pinned `sin` and the `cos` oracle.
    #[test]
    fn sin_cos_agrees_with_sin_and_cos_bit_for_bit() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        for exponent in -30..=70 {
            for _ in 0..40 {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let mantissa = (state >> 11) as f64 / (1u64 << 53) as f64;
                let x = (mantissa - 0.5) * pow(2.0, f64::from(exponent));
                let (s, c) = sin_cos(x);
                assert_eq!((s.to_bits(), c.to_bits()), (sin(x).to_bits(), cos(x).to_bits()), "{x}");
            }
        }
    }

    #[test]
    fn comparison_primitives_have_fixed_operand_order() {
        assert_eq!(min(1.0, 2.0), 1.0);
        assert_eq!(min(2.0, 1.0), 1.0);
        assert_eq!(max(1.0, 2.0), 2.0);
        assert_eq!(max(2.0, 1.0), 2.0);
        // Ties return the first operand, so the sign of a zero tie is defined.
        assert_eq!(min(0.0, -0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(min(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(max(0.0, -0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(max(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
        // A NaN second operand is ignored.
        assert_eq!(min(3.0, f64::NAN), 3.0);
        assert_eq!(max(3.0, f64::NAN), 3.0);
        assert_eq!(max(f64::NEG_INFINITY, 5.0), 5.0);
        for (x, want) in [
            (0.25, 0.25),
            (1.0, 1.0),
            (7.0, 1.0),
            (f64::INFINITY, 1.0),
            (0.0, 0.0),
            (-0.0, 0.0),
            (-3.0, 0.0),
            (f64::NAN, 0.0),
            (f64::MIN_POSITIVE, f64::MIN_POSITIVE),
        ] {
            assert_eq!(clamp_unit(x).to_bits(), want.to_bits(), "clamp_unit({x})");
        }
    }
}
