//! Exact finite-harmonic box filtering of the complete engraved material.
//!
//! The common carrier and registration phase stay separate until each Fourier
//! term is formed. In particular, the product's zero-carrier terms never lose
//! their slowly varying registration phase to subtraction of two fast phases.

use std::f64::consts::{PI, TAU};

const ORDER: usize = 8;
const MAX_HARMONIC: usize = 2 * ORDER;
const PLUS: f64 = 0.38;
const MINUS: f64 = 0.30;
const PRODUCT: f64 = 1.20;
// binomial(16, 8 - n) / 4^8, n = 0..8. All coefficients are exact binary fractions.
const COEFFICIENTS: [f64; ORDER + 1] = [
    12870.0 / 65536.0,
    11440.0 / 65536.0,
    8008.0 / 65536.0,
    4368.0 / 65536.0,
    1820.0 / 65536.0,
    560.0 / 65536.0,
    120.0 / 65536.0,
    16.0 / 65536.0,
    1.0 / 65536.0,
];
const DC: f64 = (PLUS + MINUS) * COEFFICIENTS[0] + PRODUCT * COEFFICIENTS[0] * COEFFICIENTS[0];

/// An affine phase in cycles over one centered spatial and temporal box.
#[derive(Clone, Copy, Debug)]
pub(super) struct PhaseFootprint {
    /// Phase at the center of the box.
    pub value: f64,
    /// Signed total phase excursions across the box's x, y and time axes.
    pub delta: [f64; 3],
}

/// The raw linear-material average, before envelopes, lighting or color.
#[derive(Clone, Copy, Debug)]
pub(super) struct FilteredPattern {
    /// Mathematically in [0, 1.88]; allow 1e-12 for floating-point roundoff.
    /// No clamping hides non-finite inputs or larger numerical errors.
    pub value: f64,
}

#[derive(Clone, Copy)]
struct Term {
    carrier: usize,
    half_beat: isize,
    coefficient: f64,
}

// Retain all 17*17 product terms: one DC term and 144 conjugate pairs.
// The single-family harmonics already occur in this table, so their complete
// coefficients can be added to those pairs without a second evaluation.
const TERM_COUNT: usize = ((2 * ORDER + 1) * (2 * ORDER + 1) - 1) / 2;
const TERMS: [Term; TERM_COUNT] = material_terms();

const fn material_terms() -> [Term; TERM_COUNT] {
    let mut terms = [Term { carrier: 0, half_beat: 0, coefficient: 0.0 }; TERM_COUNT];
    let mut index = 0;
    let mut carrier = 0;
    while carrier <= MAX_HARMONIC {
        let extent = (MAX_HARMONIC - carrier) as isize;
        let mut half_beat = -extent;
        while half_beat <= extent {
            // Keep exactly one of (n,k) and (-n,-k), omitting their shared DC.
            if carrier > 0 || half_beat > 0 {
                let n = isize::midpoint(carrier as isize, half_beat);
                let k = isize::midpoint(carrier as isize, -half_beat);
                let mut coefficient =
                    2.0 * PRODUCT * COEFFICIENTS[n.unsigned_abs()] * COEFFICIENTS[k.unsigned_abs()];
                if carrier <= ORDER && half_beat == carrier as isize {
                    coefficient += 2.0 * PLUS * COEFFICIENTS[carrier];
                }
                if carrier <= ORDER && half_beat == -(carrier as isize) {
                    coefficient += 2.0 * MINUS * COEFFICIENTS[carrier];
                }
                terms[index] = Term { carrier, half_beat, coefficient };
                index += 1;
            }
            half_beat += 2;
        }
        carrier += 1;
    }
    assert!(index == TERM_COUNT);
    terms
}

struct Harmonics {
    cosine: [f64; MAX_HARMONIC + 1],
    sine: [f64; MAX_HARMONIC + 1],
}
impl Harmonics {
    fn new(angle: f64) -> Self {
        let (sine, cosine) = angle.sin_cos();
        let mut result = Self { cosine: [1.0; MAX_HARMONIC + 1], sine: [0.0; MAX_HARMONIC + 1] };
        for n in 1..=MAX_HARMONIC {
            result.cosine[n] = result.cosine[n - 1] * cosine - result.sine[n - 1] * sine;
            result.sine[n] = result.sine[n - 1] * cosine + result.cosine[n - 1] * sine;
        }
        result
    }
}

struct AxisFilter {
    carrier: Harmonics,
    half_beat: Harmonics,
    carrier_delta: f64,
    half_beat_delta: f64,
}
impl AxisFilter {
    fn new(carrier_delta: f64, beat_delta: f64) -> Self {
        let half_beat_delta = beat_delta * 0.5;
        Self {
            carrier: Harmonics::new(PI * carrier_delta),
            half_beat: Harmonics::new(PI * half_beat_delta),
            carrier_delta,
            half_beat_delta,
        }
    }
    fn weight(&self, term: Term) -> f64 {
        let delta =
            term.carrier as f64 * self.carrier_delta + term.half_beat as f64 * self.half_beat_delta;
        let angle = PI * delta;
        // Use the combined argument, rather than a quotient of nearly
        // cancelling harmonic sines, whenever its magnitude is small. The
        // omitted z^10/11! term is below 2.4e-17 throughout this branch.
        if angle.abs() < 0.125 {
            let square = angle * angle;
            return 1.0
                + square
                    * (-1.0 / 6.0
                        + square * (1.0 / 120.0 + square * (-1.0 / 5040.0 + square / 362880.0)));
        }
        let beat = term.half_beat.unsigned_abs();
        let sign = if term.half_beat < 0 { -1.0 } else { 1.0 };
        let sine = self.carrier.sine[term.carrier] * self.half_beat.cosine[beat]
            + sign * self.carrier.cosine[term.carrier] * self.half_beat.sine[beat];
        sine / angle
    }
}

/// Integrate .38 C(Phi+Psi/2) + .30 C(Phi-Psi/2) + 1.20 C+ C- over an affine box.
///
/// All product phases are combined before filtering. Harmonic recurrences
/// supply both phase cosines and sinc numerators, with eight trigonometric
/// evaluations for the entire profile rather than one for each retained term.
pub(super) fn evaluate(phi: PhaseFootprint, psi: PhaseFootprint) -> FilteredPattern {
    let carrier = Harmonics::new(TAU * phi.value.rem_euclid(1.0));
    let half_beat = Harmonics::new(PI * psi.value.rem_euclid(2.0));
    let axes =
        std::array::from_fn::<_, 3, _>(|axis| AxisFilter::new(phi.delta[axis], psi.delta[axis]));
    let mut value = DC;
    let mut error = 0.0;
    for term in TERMS {
        let beat = term.half_beat.unsigned_abs();
        let sign = if term.half_beat < 0 { -1.0 } else { 1.0 };
        let cosine = carrier.cosine[term.carrier] * half_beat.cosine[beat]
            - sign * carrier.sine[term.carrier] * half_beat.sine[beat];
        let contribution = term.coefficient
            * cosine
            * axes[0].weight(term)
            * axes[1].weight(term)
            * axes[2].weight(term);
        let corrected = contribution - error;
        let next = value + corrected;
        error = (next - value) - corrected;
        value = next;
    }
    FilteredPattern { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn footprint(value: f64, delta: [f64; 3]) -> PhaseFootprint {
        PhaseFootprint { value, delta }
    }

    fn profile(phi: f64, psi: f64) -> f64 {
        let plus = (PI * (phi + psi * 0.5)).cos().powi(16);
        let minus = (PI * (phi - psi * 0.5)).cos().powi(16);
        PLUS * plus + MINUS * minus + PRODUCT * plus * minus
    }

    // Dense composite Gauss integration directly evaluates the two positive
    // powers. It shares neither Fourier coefficients nor harmonic recurrences
    // with the filter under test.
    fn dense_box_integral(phi: PhaseFootprint, psi: PhaseFootprint) -> f64 {
        const NODES: [f64; 4] =
            [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526];
        const WEIGHTS: [f64; 4] =
            [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538];
        const CELLS: usize = 32;
        let axes: [Vec<(f64, f64)>; 3] = std::array::from_fn(|axis| {
            if phi.delta[axis] == 0.0 && psi.delta[axis] == 0.0 {
                return vec![(0.0, 1.0)];
            }
            (0..CELLS)
                .flat_map(|cell| {
                    NODES.into_iter().zip(WEIGHTS).map(move |(node, weight)| {
                        (
                            (cell as f64 + 0.5 + node * 0.5) / CELLS as f64 - 0.5,
                            weight / (2 * CELLS) as f64,
                        )
                    })
                })
                .collect()
        });
        let mut total = 0.0;
        for &(x, wx) in &axes[0] {
            for &(y, wy) in &axes[1] {
                for &(t, wt) in &axes[2] {
                    let p = phi.value + phi.delta[0] * x + phi.delta[1] * y + phi.delta[2] * t;
                    let q = psi.value + psi.delta[0] * x + psi.delta[1] * y + psi.delta[2] * t;
                    total += profile(p, q) * wx * wy * wt;
                }
            }
        }
        total
    }

    #[test]
    fn complete_harmonic_expansion_reproduces_the_point_material() {
        assert_eq!(TERM_COUNT, 144);
        assert_eq!(COEFFICIENTS[0] + 2.0 * COEFFICIENTS[1..].iter().sum::<f64>(), 1.0);
        for (phi, psi, expected) in [
            (0.0, 0.0, 1.88),
            (0.5, 0.0, 0.0),
            (0.25, 0.5, 0.30),
            (0.25, -0.5, 0.38),
            (0.0, 0.5, 0.002674560546875),
        ] {
            let actual = evaluate(footprint(phi, [0.0; 3]), footprint(psi, [0.0; 3])).value;
            assert!((actual - expected).abs() < 5e-14);
        }
        for i in 0..257 {
            let phi = f64::from(i) / 257.0 - 0.5;
            let psi = f64::from((i * 61) % 257) / 257.0 * 2.0 - 1.0;
            let actual = evaluate(footprint(phi, [0.0; 3]), footprint(psi, [0.0; 3])).value;
            assert!((actual - profile(phi, psi)).abs() < 5e-14, "point {phi}, {psi}: {actual}");
        }
    }

    #[test]
    fn joint_filter_matches_dense_spatial_and_temporal_box_integration() {
        for (phi, psi) in [
            (footprint(0.13, [0.41, 0.0, 0.0]), footprint(-0.27, [-0.23, 0.0, 0.0])),
            (footprint(0.71, [0.29, -0.47, 0.0]), footprint(0.31, [0.37, 0.19, 0.0])),
            (footprint(-0.23, [0.43, -0.37, 0.29]), footprint(0.19, [-0.17, 0.21, -0.31])),
            (footprint(0.0, [0.5, 0.0, 0.0]), footprint(0.0, [-1.0, 0.0, 0.0])),
            (
                footprint(0.48, [0.000_000_01, 0.000_000_02, 0.0]),
                footprint(-0.96, [-0.000_000_02, 0.000_000_01, 0.0]),
            ),
            (footprint(0.173, [0.83, -0.47, 1.13]), footprint(-0.291, [-0.29, 0.61, 0.37])),
            (footprint(0.317, [0.9, -0.6, 1.3]), footprint(0.193, [1.8, 0.7, -2.6])),
        ] {
            let actual = evaluate(phi, psi).value;
            let reference = dense_box_integral(phi, psi);
            assert!(
                (actual - reference).abs() < 2e-10,
                "joint box {phi:?}, {psi:?}: {actual} != {reference}"
            );
        }
    }

    #[test]
    fn zero_registration_equals_the_single_registered_profile_and_its_dc() {
        for phi in [-0.5, -0.19, 0.0, 0.14, 0.5, 0.87] {
            let c = (PI * phi).cos().powi(16);
            let actual = evaluate(footprint(phi, [0.0; 3]), footprint(0.0, [0.0; 3])).value;
            assert!((actual - ((PLUS + MINUS) * c + PRODUCT * c * c)).abs() < 5e-14);
        }
        // The registered product is cos(pi Phi)^32, whose mean is choose(32,16)/4^16.
        let registered_dc = (PLUS + MINUS) * COEFFICIENTS[0] + PRODUCT * 601080390.0 / 4294967296.0;
        let actual = evaluate(footprint(0.37, [1.0, 0.0, 0.0]), footprint(0.0, [0.0; 3])).value;
        assert!((actual - registered_dc).abs() < 5e-14);
        assert!(actual > DC + 0.1);
    }

    #[test]
    fn independently_unresolved_phases_retain_the_complete_material_dc() {
        for phase in [0.0, 0.13, -3.7, 1000.25] {
            let actual = evaluate(
                footprint(phase, [1.0, 0.0, 0.0]),
                footprint(phase * 0.37, [0.0, 2.0, 0.0]),
            )
            .value;
            assert!((actual - DC).abs() < 5e-14, "independent DC {actual} != {DC}");
            let separated_families = evaluate(
                footprint(phase, [0.5, 0.5, 0.0]),
                footprint(phase * 0.37, [1.0, -1.0, 0.0]),
            )
            .value;
            assert!((separated_families - DC).abs() < 5e-14);
        }
    }

    #[test]
    fn unresolved_carrier_preserves_beat_contrast_and_its_temporal_filter() {
        let beat_average = |psi: f64, time_width: f64| {
            let mut result = DC;
            for (n, &coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
                let angle = PI * n as f64 * time_width;
                let temporal = if angle == 0.0 { 1.0 } else { angle.sin() / angle };
                result += 2.0
                    * PRODUCT
                    * coefficient
                    * coefficient
                    * (TAU * n as f64 * psi).cos()
                    * temporal;
            }
            result
        };
        for psi in [0.0, 0.13, 0.5, 0.79] {
            for time_width in [0.0, 0.27, 1.0] {
                for phi in [0.23, 1e12 + 0.25] {
                    let actual = evaluate(
                        footprint(phi, [1.0, 0.0, 0.0]),
                        footprint(psi, [0.0, 0.0, time_width]),
                    )
                    .value;
                    assert!((actual - beat_average(psi, time_width)).abs() < 5e-14);
                }
            }
        }
        assert!(beat_average(0.0, 0.0) - beat_average(0.5, 0.0) > 0.1);
    }

    #[test]
    fn affine_averages_stay_in_range_and_are_deterministic() {
        let inputs: Vec<_> = (0..256)
            .map(|i| {
                let phi = footprint(
                    f64::from(i) * 0.137,
                    [f64::from(i) / 97.0, -f64::from(i % 23) / 41.0, f64::from(i % 17) / 31.0],
                );
                let psi = footprint(
                    -f64::from(i) * 0.173,
                    [f64::from(i % 19) / 37.0, f64::from(i) / 113.0, -f64::from(i % 29) / 53.0],
                );
                (phi, psi)
            })
            .collect();
        let sequential: Vec<_> =
            inputs.iter().map(|&(phi, psi)| evaluate(phi, psi).value).collect();
        let threaded = std::thread::scope(|scope| {
            let handles: Vec<_> = inputs
                .chunks(64)
                .map(|chunk| {
                    scope.spawn(|| {
                        chunk.iter().map(|&(phi, psi)| evaluate(phi, psi).value).collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().flat_map(|handle| handle.join().unwrap()).collect::<Vec<_>>()
        });
        assert_eq!(sequential, threaded);
        assert!(sequential.iter().all(|&value| (-1e-12..=1.88 + 1e-12).contains(&value)));
    }
}
