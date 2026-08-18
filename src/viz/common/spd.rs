//! Reductions over the accumulated per-pixel SPD buffer shared by the
//! SPD-phase visualization modes.

use crate::spectrum::NUM_BINS;
use rayon::prelude::*;

/// Per-pixel total spectral energy as a compact `f32` field.
#[must_use]
pub fn energy_field(spd: &[[f64; NUM_BINS]]) -> Vec<f32> {
    spd.par_iter().map(|bins| bins.iter().sum::<f64>() as f32).collect()
}

/// Whole-image aggregate spectrum: the per-bin sum across all pixels.
#[must_use]
pub fn aggregate_spectrum(spd: &[[f64; NUM_BINS]]) -> [f64; NUM_BINS] {
    spd.par_iter()
        .fold(
            || [0.0f64; NUM_BINS],
            |mut acc, bins| {
                for (slot, &value) in acc.iter_mut().zip(bins.iter()) {
                    *slot += value;
                }
                acc
            },
        )
        .reduce(
            || [0.0f64; NUM_BINS],
            |mut lhs, rhs| {
                for (slot, &value) in lhs.iter_mut().zip(rhs.iter()) {
                    *slot += value;
                }
                lhs
            },
        )
}

/// Maximum value of an `f32` field (0 for empty fields).
#[must_use]
pub fn field_max(field: &[f32]) -> f32 {
    field.par_iter().copied().reduce(|| 0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_field_sums_bins() {
        let mut spd = vec![[0.0; NUM_BINS]; 4];
        spd[2][10] = 1.5;
        spd[2][20] = 0.5;
        let field = energy_field(&spd);
        assert_eq!(field[2], 2.0);
        assert_eq!(field[0], 0.0);
    }

    #[test]
    fn aggregate_spectrum_sums_pixels() {
        let mut spd = vec![[0.0; NUM_BINS]; 3];
        spd[0][5] = 1.0;
        spd[1][5] = 2.0;
        spd[2][7] = 4.0;
        let total = aggregate_spectrum(&spd);
        assert_eq!(total[5], 3.0);
        assert_eq!(total[7], 4.0);
        assert_eq!(total[6], 0.0);
    }
}
