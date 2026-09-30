//! Tone law: from ink presence and freshness to the pine-soot load.
//!
//! # Fields
//!
//! Every ink node carries a presence `P ∈ [0, 1]` (has this water ever been inked, diluted by
//! mixing) and, per body `i`, a freshness `E_i ∈ [0, 1]`. A parcel that body `i` last inked at
//! fluid time `t*_i` and that has not mixed since has, at frame time `t`,
//!
//! ```text
//! P = 1,    E_i = exp(-(t - t*_i) / τ)          (E_i = 0 if body i never inked it)
//! ```
//!
//! Interpolation mixes both linearly (dilution).
//!
//! # Timing in film time
//!
//! The film shows the whole orbit, `T` fluid units, in a fixed number of seconds, so the tone law
//! is timed as fractions of `T` (docs/ember-design.md §6.1):
//!
//! ```text
//! τ    = fade_fraction · T         e-folding time of the fade
//! hold = hold_fraction · T         time for which fresh ink stays full strength
//! ```
//!
//! With the defaults (0.75 s and 0.8 s of a 30-second film) ink is black for 0.8 s after a body
//! lays it and reaches the pale floor wash about six seconds later, on every orbit, whatever its
//! length in fluid units.
//!
//! # Tone law (the reservoir feed with a hold)
//!
//! With `g = exp(hold / τ)` precomputed,
//!
//! ```text
//! h_i    = min(P, g·E_i)                          full strength while younger than `hold`
//! carbon = floor·P + (1 - floor)·max_i h_i        the youngest ink of any body wins
//! ```
//!
//! For an unmixed parcel (`P = 1`) `g·E_i = exp(-(t_ref - t*_i)/τ)` with `t_ref = t - hold`, so
//! `carbon = floor + (1 - floor)·exp(-max(t_ref - t*, 0)/τ)` for the youngest uptake `t*`: the
//! museum-lab reservoir feed `ExpFeed(t_ref, τ, floor)` evaluated at the uptake time. The law is
//! exact; the numbers carry the `f32` storage and the per-frame ageing of `E` (see `ink`).
//!
//! ## Dilution
//!
//! The hold saturates at the presence, not at 1, so the law is linear under dilution with clear
//! water. A node holding a fraction `w` of an unmixed parcel of age `a` (and `1 - w` clear water)
//! has `P = w` and `E_i = w·exp(-a/τ)`, hence `h_i = w·min(1, g·exp(-a/τ))`: the carbon is `w`
//! times the unmixed one. (Saturating at 1 would draw ink diluted down to `1/g` at full
//! strength.) Mixing two *inked* parcels (`P = 1`) stays concave: the youngest ink wins. With
//! [`LookConfig::floor_tau`] set, `P` also carries the floor wash's slow fade, so the cap fades
//! the whole deposit, fresh ink included, by the same factor.
//!
//! All arithmetic is exactly rounded `f64` (one `exp` at construction, through `math`).

use super::config::LookConfig;
use super::math::{self, max, min};

/// The look: maps ink presence and per-body freshness to the pine-soot load.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Look {
    /// Strength of ink older than the reservoir memory ([`LookConfig::floor`]).
    floor: f64,
    /// `1 - floor`.
    fresh_weight: f64,
    /// Freshness gain `g = exp(hold / τ) ≥ 1`: `g·E_i ≥ 1` exactly while the ink is younger
    /// than the hold.
    hold_gain: f64,
    /// E-folding time `τ` of the fade, in fluid units.
    fade_tau: f64,
    /// Full-strength time, in fluid units.
    hold: f64,
}

impl Look {
    /// The look of `config` for an orbit lasting `duration` fluid units (finite, positive).
    pub(crate) fn new(config: &LookConfig, duration: f64) -> Self {
        let fade_tau = config.fade_fraction * duration;
        let hold = config.hold_fraction * duration;
        Self {
            floor: config.floor,
            fresh_weight: 1.0 - config.floor,
            hold_gain: math::exp(hold / fade_tau),
            fade_tau,
            hold,
        }
    }

    /// E-folding time `τ` of the fade in fluid units: the ink fields age freshness with it.
    pub(crate) fn fade_tau(&self) -> f64 {
        self.fade_tau
    }

    /// Full-strength time in fluid units.
    pub(crate) fn hold(&self) -> f64 {
        self.hold
    }

    /// Pine-soot load of a node with presence `presence` and per-body freshness `freshness`
    /// (the `f32` values the ink fields store); 0 exactly for bare water.
    pub(crate) fn carbon(&self, presence: f32, freshness: [f32; 3]) -> f64 {
        let presence = f64::from(presence);
        let h = freshness.map(|e| min(presence, f64::from(e) * self.hold_gain));
        self.floor * presence + self.fresh_weight * max(max(h[0], h[1]), h[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ember::config::EmberConfig;

    /// The production look for an orbit of `duration` fluid units.
    fn production(duration: f64) -> (Look, LookConfig) {
        let config = EmberConfig::default().look;
        (Look::new(&config, duration), config)
    }

    /// The museum-lab reservoir feed `ExpFeed(t_ref, τ, floor)` at the uptake time `t`.
    fn exp_feed(t: f64, t_ref: f64, tau: f64, floor: f64) -> f64 {
        let age = (t_ref - t).max(0.0);
        floor + (1.0 - floor) * math::exp(-age / tau)
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-300)
    }

    #[test]
    fn bare_water_is_bare_paper() {
        let (look, config) = production(19.0);
        assert_eq!(look.carbon(0.0, [0.0; 3]), 0.0);
        assert_eq!(look.carbon(0.0, [0.3, 0.2, 0.0]), 0.0, "freshness without presence is clear");
        assert_eq!(look.carbon(1.0, [0.0; 3]), config.floor, "old ink is the floor wash");
        assert!(look.carbon(1e-9, [0.0; 3]) > 0.0);
    }

    /// The timing is a fraction of the orbit: the same number of film seconds on every orbit.
    #[test]
    fn timing_is_a_fraction_of_the_film() {
        for duration in [4.0, 11.6, 19.1] {
            let (look, config) = production(duration);
            assert_eq!(look.fade_tau(), config.fade_fraction * duration);
            assert_eq!(look.hold(), config.hold_fraction * duration);
            assert!(close(math::ln(look.hold_gain), look.hold / look.fade_tau, 1e-15));
            // In a 30-second film: black for 0.8 s, then the fade's time constant is 0.75 s.
            assert!(close(look.hold() / duration * 30.0, 0.8, 1e-12));
            assert!(close(look.fade_tau() / duration * 30.0, 0.75, 1e-12));
        }
    }

    #[test]
    fn fresh_ink_is_black_for_the_hold_then_fades_to_the_floor() {
        let (look, config) = production(19.1);
        let fresh = |age: f64| math::exp(-age / look.fade_tau()) as f32;
        assert_eq!(look.carbon(1.0, [1.0, 0.0, 0.0]), 1.0);
        assert!(close(look.carbon(1.0, [fresh(look.hold()), 0.0, 0.0]), 1.0, 1e-6));
        let one_tau = look.carbon(1.0, [fresh(look.hold() + look.fade_tau()), 0.0, 0.0]);
        let expected = config.floor + (1.0 - config.floor) * math::exp(-1.0);
        assert!(close(one_tau, expected, 1e-6), "{one_tau} vs {expected}");
        // About six seconds of film after the hold the ink is within 1% of the floor's step.
        let six_seconds = 6.0 / 30.0 * 19.1;
        let late = look.carbon(1.0, [fresh(look.hold() + six_seconds), 0.0, 0.0]);
        assert!(late < config.floor + 0.01 * (1.0 - config.floor), "{late}");
    }

    /// For unmixed parcels the law is the museum-lab reservoir feed at `t_ref = t - hold`.
    #[test]
    fn unmixed_parcels_follow_the_reservoir_feed() {
        let (look, config) = production(12.0);
        let t_frame = 7.25;
        let t_ref = t_frame - look.hold();
        for age in [0.0, 0.1, 0.3, 0.32, 0.4, 0.6, 1.0, 2.5, 5.0] {
            let e = math::exp(-age / look.fade_tau()) as f32;
            let got = look.carbon(1.0, [e, 0.0, 0.0]);
            let want = exp_feed(t_frame - age, t_ref, look.fade_tau(), config.floor);
            assert!(close(got, want, 2e-7), "age {age}: {got} vs {want}");
        }
    }

    /// Dilution with clear water scales the carbon linearly.
    #[test]
    fn dilution_is_linear() {
        let (look, _) = production(19.1);
        for age in [0.0, 0.5, 1.0, 3.0] {
            let e = math::exp(-age / look.fade_tau());
            let full = look.carbon(1.0, [e as f32, 0.0, 0.0]);
            for w in [0.5f64, 0.1, 0.02] {
                let diluted = look.carbon(w as f32, [(w * e) as f32, 0.0, 0.0]);
                assert!(close(diluted, w * full, 1e-6), "age {age}, w {w}: {diluted}");
            }
        }
    }

    #[test]
    fn the_youngest_ink_wins_and_the_bodies_are_symmetric() {
        let (look, _) = production(19.1);
        let old = look.carbon(1.0, [0.001, 0.0, 0.0]);
        let young = look.carbon(1.0, [0.0, 0.9, 0.0]);
        assert_eq!(look.carbon(1.0, [0.001, 0.9, 0.0]), max(old, young));
        let e = [0.9f32, 0.07, 0.4];
        let reference = look.carbon(0.8, e);
        for perm in [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
            assert_eq!(look.carbon(0.8, perm.map(|i| e[i])), reference);
        }
    }

    #[test]
    fn carbon_is_monotone_in_freshness_and_presence() {
        let (look, config) = production(19.1);
        let grid: Vec<f32> = (0..=40).map(|i| i as f32 / 40.0).map(|x| x * x * x).collect();
        for &p in &[0.0f32, 0.3, 1.0] {
            let mut previous = -1.0;
            for &e in &grid {
                let carbon = look.carbon(p, [e, 0.0, 0.0]);
                assert!((0.0..=1.0).contains(&carbon));
                assert!(carbon >= previous);
                previous = carbon;
            }
        }
        let mut previous = -1.0;
        for &p in &grid {
            let carbon = look.carbon(p, [0.0; 3]);
            assert!(carbon > previous || (p == 0.0 && carbon == 0.0));
            previous = carbon;
        }
        assert_eq!(look.carbon(1.0, [0.0; 3]), config.floor);
    }
}
