//! Tone law and the vermilion accent: from ink presence and freshness to pigment loads.
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
//! with `τ` = [`LookConfig::fresh_tau`]. Interpolation mixes both linearly (dilution).
//!
//! # Tone law (the reservoir feed with a hold)
//!
//! With `g = exp(hold / τ)` precomputed,
//!
//! ```text
//! h_i    = min(P, g·E_i)                          full strength while younger than `hold`
//! c_i    = floor·P + (1 - floor)·h_i              species strength of body i's ink
//! c_mono = floor·P + (1 - floor)·max_i h_i        shared-carbon strength (youngest ink wins)
//! ```
//!
//! For an unmixed parcel (`P = 1`) `g·E_i = exp(-(t_ref - t*_i)/τ)` with `t_ref = t - hold`, so
//! `c_i = floor + (1 - floor)·exp(-max(t_ref - t*_i, 0)/τ)`: the law of the museum-lab reservoir
//! feed `ExpFeed(t_ref, τ, floor)` evaluated at the uptake time (equivalently its `Fade` law with
//! `T = t_ref`). A body that never inked the parcel has `c_i = floor·P` where the prototype has
//! 0. That difference never changes the loads as long as `floor ≤ meeting_threshold`
//! (production: 4e-4 ≤ 0.3), because such a `c_i ≤ floor` can then never pass the strict
//! meeting test below.
//!
//! `floor ≤ meeting_threshold` is a **precondition** of [`Look::new`], not a tuning choice. The
//! fields keep a single presence for all bodies and flush freshness below 1e-12 to 0, so they
//! cannot tell which bodies ever inked old water. A look with `floor > meeting_threshold`
//! (cinnabar wherever two bodies *ever* met, like the prototype's `ember()` default
//! `p_min = 1e-4`) is therefore not representable: it would turn single-body and stale ink
//! vermilion.
//!
//! The law is the prototype's; the numbers differ only where the stored fields do. The
//! prototype evaluates every sample's exact age, while the ink fields store `E_i` as `f32` and
//! age it by a multiplication per frame (see `ink`), so a rendered frame matches the prototype's
//! look up to that `f32` storage and per-frame ageing of `E` (relative `2⁻²⁴` per rounding).
//! What holds bit for bit is narrower: given the same `(P, E)` and an ember `K ≤ best`,
//! [`Look::loads`] equals the prototype's rule written over `(P, E)`.
//!
//! ## Dilution
//!
//! The hold saturates at the presence, not at 1, so the law is linear under dilution with clear
//! water. A node holding a fraction `w` of an unmixed parcel of age `a` (and `1 - w` clear water)
//! has `P = w` and `E_i = w·exp(-a/τ)`, hence `h_i = w·min(1, g·exp(-a/τ))`: every strength is
//! `w` times the unmixed one, and the meeting threshold below applies to the diluted strength.
//! (Saturating at 1 would draw ink diluted down to `1/g ≈ 1.6 %` at full strength, and let two
//! bodies' 2 % dilutions meet at full strength and form cinnabar.) The prototype, which ages each
//! sample exactly, does not dilute at all; for unmixed parcels the two are the same law, up to
//! the `f32` storage and per-frame ageing of `E` (above).
//! Mixing two *inked* parcels (`P = 1`) stays concave: the youngest ink wins in `c_mono`, and the
//! 50/50 interface of two fresh inks is a full-strength meeting — which is what a meeting is.
//! With [`LookConfig::floor_tau`] set, `P` also carries the floor wash's slow fade, so the cap
//! fades the whole deposit, fresh ink included, by the same factor.
//!
//! # The ember (vermilion) accent
//!
//! Cinnabar is a two-reagent product that forms only where the waters of two bodies met while
//! both were fresh (museum-lab look `vermilion` = `ember(1.2, 0.06, p_min = 0.3)`):
//!
//! ```text
//! best     = max over pairs (0,1), (1,2), (0,2) of min(c_i, c_j)
//! best     = best > meeting_threshold ? best : 0
//! carbon   = c_mono · (best > 0 ? carbon_keep : 1)
//! cinnabar = cinnabar_strength · best
//! ```
//!
//! With the production constants `best > 0.3` needs both uptakes no older than
//! `τ·ln((1 - floor)/(0.3 - floor)) ≈ 0.1445` fluid time units before `t_ref` (in unmixed water;
//! diluted water needs `P > 0.3` as well).
//!
//! # Ember memory
//!
//! In the prototype the vermilion exists only while both inks are fresh, so a still taken long
//! after the last meeting of two bodies has none. With [`LookConfig::ember_tau`] set, the cinnabar
//! instead *glows on*: every node carries an ember field `K` (in the units of `best`) that the
//! flow advects like the ink, `K ← max(best, K_prev·exp(-Δt/ember_tau))`. An ember shows while it
//! is still hotter than the meeting threshold, exactly like a fresh meeting:
//!
//! ```text
//! red      = max(best, K > meeting_threshold ? K : 0)
//! carbon   = red > 0 ? carbon_keep · max(c_mono, red) : c_mono
//! cinnabar = cinnabar_strength · red
//! ```
//!
//! The ember keeps the soot it formed with (`carbon_keep` of its strength), so a glowing ember on
//! old, pale water stays the deep vermilion of a fresh meeting instead of turning a graphic
//! pure red. For a fresh meeting `best ≤ c_mono`, so the carbon is the prototype's
//! `c_mono · carbon_keep`. The rule applies to fresh single-body ink too: ink a body lays into
//! water that still glows renders vermilion, not black (`carbon_keep·max(c_mono, red)`). This is
//! intended: the water a body carries along glows with the ember of its last meeting.
//!
//! So the vermilion of a meeting stays crisp and saturated, is stretched by the flow, shrinks as
//! mixing dilutes it below the threshold, and goes out about `ember_tau·ln(K₀/meeting_threshold)`
//! after the meeting (≈ 1.2·`ember_tau` for a full-strength meeting) — instead of lasting only
//! while both inks are fresh. It never washes out into a pale pink: the accent stays an accent.
//! Without ember memory the ink fields store `K = 0`, and with `K = 0` (indeed with any
//! `K ≤ best`) the loads are the prototype's rule above evaluated on the same `(P, E)`, bit for
//! bit; the rendered look is then the prototype's up to the `f32` storage and per-frame ageing
//! of `E`.
//!
//! All arithmetic is exactly rounded `f64` (one `exp` at construction, through `math`).

use super::config::LookConfig;
use super::math::{self, max, min};

/// Pigment loads of one ink sample (dimensionless; 1.0 = one unit of pine soot or cinnabar).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct InkLoads {
    /// Pine-soot carbon load.
    pub carbon: f64,
    /// Cinnabar (vermilion) load.
    pub cinnabar: f64,
}

impl InkLoads {
    /// Whether the sample carries no pigment at all (bare paper).
    pub(crate) fn is_bare(&self) -> bool {
        self.carbon == 0.0 && self.cinnabar == 0.0
    }
}

/// The look: maps ink presence and per-body freshness to pigment loads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Look {
    /// Strength of ink older than the reservoir memory ([`LookConfig::floor`]).
    floor: f64,
    /// `1 - floor`.
    fresh_weight: f64,
    /// Freshness gain `g = exp(hold / fresh_tau) ≥ 1`: `g·E_i ≥ 1` exactly while the ink is
    /// younger than `hold`.
    hold_gain: f64,
    /// Cinnabar load per unit of shared fresh ink.
    cinnabar_strength: f64,
    /// Fraction of the carbon kept where cinnabar forms.
    carbon_keep: f64,
    /// Both species strengths must exceed this for cinnabar to form.
    meeting_threshold: f64,
}

impl Look {
    /// Precomputes the look's constants.
    ///
    /// Preconditions: `config` satisfies [`EmberConfig::validate`] — in particular
    /// `hold / fresh_tau ≤ ln(10⁶) ≈ 13.8`, so that `g` is finite and the ink fields' freshness
    /// flush (`E < 10⁻¹²` is stored as 0) cuts at most `g·10⁻¹² ≤ 10⁻⁶` of strength — and
    /// `floor ≤ meeting_threshold`, without which a body that never inked the water could take
    /// part in a meeting (module docs). The latter is asserted in debug builds.
    ///
    /// [`EmberConfig::validate`]: super::config::EmberConfig::validate
    pub(crate) fn new(config: &LookConfig) -> Self {
        debug_assert!(
            config.floor <= config.meeting_threshold,
            "look.floor ({}) must not exceed look.meeting_threshold ({})",
            config.floor,
            config.meeting_threshold
        );
        Self {
            floor: config.floor,
            fresh_weight: 1.0 - config.floor,
            hold_gain: math::exp(config.hold / config.fresh_tau),
            cinnabar_strength: config.cinnabar_strength,
            carbon_keep: config.carbon_keep,
            meeting_threshold: config.meeting_threshold,
        }
    }

    /// Species strengths `c_i` and the shared-carbon strength `c_mono` (module docs): the hold
    /// `h_i = min(P, g·E_i)` saturates at the presence, so clear water dilutes linearly.
    fn strengths(&self, presence: f32, freshness: [f32; 3]) -> ([f64; 3], f64) {
        let presence = f64::from(presence);
        let base = self.floor * presence;
        let h = freshness.map(|e| min(presence, f64::from(e) * self.hold_gain));
        let c = h.map(|h| base + self.fresh_weight * h);
        let h_max = max(max(h[0], h[1]), h[2]);
        (c, base + self.fresh_weight * h_max)
    }

    /// `best`: the strongest meeting of two bodies' fresh inks, 0 unless it exceeds the meeting
    /// threshold (module docs). This is the cinnabar that forms now.
    pub(crate) fn meeting(&self, presence: f32, freshness: [f32; 3]) -> f64 {
        let (c, _) = self.strengths(presence, freshness);
        let meeting = max(max(min(c[0], c[1]), min(c[1], c[2])), min(c[0], c[2]));
        if meeting > self.meeting_threshold { meeting } else { 0.0 }
    }

    /// Pigment loads of a sample with presence `p`, per-body freshness `e` and ember `k` (all in
    /// `[0, 1]`, finite; the `f32` field values are widened exactly to `f64`). Without ember
    /// memory the ink fields hold `k = 0`, and the loads are then the prototype's rule (module
    /// docs) evaluated on the same `(p, e)`, bit for bit.
    ///
    /// Clear water (`p = 0`) without a glowing ember (`k ≤ meeting_threshold`) gives exactly
    /// [`InkLoads::default`] (bare paper), whatever its freshness: the hold caps every strength at
    /// the presence. With `floor > 0` and positive `carbon_keep` any other water carries pigment.
    pub(crate) fn loads(&self, presence: f32, freshness: [f32; 3], ember: f32) -> InkLoads {
        let (c, mono) = self.strengths(presence, freshness);
        let meeting = max(max(min(c[0], c[1]), min(c[1], c[2])), min(c[0], c[2]));
        let best = if meeting > self.meeting_threshold { meeting } else { 0.0 };
        let ember = f64::from(ember);
        let glowing = if ember > self.meeting_threshold { ember } else { 0.0 };
        let red = max(best, glowing);
        let carbon = if red > 0.0 { self.carbon_keep * max(mono, red) } else { mono };
        InkLoads { carbon, cinnabar: self.cinnabar_strength * red }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ember::config::EmberConfig;

    fn production() -> (Look, LookConfig) {
        let config = EmberConfig::default().look;
        (Look::new(&config), config)
    }

    /// The prototype's reservoir feed `ExpFeed(t_ref, tau, floor)` at uptake time `t`.
    fn exp_feed(t: f64, t_ref: f64, config: &LookConfig) -> f64 {
        let age = (t_ref - t).max(0.0);
        config.floor + (1.0 - config.floor) * math::exp(-age / config.fresh_tau)
    }

    /// The prototype's `ember(...)` look for per-body uptake times (`None` = never inked).
    fn prototype_loads(uptakes: [Option<f64>; 3], t_frame: f64, config: &LookConfig) -> InkLoads {
        let t_ref = t_frame - config.hold;
        let species = uptakes.map(|t| t.map_or(0.0, |t| exp_feed(t, t_ref, config)));
        let youngest = uptakes.iter().flatten().copied().fold(f64::NEG_INFINITY, f64::max);
        let mono = if youngest.is_finite() { exp_feed(youngest, t_ref, config) } else { 0.0 };
        let mut best = 0.0f64;
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            best = best.max(species[i].min(species[j]));
        }
        let best = if best > config.meeting_threshold { best } else { 0.0 };
        InkLoads {
            carbon: mono * if best > 0.0 { config.carbon_keep } else { 1.0 },
            cinnabar: config.cinnabar_strength * best,
        }
    }

    /// Unmixed-parcel fields of the given uptake times at `t_frame` (computed in f64, stored as
    /// f32 like the ink fields).
    fn fields(uptakes: [Option<f64>; 3], t_frame: f64, config: &LookConfig) -> (f32, [f32; 3]) {
        let presence = if uptakes.iter().any(Option::is_some) { 1.0 } else { 0.0 };
        let freshness =
            uptakes.map(|t| t.map_or(0.0, |t| math::exp(-(t_frame - t) / config.fresh_tau) as f32));
        (presence, freshness)
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-300)
    }

    #[test]
    fn bare_water_is_bare_paper() {
        let (look, _) = production();
        let loads = look.loads(0.0, [0.0; 3], 0.0);
        assert_eq!(loads, InkLoads::default());
        assert!(loads.is_bare());
        assert!(!look.loads(1.0, [0.0; 3], 0.0).is_bare());
        // Stray freshness (interpolation at an ink edge) or a cooled ember in clear water: bare.
        assert_eq!(look.loads(0.0, [0.3, 0.2, 0.0], 0.25), InkLoads::default());
        assert!(!look.loads(1e-9, [0.0; 3], 0.0).is_bare());
        assert!(!look.loads(0.0, [0.0; 3], 0.31).is_bare());
    }

    #[test]
    fn hold_gain_is_exp_hold_over_tau() {
        let (look, config) = production();
        assert!(close(math::ln(look.hold_gain), config.hold / config.fresh_tau, 1e-15));
        assert!(close(look.hold_gain, 64.50009306485578, 1e-12));
        // Freshly inked water (E = 1) is at full strength for exactly `hold`.
        let at_hold = math::exp(-config.hold / config.fresh_tau);
        assert_eq!(look.loads(1.0, [1.0, 0.0, 0.0], 0.0).carbon, 1.0);
        assert!(close(look.loads(1.0, [at_hold as f32, 0.0, 0.0], 0.0).carbon, 1.0, 1e-6));
        let older = math::exp(-(config.hold + 0.12) / config.fresh_tau) as f32;
        let expected = config.floor + (1.0 - config.floor) * math::exp(-1.0);
        assert!(close(look.loads(1.0, [older, 0.0, 0.0], 0.0).carbon, expected, 1e-6));
    }

    /// Compares the look with the prototype's ExpFeed/ember law (`t_ref = t_frame - hold`) on
    /// unmixed parcels, for every combination of inked bodies (including never-inked ones) over
    /// a grid of ages. Returns `(cases checked, cases with cinnabar)`.
    fn assert_matches_prototype(config: &LookConfig) -> (usize, usize) {
        let look = Look::new(config);
        let t_frame = 7.25;
        let ages = [0.0, 0.1, 0.3, 0.5, 0.55, 0.6, 0.6445, 0.65, 0.8, 1.5, 4.0];
        let mut cinnabar_cases = 0;
        let mut checked = 0;
        for &a0 in &ages {
            for &a1 in &ages {
                for pattern in 0..8u8 {
                    let pick = |bit: u8, age: f64| (pattern & bit != 0).then_some(t_frame - age);
                    let uptakes = [pick(1, a0), pick(2, a1), pick(4, 0.5 * (a0 + a1))];
                    let (p, e) = fields(uptakes, t_frame, config);
                    let got = look.loads(p, e, 0.0);
                    let want = prototype_loads(uptakes, t_frame, config);
                    // f32 storage of E is the only difference: relative 2^-24 in E.
                    assert!(
                        close(got.carbon, want.carbon, 2e-7)
                            && close(got.cinnabar, want.cinnabar, 2e-7),
                        "{config:?}, uptakes {uptakes:?}: got {got:?}, prototype {want:?}"
                    );
                    cinnabar_cases += usize::from(want.cinnabar > 0.0);
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, ages.len() * ages.len() * 8);
        (checked, cinnabar_cases)
    }

    /// For unmixed parcels the production look is the prototype's ExpFeed/ember law with
    /// `t_ref = t_frame - hold`, up to the `f32` storage of `E`, for every combination of inked
    /// bodies and ages.
    #[test]
    fn unmixed_parcels_match_the_prototype_exp_feed_law() {
        let (_, config) = production();
        assert!(config.floor <= config.meeting_threshold, "the production look is representable");
        let (checked, cinnabar_cases) = assert_matches_prototype(&config);
        assert!(cinnabar_cases > 0 && cinnabar_cases < checked);
    }

    /// The `floor·P` strength of never-inked bodies is harmless for *every* look with
    /// `floor ≤ meeting_threshold`, including the boundary `floor = meeting_threshold` (the
    /// meeting test is strict) and a threshold at zero with no floor.
    #[test]
    fn looks_with_floor_at_most_the_threshold_match_the_prototype() {
        let (_, production) = production();
        let variants = [
            (0.1, 0.3, 0.12, 0.5),
            (0.3, 0.3, 0.12, 0.5),
            (0.05, 0.05, 0.2, 0.3),
            (0.0, 0.0, 0.12, 0.5),
            (0.02, 0.6, 0.08, 0.0),
        ];
        for (floor, meeting_threshold, fresh_tau, hold) in variants {
            let config = LookConfig {
                floor,
                meeting_threshold,
                fresh_tau,
                hold,
                cinnabar_strength: 2.0,
                carbon_keep: 0.1,
                ..production.clone()
            };
            let (checked, cinnabar_cases) = assert_matches_prototype(&config);
            assert!(cinnabar_cases > 0 && cinnabar_cases < checked, "{config:?}");
            // Water only one body ever inked never turns vermilion, however stale or fresh.
            let look = Look::new(&config);
            for e in [0.0f32, 1e-9, 0.01, 1.0] {
                assert_eq!(look.loads(1.0, [e, 0.0, 0.0], 0.0).cinnabar, 0.0, "{config:?} E = {e}");
            }
        }
    }

    /// `floor > meeting_threshold` is not representable by the fields (module docs); debug
    /// builds reject it where the look is built.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "must not exceed look.meeting_threshold")]
    fn a_floor_above_the_meeting_threshold_is_rejected_in_debug_builds() {
        let (_, mut config) = production();
        config.meeting_threshold = 1e-4; // the prototype's `ember()` default p_min
        let _ = Look::new(&config);
    }

    /// `p_min = 0.3` ⇔ both contacts within `τ·ln((1 - floor)/(0.3 - floor)) ≈ 0.1445` of `t_ref`.
    #[test]
    fn meeting_threshold_is_an_age_window_before_t_ref() {
        let (look, config) = production();
        let window = config.fresh_tau
            * math::ln((1.0 - config.floor) / (config.meeting_threshold - config.floor));
        assert!((window - 0.1445).abs() < 5e-4, "{window}");
        // Ages measured from t_ref = t_frame - hold.
        let fresh =
            |age_past_ref: f64| math::exp(-(config.hold + age_past_ref) / config.fresh_tau) as f32;
        let meets = |a: f64, b: f64| look.loads(1.0, [fresh(a), fresh(b), 0.0], 0.0).cinnabar > 0.0;
        assert!(meets(0.0, 0.0));
        assert!(meets(window - 1e-4, window - 1e-4));
        assert!(!meets(window + 1e-4, window + 1e-4));
        assert!(!meets(0.0, window + 1e-4), "both bodies must be fresh");
        // Younger than t_ref counts as full strength: the meeting load saturates.
        let full = look.loads(1.0, [1.0, 1.0, 0.0], 0.0);
        assert_eq!(full.cinnabar, config.cinnabar_strength);
        // Any one pair suffices, whichever it is.
        for pair in [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]] {
            assert_eq!(look.loads(1.0, pair, 0.0).cinnabar, config.cinnabar_strength);
        }
        // A single body never makes cinnabar, however fresh.
        assert_eq!(look.loads(1.0, [1.0, 0.0, 0.0], 0.0).cinnabar, 0.0);
    }

    #[test]
    fn cinnabar_keeps_only_a_fraction_of_the_carbon() {
        let (look, config) = production();
        let loads = look.loads(1.0, [1.0, 1.0, 1.0], 0.0);
        assert_eq!(loads.carbon, config.carbon_keep);
        assert_eq!(loads.cinnabar, config.cinnabar_strength);
        // Half clear water: half the presence and half the freshness, so half the strength.
        let mixed = look.loads(0.5, [0.5, 0.5, 0.0], 0.0);
        let half = config.floor * 0.5 + (1.0 - config.floor) * 0.5;
        assert_eq!(mixed.carbon, half * config.carbon_keep);
        assert_eq!(mixed.cinnabar, config.cinnabar_strength * half);
        // Below the meeting threshold all carbon stays.
        let faint = look.loads(1.0, [0.001, 0.001, 0.0], 0.0);
        assert_eq!(faint.cinnabar, 0.0);
        let h = f64::from(0.001f32) * look.hold_gain;
        assert_eq!(faint.carbon, config.floor + (1.0 - config.floor) * h);
    }

    #[test]
    fn loads_are_monotone_in_freshness_and_presence() {
        let (look, config) = production();
        let grid: Vec<f32> = (0..=40).map(|i| i as f32 / 40.0).map(|x| x * x * x).collect();
        for &p in &[0.0f32, 0.3, 1.0] {
            for &other in &[0.0f32, 0.002, 0.05, 1.0] {
                let mut previous: Option<InkLoads> = None;
                for &e in &grid {
                    let loads = look.loads(p, [e, other, 0.0], 0.0);
                    assert!(loads.carbon >= 0.0 && loads.carbon <= 1.0);
                    assert!(loads.cinnabar >= 0.0 && loads.cinnabar <= config.cinnabar_strength);
                    if let Some(prev) = previous {
                        assert!(loads.cinnabar >= prev.cinnabar, "cinnabar grows with freshness");
                        // Carbon drops to carbon_keep only where cinnabar appears.
                        if (loads.cinnabar > 0.0) == (prev.cinnabar > 0.0) {
                            assert!(loads.carbon >= prev.carbon, "carbon grows with freshness");
                        }
                    }
                    previous = Some(loads);
                }
            }
        }
        // Presence raises the floor wash monotonically.
        let mut previous = -1.0;
        for &p in &grid {
            let carbon = look.loads(p, [0.0; 3], 0.0).carbon;
            assert!(carbon > previous || (p == 0.0 && carbon == 0.0));
            previous = carbon;
        }
        assert_eq!(look.loads(1.0, [0.0; 3], 0.0).carbon, config.floor);
    }

    #[test]
    fn loads_are_symmetric_in_the_bodies() {
        let (look, _) = production();
        let e = [0.9f32, 0.07, 0.4];
        let reference = look.loads(0.8, e, 0.0);
        for perm in [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
            assert_eq!(look.loads(0.8, perm.map(|i| e[i]), 0.0), reference);
        }
    }

    #[test]
    fn meeting_is_the_cinnabar_that_forms_now() {
        let (look, config) = production();
        for e in
            [[1.0f32, 1.0, 0.0], [0.02, 0.03, 0.0], [0.9, 0.0, 0.004], [0.0; 3], [0.3, 0.5, 0.7]]
        {
            let loads = look.loads(1.0, e, 0.0);
            assert_eq!(look.meeting(1.0, e) * config.cinnabar_strength, loads.cinnabar, "{e:?}");
        }
    }

    #[test]
    fn an_ember_glows_crisply_until_it_cools_below_the_threshold() {
        let (look, config) = production();
        let threshold = config.meeting_threshold;
        // Old single-body ink (floor wash) carrying an ember from an earlier meeting.
        let (p, e) = (1.0f32, [0.0f32; 3]);
        let bare = look.loads(p, e, 0.0);
        assert_eq!(bare.cinnabar, 0.0);
        let hot = look.loads(p, e, 0.8);
        assert_eq!(hot.cinnabar, config.cinnabar_strength * f64::from(0.8f32));
        // The ember keeps the soot it formed with: carbon_keep of its own strength.
        assert_eq!(hot.carbon, config.carbon_keep * f64::from(0.8f32));
        // At or below the threshold the ember is out: exactly the ember-free loads.
        // The largest f32 not above the threshold (0.3 rounds up in f32).
        let nearest = threshold as f32;
        let at = if f64::from(nearest) > threshold {
            f32::from_bits(nearest.to_bits() - 1)
        } else {
            nearest
        };
        assert!(f64::from(at) <= threshold);
        assert_eq!(look.loads(p, e, at), bare);
        assert_eq!(look.loads(p, e, 0.1), bare);
        // Just above it, it still glows at full saturation (no pale-pink tail).
        let above = f32::from_bits(at.to_bits() + 1);
        let glowing = look.loads(p, e, above);
        assert_eq!(glowing.cinnabar, config.cinnabar_strength * f64::from(above));
        assert_eq!(glowing.carbon, config.carbon_keep * f64::from(above));
        // A fresh meeting brighter than the ember wins; a hotter ember wins over a weak meeting.
        let fresh = look.loads(1.0, [1.0, 1.0, 0.0], 0.4);
        assert_eq!(fresh, look.loads(1.0, [1.0, 1.0, 0.0], 0.0));
        assert_eq!(look.loads(1.0, [1.0, 1.0, 0.0], 1.0).cinnabar, config.cinnabar_strength);
        // An ember on otherwise clear water still shows (the precipitate outlives the carbon).
        assert!(!look.loads(0.0, [0.0; 3], 0.5).is_bare());
    }

    /// Documented behaviour (module docs, "Ember memory"): fresh ink of a single body laid into
    /// water that still glows renders vermilion, keeping only `carbon_keep` of its carbon.
    #[test]
    fn fresh_single_body_ink_over_a_glowing_ember_is_vermilion() {
        let (look, config) = production();
        let sumi = look.loads(1.0, [1.0, 0.0, 0.0], 0.0);
        assert_eq!(sumi, InkLoads { carbon: 1.0, cinnabar: 0.0 });
        let over_ember = look.loads(1.0, [1.0, 0.0, 0.0], 0.8);
        assert_eq!(over_ember.cinnabar, config.cinnabar_strength * f64::from(0.8f32));
        assert_eq!(over_ember.carbon, config.carbon_keep * 1.0);
    }

    /// Ink diluted with clear water — a fraction `w` of an unmixed parcel, `P = w` and
    /// `E_i = w·E_i⁰` — has `w` times the unmixed strength, at every age (module docs,
    /// "Dilution").
    #[test]
    fn clear_water_dilutes_the_ink_linearly() {
        let (look, config) = production();
        for w in [1.0f32, 0.7, 0.5, 0.1, 0.02, 1e-3] {
            for age in [0.0, 0.2, 0.45, 0.5, 0.55, 0.6, 0.8, 1.5, 3.0] {
                let e0 = math::exp(-age / config.fresh_tau);
                let unmixed = look.loads(1.0, [e0 as f32, 0.0, 0.0], 0.0);
                let diluted = look.loads(w, [(f64::from(w) * e0) as f32, 0.0, 0.0], 0.0);
                assert_eq!(diluted.cinnabar, 0.0);
                // f32 storage of the diluted E is the only difference: relative 2^-24 in E.
                let want = f64::from(w) * unmixed.carbon;
                assert!(close(diluted.carbon, want, 2e-7), "w {w}, age {age}: {diluted:?}");
            }
        }
        // Unmixed parcels (`P = 1`) saturate at 1 exactly as before (the prototype's law).
        assert_eq!(look.loads(1.0, [1.0, 0.0, 0.0], 0.0).carbon, 1.0);
        // For any fields, a node never carries more than its presence of either strength.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32) / (1u32 << 24) as f32
        };
        for _ in 0..20_000 {
            let p = next();
            let e = [next(), next() * next(), next() * 0.02];
            let loads = look.loads(p, e, 0.0);
            let presence = f64::from(p);
            assert!(loads.carbon <= presence * (1.0 + 1e-15), "{p} {e:?}: {loads:?}");
            assert!(loads.cinnabar <= config.cinnabar_strength * presence * (1.0 + 1e-15));
        }
    }

    /// Faint dilutions of fresh ink neither draw at full strength nor meet: 2 % of fresh sumi
    /// draws 2 % of the carbon, and two bodies' 2 % dilutions stay far below the meeting
    /// threshold (so they neither form cinnabar nor seed an ember). A fresh meeting diluted with
    /// clear water goes out once its strength falls below the threshold, while the 50/50
    /// interface of two fresh, unmixed inks is a full-strength meeting.
    #[test]
    fn faint_dilutions_neither_draw_at_full_strength_nor_meet() {
        let (look, config) = production();
        let faint = look.loads(0.02, [0.02, 0.0, 0.0], 0.0);
        let two_percent =
            config.floor * f64::from(0.02f32) + (1.0 - config.floor) * f64::from(0.02f32);
        assert_eq!(faint, InkLoads { carbon: two_percent, cinnabar: 0.0 });
        assert_eq!(look.meeting(0.04, [0.02, 0.02, 0.0]), 0.0);
        let met = look.loads(0.04, [0.02, 0.02, 0.0], 0.0);
        assert_eq!(met.cinnabar, 0.0);
        assert!(met.carbon < 0.041, "{met:?}");
        // A diluted fresh meeting keeps a proportional strength until it falls below 0.3.
        let half = look.loads(0.5, [0.5, 0.5, 0.0], 0.0);
        assert!(close(half.cinnabar, 0.5 * config.cinnabar_strength, 1e-12), "{half:?}");
        assert_eq!(look.meeting(0.25, [0.25, 0.25, 0.0]), 0.0);
        // Two fresh unmixed inks mixed 50/50 (P = 1): a full-strength meeting.
        assert_eq!(look.loads(1.0, [0.5, 0.5, 0.0], 0.0).cinnabar, config.cinnabar_strength);
    }

    /// With no ember (`K = 0`, what the ink fields hold without ember memory) the loads are the
    /// prototype's rule of the module docs, written out literally here, bit for bit — for any
    /// presence and freshness, mixed or not.
    #[test]
    fn without_an_ember_the_loads_are_the_prototype_rule_bit_for_bit() {
        let (look, config) = production();
        let prototype = |p: f32, e: [f32; 3]| {
            let p = f64::from(p);
            let h = e.map(|e| {
                let boosted = f64::from(e) * look.hold_gain;
                if boosted < p { boosted } else { p }
            });
            let base = config.floor * p;
            let c = h.map(|h| base + look.fresh_weight * h);
            let h_max = h.iter().copied().fold(0.0, f64::max);
            let mono = base + look.fresh_weight * h_max;
            let mut best = 0.0f64;
            for (i, j) in [(0, 1), (1, 2), (0, 2)] {
                best = best.max(c[i].min(c[j]));
            }
            let best = if best > config.meeting_threshold { best } else { 0.0 };
            InkLoads {
                carbon: mono * if best > 0.0 { config.carbon_keep } else { 1.0 },
                cinnabar: config.cinnabar_strength * best,
            }
        };
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32) / (1u32 << 24) as f32
        };
        let levels = [0.0f32, 1e-12, 0.004, 0.02, 0.3, 0.5, 0.99, 1.0];
        let mut cinnabar_cases = 0;
        for &p in &levels {
            for &a in &levels {
                for &b in &levels {
                    for e in [[a, b, 0.0], [a, 0.0, b], [a, b, next()], [next(), a, b]] {
                        let got = look.loads(p, e, 0.0);
                        let want = prototype(p, e);
                        assert_eq!(
                            (got.carbon.to_bits(), got.cinnabar.to_bits()),
                            (want.carbon.to_bits(), want.cinnabar.to_bits()),
                            "P {p}, E {e:?}: {got:?} vs {want:?}"
                        );
                        cinnabar_cases += usize::from(want.cinnabar > 0.0);
                    }
                }
            }
        }
        for _ in 0..20_000 {
            let (p, e) = (next(), [next(), next(), next()]);
            let (got, want) = (look.loads(p, e, 0.0), prototype(p, e));
            assert_eq!(
                (got.carbon.to_bits(), got.cinnabar.to_bits()),
                (want.carbon.to_bits(), want.cinnabar.to_bits())
            );
            cinnabar_cases += usize::from(want.cinnabar > 0.0);
        }
        assert!(cinnabar_cases > 100, "{cinnabar_cases}");
    }
}
