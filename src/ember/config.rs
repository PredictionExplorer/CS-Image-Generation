//! Configuration of the ember edition.
//!
//! [`EmberConfig::default`] is the production look: the museum-lab prototype's vermilion look
//! ("the ember alone") plus ember memory (`look.ember_tau = 1.0`), the product's one addition.
//! Every field is recorded in the per-package certificate (`metadata/ember.json`), because the
//! rendered bits are a pure function of the orbit, the output size, the paper seed and this
//! configuration.
//!
//! Units: lengths are fluid/world units (the canvas spans `y ∈ [-1, 1]`), times are fluid time
//! units (the median body speed is [`FluidConfig::reference_speed`]), paper sizes are millimetres
//! of the physical sheet the output depicts.

use serde::{Deserialize, Serialize};

use super::error::{EmberError, EmberResult};
use super::{ink, math};

/// Every tunable of the ember edition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmberConfig {
    /// Navier–Stokes solver and snapshot cadence.
    pub fluid: FluidConfig,
    /// Projection of the 3-D orbit onto the canvas.
    pub projection: ProjectionConfig,
    /// When and where water picks up ink.
    pub contact: ContactConfig,
    /// Tone law and the vermilion accent.
    pub look: LookConfig,
    /// The kozo sheet.
    pub paper: PaperConfig,
    /// Ink raster (supersampling and margin).
    pub raster: RasterConfig,
}

/// Navier–Stokes solver parameters (mirrors the museum-lab `wake/ns.py` defaults used for the
/// masters).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidConfig {
    /// Grid rows across the box height (rounded up to an even 5-smooth count).
    pub rows: usize,
    /// Periodic-box margin beyond the canvas on every side.
    pub box_margin: f64,
    /// Reynolds number based on the body diameter and the reference speed.
    pub reynolds: f64,
    /// Median body speed; sets the orbit-to-fluid time map.
    pub reference_speed: f64,
    /// Radius of every body disc.
    pub body_radius: f64,
    /// Courant number of the adaptive time step.
    pub cfl: f64,
    /// Upper bound of the time step.
    pub max_dt: f64,
    /// Brinkman permeability as a fraction of the time step (penalisation `1/eta_ratio`).
    pub brinkman_eta_ratio: f64,
    /// Width of the tanh edge of the body mask.
    pub mask_width: f64,
    /// Vorticity damping rate of the sponge outside the canvas.
    pub sponge_rate: f64,
    /// Gap between the canvas edge and the start of the sponge ramp.
    pub sponge_pad: f64,
    /// Hyperviscosity coefficient (per grid spacing) of the `(k/kc)^24` filter.
    pub hyperviscosity: f64,
    /// Largest fluid-time gap between consecutive velocity snapshots used for tracing.
    pub max_snapshot_interval: f64,
    /// Largest body travel between consecutive snapshots, in body radii.
    pub max_snapshot_travel: f64,
}

/// Projection of the orbit onto the canvas.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionConfig {
    /// Fraction of the canvas half-height (and half-width) the orbit fills.
    pub fill: f64,
}

/// Soak-zone contact rules: where and when passing water picks up ink.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactConfig {
    /// Depth of the soak zone beyond the body radius.
    pub soak_depth: f64,
    /// Only water spinning faster than this (|ω|) picks up ink; `<= 0` disables the gate.
    pub vorticity_gate: f64,
    /// Fluid time before which nothing inks (the impulsive start settles first).
    pub pre_roll: f64,
    /// The bodies stop inking this long before the final step, so the last frame shows ink
    /// released into the flow rather than attached to the bodies.
    pub valve_lead: f64,
}

/// Tone law and vermilion accent. The default is the museum-lab look `vermilion` ("the ember
/// alone": `ember(1.2, 0.06, p_min = 0.3)` over the reservoir feed
/// `ExpFeed(tau = 0.12, floor = 0.0004)` with a 0.5 hold) plus ember memory
/// (`ember_tau = 1.0`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookConfig {
    /// Carbon load of ink older than the reservoir memory (the palest wash).
    pub floor: f64,
    /// E-folding age of the reservoir feed.
    pub fresh_tau: f64,
    /// Age below which ink is full strength. At most `ln(10⁶) ≈ 13.8` times `fresh_tau` (see
    /// [`EmberConfig::validate`]).
    pub hold: f64,
    /// Optional e-folding age of the presence `P`: the floor wash fades with it, and so (the
    /// tone law caps every strength at `P`) does the whole deposit. `None` keeps it forever.
    ///
    /// Required in JSON (as `null` or a number): a missing key is an error, not `None`, so that
    /// a certificate cannot silently lose the setting.
    #[serde(deserialize_with = "Option::deserialize")]
    pub floor_tau: Option<f64>,
    /// Ember memory: cinnabar that formed where two fresh inks met lingers in the water and cools
    /// with this e-folding age, showing (crisply) while hotter than `meeting_threshold`. `None`
    /// is the prototype's rule (vermilion only while both inks are fresh), under which the
    /// step-1M still of most orbits has no vermilion at all.
    ///
    /// Required in JSON (as `null` or a number): a missing key is an error rather than the
    /// prototype's rule.
    #[serde(deserialize_with = "Option::deserialize")]
    pub ember_tau: Option<f64>,
    /// Cinnabar load per unit of shared fresh ink.
    pub cinnabar_strength: f64,
    /// Fraction of the carbon kept where cinnabar is laid down.
    pub carbon_keep: f64,
    /// Two bodies' ink must both exceed this strength for cinnabar to appear.
    pub meeting_threshold: f64,
}

/// The kozo sheet: formation (flocs) and visible fibres.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PaperConfig {
    /// Physical width of the sheet the output depicts (millimetres).
    pub sheet_width_mm: f64,
    /// Number of random cosine modes of the formation field.
    pub formation_modes: usize,
    /// Smallest floc wavelength (mm).
    pub floc_min_mm: f64,
    /// Largest floc wavelength (mm).
    pub floc_max_mm: f64,
    /// Along-flow stretching of the flocs.
    pub anisotropy: f64,
    /// Absorption mottle per unit of formation.
    pub mottle_formation: f64,
    /// Absorption reduction per unit of fibre coverage.
    pub mottle_fibre: f64,
    /// Ink-gain reduction per unit of fibre coverage (fibres shed ink).
    pub ink_gain_fibre: f64,
    /// Visible fibres.
    pub fibres: FibreConfig,
}

/// Visible surface fibres of the kozo sheet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FibreConfig {
    /// Visible fibres per square centimetre.
    pub density_per_cm2: f64,
    /// 5th percentile of the log-normal fibre length (mm).
    pub length_p5_mm: f64,
    /// 95th percentile of the log-normal fibre length (mm).
    pub length_p95_mm: f64,
    /// Smallest fibre width (micrometres).
    pub width_min_um: f64,
    /// Largest fibre width (micrometres).
    pub width_max_um: f64,
    /// Smallest fibre visibility.
    pub visibility_min: f64,
    /// Largest fibre visibility.
    pub visibility_max: f64,
    /// Fraction of fibres aligned with the machine direction.
    pub aligned_fraction: f64,
    /// Standard deviation of the aligned fibres' angle (degrees).
    pub aligned_spread_deg: f64,
    /// Standard deviation of a fibre's constant curvature (radians per millimetre).
    pub bend_rad_per_mm: f64,
    /// Random-walk wander of the fibre direction (radians per square-root millimetre).
    pub wander_rad_per_sqrt_mm: f64,
    /// Polyline vertex spacing (mm).
    pub vertex_spacing_mm: f64,
    /// Fibres are seeded over the sheet padded by this margin (mm).
    pub margin_mm: f64,
}

/// Ink raster.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RasterConfig {
    /// Ink nodes per output pixel along each axis (samples per pixel = supersample²).
    pub supersample: u32,
    /// Ink carried beyond the canvas edge (world units); ink further out is lost.
    pub ink_margin: f64,
}

impl Default for EmberConfig {
    /// The production look: the museum-lab prototype's vermilion look ("the ember alone") on the
    /// masters' fluid, plus ember memory (`look.ember_tau = Some(1.0)`).
    fn default() -> Self {
        Self {
            fluid: FluidConfig {
                rows: 1024,
                box_margin: 0.35,
                reynolds: 300.0,
                reference_speed: 1.0,
                body_radius: 0.05,
                cfl: 0.5,
                max_dt: 2e-3,
                brinkman_eta_ratio: 0.01,
                mask_width: 0.004,
                sponge_rate: 25.0,
                sponge_pad: 0.06,
                hyperviscosity: 144.0,
                max_snapshot_interval: 2.5e-3,
                max_snapshot_travel: 0.5,
            },
            projection: ProjectionConfig { fill: 0.78 },
            contact: ContactConfig {
                soak_depth: 0.030,
                vorticity_gate: 40.0,
                pre_roll: 0.5,
                valve_lead: 0.25,
            },
            look: LookConfig {
                floor: 0.0004,
                fresh_tau: 0.12,
                hold: 0.5,
                floor_tau: None,
                ember_tau: Some(1.0),
                cinnabar_strength: 1.2,
                carbon_keep: 0.06,
                meeting_threshold: 0.3,
            },
            paper: PaperConfig {
                sheet_width_mm: 1490.0,
                formation_modes: 1024,
                floc_min_mm: 1.5,
                floc_max_mm: 8.0,
                anisotropy: 1.2,
                mottle_formation: 0.2,
                mottle_fibre: 0.6,
                ink_gain_fibre: 0.5,
                fibres: FibreConfig {
                    density_per_cm2: 1.5,
                    length_p5_mm: 5.0,
                    length_p95_mm: 15.0,
                    width_min_um: 10.0,
                    width_max_um: 20.0,
                    visibility_min: 0.3,
                    visibility_max: 1.0,
                    aligned_fraction: 0.35,
                    aligned_spread_deg: 20.0,
                    bend_rad_per_mm: 0.04,
                    wander_rad_per_sqrt_mm: 0.06,
                    vertex_spacing_mm: 0.25,
                    margin_mm: 15.0,
                },
            },
            raster: RasterConfig { supersample: 2, ink_margin: 0.15 },
        }
    }
}

/// Rejects a value unless `ok`.
fn check(ok: bool, parameter: &str, reason: &str) -> EmberResult<()> {
    if ok {
        Ok(())
    } else {
        Err(EmberError::InvalidConfig { parameter: parameter.to_string(), reason: reason.into() })
    }
}

/// `value` is finite and strictly positive.
fn positive(value: f64, parameter: &str) -> EmberResult<()> {
    check(value.is_finite() && value > 0.0, parameter, "must be finite and > 0")
}

/// `value` is finite and non-negative.
fn non_negative(value: f64, parameter: &str) -> EmberResult<()> {
    check(value.is_finite() && value >= 0.0, parameter, "must be finite and >= 0")
}

/// Largest strength the ink fields' freshness flush may cut from the tone law. Freshness below
/// [`ink::FLUSH`] is stored as exactly 0, and the look multiplies freshness by
/// `g = exp(hold / fresh_tau)`, so ink crossing the flush loses `g·FLUSH` of strength in one
/// step; beyond this, ink younger than `hold` would drop from full strength to the floor wash.
const MAX_FLUSHED_STRENGTH: f64 = 1e-6;

/// Largest supported `look.hold / look.fresh_tau`: `ln(MAX_FLUSHED_STRENGTH / FLUSH)`
/// `= -ln(FLUSH) - ln(10⁶) ≈ 13.8` (through `math`, so every CPU accepts the same configs).
fn max_hold_ratio() -> f64 {
    math::ln(MAX_FLUSHED_STRENGTH) - math::ln(ink::FLUSH)
}

/// `value` lies in the closed unit interval.
fn unit(value: f64, parameter: &str) -> EmberResult<()> {
    check(value.is_finite() && (0.0..=1.0).contains(&value), parameter, "must lie in [0, 1]")
}

impl EmberConfig {
    /// Checks every field's range. Orbit-dependent checks (e.g. that the orbit outlasts the
    /// pre-roll and valve) happen when the orbit is known.
    pub fn validate(&self) -> EmberResult<()> {
        let f = &self.fluid;
        check((16..=8192).contains(&f.rows), "fluid.rows", "must lie in [16, 8192]")?;
        positive(f.box_margin, "fluid.box_margin")?;
        positive(f.reynolds, "fluid.reynolds")?;
        positive(f.reference_speed, "fluid.reference_speed")?;
        positive(f.body_radius, "fluid.body_radius")?;
        check(f.cfl.is_finite() && f.cfl > 0.0 && f.cfl <= 1.0, "fluid.cfl", "must lie in (0, 1]")?;
        positive(f.max_dt, "fluid.max_dt")?;
        positive(f.brinkman_eta_ratio, "fluid.brinkman_eta_ratio")?;
        positive(f.mask_width, "fluid.mask_width")?;
        non_negative(f.sponge_rate, "fluid.sponge_rate")?;
        non_negative(f.sponge_pad, "fluid.sponge_pad")?;
        check(
            f.sponge_pad < f.box_margin,
            "fluid.sponge_pad",
            "must be smaller than fluid.box_margin",
        )?;
        non_negative(f.hyperviscosity, "fluid.hyperviscosity")?;
        positive(f.max_snapshot_interval, "fluid.max_snapshot_interval")?;
        positive(f.max_snapshot_travel, "fluid.max_snapshot_travel")?;

        let p = &self.projection;
        check(
            p.fill.is_finite() && p.fill > 0.0 && p.fill <= 1.0,
            "projection.fill",
            "must lie in (0, 1]",
        )?;

        let c = &self.contact;
        non_negative(c.soak_depth, "contact.soak_depth")?;
        check(c.vorticity_gate.is_finite(), "contact.vorticity_gate", "must be finite")?;
        non_negative(c.pre_roll, "contact.pre_roll")?;
        non_negative(c.valve_lead, "contact.valve_lead")?;

        let l = &self.look;
        unit(l.floor, "look.floor")?;
        check(l.floor < 1.0, "look.floor", "must be < 1")?;
        positive(l.fresh_tau, "look.fresh_tau")?;
        non_negative(l.hold, "look.hold")?;
        let max_ratio = max_hold_ratio();
        check(
            l.hold / l.fresh_tau <= max_ratio,
            "look.hold",
            &format!(
                "hold / fresh_tau is {ratio} but must be <= {max_ratio:.3}: freshness below \
                 {flush:e} is flushed to 0, which cuts exp(hold / fresh_tau)·{flush:e} of \
                 strength at once (at most {MAX_FLUSHED_STRENGTH:e} is allowed), so ink younger \
                 than the hold would drop from full strength to the floor wash; lower hold or \
                 raise fresh_tau",
                ratio = l.hold / l.fresh_tau,
                flush = ink::FLUSH,
            ),
        )?;
        if let Some(tau) = l.floor_tau {
            positive(tau, "look.floor_tau")?;
        }
        if let Some(tau) = l.ember_tau {
            positive(tau, "look.ember_tau")?;
        }
        non_negative(l.cinnabar_strength, "look.cinnabar_strength")?;
        unit(l.carbon_keep, "look.carbon_keep")?;
        unit(l.meeting_threshold, "look.meeting_threshold")?;
        // The ink fields cannot tell which bodies ever inked old water, so every body contributes
        // the floor wash to the meeting test; that is exact only while the floor cannot meet.
        check(l.floor <= l.meeting_threshold, "look.meeting_threshold", "must be >= look.floor")?;

        let paper = &self.paper;
        positive(paper.sheet_width_mm, "paper.sheet_width_mm")?;
        check(
            (1..=65_536).contains(&paper.formation_modes),
            "paper.formation_modes",
            "must lie in [1, 65536]",
        )?;
        positive(paper.floc_min_mm, "paper.floc_min_mm")?;
        check(
            paper.floc_max_mm.is_finite() && paper.floc_max_mm >= paper.floc_min_mm,
            "paper.floc_max_mm",
            "must be finite and >= paper.floc_min_mm",
        )?;
        positive(paper.anisotropy, "paper.anisotropy")?;
        non_negative(paper.mottle_formation, "paper.mottle_formation")?;
        non_negative(paper.mottle_fibre, "paper.mottle_fibre")?;
        non_negative(paper.ink_gain_fibre, "paper.ink_gain_fibre")?;
        let fib = &paper.fibres;
        non_negative(fib.density_per_cm2, "paper.fibres.density_per_cm2")?;
        positive(fib.length_p5_mm, "paper.fibres.length_p5_mm")?;
        check(
            fib.length_p95_mm.is_finite() && fib.length_p95_mm >= fib.length_p5_mm,
            "paper.fibres.length_p95_mm",
            "must be finite and >= length_p5_mm",
        )?;
        positive(fib.width_min_um, "paper.fibres.width_min_um")?;
        check(
            fib.width_max_um.is_finite() && fib.width_max_um >= fib.width_min_um,
            "paper.fibres.width_max_um",
            "must be finite and >= width_min_um",
        )?;
        unit(fib.visibility_min, "paper.fibres.visibility_min")?;
        unit(fib.visibility_max, "paper.fibres.visibility_max")?;
        check(
            fib.visibility_max >= fib.visibility_min,
            "paper.fibres.visibility_max",
            "must be >= visibility_min",
        )?;
        unit(fib.aligned_fraction, "paper.fibres.aligned_fraction")?;
        non_negative(fib.aligned_spread_deg, "paper.fibres.aligned_spread_deg")?;
        non_negative(fib.bend_rad_per_mm, "paper.fibres.bend_rad_per_mm")?;
        non_negative(fib.wander_rad_per_sqrt_mm, "paper.fibres.wander_rad_per_sqrt_mm")?;
        positive(fib.vertex_spacing_mm, "paper.fibres.vertex_spacing_mm")?;
        non_negative(fib.margin_mm, "paper.fibres.margin_mm")?;

        let r = &self.raster;
        check((1..=4).contains(&r.supersample), "raster.supersample", "must lie in [1, 4]")?;
        non_negative(r.ink_margin, "raster.ink_margin")?;
        check(
            r.ink_margin < f.box_margin,
            "raster.ink_margin",
            "must be smaller than fluid.box_margin",
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        EmberConfig::default().validate().expect("the production look validates");
    }

    #[test]
    fn default_config_matches_the_museum_lab_vermilion_master() {
        let c = EmberConfig::default();
        assert_eq!(c.fluid.reynolds, 300.0);
        assert_eq!(c.fluid.body_radius, 0.05);
        assert_eq!(c.fluid.mask_width, 0.004);
        assert_eq!(c.contact.soak_depth, 0.030);
        assert_eq!(c.contact.vorticity_gate, 40.0);
        assert_eq!(c.contact.valve_lead, 0.25);
        assert_eq!(c.contact.pre_roll, 0.5);
        assert_eq!(c.look.fresh_tau, 0.12);
        assert_eq!(c.look.floor, 0.0004);
        assert_eq!(c.look.cinnabar_strength, 1.2);
        assert_eq!(c.look.carbon_keep, 0.06);
        assert_eq!(c.look.meeting_threshold, 0.3);
        // The product's one addition to the prototype's look: embers glow on.
        assert_eq!(c.look.ember_tau, Some(1.0));
    }

    #[test]
    fn invalid_values_name_the_parameter() {
        let mut c = EmberConfig::default();
        c.fluid.cfl = 0.0;
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("fluid.cfl"), "{err}");

        let mut c = EmberConfig::default();
        c.raster.supersample = 0;
        assert!(c.validate().unwrap_err().to_string().contains("raster.supersample"));

        let mut c = EmberConfig::default();
        c.look.floor_tau = Some(f64::NAN);
        assert!(c.validate().unwrap_err().to_string().contains("look.floor_tau"));

        let mut c = EmberConfig::default();
        c.raster.ink_margin = c.fluid.box_margin;
        assert!(c.validate().unwrap_err().to_string().contains("raster.ink_margin"));

        let mut c = EmberConfig::default();
        c.look.meeting_threshold = c.look.floor / 2.0;
        assert!(c.validate().unwrap_err().to_string().contains("look.meeting_threshold"));
    }

    /// `hold / fresh_tau` is bounded by what the ink's freshness flush allows: at the bound the
    /// flush cuts `g·FLUSH = 10⁻⁶` of strength; the production look is far below it.
    #[test]
    fn hold_is_bounded_by_the_freshness_flush() {
        let max_ratio = max_hold_ratio();
        assert!((max_ratio - math::ln(1e6)).abs() < 1e-12, "{max_ratio}");
        let flushed = math::exp(max_ratio) * ink::FLUSH;
        assert!((flushed - MAX_FLUSHED_STRENGTH).abs() < 1e-18, "{flushed}");

        let production = EmberConfig::default();
        assert!(production.look.hold / production.look.fresh_tau < max_ratio / 3.0);

        let mut c = EmberConfig::default();
        c.look.hold = max_ratio * c.look.fresh_tau * (1.0 - 1e-12);
        c.validate().expect("just below the bound validates");
        // A look-development fixture that used to validate: hold 3.0 over fresh_tau 0.12 (25×).
        c.look.hold = 3.0;
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("look.hold") && err.contains("flushed"), "{err}");
        c.look.fresh_tau = 0.25; // 12×
        c.validate().expect("hold 3.0 over fresh_tau 0.25 validates");
        c.look.hold = 0.0;
        c.validate().expect("no hold validates");
    }

    #[test]
    fn config_round_trips_through_json_and_rejects_unknown_fields() {
        let config = EmberConfig::default();
        let json = serde_json::to_string(&config).expect("serialises");
        let back: EmberConfig = serde_json::from_str(&json).expect("deserialises");
        assert_eq!(back, config);
        let typo = json.replace("\"cfl\"", "\"cfll\"");
        assert!(serde_json::from_str::<EmberConfig>(&typo).is_err(), "unknown fields are errors");
    }

    /// `floor_tau` and `ember_tau` are optional values but required keys: `null` reads as
    /// `None`, a missing key is an error naming it (serde would otherwise default it to `None`,
    /// silently turning a look with ember memory into the prototype's).
    #[test]
    fn the_optional_look_fields_are_required_keys() {
        let json = serde_json::to_value(EmberConfig::default()).expect("serialises");
        for key in ["floor_tau", "ember_tau"] {
            let mut explicit_null = json.clone();
            explicit_null["look"][key] = serde_json::Value::Null;
            let back: EmberConfig = serde_json::from_value(explicit_null).expect("null is None");
            assert_eq!(
                if key == "floor_tau" { back.look.floor_tau } else { back.look.ember_tau },
                None
            );

            let mut missing = json.clone();
            missing["look"].as_object_mut().expect("look object").remove(key);
            let error = serde_json::from_value::<EmberConfig>(missing).expect_err("missing key");
            assert!(error.to_string().contains(&format!("missing field `{key}`")), "{error}");
        }
    }

    #[test]
    fn config_serialises_deterministically() {
        let a = serde_json::to_string(&EmberConfig::default()).expect("serialises");
        let b = serde_json::to_string(&EmberConfig::default()).expect("serialises");
        assert_eq!(a, b);
        assert!(a.contains("\"floor_tau\":null"));
    }
}
