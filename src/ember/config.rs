//! Configuration of the ember edition.
//!
//! [`EmberConfig::default`] is the production look: pine-soot sumi on kozo, laid by three
//! tidally stretched bodies, black while fresh and fading to a pale grey wash over about six
//! seconds of the film. Every field is recorded in the per-package certificate
//! (`metadata/ember.json`), because the rendered bits are a pure function of the orbit, the
//! output size, the paper seed and this configuration.
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
    /// When and where water picks up ink.
    pub contact: ContactConfig,
    /// The bodies' tidal stretching.
    pub tidal: TidalConfig,
    /// Tone law.
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
    /// Radius of the disc of equal area to every body (the bodies are ellipses of this area).
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
    /// Largest body travel between consecutive snapshots, in body radii (`body_radius`); with
    /// stretched bodies this should stay below the short semi-axis `R/√max_aspect`.
    pub max_snapshot_travel: f64,
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

/// Tidal stretching of the bodies (docs/ember-design.md §3.10): each body is an ellipse of the
/// disc's area, stretched along the principal axis of the other two bodies' tidal field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TidalConfig {
    /// Largest axis ratio `a/b`; 1 keeps the bodies discs.
    pub max_aspect: f64,
    /// Quantile of the orbit's tidal anisotropy at which a body shows half its extra stretch:
    /// 0.95 keeps the bodies nearly round most of the time and stretches them at the closest
    /// 5% of moments.
    pub stretch_quantile: f64,
    /// Plummer softening length of the tidal field (world units).
    pub softening: f64,
}

/// Tone law: the museum-lab reservoir feed with a hold, timed as a fraction of the film.
///
/// A parcel's ink is full strength for `hold_fraction·T` after a body last inked it, then decays
/// with the e-folding time `fade_fraction·T` towards the `floor` wash, where `T` is the orbit's
/// duration in fluid units. The film shows the whole orbit, so these are fractions of the film:
/// the defaults hold for 0.8 s and fade with a 0.75 s time constant in a 30-second film, and the
/// ink reaches the floor about six seconds after it was laid, on every orbit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookConfig {
    /// Carbon load of ink older than the reservoir memory (the palest wash).
    pub floor: f64,
    /// E-folding time of the fade, as a fraction of the orbit's duration.
    pub fade_fraction: f64,
    /// Time for which fresh ink stays full strength, as a fraction of the orbit's duration. At
    /// most `ln(10⁶) ≈ 13.8` times `fade_fraction` (see [`EmberConfig::validate`]).
    pub hold_fraction: f64,
    /// Optional e-folding age of the presence `P`: the floor wash fades with it, and so (the
    /// tone law caps every strength at `P`) does the whole deposit. `None` keeps it forever.
    ///
    /// Required in JSON (as `null` or a number): a missing key is an error, not `None`, so that
    /// a certificate cannot silently lose the setting.
    #[serde(deserialize_with = "Option::deserialize")]
    pub floor_tau: Option<f64>,
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
    /// The production look: sumi laid by tidally stretched bodies, fading to grey over about six
    /// seconds of the film, on the masters' fluid resolved 1.5 times more finely.
    fn default() -> Self {
        Self {
            fluid: FluidConfig {
                rows: 1536,
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
                max_snapshot_travel: 0.28,
            },
            contact: ContactConfig {
                soak_depth: 0.030,
                vorticity_gate: 40.0,
                pre_roll: 0.5,
                valve_lead: 0.25,
            },
            tidal: TidalConfig { max_aspect: 3.0, stretch_quantile: 0.95, softening: 0.1 },
            look: LookConfig {
                floor: 0.0004,
                fade_fraction: 0.75 / 30.0,
                hold_fraction: 0.8 / 30.0,
                floor_tau: None,
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
            raster: RasterConfig { supersample: 3, ink_margin: 0.15 },
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
/// `g = exp(hold / tau)`, so ink crossing the flush loses `g·FLUSH` of strength in one step;
/// beyond this, ink younger than the hold would drop from full strength to the floor wash.
const MAX_FLUSHED_STRENGTH: f64 = 1e-6;

/// Largest supported `look.hold_fraction / look.fade_fraction`: `ln(MAX_FLUSHED_STRENGTH / FLUSH)`
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

        let c = &self.contact;
        non_negative(c.soak_depth, "contact.soak_depth")?;
        check(c.vorticity_gate.is_finite(), "contact.vorticity_gate", "must be finite")?;
        non_negative(c.pre_roll, "contact.pre_roll")?;
        non_negative(c.valve_lead, "contact.valve_lead")?;

        let t = &self.tidal;
        check(
            t.max_aspect.is_finite() && (1.0..=10.0).contains(&t.max_aspect),
            "tidal.max_aspect",
            "must lie in [1, 10]",
        )?;
        check(
            t.stretch_quantile.is_finite() && t.stretch_quantile > 0.0 && t.stretch_quantile <= 1.0,
            "tidal.stretch_quantile",
            "must lie in (0, 1]",
        )?;
        positive(t.softening, "tidal.softening")?;

        let l = &self.look;
        unit(l.floor, "look.floor")?;
        check(l.floor < 1.0, "look.floor", "must be < 1")?;
        check(
            l.fade_fraction.is_finite() && l.fade_fraction > 0.0 && l.fade_fraction <= 1.0,
            "look.fade_fraction",
            "must lie in (0, 1]",
        )?;
        unit(l.hold_fraction, "look.hold_fraction")?;
        let max_ratio = max_hold_ratio();
        check(
            l.hold_fraction / l.fade_fraction <= max_ratio,
            "look.hold_fraction",
            &format!(
                "hold_fraction / fade_fraction is {ratio} but must be <= {max_ratio:.3}: \
                 freshness below {flush:e} is flushed to 0, which cuts exp(hold / tau)·{flush:e} \
                 of strength at once (at most {MAX_FLUSHED_STRENGTH:e} is allowed), so ink \
                 younger than the hold would drop from full strength to the floor wash; lower \
                 hold_fraction or raise fade_fraction",
                ratio = l.hold_fraction / l.fade_fraction,
                flush = ink::FLUSH,
            ),
        )?;
        if let Some(tau) = l.floor_tau {
            positive(tau, "look.floor_tau")?;
        }

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
    fn default_config_is_the_tidal_sumi_look() {
        let c = EmberConfig::default();
        // The masters' physics, resolved 1.5 times more finely.
        assert_eq!(c.fluid.rows, 1536);
        assert_eq!(c.fluid.reynolds, 300.0);
        assert_eq!(c.fluid.body_radius, 0.05);
        assert_eq!(c.fluid.mask_width, 0.004);
        assert_eq!(c.contact.soak_depth, 0.030);
        assert_eq!(c.contact.vorticity_gate, 40.0);
        assert_eq!(c.contact.valve_lead, 0.25);
        assert_eq!(c.contact.pre_roll, 0.5);
        // Tidally stretched bodies, up to 3:1 at the closest 5% of moments.
        assert_eq!(c.tidal.max_aspect, 3.0);
        assert_eq!(c.tidal.stretch_quantile, 0.95);
        assert_eq!(c.tidal.softening, 0.1);
        // Black for 0.8 s, then a 0.75 s e-folding fade, in a 30-second film.
        assert_eq!(c.look.floor, 0.0004);
        assert_eq!(c.look.hold_fraction * 30.0, 0.8);
        assert_eq!(c.look.fade_fraction * 30.0, 0.75);
        assert_eq!(c.look.floor_tau, None);
        // Nine ink nodes per output pixel.
        assert_eq!(c.raster.supersample, 3);
        // Snapshots close enough that a body moves under half its shortest semi-axis.
        let shortest = c.fluid.body_radius / c.tidal.max_aspect.sqrt();
        assert!(c.fluid.max_snapshot_travel * c.fluid.body_radius < 0.5 * shortest);
    }

    #[test]
    fn invalid_values_name_the_parameter() {
        let cases: [(&str, fn(&mut EmberConfig)); 9] = [
            ("fluid.cfl", |c| c.fluid.cfl = 0.0),
            ("raster.supersample", |c| c.raster.supersample = 0),
            ("look.floor_tau", |c| c.look.floor_tau = Some(f64::NAN)),
            ("raster.ink_margin", |c| c.raster.ink_margin = c.fluid.box_margin),
            ("tidal.max_aspect", |c| c.tidal.max_aspect = 0.5),
            ("tidal.stretch_quantile", |c| c.tidal.stretch_quantile = 0.0),
            ("tidal.softening", |c| c.tidal.softening = 0.0),
            ("look.fade_fraction", |c| c.look.fade_fraction = 0.0),
            ("look.hold_fraction", |c| c.look.hold_fraction = -0.1),
        ];
        for (parameter, break_it) in cases {
            let mut c = EmberConfig::default();
            break_it(&mut c);
            let err = c.validate().expect_err(parameter).to_string();
            assert!(err.contains(parameter), "{parameter}: {err}");
        }
    }

    /// `hold_fraction / fade_fraction` is bounded by what the ink's freshness flush allows: at the
    /// bound the flush cuts `g·FLUSH = 10⁻⁶` of strength; the production look is far below it.
    #[test]
    fn hold_is_bounded_by_the_freshness_flush() {
        let max_ratio = max_hold_ratio();
        assert!((max_ratio - math::ln(1e6)).abs() < 1e-12, "{max_ratio}");
        let flushed = math::exp(max_ratio) * ink::FLUSH;
        assert!((flushed - MAX_FLUSHED_STRENGTH).abs() < 1e-18, "{flushed}");

        let production = EmberConfig::default();
        assert!(production.look.hold_fraction / production.look.fade_fraction < max_ratio / 3.0);

        let mut c = EmberConfig::default();
        c.look.hold_fraction = max_ratio * c.look.fade_fraction * (1.0 - 1e-12);
        c.validate().expect("just below the bound validates");
        c.look.hold_fraction = 15.0 * c.look.fade_fraction;
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("look.hold_fraction") && err.contains("flushed"), "{err}");
        c.look.hold_fraction = 0.0;
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
        // A certificate of the retired vermilion look does not read as this configuration.
        let mut old = serde_json::to_value(&config).expect("serialises");
        old["look"]["ember_tau"] = serde_json::json!(1.0);
        assert!(serde_json::from_value::<EmberConfig>(old).is_err());
    }

    /// `floor_tau` is an optional value but a required key: `null` reads as `None`, a missing key
    /// is an error naming it (serde would otherwise default it silently).
    #[test]
    fn the_optional_look_field_is_a_required_key() {
        let json = serde_json::to_value(EmberConfig::default()).expect("serialises");
        let mut explicit_null = json.clone();
        explicit_null["look"]["floor_tau"] = serde_json::Value::Null;
        let back: EmberConfig = serde_json::from_value(explicit_null).expect("null is None");
        assert_eq!(back.look.floor_tau, None);

        let mut missing = json;
        missing["look"].as_object_mut().expect("look object").remove("floor_tau");
        let error = serde_json::from_value::<EmberConfig>(missing).expect_err("missing key");
        assert!(error.to_string().contains("missing field `floor_tau`"), "{error}");
    }

    #[test]
    fn config_serialises_deterministically() {
        let a = serde_json::to_string(&EmberConfig::default()).expect("serialises");
        let b = serde_json::to_string(&EmberConfig::default()).expect("serialises");
        assert_eq!(a, b);
        assert!(a.contains("\"floor_tau\":null"));
    }
}
