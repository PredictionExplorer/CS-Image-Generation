//! Photoelastic color between ideal crossed linear polarizers.
//!
//! A symmetric in-plane stress tensor supplies the principal-stress difference
//! and principal direction. At each wavelength the transmitted fraction is
//! `sin²(2(theta-alpha)) * sin²(pi * retardance_nm / wavelength_nm)`. The optical
//! retardance is the stress difference times thickness times the configured
//! stress-optic scale. Hydrostatic stress contributes no birefringence.
//!
//! This is a uniform-through-thickness, normally viewed, nondispersive
//! stress-optic plate model. Orbit-to-load mapping and the chosen stress-optic
//! scale are artistic decisions; the transmission law is the optical model.
//! There is no assigned body color, hue rotation, spectral tone curve, or bloom.
//! The repository's 64-bin observer weights are already normalized so unit
//! spectral samples sum to D65 white. They are a D65-calibrated color convention,
//! not a measured illuminant spectrum; neither dividing by 64 nor calling them
//! a blackbody spectrum would be correct.
//!
//! An optional fixed spectral filter changes the illuminating spectrum before
//! polarization. It is a broadband floor plus nonnegative Gaussian peaks,
//! normalized to unit incident Y with the same observer. Normalization models
//! adjusting lamp power after filtering; it is not a passive filter creating
//! energy. This spectral choice never follows a body or the current frame.

use crate::silk::{SilkResult, V3};
use crate::spectrum::{BIN_XYZ_LUT, NUM_BINS, wavelength_nm_for_bin, xyz_to_linear_srgb};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::sync::{Arc, LazyLock};

/// The quarter-nanometre grid resolves the shortest period by 1530 samples.
/// This also resolves the additional sensitivity of neutral-axis compression
/// near saturated magenta, where a small green-channel error changes the amount
/// of desaturation applied to a much brighter blue channel.
const TABLE_STEP_NM: f64 = 0.25;
const TABLE_MAX_NM: f64 = 8192.0;
const TABLE_INTERVALS: usize = 32_768;

/// Physical transmission controls and the declared artistic stress-optic scale.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpticsConfig {
    /// Retardance in nanometres per unit stress difference and unit thickness.
    /// This converts the normalized mechanical field to an optical material.
    pub retardance_scale_nm: f64,
    /// Fixed first-polarizer angle in degrees from the positive plate X axis.
    /// The analyzer remains exactly 90 degrees from this polarizer.
    pub polarizer_degrees: f64,
    /// Uniform incident spectral intensity after the first ideal polarizer.
    /// One supplies unit incident Y before the plate. The default is calibrated
    /// white; a filtered source can have a chromatic incident color.
    pub incident_intensity: f64,
    /// Fixed Gaussian spectral peaks added to the broadband floor. Empty keeps
    /// the established calibrated-white spectrum and its exact rendering path.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub spectral_peaks: Vec<SpectralPeak>,
    /// Nonnegative broadband part of the source filter, before Y normalization.
    /// At least this floor or one peak must be positive. Omitted at default one.
    #[serde(skip_serializing_if = "floor_is_one")]
    pub spectral_floor: f64,
}

/// One smooth, nonnegative lobe of the fixed illuminating spectral filter.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralPeak {
    /// Center wavelength in the supported 380 to 700 nanometre interval.
    pub center_nm: f64,
    /// Gaussian standard deviation in nanometres, at least 5 to resolve the
    /// peak on the existing wavelength grid; this is not its full width.
    pub width_nm: f64,
    /// Relative peak height before photopic Y normalization.
    pub weight: f64,
}

fn floor_is_one(value: &f64) -> bool {
    *value == 1.0
}

impl Default for OpticsConfig {
    fn default() -> Self {
        Self {
            retardance_scale_nm: 3000.0,
            polarizer_degrees: 0.0,
            incident_intensity: 1.0,
            spectral_peaks: Vec::new(),
            spectral_floor: 1.0,
        }
    }
}

impl OpticsConfig {
    /// Validate optical controls without constructing the shared spectral table.
    pub fn validate(&self) -> SilkResult<()> {
        if !self.retardance_scale_nm.is_finite() || self.retardance_scale_nm < 0.0 {
            return Err("crystal retardance_scale_nm must be finite and nonnegative".into());
        }
        if !self.polarizer_degrees.is_finite() {
            return Err("crystal polarizer_degrees must be finite".into());
        }
        if !self.incident_intensity.is_finite()
            || !(0.0..=1_000_000.0).contains(&self.incident_intensity)
        {
            return Err("crystal incident_intensity must be finite and in [0, 1000000]".into());
        }
        if !self.spectral_floor.is_finite()
            || !(0.0..=1_000_000.0).contains(&self.spectral_floor)
            || self.spectral_peaks.len() > 16
            || self.spectral_peaks.iter().any(|peak| {
                !peak.center_nm.is_finite()
                    || !(380.0..=700.0).contains(&peak.center_nm)
                    || !peak.width_nm.is_finite()
                    || !(5.0..=160.0).contains(&peak.width_nm)
                    || !peak.weight.is_finite()
                    || !(0.0..=1_000_000.0).contains(&peak.weight)
            })
            || (self.spectral_floor == 0.0
                && !self.spectral_peaks.iter().any(|peak| peak.weight > 0.0))
        {
            return Err("invalid crystal spectral filter: use at most 16 finite peaks in 380..700 nm, widths in 5..160 nm, and nonnegative floor/weights in 0..1000000 with some positive light".into());
        }
        Ok(())
    }

    fn is_broadband(&self) -> bool {
        self.spectral_peaks.iter().all(|peak| peak.weight == 0.0)
    }
}

/// Signed linear RGB and physical Y are interpolated before gamut compression.
/// Keeping the signed channels avoids interpolation of an already clipped hue.
#[derive(Clone, Copy, Debug)]
struct SpectralSample {
    rgb: V3,
    luminance: f64,
}

impl SpectralSample {
    fn lerp(self, other: Self, amount: f64) -> Self {
        Self {
            rgb: self.rgb.lerp(other.rgb, amount),
            luminance: self.luminance + amount * (other.luminance - self.luminance),
        }
    }

    fn nonnegative_rgb(self) -> V3 {
        let minimum = self.rgb.x.min(self.rgb.y).min(self.rgb.z);
        if minimum >= 0.0 {
            return self.rgb;
        }
        // Desaturate toward equal-luminance neutral only until the first channel
        // reaches zero. All channel differences scale by the same amount, so the
        // linear-RGB hue direction is preserved; this is not a perceptual hue
        // guarantee. The final max only removes roundoff at the gamut boundary.
        let neutral = V3::new(self.luminance, self.luminance, self.luminance);
        let amount = self.luminance / (self.luminance - minimum);
        (neutral + (self.rgb - neutral) * amount).max(V3::ZERO)
    }
}

type SpectralWeights = [V3; NUM_BINS];

fn broadband_weights() -> SpectralWeights {
    std::array::from_fn(|bin| {
        let (x, y, z, _) = BIN_XYZ_LUT[bin];
        V3::new(x, y, z)
    })
}

fn filter_value(config: &OpticsConfig, wavelength_nm: f64) -> f64 {
    config.spectral_peaks.iter().fold(config.spectral_floor, |sum, peak| {
        let coordinate = (wavelength_nm - peak.center_nm) / peak.width_nm;
        sum + peak.weight * (-0.5 * coordinate * coordinate).exp()
    })
}

fn filtered_weights(config: &OpticsConfig) -> SilkResult<SpectralWeights> {
    let mut weights = broadband_weights();
    if config.is_broadband() {
        // A constant filter disappears into the incident-Y normalization. Keep
        // the exact original coefficients instead of adding normalization noise.
        return Ok(weights);
    }
    let mut total_y = 0.0;
    for (bin, weight) in weights.iter_mut().enumerate() {
        *weight *= filter_value(config, wavelength_nm_for_bin(bin));
        total_y += weight.y;
    }
    if !total_y.is_finite() || total_y <= 0.0 {
        return Err("crystal spectral filter has no representable incident luminance".into());
    }
    for weight in &mut weights {
        *weight /= total_y;
        if !weight.is_finite() {
            return Err("crystal spectral filter normalization is not finite".into());
        }
    }
    Ok(weights)
}

fn direct_spectral_with_weights(retardance_nm: f64, weights: &SpectralWeights) -> SpectralSample {
    let mut xyz = V3::ZERO;
    for (bin, weight) in weights.iter().enumerate() {
        let wavelength = wavelength_nm_for_bin(bin);
        // sin² has period pi, hence retardance has period one wavelength.
        // Reducing first also avoids overflow for unusually large retardance.
        let phase = PI * (retardance_nm.rem_euclid(wavelength) / wavelength);
        let sine = phase.sin();
        xyz += *weight * (sine * sine);
    }
    let (r, g, b) = xyz_to_linear_srgb(xyz.x, xyz.y, xyz.z);
    SpectralSample { rgb: V3::new(r, g, b), luminance: xyz.y }
}

#[derive(Debug)]
struct SpectralTable {
    samples: Vec<SpectralSample>,
    weights: SpectralWeights,
}

impl SpectralTable {
    fn new(weights: &SpectralWeights) -> Self {
        let samples = (0..=TABLE_INTERVALS)
            .map(|index| direct_spectral_with_weights(index as f64 * TABLE_STEP_NM, weights))
            .collect();
        Self { samples, weights: *weights }
    }
}

static SPECTRAL_TABLE: LazyLock<Arc<SpectralTable>> =
    LazyLock::new(|| Arc::new(SpectralTable::new(&broadband_weights())));

/// Reusable crossed-polarizer optics with a shared deterministic spectral table.
///
/// Typical samples need one table interpolation and the rational tensor angle
/// factor. A nonzero viewing-angle offset adds one sine/cosine pair. Retardance
/// beyond the table uses the exact 64-bin calculation rather than flattening or
/// repeating the palette. No data-dependent table rebuild is performed.
#[derive(Clone, Debug)]
pub struct PolarizedOptics {
    config: OpticsConfig,
    twice_angle_sine: f64,
    twice_angle_cosine: f64,
    spectral_table: Arc<SpectralTable>,
}

impl PolarizedOptics {
    /// Validate controls and initialize the process-wide spectral table once.
    pub fn new(config: &OpticsConfig) -> SilkResult<Self> {
        config.validate()?;
        let twice_angle = 2.0 * config.polarizer_degrees.rem_euclid(180.0).to_radians();
        let (twice_angle_sine, twice_angle_cosine) = twice_angle.sin_cos();
        let spectral_table = if config.is_broadband() {
            Arc::clone(&SPECTRAL_TABLE)
        } else {
            Arc::new(SpectralTable::new(&filtered_weights(config)?))
        };
        Ok(Self { config: config.clone(), twice_angle_sine, twice_angle_cosine, spectral_table })
    }

    /// Rotate both crossed polarizers and share the existing spectral table.
    /// This is a fixed optical viewing change; no material or source spectrum
    /// is rebuilt. The offset is relative to this instance's polarizer angle.
    pub fn with_angle_offset_degrees(&self, offset_degrees: f64) -> SilkResult<Self> {
        if !offset_degrees.is_finite() {
            return Err("crystal polarizer angle offset must be finite".into());
        }
        let mut rotated = self.clone();
        rotated.config.polarizer_degrees =
            self.config.polarizer_degrees.rem_euclid(180.0) + offset_degrees.rem_euclid(180.0);
        let twice_angle = 2.0 * rotated.config.polarizer_degrees.to_radians();
        (rotated.twice_angle_sine, rotated.twice_angle_cosine) = twice_angle.sin_cos();
        Ok(rotated)
    }

    fn spectrum(&self, retardance_nm: f64) -> SpectralSample {
        if retardance_nm >= TABLE_MAX_NM {
            return direct_spectral_with_weights(retardance_nm, &self.spectral_table.weights);
        }
        let coordinate = retardance_nm / TABLE_STEP_NM;
        let index = coordinate.floor() as usize;
        self.spectral_table.samples[index]
            .lerp(self.spectral_table.samples[index + 1], coordinate - index as f64)
    }

    /// Evaluate linear, nonnegative RGB for `[sigma_xx, sigma_yy, sigma_xy]`.
    ///
    /// Thickness is a nonnegative multiple of the material reference thickness.
    /// The offset rotates both crossed polarizers together; it does not uncross
    /// them. Zero thickness, isotropic stress, or zero illumination is dark.
    /// Nonfinite inputs, negative thickness, or optical-retardance overflow
    /// return nonfinite RGB so the caller can reject invalid mechanics rather
    /// than silently turning them into a plausible black image.
    pub fn sample(&self, stress: [f64; 3], thickness: f64, angle_offset_radians: f64) -> V3 {
        if !stress.iter().all(|value| value.is_finite())
            || !thickness.is_finite()
            || thickness < 0.0
            || !angle_offset_radians.is_finite()
        {
            return V3::new(f64::NAN, f64::NAN, f64::NAN);
        }
        if thickness == 0.0
            || self.config.retardance_scale_nm == 0.0
            || self.config.incident_intensity == 0.0
        {
            return V3::ZERO;
        }
        // halved difference avoids overflowing xx-yy. Scale the two deviatoric
        // components before squaring: their ratio determines orientation.
        let half_difference = 0.5 * stress[0] - 0.5 * stress[1];
        let scale = half_difference.abs().max(stress[2].abs());
        if scale == 0.0 {
            return V3::ZERO;
        }
        let a = half_difference / scale;
        let b = stress[2] / scale;
        let norm_squared = a * a + b * b;
        let retardance_nm =
            (self.config.retardance_scale_nm * thickness) * scale * (2.0 * norm_squared.sqrt());
        if !retardance_nm.is_finite() {
            return V3::new(f64::NAN, f64::NAN, f64::NAN);
        }
        let (sine, cosine) = if angle_offset_radians == 0.0 {
            (self.twice_angle_sine, self.twice_angle_cosine)
        } else {
            let (offset_sine, offset_cosine) =
                (2.0 * angle_offset_radians.rem_euclid(PI)).sin_cos();
            (
                self.twice_angle_sine * offset_cosine + self.twice_angle_cosine * offset_sine,
                self.twice_angle_cosine * offset_cosine - self.twice_angle_sine * offset_sine,
            )
        };
        // cos(2 theta)=a/|a,b|, sin(2 theta)=b/|a,b|. The square is
        // well-defined even where theta would need an atan2 branch choice.
        let projection = b * cosine - a * sine;
        let orientation = (projection * projection / norm_squared).clamp(0.0, 1.0);
        self.spectrum(retardance_nm).nonnegative_rgb()
            * (orientation * self.config.incident_intensity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    fn maximum_error(a: V3, b: V3) -> f64 {
        let d = a - b;
        d.x.abs().max(d.y.abs()).max(d.z.abs())
    }

    fn direct_spectral(retardance_nm: f64) -> SpectralSample {
        direct_spectral_with_weights(retardance_nm, &broadband_weights())
    }

    fn jewel_filter() -> OpticsConfig {
        OpticsConfig {
            spectral_floor: 0.02,
            spectral_peaks: vec![
                SpectralPeak { center_nm: 525.0, width_nm: 18.0, weight: 1.0 },
                SpectralPeak { center_nm: 595.0, width_nm: 16.0, weight: 0.45 },
                SpectralPeak { center_nm: 440.0, width_nm: 15.0, weight: 0.55 },
            ],
            ..OpticsConfig::default()
        }
    }

    #[test]
    fn empty_or_constant_filter_preserves_original_white_table_and_recipe() {
        let original = PolarizedOptics::new(&OpticsConfig::default()).unwrap();
        let constant = PolarizedOptics::new(&OpticsConfig {
            spectral_floor: 0.03,
            spectral_peaks: vec![SpectralPeak { center_nm: 530.0, width_nm: 20.0, weight: 0.0 }],
            ..OpticsConfig::default()
        })
        .unwrap();
        assert!(Arc::ptr_eq(&original.spectral_table, &constant.spectral_table));
        for retardance in [0.0, 531.27, 2048.0, 9100.3] {
            assert_eq!(original.spectrum(retardance).rgb, constant.spectrum(retardance).rgb);
        }
        let encoded = serde_json::to_value(OpticsConfig::default()).unwrap();
        assert!(encoded.get("spectral_peaks").is_none());
        assert!(encoded.get("spectral_floor").is_none());
    }

    #[test]
    fn filtered_illumination_is_nonnegative_bounded_and_has_unit_incident_y() {
        let config = jewel_filter();
        config.validate().unwrap();
        let maximum = config.spectral_floor
            + config.spectral_peaks.iter().map(|peak| peak.weight).sum::<f64>();
        for bin in 0..NUM_BINS {
            let value = filter_value(&config, wavelength_nm_for_bin(bin));
            assert!(value.is_finite() && (0.0..=maximum).contains(&value));
        }
        let weights = filtered_weights(&config).unwrap();
        assert!((weights.iter().map(|weight| weight.y).sum::<f64>() - 1.0).abs() < 1e-14);
        assert!(weights.iter().all(|w| w.is_finite() && w.x >= 0.0 && w.y >= 0.0 && w.z >= 0.0));
        for retardance in [0.0, 130.0, 500.0, 850.0, 1400.0, 8200.0] {
            let sample = direct_spectral_with_weights(retardance, &weights);
            assert!((0.0..=1.0 + 1e-14).contains(&sample.luminance));
            let rgb = sample.nonnegative_rgb();
            assert!(rgb.is_finite() && rgb.x >= 0.0 && rgb.y >= 0.0 && rgb.z >= 0.0);
        }
    }

    #[test]
    fn custom_spectrum_lookup_matches_reference_and_rotated_views_share_it() {
        let optics = PolarizedOptics::new(&jewel_filter()).unwrap();
        let rotated = optics.with_angle_offset_degrees(23.0).unwrap();
        assert!(Arc::ptr_eq(&optics.spectral_table, &rotated.spectral_table));
        let mut largest: f64 = 0.0;
        for index in 0..1024_usize {
            let retardance = ((index as f64 + 0.381_966) * 13.738_521) % TABLE_MAX_NM;
            let reference =
                direct_spectral_with_weights(retardance, &optics.spectral_table.weights);
            largest = largest.max(maximum_error(
                optics.spectrum(retardance).nonnegative_rgb(),
                reference.nonnegative_rgb(),
            ));
        }
        assert!(largest < 1e-5, "custom spectral lookup error: {largest}");
        let stress = [0.17, -0.05, 0.013];
        assert!(
            maximum_error(
                rotated.sample(stress, 0.85, 0.0),
                optics.sample(stress, 0.85, 23.0_f64.to_radians()),
            ) < 1e-14
        );
        let retardance = TABLE_MAX_NM + 91.37;
        assert_eq!(
            optics.spectrum(retardance).rgb,
            direct_spectral_with_weights(retardance, &optics.spectral_table.weights).rgb,
        );
        assert!(optics.with_angle_offset_degrees(f64::NAN).is_err());
    }

    #[test]
    fn invalid_spectral_filters_are_rejected() {
        let base_peak = SpectralPeak { center_nm: 525.0, width_nm: 18.0, weight: 1.0 };
        for peak in [
            SpectralPeak { center_nm: f64::NAN, ..base_peak },
            SpectralPeak { center_nm: 750.0, ..base_peak },
            SpectralPeak { width_nm: 0.0, ..base_peak },
            SpectralPeak { width_nm: 4.9, ..base_peak },
            SpectralPeak { width_nm: f64::INFINITY, ..base_peak },
            SpectralPeak { weight: -0.1, ..base_peak },
            SpectralPeak { weight: f64::NAN, ..base_peak },
        ] {
            assert!(
                OpticsConfig { spectral_peaks: vec![peak], ..OpticsConfig::default() }
                    .validate()
                    .is_err()
            );
        }
        for floor in [-1.0, f64::NAN, f64::INFINITY, 0.0] {
            assert!(
                OpticsConfig { spectral_floor: floor, ..OpticsConfig::default() }
                    .validate()
                    .is_err()
            );
        }
        assert!(
            OpticsConfig { spectral_peaks: vec![base_peak; 17], ..OpticsConfig::default() }
                .validate()
                .is_err()
        );
        let mut positive_peak_without_floor = jewel_filter();
        positive_peak_without_floor.spectral_floor = 0.0;
        positive_peak_without_floor.validate().unwrap();
    }

    #[test]
    fn normalized_observer_integrates_unit_samples_to_white() {
        let xyz = BIN_XYZ_LUT.iter().fold(V3::ZERO, |sum, &(x, y, z, _)| sum + V3::new(x, y, z));
        let (r, g, b) = xyz_to_linear_srgb(xyz.x, xyz.y, xyz.z);
        assert!((xyz.y - 1.0).abs() < 1e-14);
        assert!(maximum_error(V3::new(r, g, b), V3::new(1.0, 1.0, 1.0)) < 1e-6);
    }

    #[test]
    fn zero_retardance_and_hydrostatic_stress_are_dark() {
        let optics = PolarizedOptics::new(&OpticsConfig::default()).unwrap();
        for stress in [[0.0; 3], [1.0, 1.0, 0.0], [-19.0, -19.0, 0.0]] {
            assert_eq!(optics.sample(stress, 1.0, 0.37), V3::ZERO);
        }
        assert_eq!(optics.sample([2.0, -1.0, 0.3], 0.0, 0.0), V3::ZERO);
        assert_eq!(direct_spectral(0.0).rgb, V3::ZERO);
    }

    #[test]
    fn hydrostatic_offset_and_stress_sign_leave_transmission_unchanged() {
        let optics = PolarizedOptics::new(&OpticsConfig::default()).unwrap();
        let reference = optics.sample([1.0, -2.0, 0.25], 0.2, 0.17);
        assert_eq!(reference, optics.sample([1001.0, 998.0, 0.25], 0.2, 0.17));
        assert_eq!(reference, optics.sample([-1.0, 2.0, -0.25], 0.2, 0.17));
    }

    #[test]
    fn tensor_orientation_matches_known_polarizer_directions() {
        let config = OpticsConfig { retardance_scale_nm: 375.0, ..OpticsConfig::default() };
        let optics = PolarizedOptics::new(&config).unwrap();
        let fully_transmitted = direct_spectral(750.0).nonnegative_rgb();
        assert_eq!(optics.sample([1.0, -1.0, 0.0], 1.0, 0.0), V3::ZERO);
        assert!(maximum_error(optics.sample([0.0, 0.0, 1.0], 1.0, 0.0), fully_transmitted) < 1e-14);
        assert!(
            maximum_error(optics.sample([1.0, -1.0, 0.0], 1.0, FRAC_PI_4), fully_transmitted)
                < 1e-14
        );
        assert!(optics.sample([0.0, 0.0, 1.0], 1.0, FRAC_PI_4).length() < 1e-28);
    }

    #[test]
    fn rotating_crossed_polarizers_has_half_pi_period() {
        let optics = PolarizedOptics::new(&OpticsConfig {
            polarizer_degrees: 23.0,
            ..OpticsConfig::default()
        })
        .unwrap();
        for angle in [-0.8, 0.0, 0.27, 1.5, 8.0] {
            let a = optics.sample([0.27, -0.11, 0.03], 1.3, angle);
            let b = optics.sample([0.27, -0.11, 0.03], 1.3, angle + FRAC_PI_2);
            assert!(maximum_error(a, b) < 1e-14);
        }
    }

    #[test]
    fn lookup_matches_spectral_reference_and_refinement_converges() {
        let optics = PolarizedOptics::new(&OpticsConfig::default()).unwrap();
        let mut fine_error: f64 = 0.0;
        let mut coarse_error: f64 = 0.0;
        for index in 0..2048_usize {
            let retardance = ((index as f64 + 0.381_966) * 13.738_521) % TABLE_MAX_NM;
            let reference = direct_spectral(retardance).nonnegative_rgb();
            fine_error = fine_error
                .max(maximum_error(optics.spectrum(retardance).nonnegative_rgb(), reference));
            let coarse_step = 2.0;
            let lo = (retardance / coarse_step).floor() * coarse_step;
            let coarse = direct_spectral(lo)
                .lerp(direct_spectral(lo + coarse_step), (retardance - lo) / coarse_step)
                .nonnegative_rgb();
            coarse_error = coarse_error.max(maximum_error(coarse, reference));
        }
        assert!(fine_error < 1e-5, "fine table error: {fine_error}");
        assert!(fine_error < coarse_error * 0.1, "fine {fine_error}, coarse {coarse_error}");
        for retardance in [TABLE_MAX_NM, TABLE_MAX_NM + 0.17, 100_000.3] {
            assert_eq!(optics.spectrum(retardance).rgb, direct_spectral(retardance).rgb);
        }
    }

    #[test]
    fn gamut_compression_preserves_neutral_axis_direction_and_luminance() {
        let rgb = V3::new(-0.2, 0.4, 0.7);
        let luminance = 0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z;
        let compressed = SpectralSample { rgb, luminance }.nonnegative_rgb();
        let result_luminance =
            0.2126 * compressed.x + 0.7152 * compressed.y + 0.0722 * compressed.z;
        assert!(compressed.x.abs() < 1e-15);
        assert!((result_luminance - luminance).abs() < 1e-15);
        assert!(
            ((compressed.y - compressed.x) / (compressed.z - compressed.x)
                - (rgb.y - rgb.x) / (rgb.z - rgb.x))
                .abs()
                < 1e-14
        );
    }

    #[test]
    fn samples_are_finite_nonnegative_and_scale_linearly_with_illumination() {
        let optics = PolarizedOptics::new(&OpticsConfig::default()).unwrap();
        let brighter = PolarizedOptics::new(&OpticsConfig {
            incident_intensity: 3.0,
            ..OpticsConfig::default()
        })
        .unwrap();
        for index in 0..128_usize {
            let value = index as f64 * 0.137;
            let stress = [0.3 * value.cos(), 0.12 * value.sin(), 0.25 * (1.7 * value).sin()];
            let color = optics.sample(stress, 0.1 + value, 0.23);
            assert!(color.is_finite());
            assert!(color.x >= 0.0 && color.y >= 0.0 && color.z >= 0.0);
            // Physical Y is at most one. Signed sRGB conversion can exceed
            // unit channels; 3.25 safely bounds the positive matrix terms.
            assert!(color.x <= 3.25 && color.y <= 3.25 && color.z <= 3.25);
            assert!(maximum_error(brighter.sample(stress, 0.1 + value, 0.23), color * 3.0) < 1e-14);
        }
    }

    #[test]
    fn invalid_configuration_and_sample_inputs_are_rejected() {
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                OpticsConfig { retardance_scale_nm: bad, ..OpticsConfig::default() }
                    .validate()
                    .is_err()
            );
            assert!(
                OpticsConfig { incident_intensity: bad, ..OpticsConfig::default() }
                    .validate()
                    .is_err()
            );
        }
        assert!(
            OpticsConfig { polarizer_degrees: f64::INFINITY, ..OpticsConfig::default() }
                .validate()
                .is_err()
        );
        assert!(serde_json::from_str::<OpticsConfig>(r#"{"invented_color":1}"#).is_err());
        let optics = PolarizedOptics::new(&OpticsConfig::default()).unwrap();
        assert!(!optics.sample([f64::NAN, 0.0, 0.0], 1.0, 0.0).is_finite());
        assert!(!optics.sample([1.0, 0.0, 0.0], -1.0, 0.0).is_finite());
        assert!(!optics.sample([1.0, 0.0, 0.0], 1.0, f64::NAN).is_finite());
        assert!(!optics.sample([f64::MAX, -f64::MAX, 0.0], 1.0, 0.0).is_finite());
    }
}
