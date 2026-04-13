//! Drift configuration resolution with random generation support
//!
//! This module handles the logic for resolving drift parameters, either from
//! user-provided values or by generating random values based on the mode.

use crate::drift::DriftParameters;
use crate::error::ConfigError;
use crate::sim::Sha3RandomByteStream;
use tracing::info;

const DRIFT_SCALE_MIN: f64 = 0.8;
const DRIFT_SCALE_RANGE: f64 = 1.2;
const DRIFT_ARC_FRACTION_RANGE: f64 = 0.8;
const DRIFT_ECCENTRICITY_MIN: f64 = 0.4;
const DRIFT_ECCENTRICITY_RANGE: f64 = 0.1;

/// Resolved drift configuration ready for use
#[derive(Debug, Clone)]
pub struct ResolvedDriftConfig {
    pub scale: f64,
    pub arc_fraction: f64,
    pub orbit_eccentricity: f64,
    pub was_randomized: bool,
}

impl ResolvedDriftConfig {
    /// Create drift configuration from explicit values
    #[must_use]
    pub fn from_values(scale: f64, arc_fraction: f64, orbit_eccentricity: f64) -> Self {
        Self { scale, arc_fraction, orbit_eccentricity, was_randomized: false }
    }

    /// Generate random drift configuration with curated ranges.
    pub fn generate_random(rng: &mut Sha3RandomByteStream) -> Self {
        let scale = DRIFT_SCALE_MIN + rng.next_f64() * DRIFT_SCALE_RANGE; // 0.8 to 2.0
        let arc_fraction = rng.next_f64() * DRIFT_ARC_FRACTION_RANGE; // 0.0 to 0.8
        let orbit_eccentricity = DRIFT_ECCENTRICITY_MIN + rng.next_f64() * DRIFT_ECCENTRICITY_RANGE; // 0.4 to 0.5

        info!("Generated random drift parameters:");
        info!("  scale: {:.3}", scale);
        info!("  arc_fraction: {:.3}", arc_fraction);
        info!("  orbit_eccentricity: {:.3}", orbit_eccentricity);

        Self { scale, arc_fraction, orbit_eccentricity, was_randomized: true }
    }

    /// Convert to DriftParameters for use in the drift system
    pub fn to_drift_parameters(&self) -> DriftParameters {
        DriftParameters::new(self.scale, self.arc_fraction, self.orbit_eccentricity)
    }
}

/// Helper to resolve drift configuration from optional command-line args.
///
/// Returns an error if only some drift parameters are provided (must be all or none).
pub fn resolve_drift_config(
    scale_opt: Option<f64>,
    arc_fraction_opt: Option<f64>,
    eccentricity_opt: Option<f64>,
    rng: &mut Sha3RandomByteStream,
) -> std::result::Result<ResolvedDriftConfig, ConfigError> {
    match (scale_opt, arc_fraction_opt, eccentricity_opt) {
        (Some(scale), Some(arc), Some(ecc)) => {
            info!("Using user-specified drift parameters:");
            info!("  scale: {:.3}", scale);
            info!("  arc_fraction: {:.3}", arc);
            info!("  orbit_eccentricity: {:.3}", ecc);
            Ok(ResolvedDriftConfig::from_values(scale, arc, ecc))
        }
        (None, None, None) => {
            info!("No drift parameters specified, generating random values...");
            Ok(ResolvedDriftConfig::generate_random(rng))
        }
        _ => Err(ConfigError::InvalidDriftConfig {
            scale: scale_opt,
            arc_fraction: arc_fraction_opt,
            eccentricity: eccentricity_opt,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> Sha3RandomByteStream {
        let seed = [0x42u8; 32];
        Sha3RandomByteStream::new(&seed, 1.0, 2.0, 1.0, 1.0)
    }

    #[test]
    fn test_from_values() {
        let config = ResolvedDriftConfig::from_values(1.5, 0.3, 0.2);
        assert_eq!(config.scale, 1.5);
        assert_eq!(config.arc_fraction, 0.3);
        assert_eq!(config.orbit_eccentricity, 0.2);
        assert!(!config.was_randomized);
    }

    #[test]
    fn test_generate_random() {
        let mut rng = make_rng();
        let config = ResolvedDriftConfig::generate_random(&mut rng);

        assert!(
            config.scale >= 0.8 && config.scale <= 2.0,
            "drift_scale {} outside [0.8, 2.0]",
            config.scale
        );
        assert!(config.arc_fraction >= 0.0 && config.arc_fraction <= 0.8);
        assert!(config.orbit_eccentricity >= 0.4 && config.orbit_eccentricity <= 0.5);
        assert!(config.was_randomized);
    }

    #[test]
    fn test_resolve_all_provided() {
        let mut rng = make_rng();
        let config = resolve_drift_config(Some(1.0), Some(0.5), Some(0.3), &mut rng).unwrap();
        assert_eq!(config.scale, 1.0);
        assert!(!config.was_randomized);
    }

    #[test]
    fn test_resolve_none_provided() {
        let mut rng = make_rng();
        let config = resolve_drift_config(None, None, None, &mut rng).unwrap();
        assert!(config.was_randomized);
    }

    #[test]
    fn test_resolve_partial_returns_error() {
        let mut rng = make_rng();
        let result = resolve_drift_config(Some(1.0), None, None, &mut rng);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("must be either all specified or all omitted"));
    }

    #[test]
    fn test_drift_scale_always_above_floor() {
        for seed_byte in 0u8..=255 {
            let seed = [seed_byte; 32];
            let mut rng = Sha3RandomByteStream::new(&seed, 1.0, 2.0, 1.0, 1.0);
            let config = ResolvedDriftConfig::generate_random(&mut rng);

            assert!(
                config.scale >= 0.8,
                "seed {} produced drift_scale {} below 0.8 floor",
                seed_byte,
                config.scale
            );
            assert!(
                config.scale <= 2.0,
                "seed {} produced drift_scale {} above 2.0 ceiling",
                seed_byte,
                config.scale
            );
        }
    }
}
