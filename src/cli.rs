//! Command-line parsing and conversion into pipeline requests.

use clap::{Parser, ValueEnum};
use three_body_problem::pipeline::{
    BordaWeightOptions, DEFAULT_LOG_LEVEL, DEFAULT_NUM_SIMS, DEFAULT_NUM_STEPS,
    DEFAULT_OUTPUT_NAME, DEFAULT_RESOLUTION, GenerationDriftMode, GenerationRequest, MAX_NUM_SIMS,
    MAX_NUM_STEPS,
};
use tracing_subscriber::EnvFilter;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputResolution {
    width: u32,
    height: u32,
}

fn parse_resolution(value: &str) -> std::result::Result<OutputResolution, String> {
    let (width, height) = value
        .split_once('x')
        .ok_or_else(|| "resolution must use WIDTHxHEIGHT format".to_string())?;
    let width = width
        .parse::<u32>()
        .map_err(|_| "resolution width must be a positive integer".to_string())?;
    let height = height
        .parse::<u32>()
        .map_err(|_| "resolution height must be a positive integer".to_string())?;

    if width == 0 || height == 0 {
        return Err("resolution dimensions must be greater than zero".to_string());
    }

    Ok(OutputResolution { width, height })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DriftModeArg {
    None,
    Linear,
    Brownian,
    Elliptical,
}

impl From<DriftModeArg> for GenerationDriftMode {
    fn from(value: DriftModeArg) -> Self {
        match value {
            DriftModeArg::None => Self::None,
            DriftModeArg::Linear => Self::Linear,
            DriftModeArg::Brownian => Self::Brownian,
            DriftModeArg::Elliptical => Self::Elliptical,
        }
    }
}

/// Command-line arguments.
#[derive(Parser, Debug)]
#[command(author, version, about = "Generate a curated three-body image and video from a seed.")]
pub(crate) struct Args {
    #[arg(long, default_value = "0x100033")]
    seed: String,

    #[arg(short, long, default_value = DEFAULT_OUTPUT_NAME)]
    output: String,

    #[arg(long, default_value_t = DEFAULT_NUM_SIMS, value_parser = parse_bounded_sims)]
    sims: usize,

    #[arg(long, default_value_t = DEFAULT_NUM_STEPS, value_parser = parse_bounded_steps)]
    steps: usize,

    #[arg(short = 'r', long, default_value = DEFAULT_RESOLUTION, value_parser = parse_resolution)]
    resolution: OutputResolution,

    #[arg(long, value_enum, default_value_t = DriftModeArg::Elliptical)]
    drift: DriftModeArg,

    #[arg(long, default_value_t = false)]
    fast_encode: bool,

    #[arg(long, default_value = DEFAULT_LOG_LEVEL)]
    pub(crate) log_level: String,

    /// Borda weight for chaos (FFT regularity) rank points.
    /// Omit to randomize from a curated range.
    #[arg(long, value_parser = parse_borda_weight)]
    chaos_weight: Option<f64>,

    /// Borda weight for equilateralness (triangle balance) rank points.
    /// Omit to randomize from a curated range.
    #[arg(long, value_parser = parse_borda_weight)]
    equil_weight: Option<f64>,

    /// Borda weight for curvature-entropy (turning-angle diversity) rank points.
    /// Omit to randomize from a curated range.
    #[arg(long, value_parser = parse_borda_weight)]
    curvature_weight: Option<f64>,

    /// Borda weight for permutation-entropy (Bandt-Pompe complexity) rank points.
    /// Omit to randomize from a curated range.
    #[arg(long, value_parser = parse_borda_weight)]
    permutation_weight: Option<f64>,
}

impl Args {
    pub(crate) fn parse_args() -> Self {
        Self::parse()
    }

    pub(crate) fn to_generation_request(&self) -> GenerationRequest {
        GenerationRequest {
            seed: self.seed.clone(),
            output: self.output.clone(),
            sims: self.sims,
            steps: self.steps,
            width: self.resolution.width,
            height: self.resolution.height,
            drift_mode: self.drift.into(),
            fast_encode: self.fast_encode,
            borda_weights: BordaWeightOptions {
                chaos: self.chaos_weight,
                equil: self.equil_weight,
                curvature: self.curvature_weight,
                permutation: self.permutation_weight,
            },
        }
    }
}

fn parse_bounded_sims(value: &str) -> std::result::Result<usize, String> {
    let n: usize = value.parse().map_err(|_| "sims must be a positive integer".to_string())?;
    if n == 0 || n > MAX_NUM_SIMS {
        return Err(format!("sims must be between 1 and {MAX_NUM_SIMS}"));
    }
    Ok(n)
}

fn parse_bounded_steps(value: &str) -> std::result::Result<usize, String> {
    let n: usize = value.parse().map_err(|_| "steps must be a positive integer".to_string())?;
    if n == 0 || n > MAX_NUM_STEPS {
        return Err(format!("steps must be between 1 and {MAX_NUM_STEPS}"));
    }
    Ok(n)
}

fn parse_borda_weight(value: &str) -> std::result::Result<f64, String> {
    let n: f64 = value.parse().map_err(|_| "weight must be a finite number".to_string())?;
    if !n.is_finite() || n <= 0.0 {
        return Err("weight must be finite and greater than zero".to_string());
    }
    Ok(n)
}

pub(crate) fn setup_logging(level: &str) {
    let env_filter =
        EnvFilter::try_new(level).unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_LEVEL));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_thread_ids(false)
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_defaults() {
        let args = Args::parse_from(["three_body_problem"]);

        assert_eq!(args.output, DEFAULT_OUTPUT_NAME);
        assert_eq!(args.sims, DEFAULT_NUM_SIMS);
        assert_eq!(args.steps, DEFAULT_NUM_STEPS);
        assert_eq!(args.resolution, OutputResolution { width: 1920, height: 1080 });
        assert_eq!(args.drift, DriftModeArg::Elliptical);
        assert!(!args.fast_encode);
        assert_eq!(args.log_level, DEFAULT_LOG_LEVEL);
        assert!(args.chaos_weight.is_none());
        assert!(args.equil_weight.is_none());
        assert!(args.curvature_weight.is_none());
        assert!(args.permutation_weight.is_none());
    }

    #[test]
    fn test_parse_custom_resolution_and_drift() {
        let args = Args::parse_from([
            "three_body_problem",
            "--output",
            "gallery-piece",
            "--resolution",
            "1280x720",
            "--drift",
            "none",
            "--sims",
            "5000",
            "--steps",
            "12000",
            "--fast-encode",
        ]);

        assert_eq!(args.output, "gallery-piece");
        assert_eq!(args.resolution, OutputResolution { width: 1280, height: 720 });
        assert_eq!(args.drift, DriftModeArg::None);
        assert_eq!(args.sims, 5000);
        assert_eq!(args.steps, 12000);
        assert!(args.fast_encode);
    }

    #[test]
    fn test_parse_explicit_borda_weights() {
        let args = Args::parse_from([
            "three_body_problem",
            "--chaos-weight",
            "1.5",
            "--equil-weight",
            "8.0",
            "--curvature-weight",
            "3.3",
            "--permutation-weight",
            "0.4",
        ]);
        assert_eq!(args.chaos_weight, Some(1.5));
        assert_eq!(args.equil_weight, Some(8.0));
        assert_eq!(args.curvature_weight, Some(3.3));
        assert_eq!(args.permutation_weight, Some(0.4));
    }

    #[test]
    fn test_reject_invalid_borda_weights() {
        for (flag, value) in [
            ("--chaos-weight", "0"),
            ("--equil-weight", "-1"),
            ("--curvature-weight", "NaN"),
            ("--permutation-weight", "inf"),
        ] {
            let result = Args::try_parse_from(["three_body_problem", flag, value]);
            assert!(result.is_err(), "{flag}={value} should be rejected");
        }
    }

    #[test]
    fn test_to_generation_request_maps_all_fields() {
        let args = Args::parse_from([
            "three_body_problem",
            "--seed",
            "0xfeed",
            "--output",
            "mapped",
            "--resolution",
            "32x24",
            "--drift",
            "brownian",
            "--sims",
            "10",
            "--steps",
            "20",
            "--fast-encode",
            "--chaos-weight",
            "1.0",
            "--equil-weight",
            "2.0",
            "--curvature-weight",
            "3.0",
            "--permutation-weight",
            "4.0",
        ]);

        let request = args.to_generation_request();
        assert_eq!(request.seed, "0xfeed");
        assert_eq!(request.output, "mapped");
        assert_eq!(request.width, 32);
        assert_eq!(request.height, 24);
        assert_eq!(request.drift_mode, GenerationDriftMode::Brownian);
        assert_eq!(request.sims, 10);
        assert_eq!(request.steps, 20);
        assert!(request.fast_encode);
        assert_eq!(request.borda_weights.chaos, Some(1.0));
        assert_eq!(request.borda_weights.equil, Some(2.0));
        assert_eq!(request.borda_weights.curvature, Some(3.0));
        assert_eq!(request.borda_weights.permutation, Some(4.0));
    }

    #[test]
    fn test_reject_invalid_resolution() {
        let result = Args::try_parse_from(["three_body_problem", "--resolution", "wide-by-tall"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_zero_resolution() {
        let result = Args::try_parse_from(["three_body_problem", "--resolution", "0x1080"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_zero_sims() {
        let result = Args::try_parse_from(["three_body_problem", "--sims", "0"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_steps_above_limit() {
        let result = Args::try_parse_from([
            "three_body_problem",
            "--steps",
            &(MAX_NUM_STEPS + 1).to_string(),
        ]);
        assert!(result.is_err());
    }
}
