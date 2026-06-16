//! Generation-log records for seed-resolved visual parameters.
//!
//! The `CosmicSignature` profile resolves its parameters directly from the
//! deterministic seed stream. This module only keeps the serializable record
//! types used by generation logs.

/// Tracks randomization decisions for logging.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RandomizationRecord {
    /// Name of the effect that was randomized.
    pub effect_name: String,
    /// Whether the effect was enabled after randomization.
    pub enabled: bool,
    /// Whether the effect's enable state was determined randomly.
    pub was_randomized: bool,
    /// Individual parameter values that were recorded.
    pub parameters: Vec<RandomizedParameter>,
}

/// A single randomized parameter value.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RandomizedParameter {
    /// Parameter name.
    pub name: String,
    /// Formatted parameter value.
    pub value: String,
    /// Whether this parameter was randomly generated.
    pub was_randomized: bool,
    /// Formatted range string (e.g. `[0.0, 1.0]`).
    pub range_used: String,
}

impl RandomizationRecord {
    /// Create a new record for the named effect.
    #[must_use]
    pub fn new(effect_name: impl Into<String>, enabled: bool, was_randomized: bool) -> Self {
        Self { effect_name: effect_name.into(), enabled, was_randomized, parameters: Vec::new() }
    }

    /// Record a float parameter value and the range it was sampled from.
    pub fn add_float(
        &mut self,
        name: impl Into<String>,
        value: f64,
        was_randomized: bool,
        range: (f64, f64),
    ) {
        self.parameters.push(RandomizedParameter {
            name: name.into(),
            value: format!("{value:.4}"),
            was_randomized,
            range_used: format!("[{:.4}, {:.4}]", range.0, range.1),
        });
    }

    /// Record an integer parameter value and the range it was sampled from.
    pub fn add_int(
        &mut self,
        name: impl Into<String>,
        value: usize,
        was_randomized: bool,
        range: (usize, usize),
    ) {
        self.parameters.push(RandomizedParameter {
            name: name.into(),
            value: value.to_string(),
            was_randomized,
            range_used: format!("[{}, {}]", range.0, range.1),
        });
    }
}

/// Collection of all randomization records for a render session.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RandomizationLog {
    /// Ordered list of per-effect randomization records.
    pub effects: Vec<RandomizationRecord>,
}

impl RandomizationLog {
    /// Create an empty randomization log.
    #[must_use]
    pub fn new() -> Self {
        Self { effects: Vec::new() }
    }

    /// Append a randomization record to the log.
    pub fn add_record(&mut self, record: RandomizationRecord) {
        self.effects.push(record);
    }
}
