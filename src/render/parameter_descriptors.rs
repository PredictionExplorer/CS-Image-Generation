//! Parameter descriptors for active seed-resolved controls.

/// Descriptor for a floating-point parameter with bounded range.
#[derive(Clone, Debug)]
pub struct FloatParamDescriptor {
    /// Machine-readable parameter name used for logging and serialization.
    pub name: &'static str,
    /// Minimum allowed value (inclusive).
    pub min: f64,
    /// Maximum allowed value (inclusive).
    pub max: f64,
    /// Human-readable description of what this parameter controls.
    pub description: &'static str,
}

/// Equilateralness-to-chaos Borda weight ratio descriptor.
///
/// Sampled log-uniformly with a moderate bias toward equilateralness.
/// Ratio range: 1/5 to 50 (median ~3.16).
/// At ratio < 1 chaos dominates; at ratio > 1 equilateralness dominates.
pub const EQUIL_CHAOS_RATIO: FloatParamDescriptor = FloatParamDescriptor {
    name: "equil_chaos_ratio",
    min: 0.2,
    max: 50.0,
    description: "Equilateralness-to-chaos Borda weight ratio (log-uniform, 1/5x to 50x)",
};

const _: () = {
    assert!(EQUIL_CHAOS_RATIO.min < 1.0);
    assert!(EQUIL_CHAOS_RATIO.max > 1.0);
    assert!(EQUIL_CHAOS_RATIO.max / EQUIL_CHAOS_RATIO.min > 100.0);
};
