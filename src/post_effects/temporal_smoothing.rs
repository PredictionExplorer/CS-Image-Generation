//! Temporal smoothing effect for buttery-smooth video quality.
//!
//! This effect blends each frame with the previous frame to reduce temporal jitter
//! and create a more cinematic, fluid motion. Creates the perception of higher
//! frame rate and more organic movement without actual interpolation.
//!
//! Note: This effect is stateful and is wired directly into the video renderer
//! instead of going through the stateless `PostEffect` trait.

use super::PixelBuffer;
use rayon::prelude::*;
use std::sync::Mutex;

/// Configuration for temporal smoothing effect
#[derive(Clone, Debug)]
pub struct TemporalSmoothingConfig {
    /// Blend factor with previous frame (0.0 = no smoothing, 1.0 = full blend)
    /// Typical values: 0.15-0.35
    pub blend_factor: f64,
    /// Minimum alpha threshold to apply smoothing (prevents ghosting in empty areas)
    pub alpha_threshold: f64,
}

impl Default for TemporalSmoothingConfig {
    fn default() -> Self {
        Self::special_mode()
    }
}

impl TemporalSmoothingConfig {
    /// Create configuration for special mode (stronger smoothing)
    pub fn special_mode() -> Self {
        Self {
            blend_factor: 0.25,    // 25% of previous frame
            alpha_threshold: 0.01, // Only smooth visible pixels
        }
    }
}

/// Temporal smoothing effect with frame history
///
/// Note: This effect maintains state between frames, so it must be used carefully.
/// It's designed for video rendering where frames are processed sequentially.
pub struct TemporalSmoothing {
    config: TemporalSmoothingConfig,
    enabled: bool,
    // Thread-safe frame buffer for video processing
    previous_frame: Mutex<Option<PixelBuffer>>,
}

impl TemporalSmoothing {
    pub fn new(config: TemporalSmoothingConfig) -> Self {
        let enabled = config.blend_factor > 0.0;
        Self { config, enabled, previous_frame: Mutex::new(None) }
    }

    /// Process a frame with temporal smoothing
    ///
    /// This is NOT a PostEffect trait implementation because it needs mutable state.
    /// Called directly from the rendering pipeline for video frames.
    pub fn process_frame(&self, current: PixelBuffer) -> PixelBuffer {
        if !self.enabled {
            return current;
        }

        let mut prev_guard = self.previous_frame.lock().unwrap();

        let result = if let Some(prev) = prev_guard.as_ref() {
            if prev.len() != current.len() {
                *prev_guard = Some(current.clone());
                return current;
            }

            let blend_factor = self.config.blend_factor;
            let inv_blend = 1.0 - blend_factor;
            let threshold = self.config.alpha_threshold;

            current
                .par_iter()
                .zip(prev.par_iter())
                .map(|(&(cr, cg, cb, ca), &(pr, pg, pb, pa))| {
                    if ca > threshold && pa > threshold {
                        (
                            cr * inv_blend + pr * blend_factor,
                            cg * inv_blend + pg * blend_factor,
                            cb * inv_blend + pb * blend_factor,
                            ca * inv_blend + pa * blend_factor,
                        )
                    } else {
                        (cr, cg, cb, ca)
                    }
                })
                .collect()
        } else {
            current.clone()
        };

        *prev_guard = Some(current);

        result
    }

    /// Reset temporal buffer (call when starting new video or after seeking)
    pub fn reset(&self) {
        let mut prev_guard = self.previous_frame.lock().unwrap();
        *prev_guard = None;
    }

    /// Check if effect is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_smoothing_disabled() {
        let config = TemporalSmoothingConfig { blend_factor: 0.0, alpha_threshold: 0.01 };
        let smoother = TemporalSmoothing::new(config);
        assert!(!smoother.is_enabled());
    }

    #[test]
    fn test_temporal_smoothing_enabled() {
        let config = TemporalSmoothingConfig::special_mode();
        let smoother = TemporalSmoothing::new(config);
        assert!(smoother.is_enabled());
    }

    #[test]
    fn test_first_frame_passthrough() {
        let config = TemporalSmoothingConfig::special_mode();
        let smoother = TemporalSmoothing::new(config);

        let frame = vec![(1.0, 0.5, 0.25, 1.0); 100];
        let result = smoother.process_frame(frame.clone());

        // First frame should pass through unchanged
        assert_eq!(result, frame);
    }

    #[test]
    fn test_frame_blending() {
        let config = TemporalSmoothingConfig {
            blend_factor: 0.5, // 50% blend for clear testing
            alpha_threshold: 0.01,
        };
        let smoother = TemporalSmoothing::new(config);

        // First frame
        let frame1 = vec![(1.0, 1.0, 1.0, 1.0); 100];
        let _result1 = smoother.process_frame(frame1);

        // Second frame (black)
        let frame2 = vec![(0.0, 0.0, 0.0, 1.0); 100];
        let result2 = smoother.process_frame(frame2);

        // Should be 50% blend: (0.0 * 0.5 + 1.0 * 0.5) = 0.5
        assert!((result2[0].0 - 0.5).abs() < 0.001);
        assert!((result2[0].1 - 0.5).abs() < 0.001);
        assert!((result2[0].2 - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let config = TemporalSmoothingConfig::special_mode();
        let smoother = TemporalSmoothing::new(config);

        // Process a frame
        let frame1 = vec![(1.0, 1.0, 1.0, 1.0); 100];
        let _result1 = smoother.process_frame(frame1);

        // Reset
        smoother.reset();

        // Next frame should be treated as first frame
        let frame2 = vec![(0.5, 0.5, 0.5, 1.0); 100];
        let result2 = smoother.process_frame(frame2.clone());

        // Should pass through unchanged (no blending)
        assert_eq!(result2, frame2);
    }

    #[test]
    fn test_alpha_threshold() {
        let config = TemporalSmoothingConfig {
            blend_factor: 0.5,
            alpha_threshold: 0.5,
        };
        let smoother = TemporalSmoothing::new(config);

        let frame1 = vec![(1.0, 1.0, 1.0, 0.3); 100];
        let _result1 = smoother.process_frame(frame1);

        let frame2 = vec![(0.0, 0.0, 0.0, 0.3); 100];
        let result2 = smoother.process_frame(frame2.clone());

        assert_eq!(result2, frame2);
    }

    #[test]
    fn test_parallel_smoothing_deterministic() {
        let config = TemporalSmoothingConfig { blend_factor: 0.3, alpha_threshold: 0.01 };
        let n = 5000;

        let frame1: PixelBuffer = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (t, 1.0 - t, t * 0.5, 0.8)
            })
            .collect();
        let frame2: PixelBuffer = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (1.0 - t, t, 0.5, 0.9)
            })
            .collect();

        let smoother_a = TemporalSmoothing::new(config.clone());
        let _ = smoother_a.process_frame(frame1.clone());
        let result_a = smoother_a.process_frame(frame2.clone());

        let smoother_b = TemporalSmoothing::new(config);
        let _ = smoother_b.process_frame(frame1);
        let result_b = smoother_b.process_frame(frame2);

        for (i, (a, b)) in result_a.iter().zip(result_b.iter()).enumerate() {
            assert_eq!(a.0.to_bits(), b.0.to_bits(), "R differs at pixel {i}");
            assert_eq!(a.1.to_bits(), b.1.to_bits(), "G differs at pixel {i}");
            assert_eq!(a.2.to_bits(), b.2.to_bits(), "B differs at pixel {i}");
            assert_eq!(a.3.to_bits(), b.3.to_bits(), "A differs at pixel {i}");
        }
    }

    #[test]
    fn test_process_frame_blends_known_values() {
        let config = TemporalSmoothingConfig { blend_factor: 0.25, alpha_threshold: 0.01 };
        let smoother = TemporalSmoothing::new(config);

        let frame1 = vec![(1.0, 0.0, 0.5, 1.0)];
        let _ = smoother.process_frame(frame1);

        let frame2 = vec![(0.0, 1.0, 0.5, 1.0)];
        let result = smoother.process_frame(frame2);

        // blend_factor=0.25: result = current*0.75 + prev*0.25
        assert!((result[0].0 - 0.25).abs() < 1e-10, "R: {} vs 0.25", result[0].0);
        assert!((result[0].1 - 0.75).abs() < 1e-10, "G: {} vs 0.75", result[0].1);
        assert!((result[0].2 - 0.50).abs() < 1e-10, "B: {} vs 0.50", result[0].2);
    }

    #[test]
    fn test_disabled_smoother_is_passthrough() {
        let config = TemporalSmoothingConfig { blend_factor: 0.0, alpha_threshold: 0.01 };
        let smoother = TemporalSmoothing::new(config);

        let frame = vec![(0.42, 0.84, 0.13, 0.99); 50];
        let result = smoother.process_frame(frame.clone());
        assert_eq!(result, frame, "disabled smoother must return input unchanged");
    }
}
