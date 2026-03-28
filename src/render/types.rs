//! Common tone-mapping types for the rendering pipeline.

/// Tone-mapping controls shared across render stages.
#[derive(Clone, Copy, Debug)]
pub struct ToneMappingControls {
    pub exposure_scale: f64,
    pub paper_white: f64,
    pub highlight_rolloff: f64,
}

impl Default for ToneMappingControls {
    fn default() -> Self {
        Self {
            exposure_scale: 1.0,
            paper_white: crate::render::constants::DEFAULT_TONEMAP_PAPER_WHITE,
            highlight_rolloff: crate::render::constants::DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF,
        }
    }
}

/// Channel levels for tone mapping.
#[derive(Clone, Copy, Debug)]
pub struct ChannelLevels {
    pub black: [f64; 3],
    pub range: [f64; 3],
    pub exposure_scale: f64,
    pub paper_white: f64,
    pub highlight_rolloff: f64,
}

impl ChannelLevels {
    /// Create channel levels from black/white points.
    #[inline]
    pub fn new(
        black_r: f64,
        white_r: f64,
        black_g: f64,
        white_g: f64,
        black_b: f64,
        white_b: f64,
    ) -> Self {
        Self::with_tone_mapping(
            black_r,
            white_r,
            black_g,
            white_g,
            black_b,
            white_b,
            ToneMappingControls::default(),
        )
    }

    /// Create channel levels with explicit tone-mapping controls.
    #[inline]
    pub fn with_tone_mapping(
        black_r: f64,
        white_r: f64,
        black_g: f64,
        white_g: f64,
        black_b: f64,
        white_b: f64,
        controls: ToneMappingControls,
    ) -> Self {
        Self {
            black: [black_r, black_g, black_b],
            range: [
                (white_r - black_r).max(1e-14),
                (white_g - black_g).max(1e-14),
                (white_b - black_b).max(1e-14),
            ],
            exposure_scale: controls.exposure_scale,
            paper_white: controls.paper_white.clamp(0.5, 0.99),
            highlight_rolloff: controls.highlight_rolloff.max(0.1),
        }
    }

    #[inline]
    pub fn black_point(&self, channel: usize) -> f64 {
        self.black[channel]
    }

    #[inline]
    pub fn range(&self, channel: usize) -> f64 {
        self.range[channel]
    }

    #[inline]
    pub fn exposure_scale(&self) -> f64 {
        self.exposure_scale
    }

    #[inline]
    pub fn paper_white(&self) -> f64 {
        self.paper_white
    }

    #[inline]
    pub fn highlight_rolloff(&self) -> f64 {
        self.highlight_rolloff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::constants;

    #[test]
    fn test_channel_levels() {
        let levels = ChannelLevels::new(0.0, 1.0, 0.1, 0.9, 0.2, 0.8);

        assert_eq!(levels.black_point(0), 0.0);
        assert_eq!(levels.black_point(1), 0.1);
        assert_eq!(levels.black_point(2), 0.2);

        assert!((levels.range(0) - 1.0).abs() < 1e-10);
        assert!((levels.range(1) - 0.8).abs() < 1e-10);
        assert!((levels.range(2) - 0.6).abs() < 1e-10);
        assert!((levels.exposure_scale() - 1.0).abs() < 1e-10);
        assert!(levels.paper_white() > 0.8 && levels.paper_white() < 1.0);
        assert!(levels.highlight_rolloff() > 0.0);
    }

    #[test]
    fn test_channel_levels_new_computes_ranges() {
        let black_r = 0.05;
        let white_r = 0.95;
        let black_g = 0.1;
        let white_g = 0.7;
        let black_b = 0.0;
        let white_b = 1.0;

        let levels = ChannelLevels::new(black_r, white_r, black_g, white_g, black_b, white_b);

        let expected = [
            white_r - black_r,
            white_g - black_g,
            white_b - black_b,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((levels.range[i] - exp).abs() < 1e-12, "channel {i}");
        }
    }

    #[test]
    fn test_channel_levels_with_tone_mapping_stores_controls() {
        let controls = ToneMappingControls {
            exposure_scale: 1.25,
            paper_white: 0.88,
            highlight_rolloff: 4.0,
        };
        let levels = ChannelLevels::with_tone_mapping(
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, controls,
        );

        assert!((levels.exposure_scale - controls.exposure_scale).abs() < 1e-12);
        assert!((levels.paper_white - controls.paper_white).abs() < 1e-12);
        assert!((levels.highlight_rolloff - controls.highlight_rolloff).abs() < 1e-12);
    }

    #[test]
    fn test_channel_levels_default_tone_mapping() {
        let levels = ChannelLevels::new(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        assert!((levels.exposure_scale - 1.0).abs() < 1e-12);
        assert!((levels.paper_white - constants::DEFAULT_TONEMAP_PAPER_WHITE).abs() < 1e-12);
        assert!(
            (levels.highlight_rolloff - constants::DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF).abs() < 1e-12
        );
    }

    #[test]
    fn test_channel_levels_range_never_negative() {
        let levels = ChannelLevels::new(
            0.5, 0.2,
            0.0, 1.0,
            0.0, 1.0,
        );

        assert!(levels.range[0] >= 0.0);
        let raw = 0.2_f64 - 0.5_f64;
        let expected = raw.max(1e-14);
        assert!((levels.range[0] - expected).abs() < 1e-20);
        assert!((levels.range[1] - 1.0).abs() < 1e-12);
        assert!((levels.range[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_tone_mapping_controls_applied_to_levels() {
        let base = (
            0.0_f64, 1.0_f64,
            0.0_f64, 1.0_f64,
            0.0_f64, 1.0_f64,
        );
        let a = ChannelLevels::with_tone_mapping(
            base.0, base.1, base.2, base.3, base.4, base.5,
            ToneMappingControls {
                exposure_scale: 0.5,
                paper_white: 0.6,
                highlight_rolloff: 1.5,
            },
        );
        let b = ChannelLevels::with_tone_mapping(
            base.0, base.1, base.2, base.3, base.4, base.5,
            ToneMappingControls {
                exposure_scale: 2.0,
                paper_white: 0.95,
                highlight_rolloff: 5.0,
            },
        );

        assert_ne!(a.exposure_scale, b.exposure_scale);
        assert_ne!(a.paper_white, b.paper_white);
        assert_ne!(a.highlight_rolloff, b.highlight_rolloff);
    }
}
