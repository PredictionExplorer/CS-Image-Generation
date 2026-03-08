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
    fn test_channel_levels_with_tone_mapping() {
        let levels = ChannelLevels::with_tone_mapping(
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            ToneMappingControls { exposure_scale: 0.75, paper_white: 0.9, highlight_rolloff: 3.0 },
        );

        assert!((levels.exposure_scale() - 0.75).abs() < 1e-10);
        assert!((levels.paper_white() - 0.9).abs() < 1e-10);
        assert!((levels.highlight_rolloff() - 3.0).abs() < 1e-10);
    }
}
