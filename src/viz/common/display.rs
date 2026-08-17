//! Spectral canvas: draw production-grade crisp spectral strokes into a
//! standalone SPD buffer and grade the result through the exact production
//! conversion, auto-levels, `AgX` tonemap, and Display P3 quantization.
//!
//! This is the shared "synthetic scene" path used by diagram-like modes
//! (braid, syzygy wheel, and future vocabulary experiments) so their strokes
//! carry the same spectral emission physics as the master render.

use crate::render::OklabColor;
use crate::render::constants::{DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF, DEFAULT_TONEMAP_PAPER_WHITE};
use crate::render::effects::convert_spd_buffer_to_rgba;
use crate::render::histogram::analyze_tonemapping;
use crate::render::{
    ChannelLevels, ImageBuffer, LineVertex, Rgb, SpectralLineSegment, ToneMappingControls,
    draw_line_segment_aa_spectral, quantize_display_buffer_to_16bit, tonemap_to_display_buffer,
};
use crate::spectrum::NUM_BINS;

/// Maximum number of pixels sampled for the auto-levels histogram.
const LEVELS_SAMPLE_CAP: usize = 2_000_000;

/// A standalone spectral accumulation canvas with production grading.
pub struct SpdCanvas {
    /// Per-pixel 64-bin spectral power distribution.
    spd: Vec<[f64; NUM_BINS]>,
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
}

impl SpdCanvas {
    /// Create a zeroed canvas.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self { spd: vec![[0.0; NUM_BINS]; width as usize * height as usize], width, height }
    }

    /// Splat one anti-aliased spectral segment (production stroke renderer).
    pub fn draw_segment(&mut self, segment: SpectralLineSegment) {
        draw_line_segment_aa_spectral(&mut self.spd, self.width, self.height, segment);
    }

    /// Convenience: splat a stroke between two points with uniform color.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_stroke(
        &mut self,
        from: (f32, f32),
        to: (f32, f32),
        color_start: OklabColor,
        color_end: OklabColor,
        alpha: f64,
        energy: f64,
        thickness_factor: f64,
    ) {
        self.draw_segment(SpectralLineSegment {
            start: LineVertex { x: from.0, y: from.1, z: 0.0, color: color_start, alpha },
            end: LineVertex { x: to.0, y: to.1, z: 0.0, color: color_end, alpha },
            hdr_scale: energy,
            thickness_factor,
        });
    }

    /// Convert, auto-level, tonemap, and quantize to a 16-bit Display P3
    /// image using the production pipeline.
    ///
    /// `clip_black` / `clip_white` follow the run's resolved percentiles;
    /// `exposure_key` biases display exposure exactly like the profile's
    /// seed key (1.0 = neutral).
    #[must_use]
    pub fn into_png16(
        self,
        clip_black: f64,
        clip_white: f64,
        exposure_key: f64,
    ) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
        let width = self.width as usize;
        let height = self.height as usize;
        let mut rgba = vec![(0.0, 0.0, 0.0, 0.0); width * height];
        convert_spd_buffer_to_rgba(&self.spd, &mut rgba, width, height);

        // Auto-levels from premultiplied samples, mirroring pass 1.
        let stride = (rgba.len() / LEVELS_SAMPLE_CAP).max(1);
        let samples: Vec<[f64; 3]> =
            rgba.iter().step_by(stride).map(|&(r, g, b, a)| [r * a, g * a, b * a]).collect();
        let analysis = analyze_tonemapping(&samples, clip_black, clip_white);
        let levels = ChannelLevels::with_tone_mapping(
            analysis.black_r,
            analysis.white_r,
            analysis.black_g,
            analysis.white_g,
            analysis.black_b,
            analysis.white_b,
            ToneMappingControls {
                exposure_scale: analysis.exposure_scale * exposure_key.clamp(0.5, 1.5),
                paper_white: DEFAULT_TONEMAP_PAPER_WHITE,
                highlight_rolloff: DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF,
            },
        );

        let display = tonemap_to_display_buffer(&rgba, &levels);
        let quantized = quantize_display_buffer_to_16bit(&display);
        ImageBuffer::from_raw(self.width, self.height, quantized)
            .expect("quantized buffer has exactly width*height*3 samples")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stroke_deposits_energy_and_grades_to_nonblack_png() {
        let mut canvas = SpdCanvas::new(64, 64);
        canvas.draw_stroke(
            (8.0, 32.0),
            (56.0, 32.0),
            (0.8, 0.1, 0.05),
            (0.8, 0.1, 0.05),
            0.9,
            0.05,
            1.0,
        );
        let image = canvas.into_png16(0.01, 0.99, 1.0);
        assert_eq!(image.width(), 64);
        let max = image.as_raw().iter().copied().max().unwrap_or(0);
        assert!(max > 8_000, "graded stroke should be clearly visible, max={max}");
    }
}
