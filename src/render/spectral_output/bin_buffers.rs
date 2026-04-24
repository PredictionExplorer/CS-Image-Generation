//! Per-wavelength bin buffers shared by spectral gallery and sweep output.

use crate::render::constants;
#[cfg(test)]
use crate::render::context::PixelBuffer;
use crate::render::error::{RenderError, Result};
use crate::spectrum::{NUM_BINS, wavelength_nm_for_bin, wavelength_to_rgb};
use rayon::prelude::*;

/// Pre-computed per-bin float RGB images used by both the gallery and sweep video.
pub struct BinBuffers {
    pub(crate) buffers: Vec<Vec<[f32; 3]>>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl BinBuffers {
    fn validate_image_shape(
        accum_spd: &[[f64; NUM_BINS]],
        width: u32,
        height: u32,
    ) -> Result<(usize, usize, usize)> {
        let pixel_count = checked_pixel_count(width, height)?;
        if accum_spd.len() != pixel_count {
            return Err(RenderError::InvalidScene {
                reason: format!(
                    "accumulated SPD length ({}) does not match dimensions {width}x{height} ({pixel_count} pixels)",
                    accum_spd.len()
                ),
            });
        }

        let width_usize =
            usize::try_from(width).map_err(|_| RenderError::InvalidDimensions { width, height })?;
        let height_usize = usize::try_from(height)
            .map_err(|_| RenderError::InvalidDimensions { width, height })?;

        Ok((width_usize, height_usize, pixel_count))
    }

    /// Try to build 64 bin images from an accumulated SPD buffer.
    ///
    /// Each bin image normalizes that bin's energy across all pixels, tints by the
    /// bin's wavelength color, and applies display gamma.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions are zero, dimensions overflow the host
    /// address space, or `accum_spd.len()` does not equal `width * height`.
    pub fn try_new(accum_spd: &[[f64; NUM_BINS]], width: u32, height: u32) -> Result<Self> {
        let (width, height, pixel_count) = Self::validate_image_shape(accum_spd, width, height)?;
        Ok(Self::build(accum_spd, width, height, pixel_count))
    }

    /// Build 64 bin images from the accumulated SPD buffer.
    ///
    /// Each bin image normalizes that bin's energy across all pixels, tints by the
    /// bin's wavelength color, and applies display gamma.
    ///
    /// Prefer [`Self::try_new`] when dimensions or buffer shape come from a
    /// caller or other untrusted boundary.
    ///
    /// # Panics
    ///
    /// Panics when `width * height` overflows `usize` or `accum_spd.len()` does
    /// not equal `width * height`.
    #[must_use]
    pub fn new(accum_spd: &[[f64; NUM_BINS]], width: usize, height: usize) -> Self {
        let pixel_count = width.checked_mul(height).expect("image dimensions overflow usize");
        assert_eq!(accum_spd.len(), pixel_count);

        Self::build(accum_spd, width, height, pixel_count)
    }

    fn build(
        accum_spd: &[[f64; NUM_BINS]],
        width: usize,
        height: usize,
        pixel_count: usize,
    ) -> Self {
        let inv_gamma = 1.0 / constants::DISPLAY_GAMMA;

        let buffers: Vec<Vec<[f32; 3]>> = (0..NUM_BINS)
            .into_par_iter()
            .map(|bin| {
                let wavelength = wavelength_nm_for_bin(bin);
                let (tint_r, tint_g, tint_b) = wavelength_to_rgb(wavelength);

                let max_val =
                    accum_spd.iter().map(|spd| spd[bin]).fold(0.0f64, f64::max).max(1e-10);

                let mut buf = vec![[0.0f32; 3]; pixel_count];
                for (i, pixel) in buf.iter_mut().enumerate() {
                    let normalized = (accum_spd[i][bin] / max_val).clamp(0.0, 1.0);
                    pixel[0] = ((normalized * tint_r).powf(inv_gamma)) as f32;
                    pixel[1] = ((normalized * tint_g).powf(inv_gamma)) as f32;
                    pixel[2] = ((normalized * tint_b).powf(inv_gamma)) as f32;
                }
                buf
            })
            .collect();

        Self { buffers, width, height }
    }

    /// Number of pixels (`width * height`) per bin buffer.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Return the first and last bin indices that contain visible energy.
    ///
    /// A bin is considered active when at least one pixel has an RGB component
    /// above a small threshold.  The result is padded by 2 bins on each side
    /// and clamped to `[0, NUM_BINS - 1]`, falling back to the static
    /// `SWEEP_BIN_START..=SWEEP_BIN_END` range when no bins are active.
    #[must_use]
    pub fn active_bin_range(&self) -> (usize, usize) {
        const THRESHOLD: f32 = 1e-4;
        const PAD: usize = 2;

        let first_active = self.buffers.iter().position(|buf| {
            buf.iter().any(|px| px[0] > THRESHOLD || px[1] > THRESHOLD || px[2] > THRESHOLD)
        });
        let last_active = self.buffers.iter().rposition(|buf| {
            buf.iter().any(|px| px[0] > THRESHOLD || px[1] > THRESHOLD || px[2] > THRESHOLD)
        });

        match (first_active, last_active) {
            (Some(lo), Some(hi)) => {
                let lo = lo.saturating_sub(PAD);
                let hi = (hi + PAD).min(NUM_BINS - 1);
                (lo, hi)
            }
            _ => (constants::SWEEP_BIN_START, constants::SWEEP_BIN_END),
        }
    }

    /// Average all bin images into a single full-spectrum composite `PixelBuffer`.
    #[cfg(test)]
    pub(super) fn composite(&self) -> PixelBuffer {
        let pixel_count = self.pixel_count();
        let inv_bins = 1.0 / NUM_BINS as f64;
        (0..pixel_count)
            .into_par_iter()
            .map(|i| {
                let (mut r, mut g, mut b) = (0.0f64, 0.0f64, 0.0f64);
                for bin in &self.buffers {
                    r += f64::from(bin[i][0]);
                    g += f64::from(bin[i][1]);
                    b += f64::from(bin[i][2]);
                }
                (r * inv_bins, g * inv_bins, b * inv_bins, 1.0)
            })
            .collect()
    }
}

pub(super) fn checked_pixel_count(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(RenderError::InvalidDimensions { width, height });
    }

    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or(RenderError::InvalidDimensions { width, height })
}

impl std::fmt::Debug for BinBuffers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinBuffers")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("num_bins", &self.buffers.len())
            .finish()
    }
}
