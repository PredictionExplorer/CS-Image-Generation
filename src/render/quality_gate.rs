//! Post-render image quality gate.
//!
//! A final sanity check between the 16-bit RGB tonemapped output and
//! `save_image_as_png_16bit`. The earlier pipeline fixes (scene-linear
//! ceiling, metering, per-effect caps, stack guard) are sufficient to
//! prevent blowouts in virtually all cases, but a post-render gate is
//! the "belt and suspenders" layer: if something still slips through,
//! we reject the image rather than ship it.
//!
//! The gate specifically targets the two failure modes observed in the
//! wild:
//! 1. **White blob / flat-white frame** — compact bright cores that
//!    clipped all three channels to near-max.
//! 2. **Empty-canvas frame** — subject completely outside the viewport
//!    (framing failure) leaving a nearly black image with, at most, a
//!    few stray specks.

use image::{ImageBuffer, Rgb};

/// Threshold, in the `[0, 65535]` PNG range, above which all three
/// channels are considered "near-white". Chosen high enough that normal
/// specular highlights don't register but low enough to catch clipped
/// cores well before they would fully saturate.
const NEAR_WHITE_THRESHOLD: u16 = 63_000;

/// Luminance threshold below which a pixel is considered "near-black"
/// in the 16-bit intermediate space.
const NEAR_BLACK_LUMA_THRESHOLD: f64 = 500.0;

/// Maximum allowed fraction of near-white pixels in an accepted image.
/// Empirically, a good render rarely exceeds ~3 %; flagging at 15 %
/// gives substantial headroom for intentionally bright compositions
/// while still catching blown-out frames.
const MAX_NEAR_WHITE_FRACTION: f64 = 0.15;

/// Maximum fraction of near-black pixels before we call the frame
/// "empty canvas". Museum-quality output can be dark, but 97 % near-
/// black implies the subject never made it into the frame.
const MAX_NEAR_BLACK_FRACTION: f64 = 0.97;

/// Verdict returned by [`analyze_white_blob`].
#[derive(Debug, Clone, PartialEq)]
pub enum QaVerdict {
    /// The frame passed all checks.
    Pass,
    /// The frame failed; `reason` describes which check fired and with
    /// what numbers. Suitable for logging and for inclusion in a
    /// `RenderError::QualityGateRejected`.
    Reject {
        /// Human-readable description of the failure.
        reason: String,
    },
}

/// Analyse a tonemapped 16-bit RGB image for catastrophic quality
/// failures (white blob, empty canvas).
///
/// This is intentionally cheap — a single pass over the pixels counting
/// near-white and near-black fractions. Runs after all rendering is
/// complete but before the PNG is written, so the caller can delete
/// partial outputs and surface a rescue-regeneratable error on failure.
pub fn analyze_white_blob(buf: &ImageBuffer<Rgb<u16>, Vec<u16>>) -> QaVerdict {
    let total = (buf.width() as usize) * (buf.height() as usize);
    if total == 0 {
        return QaVerdict::Reject { reason: "empty framebuffer".into() };
    }

    let mut near_white = 0usize;
    let mut near_black = 0usize;
    for Rgb([r, g, b]) in buf.pixels().copied() {
        if r >= NEAR_WHITE_THRESHOLD && g >= NEAR_WHITE_THRESHOLD && b >= NEAR_WHITE_THRESHOLD {
            near_white += 1;
        }
        let luma = 0.2126 * f64::from(r) + 0.7152 * f64::from(g) + 0.0722 * f64::from(b);
        if luma < NEAR_BLACK_LUMA_THRESHOLD {
            near_black += 1;
        }
    }

    let total_f = total as f64;
    let near_white_frac = near_white as f64 / total_f;
    let near_black_frac = near_black as f64 / total_f;

    if near_white_frac > MAX_NEAR_WHITE_FRACTION {
        return QaVerdict::Reject {
            reason: format!(
                "white-blob detected: near-white fraction {:.3} > {:.3}",
                near_white_frac, MAX_NEAR_WHITE_FRACTION
            ),
        };
    }
    if near_black_frac > MAX_NEAR_BLACK_FRACTION {
        return QaVerdict::Reject {
            reason: format!(
                "empty-canvas detected: near-black fraction {:.3} > {:.3}",
                near_black_frac, MAX_NEAR_BLACK_FRACTION
            ),
        };
    }

    QaVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, rgb: [u16; 3]) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
        ImageBuffer::from_fn(width, height, |_, _| Rgb(rgb))
    }

    #[test]
    fn pass_on_mid_gray_frame() {
        let img = solid(16, 16, [20_000, 20_000, 20_000]);
        assert_eq!(analyze_white_blob(&img), QaVerdict::Pass);
    }

    #[test]
    fn reject_solid_white() {
        let img = solid(16, 16, [u16::MAX, u16::MAX, u16::MAX]);
        let v = analyze_white_blob(&img);
        assert!(matches!(v, QaVerdict::Reject { .. }));
    }

    #[test]
    fn reject_solid_black() {
        let img = solid(16, 16, [0, 0, 0]);
        let v = analyze_white_blob(&img);
        assert!(matches!(v, QaVerdict::Reject { .. }));
    }

    #[test]
    fn pass_with_small_white_highlight() {
        // ~5 % white pixels in a dim background — should pass.
        let w = 20u32;
        let h = 20u32;
        let mut img: ImageBuffer<Rgb<u16>, Vec<u16>> = solid(w, h, [8_000, 8_000, 8_000]);
        let white_count = ((w * h) as f64 * 0.05) as u32;
        for i in 0..white_count {
            let x = i % w;
            let y = i / w;
            img.put_pixel(x, y, Rgb([u16::MAX, u16::MAX, u16::MAX]));
        }
        assert_eq!(analyze_white_blob(&img), QaVerdict::Pass);
    }

    #[test]
    fn reject_with_large_white_cluster() {
        // ~25 % white pixels — over the 15 % threshold.
        let w = 20u32;
        let h = 20u32;
        let mut img: ImageBuffer<Rgb<u16>, Vec<u16>> = solid(w, h, [8_000, 8_000, 8_000]);
        let white_count = ((w * h) as f64 * 0.25) as u32;
        for i in 0..white_count {
            let x = i % w;
            let y = i / w;
            img.put_pixel(x, y, Rgb([u16::MAX, u16::MAX, u16::MAX]));
        }
        let v = analyze_white_blob(&img);
        assert!(matches!(v, QaVerdict::Reject { .. }));
    }
}
