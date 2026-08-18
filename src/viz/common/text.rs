//! Typography for poster modes (master plan II.10): an `ab_glyph`-based
//! rasterizer over the bundled IBM Plex faces (OFL, `assets/fonts/`),
//! drawing anti-aliased text into linear-light `Rgb64` buffers.
//!
//! Latin + digits + basic punctuation only (no shaping engine); tabular
//! numeral mode guarantees fixed digit advances for data columns, and
//! small caps are faked with scaled uppercase glyphs (Plex ships no
//! small-cap cut).

use crate::viz::common::raster::Rgb64;
use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use std::sync::OnceLock;

/// Bundled typefaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    /// IBM Plex Sans Regular.
    Sans,
    /// IBM Plex Sans Italic.
    SansItalic,
    /// IBM Plex Mono Regular.
    Mono,
}

static SANS_BYTES: &[u8] = include_bytes!("../../../assets/fonts/IBMPlexSans-Regular.ttf");
static SANS_ITALIC_BYTES: &[u8] = include_bytes!("../../../assets/fonts/IBMPlexSans-Italic.ttf");
static MONO_BYTES: &[u8] = include_bytes!("../../../assets/fonts/IBMPlexMono-Regular.ttf");

/// Parsed font for a face (parsed once, process-wide).
fn font(face: Face) -> &'static FontRef<'static> {
    static FONTS: OnceLock<[FontRef<'static>; 3]> = OnceLock::new();
    let fonts = FONTS.get_or_init(|| {
        [
            FontRef::try_from_slice(SANS_BYTES).expect("bundled Plex Sans parses"),
            FontRef::try_from_slice(SANS_ITALIC_BYTES).expect("bundled Plex Sans Italic parses"),
            FontRef::try_from_slice(MONO_BYTES).expect("bundled Plex Mono parses"),
        ]
    });
    match face {
        Face::Sans => &fonts[0],
        Face::SansItalic => &fonts[1],
        Face::Mono => &fonts[2],
    }
}

/// Horizontal anchoring of the drawn string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    /// `x` is the left edge.
    Left,
    /// `x` is the center.
    Center,
    /// `x` is the right edge.
    Right,
}

/// Text style for one draw call.
#[derive(Clone, Copy, Debug)]
pub struct TextStyle {
    /// Typeface.
    pub face: Face,
    /// Pixel size (cap-to-descender scale).
    pub px: f64,
    /// Extra tracking as a fraction of `px` added to every advance.
    pub tracking: f64,
    /// Ink color (linear Rec.2020).
    pub color: Rgb64,
    /// Ink opacity.
    pub opacity: f64,
    /// Fake small caps (lowercase drawn as scaled uppercase).
    pub small_caps: bool,
    /// Tabular numerals (digits centered in a shared fixed advance).
    pub tabular: bool,
    /// Horizontal anchoring.
    pub align: Align,
}

impl TextStyle {
    /// A plain left-aligned style in the given face, size, and color.
    #[must_use]
    pub fn new(face: Face, px: f64, color: Rgb64) -> Self {
        Self {
            face,
            px,
            tracking: 0.0,
            color,
            opacity: 1.0,
            small_caps: false,
            tabular: false,
            align: Align::Left,
        }
    }

    /// Archival caption variant: small caps with +8% tracking (II.14).
    #[must_use]
    pub fn caption(face: Face, px: f64, color: Rgb64) -> Self {
        Self { small_caps: true, tracking: 0.08, ..Self::new(face, px, color) }
    }
}

/// One positioned glyph after layout.
struct Placed {
    glyph_id: ab_glyph::GlyphId,
    /// Pen x offset from the string origin.
    x: f64,
    /// Glyph pixel scale (differs from the base for fake small caps).
    scale: f64,
}

/// Widest digit advance at the style's scale (the tabular column width).
fn tabular_advance(face: Face, px: f64) -> f64 {
    let scaled = font(face).as_scaled(PxScale::from(px as f32));
    ('0'..='9')
        .map(|digit| f64::from(scaled.h_advance(scaled.font().glyph_id(digit))))
        .fold(0.0, f64::max)
}

/// Lay out a string: per-glyph positions plus the total advance width.
fn layout(style: &TextStyle, text: &str) -> (Vec<Placed>, f64) {
    let base_font = font(style.face);
    let base_scale = style.px;
    let small_scale = style.px * 0.78;
    let tab_advance = if style.tabular {
        tabular_advance(style.face, base_scale) + style.tracking * style.px
    } else {
        0.0
    };

    let mut placed = Vec::with_capacity(text.chars().count());
    let mut pen = 0.0f64;
    let mut previous: Option<(ab_glyph::GlyphId, f64)> = None;
    for ch in text.chars() {
        let (drawn_char, scale) = if style.small_caps && ch.is_lowercase() {
            (ch.to_uppercase().next().unwrap_or(ch), small_scale)
        } else {
            (ch, base_scale)
        };
        let glyph_id = base_font.glyph_id(drawn_char);
        let scaled = base_font.as_scaled(PxScale::from(scale as f32));
        if let Some((previous_id, previous_scale)) = previous
            && (previous_scale - scale).abs() < f64::EPSILON
        {
            pen += f64::from(scaled.kern(previous_id, glyph_id));
        }
        if style.tabular && drawn_char.is_ascii_digit() {
            // Center the digit inside the shared tabular column.
            let advance = f64::from(scaled.h_advance(glyph_id));
            placed.push(Placed { glyph_id, x: pen + (tab_advance - advance) / 2.0, scale });
            pen += tab_advance;
        } else {
            placed.push(Placed { glyph_id, x: pen, scale });
            pen += f64::from(scaled.h_advance(glyph_id)) + style.tracking * style.px;
        }
        previous = Some((glyph_id, scale));
    }
    (placed, pen)
}

/// Measured advance width and (ascent, descent) of a string in pixels.
#[must_use]
pub fn measure(style: &TextStyle, text: &str) -> (f64, f64, f64) {
    let (_, width) = layout(style, text);
    let scaled = font(style.face).as_scaled(PxScale::from(style.px as f32));
    (width, f64::from(scaled.ascent()), f64::from(-scaled.descent()))
}

/// Draw a string into a linear `Rgb64` buffer; `(x, y)` is the baseline
/// anchor interpreted per `style.align`.
#[allow(clippy::similar_names)]
pub fn draw_text(
    buffer: &mut [Rgb64],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
    style: &TextStyle,
    text: &str,
) {
    let (placed, total) = layout(style, text);
    let origin = match style.align {
        Align::Left => x,
        Align::Center => x - total / 2.0,
        Align::Right => x - total,
    };
    let base_font = font(style.face);
    for glyph in placed {
        let positioned = glyph.glyph_id.with_scale_and_position(
            PxScale::from(glyph.scale as f32),
            ab_glyph::point((origin + glyph.x) as f32, y as f32),
        );
        let Some(outline) = base_font.outline_glyph(positioned) else {
            continue;
        };
        let bounds = outline.px_bounds();
        outline.draw(|gx, gy, coverage| {
            let px = bounds.min.x as i64 + i64::from(gx);
            let py = bounds.min.y as i64 + i64::from(gy);
            if px < 0 || py < 0 || px >= width as i64 || py >= height as i64 {
                return;
            }
            let alpha = f64::from(coverage).clamp(0.0, 1.0) * style.opacity;
            if alpha <= 0.0 {
                return;
            }
            let pixel = &mut buffer[py as usize * width + px as usize];
            pixel.0 = pixel.0 * (1.0 - alpha) + style.color.0 * alpha;
            pixel.1 = pixel.1 * (1.0 - alpha) + style.color.1 * alpha;
            pixel.2 = pixel.2 * (1.0 - alpha) + style.color.2 * alpha;
        });
    }
}

/// The character set every poster string is drawn from; a unit test pins
/// glyph coverage for all three faces.
pub const POSTER_CHARSET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ\
     abcdefghijklmnopqrstuvwxyz0123456789 .,:;+-=()[]/#%*'\"_|<>{}!?@&\u{b0}\u{d7}\u{2026}\u{b7}";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_faces_cover_the_poster_charset() {
        for face in [Face::Sans, Face::SansItalic, Face::Mono] {
            let parsed = font(face);
            for ch in POSTER_CHARSET.chars() {
                if ch == ' ' {
                    continue;
                }
                assert_ne!(parsed.glyph_id(ch).0, 0, "face {face:?} is missing glyph for {ch:?}");
            }
        }
    }

    #[test]
    fn tabular_digits_share_one_advance() {
        let style =
            TextStyle { tabular: true, ..TextStyle::new(Face::Sans, 32.0, (1.0, 1.0, 1.0)) };
        let (w0, _, _) = measure(&style, "111");
        let (w1, _, _) = measure(&style, "888");
        assert!((w0 - w1).abs() < 1e-9, "tabular widths must match: {w0} vs {w1}");
    }

    #[test]
    fn tracking_widens_the_measure() {
        let plain = TextStyle::new(Face::Sans, 32.0, (1.0, 1.0, 1.0));
        let tracked = TextStyle { tracking: 0.1, ..plain };
        let (w_plain, ascent, descent) = measure(&plain, "MEASURE");
        let (w_tracked, _, _) = measure(&tracked, "MEASURE");
        assert!(w_tracked > w_plain + 5.0);
        assert!(ascent > 0.0 && descent > 0.0);
    }

    #[test]
    fn draw_places_ink_inside_the_buffer() {
        let mut buffer = vec![(0.0, 0.0, 0.0); 200 * 60];
        let style = TextStyle::new(Face::Mono, 30.0, (1.0, 0.9, 0.8));
        draw_text(&mut buffer, 200, 60, 8.0, 44.0, &style, "T+0142");
        let ink: f64 = buffer.iter().map(|&(r, _, _)| r).sum();
        assert!(ink > 10.0, "expected visible ink, got {ink}");
    }

    #[test]
    fn small_caps_render_lowercase_as_smaller_capitals() {
        let caps = TextStyle::caption(Face::Sans, 32.0, (1.0, 1.0, 1.0));
        let (lower_width, _, _) = measure(&caps, "abc");
        let upper = TextStyle { small_caps: false, ..caps };
        let (upper_width, _, _) = measure(&upper, "ABC");
        assert!(lower_width < upper_width, "small caps must be narrower than full caps");
        assert!(lower_width > upper_width * 0.5);
    }
}
