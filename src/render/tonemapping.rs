//! Display tonemapping and 16-bit output quantization.

use super::constants;
use super::context::PixelBuffer;
use super::types::ChannelLevels;
use rayon::prelude::*;

#[inline]
fn compress_display_highlights(rgb: [f64; 3], paper_white: f64, rolloff: f64) -> [f64; 3] {
    let luminance = constants::rec709_luminance(rgb[0], rgb[1], rgb[2]);
    if luminance <= paper_white || luminance <= 1e-10 {
        return rgb;
    }

    let shoulder_span = (1.0 - paper_white).max(1e-6);
    let excess = luminance - paper_white;
    let compressed_luminance =
        paper_white + shoulder_span * (1.0 - (-(excess * rolloff) / shoulder_span).exp());
    let scale = compressed_luminance / luminance;

    [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
}

/// Core tonemapping function shared by preview and final output paths.
#[inline]
pub(super) fn tonemap_core(
    fr: f64,
    fg: f64,
    fb: f64,
    fa: f64,
    levels: &ChannelLevels,
    aces_tweak: bool,
) -> [f64; 3] {
    let alpha = fa.clamp(0.0, 1.0);
    if alpha <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let source = [fr.max(0.0), fg.max(0.0), fb.max(0.0)];
    let premult = [source[0] * alpha, source[1] * alpha, source[2] * alpha];
    if premult[0] <= 0.0 && premult[1] <= 0.0 && premult[2] <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let mut leveled = [0.0; 3];
    for i in 0..3 {
        leveled[i] =
            (((premult[i] - levels.black[i]).max(0.0)) / levels.range[i]) * levels.exposure_scale;
    }

    let r = leveled[0];
    let g = leveled[1];
    let b = leveled[2];

    let r_in = 0.842479062253094 * r + 0.0784335999999992 * g + 0.0792237451477643 * b;
    let g_in = 0.0423282422610123 * r + 0.878468636469772 * g + 0.0791661274605434 * b;
    let b_in = 0.0423756549057051 * r + 0.0784336000000000 * g + 0.877456439033405 * b;

    let min_ev = -10.0;
    let max_ev = 2.5;
    let range = max_ev - min_ev;

    let allocate = |v: f64| -> f64 {
        let val = v.max(1e-10).log2();
        ((val - min_ev) / range).clamp(0.0, 1.0)
    };

    let r_alloc = allocate(r_in);
    let g_alloc = allocate(g_in);
    let b_alloc = allocate(b_in);

    let spline = |x: f64| -> f64 {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x2 * x2;
        let x5 = x4 * x;
        let x6 = x5 * x;
        12.0625 * x6 - 36.3262 * x5 + 39.5298 * x4 - 17.6534 * x3 + 3.0135 * x2 + 0.3707 * x
    };

    let r_spline = spline(r_alloc);
    let g_spline = spline(g_alloc);
    let b_spline = spline(b_alloc);

    let (r_out, g_out, b_out) = if aces_tweak {
        (
            1.133276 * r_spline - 0.117109 * g_spline - 0.016167 * b_spline,
            -0.097008 * r_spline + 1.148151 * g_spline - 0.051143 * b_spline,
            -0.008107 * r_spline - 0.031776 * g_spline + 1.039883 * b_spline,
        )
    } else {
        (
            1.0987524 * r_spline - 0.0880758 * g_spline - 0.0106766 * b_spline,
            -0.0729567 * r_spline + 1.1114562 * g_spline - 0.0384995 * b_spline,
            -0.0060957 * r_spline - 0.0238959 * g_spline + 1.0299916 * b_spline,
        )
    };

    let compressed = compress_display_highlights(
        [r_out.max(0.0), g_out.max(0.0), b_out.max(0.0)],
        levels.paper_white,
        levels.highlight_rolloff,
    );

    [compressed[0].clamp(0.0, 1.0), compressed[1].clamp(0.0, 1.0), compressed[2].clamp(0.0, 1.0)]
}

/// Tonemap to 16-bit output for unit-level range tests.
#[cfg(test)]
#[inline]
pub(super) fn tonemap_to_16bit(
    fr: f64,
    fg: f64,
    fb: f64,
    fa: f64,
    levels: &ChannelLevels,
    aces_tweak: bool,
) -> [u16; 3] {
    let channels = tonemap_core(fr, fg, fb, fa, levels, aces_tweak);
    [
        crate::utils::f64_to_u16_saturating((channels[0] * constants::U16_MAX_F64).round()),
        crate::utils::f64_to_u16_saturating((channels[1] * constants::U16_MAX_F64).round()),
        crate::utils::f64_to_u16_saturating((channels[2] * constants::U16_MAX_F64).round()),
    ]
}

pub(super) fn tonemap_to_display_buffer(
    pixels: &PixelBuffer,
    levels: &ChannelLevels,
    aces_tweak: bool,
) -> PixelBuffer {
    pixels
        .par_iter()
        .map(|&(fr, fg, fb, fa)| {
            let mapped = tonemap_core(fr, fg, fb, fa, levels, aces_tweak);
            (mapped[0], mapped[1], mapped[2], fa.clamp(0.0, 1.0))
        })
        .collect()
}

pub(super) fn quantize_display_buffer_to_16bit(pixels: &PixelBuffer) -> Vec<u16> {
    let mut buf_16bit = vec![0u16; pixels.len() * 3];
    buf_16bit.par_chunks_mut(3).zip(pixels.par_iter()).for_each(|(chunk, &(r, g, b, _a))| {
        chunk[0] = (r.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
        chunk[1] = (g.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
        chunk[2] = (b.clamp(0.0, 1.0) * constants::U16_MAX_F64).round() as u16;
    });
    buf_16bit
}
