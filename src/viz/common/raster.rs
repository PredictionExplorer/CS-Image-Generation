//! Direct-color rasterization helpers: anti-aliased hairlines and markers
//! blended into linear RGB buffers, plus separable Gaussian blurs for scalar
//! fields. Used by the ink-on-paper poster modes (contours, spectrum card).

/// Linear RGB pixel.
pub type Rgb64 = (f64, f64, f64);

/// Blend `color` over the buffer with coverage `alpha`.
#[inline]
fn blend(pixel: &mut Rgb64, color: Rgb64, alpha: f64) {
    pixel.0 = pixel.0 * (1.0 - alpha) + color.0 * alpha;
    pixel.1 = pixel.1 * (1.0 - alpha) + color.1 * alpha;
    pixel.2 = pixel.2 * (1.0 - alpha) + color.2 * alpha;
}

/// Draw an anti-aliased line segment with the given stroke width (pixels).
#[allow(clippy::too_many_arguments)]
pub fn draw_line(
    buffer: &mut [Rgb64],
    width: usize,
    height: usize,
    from: (f32, f32),
    to: (f32, f32),
    color: Rgb64,
    stroke_px: f64,
    opacity: f64,
) {
    let (x0, y0) = (f64::from(from.0), f64::from(from.1));
    let (x1, y1) = (f64::from(to.0), f64::from(to.1));
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len_sq = dx * dx + dy * dy;
    let half = stroke_px * 0.5;
    let pad = (half + 1.5).ceil() as i64;

    let min_x = ((x0.min(x1)) as i64 - pad).max(0);
    let max_x = ((x0.max(x1)) as i64 + pad).min(width as i64 - 1);
    let min_y = ((y0.min(y1)) as i64 - pad).max(0);
    let max_y = ((y0.max(y1)) as i64 + pad).min(height as i64 - 1);
    if min_x > max_x || min_y > max_y {
        return;
    }

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let sx = px as f64 + 0.5;
            let sy = py as f64 + 0.5;
            let t = if len_sq > 1e-12 {
                (((sx - x0) * dx + (sy - y0) * dy) / len_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let cx = x0 + dx * t;
            let cy = y0 + dy * t;
            let dist = ((sx - cx).powi(2) + (sy - cy).powi(2)).sqrt();
            // Smooth coverage: full inside the stroke, 1-pixel falloff.
            let coverage = (half + 0.5 - dist).clamp(0.0, 1.0);
            if coverage > 0.0 {
                blend(&mut buffer[py as usize * width + px as usize], color, coverage * opacity);
            }
        }
    }
}

/// Draw an anti-aliased line into an RGBA buffer (rgb blended by coverage,
/// alpha raised to at least the coverage). Used for compositing display-
/// intent ink over tonemapped frames, where alpha is ignored downstream.
#[allow(clippy::too_many_arguments)]
pub fn draw_line_rgba(
    buffer: &mut [(f64, f64, f64, f64)],
    width: usize,
    height: usize,
    from: (f32, f32),
    to: (f32, f32),
    color: Rgb64,
    stroke_px: f64,
    opacity: f64,
) {
    let (x0, y0) = (f64::from(from.0), f64::from(from.1));
    let (x1, y1) = (f64::from(to.0), f64::from(to.1));
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len_sq = dx * dx + dy * dy;
    let half = stroke_px * 0.5;
    let pad = (half + 1.5).ceil() as i64;

    let min_x = ((x0.min(x1)) as i64 - pad).max(0);
    let max_x = ((x0.max(x1)) as i64 + pad).min(width as i64 - 1);
    let min_y = ((y0.min(y1)) as i64 - pad).max(0);
    let max_y = ((y0.max(y1)) as i64 + pad).min(height as i64 - 1);
    if min_x > max_x || min_y > max_y {
        return;
    }

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let sx = px as f64 + 0.5;
            let sy = py as f64 + 0.5;
            let t = if len_sq > 1e-12 {
                (((sx - x0) * dx + (sy - y0) * dy) / len_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let cx = x0 + dx * t;
            let cy = y0 + dy * t;
            let dist = ((sx - cx).powi(2) + (sy - cy).powi(2)).sqrt();
            let coverage = (half + 0.5 - dist).clamp(0.0, 1.0) * opacity;
            if coverage > 0.0 {
                let pixel = &mut buffer[py as usize * width + px as usize];
                pixel.0 = pixel.0 * (1.0 - coverage) + color.0 * coverage;
                pixel.1 = pixel.1 * (1.0 - coverage) + color.1 * coverage;
                pixel.2 = pixel.2 * (1.0 - coverage) + color.2 * coverage;
                pixel.3 = pixel.3.max(coverage);
            }
        }
    }
}

/// Draw a small plus-shaped marker (spot heights, registration marks).
#[allow(clippy::too_many_arguments)]
pub fn draw_cross(
    buffer: &mut [Rgb64],
    width: usize,
    height: usize,
    center: (f32, f32),
    half_size: f32,
    color: Rgb64,
    stroke_px: f64,
    opacity: f64,
) {
    let (cx, cy) = center;
    draw_line(
        buffer,
        width,
        height,
        (cx - half_size, cy),
        (cx + half_size, cy),
        color,
        stroke_px,
        opacity,
    );
    draw_line(
        buffer,
        width,
        height,
        (cx, cy - half_size),
        (cx, cy + half_size),
        color,
        stroke_px,
        opacity,
    );
}

/// Additively splat a line segment into weight and weighted-value fields
/// (used for energy-weighted depth accumulation). `value` is interpolated
/// along the segment; deposit falls off smoothly across `stroke_px`.
#[allow(clippy::too_many_arguments)]
pub fn splat_line_additive(
    weight_field: &mut [f32],
    value_field: &mut [f32],
    width: usize,
    height: usize,
    from: (f32, f32),
    to: (f32, f32),
    value_from: f32,
    value_to: f32,
    stroke_px: f64,
) {
    let (x0, y0) = (f64::from(from.0), f64::from(from.1));
    let (x1, y1) = (f64::from(to.0), f64::from(to.1));
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len_sq = dx * dx + dy * dy;
    let half = stroke_px * 0.5;
    let pad = (half + 1.5).ceil() as i64;

    let min_x = ((x0.min(x1)) as i64 - pad).max(0);
    let max_x = ((x0.max(x1)) as i64 + pad).min(width as i64 - 1);
    let min_y = ((y0.min(y1)) as i64 - pad).max(0);
    let max_y = ((y0.max(y1)) as i64 + pad).min(height as i64 - 1);
    if min_x > max_x || min_y > max_y {
        return;
    }

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let sx = px as f64 + 0.5;
            let sy = py as f64 + 0.5;
            let t = if len_sq > 1e-12 {
                (((sx - x0) * dx + (sy - y0) * dy) / len_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let cx = x0 + dx * t;
            let cy = y0 + dy * t;
            let dist = ((sx - cx).powi(2) + (sy - cy).powi(2)).sqrt();
            let coverage = (half + 0.5 - dist).clamp(0.0, 1.0) as f32;
            if coverage > 0.0 {
                let index = py as usize * width + px as usize;
                let value = value_from + (value_to - value_from) * t as f32;
                weight_field[index] += coverage;
                value_field[index] += coverage * value;
            }
        }
    }
}

/// Separable Gaussian blur of a scalar field, in place (allocates one temp).
pub fn blur_field(field: &mut [f32], width: usize, height: usize, sigma: f64) {
    if sigma <= 0.05 {
        return;
    }
    let radius = (sigma * 3.0).ceil() as i64;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|offset| (-(offset as f64).powi(2) / (2.0 * sigma * sigma)).exp())
        .collect();
    let kernel_sum: f64 = kernel.iter().sum();

    let mut temp = vec![0.0f32; field.len()];
    // Horizontal pass.
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f64;
            for (index, &weight) in kernel.iter().enumerate() {
                let sample_x =
                    (x as i64 + index as i64 - radius).clamp(0, width as i64 - 1) as usize;
                sum += f64::from(field[y * width + sample_x]) * weight;
            }
            temp[y * width + x] = (sum / kernel_sum) as f32;
        }
    }
    // Vertical pass.
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f64;
            for (index, &weight) in kernel.iter().enumerate() {
                let sample_y =
                    (y as i64 + index as i64 - radius).clamp(0, height as i64 - 1) as usize;
                sum += f64::from(temp[sample_y * width + x]) * weight;
            }
            field[y * width + x] = (sum / kernel_sum) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_deposits_ink_along_its_path() {
        let mut buffer = vec![(0.0, 0.0, 0.0); 32 * 32];
        // Draw through pixel centers (y = 16.5 is the center of row 16).
        draw_line(&mut buffer, 32, 32, (4.0, 16.5), (28.0, 16.5), (1.0, 1.0, 1.0), 1.5, 1.0);
        assert!(buffer[16 * 32 + 16].0 > 0.9, "center of stroke should be inked");
        assert!(buffer[2 * 32 + 16].0 < 1e-9, "far row should stay clean");
    }

    #[test]
    fn blur_preserves_mass_approximately() {
        let mut field = vec![0.0f32; 64 * 64];
        field[32 * 64 + 32] = 100.0;
        let before: f64 = field.iter().map(|&v| f64::from(v)).sum();
        blur_field(&mut field, 64, 64, 3.0);
        let after: f64 = field.iter().map(|&v| f64::from(v)).sum();
        assert!((before - after).abs() / before < 0.01);
    }
}
