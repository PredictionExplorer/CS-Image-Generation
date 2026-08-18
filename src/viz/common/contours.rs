//! Marching-squares contour extraction over scalar fields.
//!
//! Returns raw line segments (endpoint pairs with linear interpolation along
//! cell edges); hairline rendering does not require polyline assembly.

/// One contour line segment in pixel coordinates.
pub type Segment = ((f32, f32), (f32, f32));

/// Interpolated crossing position along an edge between two samples.
#[inline]
fn interpolate(level: f32, a: f32, b: f32) -> f32 {
    if (b - a).abs() < 1e-12 { 0.5 } else { ((level - a) / (b - a)).clamp(0.0, 1.0) }
}

/// Extract iso-level segments from a row-major scalar field.
#[must_use]
pub fn marching_squares(field: &[f32], width: usize, height: usize, level: f32) -> Vec<Segment> {
    let mut segments = Vec::new();
    if width < 2 || height < 2 {
        return segments;
    }
    for y in 0..height - 1 {
        for x in 0..width - 1 {
            let tl = field[y * width + x];
            let tr = field[y * width + x + 1];
            let br = field[(y + 1) * width + x + 1];
            let bl = field[(y + 1) * width + x];

            let mut case = 0u8;
            if tl >= level {
                case |= 1;
            }
            if tr >= level {
                case |= 2;
            }
            if br >= level {
                case |= 4;
            }
            if bl >= level {
                case |= 8;
            }
            if case == 0 || case == 15 {
                continue;
            }

            let xf = x as f32;
            let yf = y as f32;
            // Edge crossing points: top, right, bottom, left.
            let top = (xf + interpolate(level, tl, tr), yf);
            let right = (xf + 1.0, yf + interpolate(level, tr, br));
            let bottom = (xf + interpolate(level, bl, br), yf + 1.0);
            let left = (xf, yf + interpolate(level, tl, bl));

            match case {
                1 | 14 => segments.push((left, top)),
                2 | 13 => segments.push((top, right)),
                3 | 12 => segments.push((left, right)),
                4 | 11 => segments.push((right, bottom)),
                6 | 9 => segments.push((top, bottom)),
                7 | 8 => segments.push((left, bottom)),
                5 => {
                    // Saddle: resolve by center average.
                    let center = (tl + tr + br + bl) * 0.25;
                    if center >= level {
                        segments.push((left, top));
                        segments.push((right, bottom));
                    } else {
                        segments.push((top, right));
                        segments.push((left, bottom));
                    }
                }
                10 => {
                    let center = (tl + tr + br + bl) * 0.25;
                    if center >= level {
                        segments.push((top, right));
                        segments.push((left, bottom));
                    } else {
                        segments.push((left, top));
                        segments.push((right, bottom));
                    }
                }
                _ => unreachable!("cases 0 and 15 are filtered above"),
            }
        }
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment_length(segment: Segment) -> f64 {
        let dx = f64::from(segment.1.0 - segment.0.0);
        let dy = f64::from(segment.1.1 - segment.0.1);
        (dx * dx + dy * dy).sqrt()
    }

    #[test]
    fn circle_contour_length_matches_circumference() {
        let size = 128usize;
        let radius = 40.0f32;
        let field: Vec<f32> = (0..size * size)
            .map(|index| {
                let x = (index % size) as f32 - size as f32 / 2.0;
                let y = (index / size) as f32 - size as f32 / 2.0;
                radius - (x * x + y * y).sqrt()
            })
            .collect();
        let segments = marching_squares(&field, size, size, 0.0);
        let total: f64 = segments.iter().map(|&segment| segment_length(segment)).sum();
        let circumference = 2.0 * std::f64::consts::PI * f64::from(radius);
        assert!(
            (total - circumference).abs() / circumference < 0.05,
            "contour length {total:.1} should approximate circumference {circumference:.1}"
        );
    }

    #[test]
    fn flat_field_yields_no_segments() {
        let field = vec![0.5f32; 64];
        assert!(marching_squares(&field, 8, 8, 0.7).is_empty());
    }
}
