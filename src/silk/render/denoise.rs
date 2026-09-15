//! Small deterministic à-trous filter guided by ray-traced surface properties.
//!
//! This is an explicit rendering treatment. It does not alter cloth geometry,
//! fill silhouettes, repair intersections, or consult any external model.
use super::super::V3;
use rayon::prelude::*;

#[derive(Clone, Copy, Default)]
pub(super) struct Guide {
    pub normal: V3,
    pub depth: f64,
    pub albedo: V3,
    pub surface: bool,
}

pub(super) fn luminance(c: V3) -> f64 {
    c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722
}

pub(super) fn filter(
    input: &[V3],
    guides: &[Guide],
    variance: &[f64],
    width: usize,
    height: usize,
    passes: u32,
) -> Vec<V3> {
    let mut source = input.to_vec();
    let mut noise = variance.to_vec();
    for pass in 0..passes {
        let stride = 1_isize << pass;
        let mut target = vec![(V3::ZERO, 0.0); input.len()];
        target.par_iter_mut().enumerate().for_each(|(pixel, out)| {
            let guide = guides[pixel];
            let center = source[pixel];
            if !guide.surface {
                *out = (center, noise[pixel]);
                return;
            }
            let x = (pixel % width) as isize;
            let y = (pixel / width) as isize;
            let mut total = V3::ZERO;
            let mut weight_sum = 0.0;
            let mut noise_sum = 0.0;
            for dy in -1_isize..=1 {
                for dx in -1_isize..=1 {
                    let nx = x + dx * stride;
                    let ny = y + dy * stride;
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    let neighbor = ny as usize * width + nx as usize;
                    let other = guides[neighbor];
                    if !other.surface {
                        continue;
                    }
                    let normal_agreement = guide.normal.dot(other.normal).clamp(0.0, 1.0);
                    // Hard rejection protects opposite folds and material boundaries.
                    if normal_agreement < 0.75
                        || (guide.albedo - other.albedo).length_squared() > 0.035
                    {
                        continue;
                    }
                    let depth_sigma =
                        (guide.depth.abs() * 0.012 * (1.0 + stride as f64 * 0.5)).max(1e-5);
                    let depth_delta = (guide.depth - other.depth) / depth_sigma;
                    if depth_delta.abs() > 3.0 {
                        continue;
                    }
                    let l0 = luminance(center).max(0.0);
                    let l1 = luminance(source[neighbor]).max(0.0);
                    let sigma = 3.5 * (noise[pixel] + noise[neighbor]).max(0.0).sqrt()
                        + 0.012 * l0.max(l1).sqrt()
                        + 0.003;
                    let color_delta =
                        (center - source[neighbor]).length_squared() / (3.0 * sigma * sigma);
                    let spatial = if dx == 0 { 2.0 } else { 1.0 } * if dy == 0 { 2.0 } else { 1.0 };
                    let weight = spatial
                        * normal_agreement.powi(64)
                        * (-0.5 * depth_delta * depth_delta - color_delta).exp();
                    total += source[neighbor] * weight;
                    weight_sum += weight;
                    noise_sum += noise[neighbor] * weight * weight;
                }
            }
            *out = if weight_sum > 1e-14 {
                (total / weight_sum, noise_sum / (weight_sum * weight_sum))
            } else {
                (center, noise[pixel])
            };
        });
        for (index, (color, var)) in target.into_iter().enumerate() {
            source[index] = color;
            noise[index] = var;
        }
    }
    source
}

#[cfg(test)]
mod tests {
    use super::*;
    fn guide() -> Guide {
        Guide {
            normal: V3::new(0.0, 0.0, 1.0),
            depth: 2.0,
            albedo: V3::new(0.5, 0.5, 0.5),
            surface: true,
        }
    }
    #[test]
    fn a_constant_surface_is_unchanged() {
        let color = V3::new(0.2, 0.4, 0.6);
        let result = filter(&vec![color; 49], &vec![guide(); 49], &[0.02; 49], 7, 7, 3);
        for pixel in result {
            assert!((pixel - color).length() < 1e-12);
        }
    }
    #[test]
    fn geometry_material_and_luminance_steps_remain_sharp() {
        for boundary in 0..4 {
            let mut colors = vec![V3::ZERO; 64];
            let mut guides = vec![guide(); 64];
            for y in 0..8 {
                for x in 4..8 {
                    let p = y * 8 + x;
                    colors[p] = V3::new(1.0, 0.5, 0.2);
                    match boundary {
                        0 => guides[p].normal = V3::new(1.0, 0.0, 0.0),
                        1 => guides[p].depth = 4.0,
                        2 => guides[p].surface = false,
                        _ => {}
                    }
                }
            }
            let result = filter(&colors, &guides, &[0.0; 64], 8, 8, 3);
            for (actual, expected) in result.iter().zip(&colors) {
                assert!((*actual - *expected).length() < 1e-10, "boundary {boundary} was blurred");
            }
        }
    }
    #[test]
    fn filtering_reduces_sampling_noise_and_is_thread_independent() {
        let colors: Vec<_> = (0..256)
            .map(|i| {
                let value = 0.4 + if (i / 16 + i % 16) % 2 == 0 { 0.08 } else { -0.08 };
                V3::new(value, value, value)
            })
            .collect();
        let guides = vec![guide(); 256];
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| filter(&colors, &guides, &[0.0064; 256], 16, 16, 2))
        };
        let a = run(1);
        let b = run(4);
        assert_eq!(a, b);
        let error = a.iter().map(|c| (c.x - 0.4).powi(2)).sum::<f64>();
        assert!(error < 0.0064 * 256.0 * 0.1, "filter should remove most spatial sampling noise");
    }
}
