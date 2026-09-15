//! Direct specular multiple-importance sampling in solid-angle measure.
//!
//! Each rectangle has N uniform-area samples, paired with N samples of the
//! anisotropic GGX normal distribution. Balance weights sum to one for each
//! emitter. Rejected below-surface reflections retain their zero contribution;
//! the distribution is never conditioned on sample acceptance.
use super::{Light, Material, V3, geometry::Surface};
use std::f64::consts::{PI, TAU};

pub(super) struct Specular {
    /// BSDF multiplied by the incident cosine, before incoming radiance.
    pub response: V3,
    /// Reflected-direction density per steradian, including null outcomes.
    pub pdf: f64,
}

fn axes(material: &Material) -> (f64, f64) {
    let alpha = material.roughness * material.roughness;
    ((alpha * 0.48).max(0.015), (alpha / 0.48).max(0.025))
}

pub(super) fn evaluate(surface: &Surface, material: &Material, view: V3, incoming: V3) -> Specular {
    let nl = surface.normal.dot(incoming);
    let nv = surface.normal.dot(view);
    if nl <= 0.0 || nv <= 0.0 {
        return Specular { response: V3::ZERO, pdf: 0.0 };
    }
    let half = (view + incoming).normalized();
    let nh = surface.normal.dot(half);
    let vh = view.dot(half);
    if nh <= 0.0 || vh <= 0.0 {
        return Specular { response: V3::ZERO, pdf: 0.0 };
    }
    let tangent = surface.tangent;
    let bitangent = surface.normal.cross(tangent).normalized();
    let (ax, ay) = axes(material);
    let denominator =
        (half.dot(tangent) / ax).powi(2) + (half.dot(bitangent) / ay).powi(2) + nh * nh;
    let distribution = 1.0 / (PI * ax * ay * denominator.powi(2));
    let lambda = |direction: V3| {
        let z = surface.normal.dot(direction).abs().max(0.001);
        0.5 * ((1.0
            + ((direction.dot(tangent) * ax).powi(2) + (direction.dot(bitangent) * ay).powi(2))
                / (z * z))
            .sqrt()
            - 1.0)
    };
    let geometry = 1.0 / (1.0 + lambda(view) + lambda(incoming));
    let white = V3::new(1.0, 1.0, 1.0);
    let f0 = V3::new(0.045, 0.045, 0.045).lerp(material.color, material.metallic);
    let fresnel = f0 + (white - f0) * (1.0 - vh).powi(5);
    Specular {
        response: fresnel * (distribution * geometry / (4.0 * nv.max(0.01))),
        pdf: distribution * nh / (4.0 * vh),
    }
}

pub(super) fn sample(
    surface: &Surface,
    material: &Material,
    view: V3,
    u: f64,
    v: f64,
) -> Option<V3> {
    let (ax, ay) = axes(material);
    let radius = (u / (1.0 - u)).sqrt();
    let bitangent = surface.normal.cross(surface.tangent).normalized();
    let half = (surface.tangent * (ax * radius * (TAU * v).cos())
        + bitangent * (ay * radius * (TAU * v).sin())
        + surface.normal)
        .normalized();
    let vh = view.dot(half);
    if vh <= 0.0 {
        return None;
    }
    let direction = half * (2.0 * vh) - view;
    (direction.dot(surface.normal) > 0.0).then_some(direction)
}

/// Distance to an emitting rectangle and its matching directional PDF.
pub(super) fn rectangle(light: &Light, origin: V3, direction: V3) -> Option<(f64, f64)> {
    let cosine = light.normal.dot(-direction);
    if cosine <= 1e-12 {
        return None;
    }
    let distance = light.normal.dot(origin - light.center) / cosine;
    if distance <= 1e-6 {
        return None;
    }
    let point = origin + direction * distance - light.center;
    let across = point.dot(light.across) / light.across.length_squared();
    let vertical = point.dot(light.vertical) / light.vertical.length_squared();
    if across.abs() > 0.5 || vertical.abs() > 0.5 {
        return None;
    }
    Some((distance, distance * distance / (light.area * cosine)))
}

pub(super) fn balance(selected: f64, other: f64) -> f64 {
    if selected <= 0.0 { 0.0 } else { selected / (selected + other) }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn surface() -> Surface {
        Surface {
            point: V3::ZERO,
            normal: V3::new(0.0, 0.0, 1.0),
            geometric: V3::new(0.0, 0.0, 1.0),
            tangent: V3::new(1.0, 0.0, 0.0),
            uv: [0.0, 0.0],
            hem_distance: 1.0,
        }
    }
    fn material(roughness: f64) -> Material {
        Material { color: V3::new(0.4, 0.4, 0.4), metallic: 0.0, roughness }
    }
    fn light() -> Light {
        Light {
            center: V3::new(0.0, 0.0, 2.0),
            across: V3::new(3.0, 0.0, 0.0),
            vertical: V3::new(0.0, 3.0, 0.0),
            normal: V3::new(0.0, 0.0, -1.0),
            area: 9.0,
            radiance: V3::new(1.0, 1.0, 1.0),
        }
    }
    fn radical(mut index: u32, base: u32) -> f64 {
        let mut x = 0.0;
        let mut f = 1.0 / f64::from(base);
        while index > 0 {
            x += f64::from(index % base) * f;
            index /= base;
            f /= f64::from(base);
        }
        x
    }
    #[test]
    fn reflected_samples_have_finite_matching_pdfs() {
        let s = surface();
        for roughness in [0.08, 0.3, 0.8] {
            let m = material(roughness);
            for view in [V3::new(0.0, 0.0, 1.0), V3::new(0.98, 0.0, 0.2).normalized()] {
                for index in 1..4096 {
                    if let Some(direction) =
                        sample(&s, &m, view, radical(index, 2), radical(index, 3))
                    {
                        assert!((direction.length() - 1.0).abs() < 1e-12);
                        let evaluated = evaluate(&s, &m, view, direction);
                        assert!(evaluated.pdf.is_finite() && evaluated.pdf > 0.0);
                        assert!(evaluated.response.is_finite() && evaluated.response.x >= 0.0);
                    }
                }
            }
        }
        assert_eq!(
            evaluate(&s, &material(0.3), V3::new(0.0, 0.0, 1.0), V3::new(0.0, 0.0, -1.0)).pdf,
            0.0
        );
    }
    #[test]
    fn analytic_rectangle_requires_front_face_and_valid_bounds() {
        let light = light();
        let hit = rectangle(&light, V3::ZERO, V3::new(0.0, 0.0, 1.0)).unwrap();
        assert_eq!(hit, (2.0, 4.0 / 9.0));
        assert!(rectangle(&light, V3::new(2.0, 0.0, 0.0), V3::new(0.0, 0.0, 1.0)).is_none());
        assert!(rectangle(&light, V3::new(0.0, 0.0, 3.0), V3::new(0.0, 0.0, -1.0)).is_none());
    }
    #[test]
    fn balanced_estimators_agree_with_area_integral_without_acceptance_renormalization() {
        let s = surface();
        let m = material(0.3);
        let light = light();
        let view = V3::new(0.35, 0.1, 1.0).normalized();
        let mut reference = 0.0;
        const EDGE: u32 = 320;
        for y in 0..EDGE {
            for x in 0..EDGE {
                let p = light.center
                    + light.across * ((f64::from(x) + 0.5) / f64::from(EDGE) - 0.5)
                    + light.vertical * ((f64::from(y) + 0.5) / f64::from(EDGE) - 0.5);
                let direction = p.normalized();
                let (_, pdf) = rectangle(&light, V3::ZERO, direction).unwrap();
                reference += evaluate(&s, &m, view, direction).response.x / pdf;
            }
        }
        reference /= f64::from(EDGE * EDGE);
        let mut combined = 0.0;
        const SAMPLES: u32 = 65_536;
        for index in 1..=SAMPLES {
            let (u, v) = (radical(index, 2), radical(index, 3));
            let p = light.center + light.across * (u - 0.5) + light.vertical * (v - 0.5);
            let direction = p.normalized();
            let (_, p_light) = rectangle(&light, V3::ZERO, direction).unwrap();
            let spec = evaluate(&s, &m, view, direction);
            combined += spec.response.x * balance(p_light, spec.pdf) / p_light;
            if let Some(direction) = sample(&s, &m, view, radical(index, 5), radical(index, 7))
                && let Some((_, p_light)) = rectangle(&light, V3::ZERO, direction)
            {
                let spec = evaluate(&s, &m, view, direction);
                combined += spec.response.x * balance(spec.pdf, p_light) / spec.pdf;
            }
        }
        combined /= f64::from(SAMPLES);
        assert!(
            (combined - reference).abs() / reference < 0.015,
            "MIS {combined} versus reference {reference}"
        );
    }
}
