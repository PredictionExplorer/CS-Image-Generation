//! Quiet analytic studio lighting and thin colored optical layers.
use super::{CameraFrame, Material, RenderConfig, V3};
use std::f64::consts::{PI, TAU};

#[derive(Clone, Copy)]
struct Strip {
    center: V3,
    normal: V3,
    long_axis: V3,
    short_axis: V3,
    color: V3,
    half_length: f64,
    half_width: f64,
    reference_squared: f64,
}

pub(super) struct Studio {
    view: V3,
    strips: [Strip; 2],
    fill: V3,
}

#[derive(Clone, Copy)]
pub(super) struct Optical {
    pub light: V3,
    pub transmission: V3,
}

impl Studio {
    pub fn new(camera: &CameraFrame, config: &RenderConfig) -> Self {
        let angle = config.light_rotation_degrees.to_radians();
        let horizontal = camera.right * angle.cos() + camera.up * angle.sin();
        let vertical = camera.up * angle.cos() - camera.right * angle.sin();
        let view = -camera.forward;
        let make = |offset: V3, color: V3, half_length: f64| {
            let normal = (-offset).normalized();
            let long_axis = (vertical - normal * vertical.dot(normal)).normalized();
            Strip {
                center: camera.target + offset,
                normal,
                long_axis,
                short_axis: normal.cross(long_axis).normalized(),
                color,
                half_length,
                half_width: 0.085,
                reference_squared: offset.length_squared(),
            }
        };
        Self {
            view,
            strips: [
                make(
                    view * 3.5 - horizontal * 1.8 + vertical * 2.8,
                    V3::new(1.90, 1.78, 1.59) * config.key_strength,
                    1.30,
                ),
                make(
                    -view * 3.3 + horizontal * 2.0 + vertical * 2.8,
                    V3::new(1.15, 1.42, 1.86) * config.rim_strength,
                    1.55,
                ),
            ],
            fill: V3::new(0.50, 0.56, 0.68) * config.fill_strength,
        }
    }

    /// Finite-position strips provide deterministic direction and distance falloff.
    /// Their angular covariance broadens the highlight analytically; this is a
    /// smooth emitter approximation, not Monte Carlo transport or a cast shadow.
    pub fn shade(
        &self,
        material: &Material,
        point: V3,
        normal: V3,
        tangent: V3,
        uv: [f64; 2],
        transverse_variance: f64,
    ) -> Optical {
        let front = normal.dot(self.view) >= 0.0;
        let base_normal = if front { normal } else { -normal };
        let optical_cosine = base_normal.dot(self.view).max(0.0);
        let normal = base_normal;
        let tangent = (tangent - normal * tangent.dot(normal)).normalized();
        let tangent = if tangent.length_squared() > 0.5 { tangent } else { perpendicular(normal) };
        let bitangent = normal.cross(tangent).normalized();
        let base = if front { material.front_color } else { material.back_color };
        // Integrate the high-frequency transverse weave over the AA footprint.
        // Unresolvable fibers fade instead of making moire or temporal glitter.
        let frequency = material.fiber_frequency;
        let filter = (-2.0 * PI * PI * frequency * frequency * transverse_variance).exp();
        let fiber = (uv[1] * TAU * frequency).sin() * material.fiber_strength * filter;
        // Resolved fibers bend the reflection normal across the weave. Unresolved
        // fiber slopes broaden the lobe instead of aliasing or vanishing entirely.
        let normal = (base_normal + bitangent * (fiber * 0.24)).normalized();
        let tangent = (tangent - normal * tangent.dot(normal)).normalized();
        let bitangent = normal.cross(tangent).normalized();
        let nv = normal.dot(self.view).max(0.0);
        let unresolved_slope_variance =
            (material.fiber_strength * 0.24).powi(2) * 0.5 * (1.0 - filter * filter);
        let base = base * (1.0 + fiber * 0.018);
        let roughness = (material.roughness * (1.0 + fiber * 0.08)).clamp(0.025, 1.0);
        let slant = 1.0 / optical_cosine.max(0.055);
        let transmission = V3::new(
            (-material.optical_depth.x * slant).exp(),
            (-material.optical_depth.y * slant).exp(),
            (-material.optical_depth.z * slant).exp(),
        );
        let intercepted = V3::new(1.0, 1.0, 1.0) - transmission;
        // Absorbed red/green/blue energy is not the reflected dye. Multiplying
        // it directly by front_color can neutralize the intended hue. Allocate
        // the common extinction share to bounded scattering; residual channel
        // differences remain colored absorption in the camera transmission.
        let common =
            material.optical_depth.x.min(material.optical_depth.y).min(material.optical_depth.z);
        let scattered_channel = |depth: f64, removed: f64| {
            if depth > 0.0 { (common / depth) * removed } else { 0.0 }
        };
        let scattered = V3::new(
            scattered_channel(material.optical_depth.x, intercepted.x),
            scattered_channel(material.optical_depth.y, intercepted.y),
            scattered_channel(material.optical_depth.z, intercepted.z),
        );
        let opacity = (intercepted.x + intercepted.y + intercepted.z) / 3.0;
        let mut light = base.hadamard(self.fill).hadamard(scattered);
        for strip in &self.strips {
            let offset = strip.center - point;
            let distance_squared = offset.length_squared();
            if distance_squared <= 1e-12 {
                continue;
            }
            let distance = distance_squared.sqrt();
            let direction = offset / distance;
            let light_cosine = strip.normal.dot(-direction).max(0.0);
            if light_cosine <= 0.0 {
                continue;
            }
            let area = 4.0 * strip.half_length * strip.half_width;
            // The area term regularizes the near field of a finite emitter.
            // Normalization keeps key/rim strength intuitive at camera.target.
            let falloff = (strip.reference_squared + area * 0.25)
                / (distance_squared + area * 0.25)
                * light_cosine;
            let incoming = strip.color * falloff;
            let signed_nl = normal.dot(direction);
            let nl = signed_nl.max(0.0);
            let backlight = (-signed_nl).max(0.0);
            // A thin textile scatters some back illumination while also allowing
            // the camera to see farther surfaces through its colored transmission.
            light += base.hadamard(incoming).hadamard(scattered) * (0.70 * nl + 0.42 * backlight);
            if nl > 0.0 && nv > 1e-6 {
                let half_length = (self.view + direction).length().max(1e-6);
                let half = (self.view + direction) / half_length;
                let nh = normal.dot(half).max(0.0);
                let alpha = roughness * roughness;
                let aspect = (1.0 - 0.88 * material.anisotropy).sqrt();
                let half_delta = |axis: V3, extent: f64| {
                    let incident_delta =
                        (axis - direction * axis.dot(direction)) * (extent / distance);
                    (incident_delta - half * incident_delta.dot(half)) / half_length
                };
                let long_delta = half_delta(strip.long_axis, strip.half_length);
                let short_delta = half_delta(strip.short_axis, strip.half_width);
                let spread =
                    |axis: V3| (axis.dot(long_delta).powi(2) + axis.dot(short_delta).powi(2)) / 3.0;
                let ax = ((alpha * aspect).powi(2) + spread(tangent)).sqrt().max(0.006);
                let ay = ((alpha / aspect).powi(2) + spread(bitangent) + unresolved_slope_variance)
                    .sqrt()
                    .max(0.006);
                let denominator =
                    (half.dot(tangent) / ax).powi(2) + (half.dot(bitangent) / ay).powi(2) + nh * nh;
                let distribution = 1.0 / (PI * ax * ay * denominator.max(1e-8).powi(2));
                let lambda = |direction: V3, cosine: f64| {
                    0.5 * ((1.0
                        + ((direction.dot(tangent) * ax).powi(2)
                            + (direction.dot(bitangent) * ay).powi(2))
                            / (cosine * cosine).max(1e-6))
                    .sqrt()
                        - 1.0)
                };
                let masking = 1.0 / (1.0 + lambda(self.view, nv) + lambda(direction, nl));
                let fresnel = 0.045 + 0.955 * (1.0 - self.view.dot(half).max(0.0)).powi(5);
                let specular = distribution * masking * fresnel / (4.0 * nv.max(0.035));
                light += incoming * (specular * opacity * 0.78);
            }
            let edge = (1.0 - nv).powi(2);
            let velvet = material.sheen * edge * (nl * 0.45 + backlight * 0.65);
            light +=
                base.lerp(V3::new(1.0, 1.0, 1.0), 0.70).hadamard(incoming) * (velvet * opacity);
        }
        Optical { light: light + material.emission, transmission }
    }
}

pub(super) fn perpendicular(normal: V3) -> V3 {
    let axis = if normal.z.abs() < 0.8 { V3::new(0.0, 0.0, 1.0) } else { V3::new(0.0, 1.0, 0.0) };
    normal.cross(axis).normalized()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn studio() -> Studio {
        Studio::new(
            &CameraFrame {
                forward: V3::new(0.0, 0.0, -1.0),
                right: V3::new(1.0, 0.0, 0.0),
                up: V3::new(0.0, 1.0, 0.0),
                position: V3::new(0.0, 0.0, 5.0),
                target: V3::ZERO,
                scale: 1.0,
                width: 2.0,
                height: 2.0,
            },
            &RenderConfig {
                key_strength: 1.0,
                rim_strength: 0.0,
                fill_strength: 0.0,
                light_rotation_degrees: 0.0,
                ..RenderConfig::default()
            },
        )
    }

    #[test]
    fn finite_strip_changes_illumination_along_equal_normal_surface() {
        let studio = studio();
        let material = Material { fiber_strength: 0.0, ..Material::default() };
        let shade = |point| {
            studio.shade(
                &material,
                point,
                V3::new(0.0, 0.0, 1.0),
                V3::new(1.0, 0.0, 0.0),
                [0.0, 0.0],
                0.0,
            )
        };
        let near = shade(V3::new(-1.5, 1.0, 0.0));
        let far = shade(V3::new(1.5, -1.0, 0.0));
        assert!(near.light.length() > far.light.length() * 1.2);
        assert_eq!(near.transmission, far.transmission);
    }

    #[test]
    fn unresolved_microfibers_filter_without_changing_bulk_transmission() {
        let studio = studio();
        let material =
            Material { fiber_strength: 0.4, fiber_frequency: 24.0, ..Material::default() };
        let shade = |phase, variance| {
            studio.shade(
                &material,
                V3::ZERO,
                V3::new(0.0, 0.0, 1.0),
                V3::new(1.0, 0.0, 0.0),
                [0.0, phase / material.fiber_frequency],
                variance,
            )
        };
        let resolved_a = shade(0.25, 0.0);
        let resolved_b = shade(0.75, 0.0);
        assert!((resolved_a.light - resolved_b.light).length() > 1e-5);
        assert_eq!(resolved_a.transmission, resolved_b.transmission);
        let variance = 4.0 / material.fiber_frequency.powi(2);
        let filtered_a = shade(0.25, variance);
        let filtered_b = shade(0.75, variance);
        assert!((filtered_a.light - filtered_b.light).length() < 1e-12);
    }

    #[test]
    fn reflected_gold_dye_is_not_neutralized_by_chromatic_absorption() {
        let material = Material {
            front_color: V3::new(0.78, 0.73, 0.62),
            optical_depth: V3::new(0.40, 0.45, 0.54),
            roughness: 0.29,
            anisotropy: 0.82,
            sheen: 0.0,
            fiber_strength: 0.0,
            ..Material::default()
        };
        let result = studio().shade(
            &material,
            V3::ZERO,
            V3::new(0.0, 0.0, 1.0),
            V3::new(1.0, 0.0, 0.0),
            [0.0, 0.0],
            0.0,
        );
        assert!(
            result.light.x > result.light.z * 1.45,
            "gold was washed toward gray: {:?}",
            result.light
        );
        assert!((result.transmission.x - (-0.40_f64).exp()).abs() < 1e-12);
        assert!((result.transmission.z - (-0.54_f64).exp()).abs() < 1e-12);
    }
}
