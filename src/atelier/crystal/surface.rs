//! A shallow polished dome under a fixed, dark photographic studio environment.
//!
//! The existing polar outline is retained: in rotated ellipse coordinates,
//! `f(phi) = 1 + a * (cos(3 phi) + 0.5 sin(2 phi))`, `q = r² / f²` and
//! `z = depth * sqrt(1-q)`. The exact first derivatives of q determine the
//! surface normal. Multiplying by the dome height before normalization avoids
//! the infinite height derivative at the silhouette.
//!
//! Reflection samples three distant fourth-order super-Gaussian softboxes along
//! the ideal mirror direction, weighted by exact unpolarized dielectric Fresnel
//! reflectance. This is an analytic studio approximation with no cast shadows
//! or multiple scattering. A frontal camera ray refracts once through the dome
//! using geometric Snell transmission and intersects the internal stress plane
//! at z=0. This models a clear viewing dome above that plane; it does not trace
//! a second interface or transport polarization through the curved surface.
//! The separate optional optical path multiplier still describes a locally
//! parallel slab, not the complete multi-interface optical system.

use crate::atelier::{SilkResult, V3};
use serde::{Deserialize, Serialize};

/// Fixed specimen geometry and neutral studio reflection controls.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SurfaceConfig {
    /// Horizontal and vertical semi-axes before the in-plane rotation.
    pub semi_axes: [f64; 2],
    /// Counterclockwise rotation of the specimen in degrees.
    pub rotation_degrees: f64,
    /// Smooth angular outline perturbation, from zero to 0.15.
    pub outline_asymmetry: f64,
    /// Height of the front dome at its center, in the same units as the axes.
    pub depth: f64,
    /// Glass-to-air refractive-index ratio, strictly greater than one.
    pub refractive_index: f64,
    /// Fixed studio illumination multiplier; one preserves the supplied studio.
    pub reflection_strength: f64,
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        Self {
            semi_axes: [3.55, 2.15],
            rotation_degrees: -12.0,
            outline_asymmetry: 0.055,
            depth: 0.65,
            refractive_index: 1.5,
            reflection_strength: 1.0,
        }
    }
}

impl SurfaceConfig {
    /// Validate material controls before sampling or allocating an image.
    pub fn validate(&self) -> SilkResult<()> {
        if self.semi_axes.iter().any(|v| !v.is_finite() || !(1e-6..=1e6).contains(v))
            || !self.rotation_degrees.is_finite()
            || !self.outline_asymmetry.is_finite()
            || !(0.0..=0.15).contains(&self.outline_asymmetry)
            || !self.depth.is_finite()
            || self.depth < 0.0
            || self.depth > self.semi_axes[0].min(self.semi_axes[1])
            || !self.refractive_index.is_finite()
            || !(1.01..=3.0).contains(&self.refractive_index)
            || !self.reflection_strength.is_finite()
            || !(0.0..=8.0).contains(&self.reflection_strength)
        {
            return Err("invalid polished crystal surface configuration".into());
        }
        Ok(())
    }
}

/// Geometry and reflected radiance at one point inside the visible specimen.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceSample {
    /// Outline-relative radius, from zero at the center to one at the rim.
    pub rho: f64,
    /// Unit dome profile, `sqrt(1-rho²)`.
    pub dome: f64,
    /// Front surface elevation above the specimen's base plane.
    pub height: f64,
    /// Unit outward surface normal in the original world coordinates.
    pub normal: V3,
    /// World XY where the once-refracted camera ray reaches the stress plane z=0.
    pub refracted_point: [f64; 2],
    /// Scene-linear RGB from the fixed studio, including Fresnel weighting.
    pub reflection: V3,
    /// Unpolarized dielectric Fresnel reflection fraction before artistic scaling.
    pub fresnel: f64,
    /// Local slab path length relative to its normal thickness, `1/cos(theta_t)`.
    pub optical_path_scale: f64,
}

#[derive(Clone, Copy, Debug)]
struct Softbox {
    center: V3,
    horizontal: V3,
    vertical: V3,
    widths: [f64; 2],
    radiance: V3,
}

impl Softbox {
    fn new(center: V3, widths: [f64; 2], radiance: V3) -> Self {
        let center = center.normalized();
        let horizontal = V3::new(center.z, 0.0, -center.x).normalized();
        let vertical = center.cross(horizontal).normalized();
        Self { center, horizontal, vertical, widths, radiance }
    }

    fn sample(self, direction: V3) -> V3 {
        let facing = direction.dot(self.center).max(0.0);
        let u = direction.dot(self.horizontal) / self.widths[0];
        let v = direction.dot(self.vertical) / self.widths[1];
        self.radiance * ((-0.5 * (u.powi(4) + v.powi(4))).exp() * facing.powi(8))
    }
}

/// Prepared analytic shape and fixed studio; independent of source time or stress.
#[derive(Clone, Debug)]
pub struct Surface {
    config: SurfaceConfig,
    sine: f64,
    cosine: f64,
    lights: [Softbox; 3],
}

impl Surface {
    /// Prepare a shallow dome and a restrained upper-left studio catch.
    pub fn new(config: &SurfaceConfig) -> SilkResult<Self> {
        config.validate()?;
        let (sine, cosine) = config.rotation_degrees.to_radians().sin_cos();
        Ok(Self {
            config: config.clone(),
            sine,
            cosine,
            lights: [
                // A long neutral strip reflected near the upper-left shoulder.
                Softbox::new(V3::new(-0.48, 0.64, 0.60), [0.48, 0.075], V3::new(2.30, 2.35, 2.45)),
                // A very faint broad fill models a dark studio wall, not a glaze.
                Softbox::new(V3::new(0.52, 0.25, 0.82), [0.52, 0.22], V3::new(0.11, 0.13, 0.16)),
                // The rear strip is visible only in a localized grazing reflection.
                Softbox::new(V3::new(0.60, -0.25, -0.76), [0.30, 0.085], V3::new(0.48, 0.39, 0.28)),
            ],
        })
    }

    /// Sample world XY; return None outside the outline or for nonfinite inputs.
    pub fn sample(&self, x: f64, y: f64) -> Option<SurfaceSample> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let config = &self.config;
        let u = (x * self.cosine + y * self.sine) / config.semi_axes[0];
        let v = (-x * self.sine + y * self.cosine) / config.semi_axes[1];
        let radius = u.hypot(v);
        if radius >= 1.0 + 1.5 * config.outline_asymmetry {
            return None;
        }
        let phi = v.atan2(u);
        let outline =
            1.0 + config.outline_asymmetry * ((3.0 * phi).cos() + 0.5 * (2.0 * phi).sin());
        let outline_phi = config.outline_asymmetry * (-3.0 * (3.0 * phi).sin() + (2.0 * phi).cos());
        let rho = radius / outline;
        if rho >= 1.0 {
            return None;
        }
        let dome = (1.0 - rho * rho).max(0.0).sqrt();
        // These derivatives extend continuously to zero at the origin. They
        // avoid atan2's 1/r² derivative by cancelling it symbolically first.
        let inverse_squared = 1.0 / (outline * outline);
        let angular = outline_phi / outline;
        let q_u = 2.0 * inverse_squared * (u + v * angular);
        let q_v = 2.0 * inverse_squared * (v - u * angular);
        let q_x = q_u * self.cosine / config.semi_axes[0] - q_v * self.sine / config.semi_axes[1];
        let q_y = q_u * self.sine / config.semi_axes[0] + q_v * self.cosine / config.semi_axes[1];
        let normal = V3::new(config.depth * q_x, config.depth * q_y, 2.0 * dome).normalized();
        let cosine_i = normal.z.clamp(0.0, 1.0);
        let cosine_t =
            (1.0 - (1.0 - cosine_i * cosine_i) / config.refractive_index.powi(2)).max(0.0).sqrt();
        let fresnel = dielectric_fresnel(cosine_i, cosine_t, config.refractive_index);
        let height = config.depth * dome;
        let eta = 1.0 / config.refractive_index;
        let transmitted = V3::new(0.0, 0.0, -eta) + normal * (eta * cosine_i - cosine_t);
        // Tz remains negative for air-to-glass transmission: both terms in
        // -Tz = eta*(1-cos_i²) + cos_t*cos_i are nonnegative. No grazing clamp
        // or artificial displacement limit is needed for the permitted IOR.
        let travel = height / -transmitted.z;
        let refracted_point = [x + travel * transmitted.x, y + travel * transmitted.y];
        let reflected = normal * (2.0 * cosine_i) - V3::new(0.0, 0.0, 1.0);
        let environment =
            self.lights.iter().fold(V3::ZERO, |sum, light| sum + light.sample(reflected));
        Some(SurfaceSample {
            rho,
            dome,
            height,
            normal,
            refracted_point,
            reflection: environment * (fresnel * config.reflection_strength),
            fresnel,
            optical_path_scale: 1.0 / cosine_t,
        })
    }
}

fn dielectric_fresnel(cosine_i: f64, cosine_t: f64, index: f64) -> f64 {
    let rs = (cosine_i - index * cosine_t) / (cosine_i + index * cosine_t);
    let rp = (index * cosine_i - cosine_t) / (index * cosine_i + cosine_t);
    0.5 * (rs * rs + rp * rp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn analytic_normal_matches_differentiated_rotated_asymmetric_height() {
        let surface = Surface::new(&SurfaceConfig::default()).unwrap();
        for (x, y) in [(0.4, 0.3), (-1.4, 0.7), (1.8, -0.6), (-0.7, -1.1)] {
            let h = 1e-5;
            let sample = surface.sample(x, y).unwrap();
            let dx = (surface.sample(x + h, y).unwrap().height
                - surface.sample(x - h, y).unwrap().height)
                / (2.0 * h);
            let dy = (surface.sample(x, y + h).unwrap().height
                - surface.sample(x, y - h).unwrap().height)
                / (2.0 * h);
            assert!((dx + sample.normal.x / sample.normal.z).abs() < 1e-8);
            assert!((dy + sample.normal.y / sample.normal.z).abs() < 1e-8);
        }
    }

    #[test]
    fn center_and_grazing_samples_are_finite_and_obey_dielectric_bounds() {
        let config = SurfaceConfig::default();
        let surface = Surface::new(&config).unwrap();
        let center = surface.sample(0.0, 0.0).unwrap();
        assert_eq!(center.normal, V3::new(0.0, 0.0, 1.0));
        assert!((center.fresnel - 0.04).abs() < 1e-14);
        assert_eq!(center.optical_path_scale, 1.0);
        for step in 0..128 {
            let phi = TAU * f64::from(step) / 128.0;
            let radius = (1.0
                + config.outline_asymmetry * ((3.0 * phi).cos() + 0.5 * (2.0 * phi).sin()))
                * (1.0 - 1e-10);
            let x = config.semi_axes[0] * radius * phi.cos();
            let y = config.semi_axes[1] * radius * phi.sin();
            let sample = surface
                .sample(
                    x * surface.cosine - y * surface.sine,
                    x * surface.sine + y * surface.cosine,
                )
                .unwrap();
            assert!(sample.normal.is_finite() && sample.reflection.is_finite());
            assert!((sample.normal.length_squared() - 1.0).abs() < 1e-12);
            assert!((0.04..=1.0).contains(&sample.fresnel));
            assert!((1.0..=1.35).contains(&sample.optical_path_scale));
        }
        assert!(surface.sample(100.0, 0.0).is_none());
        assert!(surface.sample(f64::NAN, 0.0).is_none());
    }

    #[test]
    fn studio_catches_are_directional_and_zero_strength_only_removes_reflection() {
        let config = SurfaceConfig::default();
        let surface = Surface::new(&config).unwrap();
        let dark = Surface::new(&SurfaceConfig { reflection_strength: 0.0, ..config }).unwrap();
        let key = surface.lights[0];
        assert!(key.sample(key.center).x > 100.0 * key.sample(V3::new(0.0, 0.0, 1.0)).x);
        for (x, y) in [(-1.3, 1.1), (0.0, 0.0), (2.0, -0.4)] {
            let a = surface.sample(x, y).unwrap();
            let b = dark.sample(x, y).unwrap();
            assert_eq!(a.normal, b.normal);
            assert_eq!(a.refracted_point, b.refracted_point);
            assert_eq!(a.optical_path_scale, b.optical_path_scale);
            assert_eq!(b.reflection, V3::ZERO);
        }
    }

    #[test]
    fn refraction_preserves_center_and_a_zero_depth_surface() {
        let config = SurfaceConfig::default();
        let surface = Surface::new(&config).unwrap();
        assert_eq!(surface.sample(0.0, 0.0).unwrap().refracted_point, [0.0, 0.0]);
        let flat = Surface::new(&SurfaceConfig { depth: 0.0, ..config }).unwrap();
        for (x, y) in [(-1.3, 1.1), (0.0, 0.0), (2.0, -0.4)] {
            assert_eq!(flat.sample(x, y).unwrap().refracted_point, [x, y]);
        }
    }

    #[test]
    fn symmetric_dome_refracts_symmetrically_toward_its_center() {
        let surface = Surface::new(&SurfaceConfig {
            outline_asymmetry: 0.0,
            rotation_degrees: 0.0,
            ..SurfaceConfig::default()
        })
        .unwrap();
        for (x, y) in [(0.4, 0.3), (1.4, 0.7), (1.8, 0.6), (0.7, 1.1)] {
            let point = surface.sample(x, y).unwrap().refracted_point;
            let mirror_x = surface.sample(-x, y).unwrap().refracted_point;
            let mirror_y = surface.sample(x, -y).unwrap().refracted_point;
            assert!((point[0] + mirror_x[0]).abs() < 1e-12);
            assert!((point[1] - mirror_x[1]).abs() < 1e-12);
            assert!((point[0] - mirror_y[0]).abs() < 1e-12);
            assert!((point[1] + mirror_y[1]).abs() < 1e-12);
            assert!(point[0] > 0.0 && point[0] < x);
            assert!(point[1] > 0.0 && point[1] < y);
        }
    }

    #[test]
    fn refracted_stress_points_remain_finite_inside_the_specimen() {
        let config = SurfaceConfig::default();
        let surface = Surface::new(&config).unwrap();
        for step in 0..128 {
            let phi = TAU * f64::from(step) / 128.0;
            let outline =
                1.0 + config.outline_asymmetry * ((3.0 * phi).cos() + 0.5 * (2.0 * phi).sin());
            for radius in [0.05, 0.3, 0.6, 0.9, 0.99, 1.0 - 1e-10] {
                let local_x = config.semi_axes[0] * outline * radius * phi.cos();
                let local_y = config.semi_axes[1] * outline * radius * phi.sin();
                let x = local_x * surface.cosine - local_y * surface.sine;
                let y = local_x * surface.sine + local_y * surface.cosine;
                let sample = surface.sample(x, y).unwrap();
                let [px, py] = sample.refracted_point;
                assert!(px.is_finite() && py.is_finite());
                assert!(surface.sample(px, py).is_some());
                assert!(px.hypot(py) <= x.hypot(y) + 1e-12);
                assert!((px - x) * x + (py - y) * y <= 1e-12);
            }
        }
    }
}
