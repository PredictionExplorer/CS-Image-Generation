//! Deterministic, CPU ray-traced silk with soft studio illumination.
//!
//! A fixed-topology BVH supplies two-sided visibility, interpolated normals,
//! UV-aligned anisotropic reflection, diffuse transmission, and soft self-shadow.
//! Pixel/sample streams are independent of Rayon scheduling and thread count.
mod denoise;
mod geometry;
mod sampling;
mod subdivision;

use super::{ClothBake, SilkResult, V3};
use geometry::{Geometry, Ray, Surface, basis};
use image::{ImageBuffer, Rgb, RgbImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

/// Camera, sampling and material controls independent of the cloth bake.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RenderConfig {
    /// Output image width.
    pub width: u32,
    /// Output image height.
    pub height: u32,
    /// Independent camera samples per output pixel.
    pub samples_per_pixel: u32,
    /// Samples per rectangular light at each surface encounter.
    pub light_samples: u32,
    /// Additional diffuse surface bounces after the camera intersection.
    pub max_bounces: u32,
    /// Optional geometry- and variance-guided noise filtering passes, from zero to three.
    pub denoise_passes: u32,
    /// Fixed sampling seed. Identical configurations produce identical pixels.
    pub seed: u64,
    /// Camera rotation about the vertical Y axis.
    pub azimuth_degrees: f64,
    /// Camera elevation above the XZ plane.
    pub elevation_degrees: f64,
    /// Camera roll about its viewing direction.
    pub roll_degrees: f64,
    /// Multiplier over the distance needed to fit the cloth; above one adds margin.
    pub distance_scale: f64,
    /// Vertical perspective field of view.
    pub fov_degrees: f64,
    /// Photographic exposure adjustment in stops.
    pub exposure: f64,
    /// Material family: `pearl`, `indigo`, or `rose`.
    pub palette: String,
    /// Surface roughness, before restrained UV weave variation.
    pub roughness: f64,
    /// Diffuse light transmitted through each cloth layer.
    pub transmission: f64,
    /// Grazing-angle microfiber reflection strength.
    pub sheen: f64,
    /// Subtle woven-thread surface modulation; zero disables it.
    pub weave_strength: f64,
    /// Render-only smooth surface refinement, from zero to two levels.
    pub subdivision_levels: u32,
    /// Rotate the studio rig around the camera's viewing axis.
    pub light_rotation_degrees: f64,
    /// Intensity multiplier for the long warm key light.
    pub key_light_strength: f64,
    /// Intensity multiplier for the soft violet fill light.
    pub fill_light_strength: f64,
    /// Intensity multiplier for the cool backlight.
    pub rim_light_strength: f64,
    /// Keep a consistent camera fit over every baked frame.
    pub fit_all_frames: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 1280,
            samples_per_pixel: 16,
            light_samples: 2,
            max_bounces: 1,
            denoise_passes: 0,
            seed: 0x5349_4c4b_2026,
            azimuth_degrees: -22.0,
            elevation_degrees: 24.0,
            roll_degrees: -8.0,
            distance_scale: 1.12,
            fov_degrees: 38.0,
            exposure: 0.0,
            palette: "pearl".into(),
            roughness: 0.38,
            transmission: 0.24,
            sheen: 0.34,
            weave_strength: 0.16,
            subdivision_levels: 1,
            light_rotation_degrees: 0.0,
            key_light_strength: 1.0,
            fill_light_strength: 1.0,
            rim_light_strength: 1.0,
            fit_all_frames: true,
        }
    }
}

struct Camera {
    origin: V3,
    forward: V3,
    right: V3,
    up: V3,
    tangent: f64,
    aspect: f64,
}
struct Light {
    center: V3,
    across: V3,
    vertical: V3,
    normal: V3,
    area: f64,
    radiance: V3,
}
struct Scene<'a> {
    geometry: Geometry,
    camera: Camera,
    lights: [Light; 3],
    config: &'a RenderConfig,
}

/// Render one baked frame to a display-encoded RGB image.
///
/// Image rays and all secondary samples are reproducible across thread counts.
/// The bake is immutable; changing lighting never requires re-simulation.
pub fn render_frame(
    bake: &ClothBake,
    frame_index: usize,
    config: &RenderConfig,
) -> SilkResult<RgbImage> {
    let linear = render_linear(bake, frame_index, config)?;
    let pixels: Vec<u8> = linear
        .into_iter()
        .flat_map(|color| display(color, config.exposure).map(|v| (v * 255.0 + 0.5) as u8))
        .collect();
    ImageBuffer::from_raw(config.width, config.height, pixels)
        .ok_or_else(|| "invalid RGB output size".into())
}

/// Render a 16-bit display-encoded RGB master, preserving subtle fabric gradients.
pub fn render_frame16(
    bake: &ClothBake,
    frame_index: usize,
    config: &RenderConfig,
) -> SilkResult<ImageBuffer<Rgb<u16>, Vec<u16>>> {
    let linear = render_linear(bake, frame_index, config)?;
    let pixels: Vec<u16> = linear
        .into_iter()
        .flat_map(|color| display(color, config.exposure).map(|v| (v * 65535.0 + 0.5) as u16))
        .collect();
    ImageBuffer::from_raw(config.width, config.height, pixels)
        .ok_or_else(|| "invalid RGB16 output size".into())
}

fn validate(bake: &ClothBake, frame: usize, c: &RenderConfig) -> SilkResult<()> {
    if c.width == 0 || c.height == 0 || c.width > 32768 || c.height > 32768 {
        return Err("render dimensions must be between 1 and 32768".into());
    }
    if c.samples_per_pixel == 0
        || c.samples_per_pixel > 65536
        || c.light_samples == 0
        || c.light_samples > 1024
        || c.max_bounces > 8
        || c.subdivision_levels > 2
        || c.denoise_passes > 3
    {
        return Err("invalid render sampling budget".into());
    }
    if ![
        c.azimuth_degrees,
        c.elevation_degrees,
        c.roll_degrees,
        c.distance_scale,
        c.fov_degrees,
        c.exposure,
        c.roughness,
        c.transmission,
        c.sheen,
        c.weave_strength,
        c.light_rotation_degrees,
        c.key_light_strength,
        c.fill_light_strength,
        c.rim_light_strength,
    ]
    .iter()
    .all(|v| v.is_finite())
    {
        return Err("render controls must be finite".into());
    }
    if !(5.0..120.0).contains(&c.fov_degrees)
        || c.distance_scale < 0.25
        || c.roughness <= 0.0
        || c.roughness > 1.0
        || !(0.0..=1.0).contains(&c.transmission)
        || !(0.0..=1.0).contains(&c.sheen)
        || !(0.0..=1.0).contains(&c.weave_strength)
        || !(0.0..=20.0).contains(&c.key_light_strength)
        || !(0.0..=20.0).contains(&c.fill_light_strength)
        || !(0.0..=20.0).contains(&c.rim_light_strength)
    {
        return Err("render controls outside supported ranges".into());
    }
    if !matches!(c.palette.as_str(), "pearl" | "indigo" | "rose") {
        return Err("palette must be pearl, indigo, or rose".into());
    }
    let positions = bake.frames.get(frame).ok_or("render frame does not exist")?;
    if positions.is_empty()
        || positions.len() != bake.mesh.uv.len()
        || bake.mesh.triangles.is_empty()
    {
        return Err("cloth geometry has no renderable surface or mismatched UVs".into());
    }
    if bake.mesh.triangles.iter().flatten().any(|&i| i as usize >= positions.len())
        || !positions.iter().all(|p| p.is_finite())
        || !bake.mesh.uv.iter().flatten().all(|v| v.is_finite())
    {
        return Err("invalid cloth mesh coordinates or topology".into());
    }
    if c.fit_all_frames
        && bake
            .frames
            .iter()
            .any(|f| f.len() != positions.len() || !f.iter().all(|p| p.is_finite()))
    {
        return Err("invalid geometry in camera fit frames".into());
    }
    Ok(())
}

/// Project cloth-world points through the exact camera used to render a frame.
///
/// The result uses normalized image coordinates: `(0, 0)` is the top-left
/// corner and `(1, 1)` the bottom-right. Coordinates are intentionally not
/// clamped, so callers can identify supports outside the image. Raw orbital
/// points must first receive the bake recipe's `world_center`/`world_scale`.
pub fn project_points(
    bake: &ClothBake,
    frame_index: usize,
    config: &RenderConfig,
    points: &[V3],
) -> SilkResult<Vec<[f64; 2]>> {
    validate(bake, frame_index, config)?;
    if !points.iter().all(|point| point.is_finite()) {
        return Err("projection points must be finite".into());
    }
    let (camera, center, radius) = camera_fit(bake, frame_index, config);
    points
        .iter()
        .map(|&point| {
            let relative = (point - center) / radius - camera.origin;
            let depth = relative.dot(camera.forward);
            if depth <= 1e-12 {
                return Err("projection point is behind the camera".into());
            }
            Ok([
                0.5 + 0.5 * relative.dot(camera.right) / (depth * camera.tangent * camera.aspect),
                0.5 - 0.5 * relative.dot(camera.up) / (depth * camera.tangent),
            ])
        })
        .collect()
}

fn camera_fit(bake: &ClothBake, frame_index: usize, config: &RenderConfig) -> (Camera, V3, f64) {
    let selected = &bake.frames[frame_index];
    let fitting: &[Vec<V3>] =
        if config.fit_all_frames { &bake.frames } else { std::slice::from_ref(selected) };
    let mut minimum = V3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut maximum = -minimum;
    for &p in fitting.iter().flatten() {
        minimum = minimum.min(p);
        maximum = maximum.max(p);
    }
    let center = (minimum + maximum) * 0.5;
    let radius =
        fitting.iter().flatten().map(|&p| (p - center).length()).fold(0.0, f64::max).max(1e-9);
    let azimuth = config.azimuth_degrees.to_radians();
    let elevation = config.elevation_degrees.to_radians();
    let eye =
        V3::new(azimuth.sin() * elevation.cos(), elevation.sin(), azimuth.cos() * elevation.cos());
    let right0 = V3::new(azimuth.cos(), 0.0, -azimuth.sin());
    let up0 = eye.cross(right0);
    let roll = config.roll_degrees.to_radians();
    let right = right0 * roll.cos() + up0 * roll.sin();
    let up = up0 * roll.cos() - right0 * roll.sin();
    let tangent = (config.fov_degrees.to_radians() * 0.5).tan();
    let aspect = f64::from(config.width) / f64::from(config.height);
    let distance = fitting
        .iter()
        .flatten()
        .map(|&p| {
            let p = (p - center) / radius;
            p.dot(eye) + (p.dot(right).abs() / (tangent * aspect)).max(p.dot(up).abs() / tangent)
        })
        .fold(0.5, f64::max)
        * config.distance_scale;
    let camera = Camera { origin: eye * distance, forward: -eye, right, up, tangent, aspect };
    (camera, center, radius)
}

fn render_linear(
    bake: &ClothBake,
    frame_index: usize,
    config: &RenderConfig,
) -> SilkResult<Vec<V3>> {
    validate(bake, frame_index, config)?;
    let (camera, center, radius) = camera_fit(bake, frame_index, config);
    let positions: Vec<V3> =
        bake.frames[frame_index].iter().map(|&p| (p - center) / radius).collect();
    let eye = -camera.forward;
    let right = camera.right;
    let up = camera.up;
    let light_angle = config.light_rotation_degrees.to_radians();
    let light_right = right * light_angle.cos() + up * light_angle.sin();
    let light_up = up * light_angle.cos() - right * light_angle.sin();
    let lights = [
        make_light(
            light_right * (-2.5) + light_up * 2.9 + eye * 2.6,
            1.2,
            5.0,
            V3::new(13.8, 12.7, 11.3) * config.key_light_strength,
            light_up,
        ),
        make_light(
            light_right * 3.0 + light_up * 0.8 + eye * 0.5,
            4.2,
            4.2,
            V3::new(0.92, 0.70, 1.48) * config.fill_light_strength,
            light_up,
        ),
        make_light(
            light_right * 0.8 + light_up * 2.5 - eye * 2.2,
            1.8,
            3.8,
            V3::new(6.0, 6.4, 7.5) * config.rim_light_strength,
            light_up,
        ),
    ];
    let refined = subdivision::smooth(&bake.mesh, positions, config.subdivision_levels);
    let scene = Scene {
        geometry: Geometry::new(&refined, refined.positions.clone()),
        camera,
        lights,
        config,
    };
    let count = config.width as usize * config.height as usize;
    let mut output = vec![(V3::ZERO, 0.0, denoise::Guide::default()); count];
    output.par_iter_mut().enumerate().for_each(|(pixel, slot)| {
        let x = pixel % config.width as usize;
        let y = pixel / config.width as usize;
        let mut total = V3::ZERO;
        let mut squared_luminance = 0.0;
        for sample in 0..config.samples_per_pixel {
            let mut rng = Random::new(
                config.seed ^ (pixel as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                sample,
            );
            let jitter_x = rng.next();
            let jitter_y = rng.next();
            let sx = ((x as f64 + jitter_x) / f64::from(config.width) * 2.0 - 1.0)
                * scene.camera.aspect
                * scene.camera.tangent;
            let sy = (1.0 - (y as f64 + jitter_y) / f64::from(config.height) * 2.0)
                * scene.camera.tangent;
            let direction = (scene.camera.forward + scene.camera.right * sx + scene.camera.up * sy)
                .normalized();
            let sample_color =
                radiance(&scene, Ray { origin: scene.camera.origin, direction }, &mut rng, 0);
            total += sample_color;
            if config.denoise_passes > 0 {
                squared_luminance += denoise::luminance(sample_color).powi(2);
            }
        }
        let samples = f64::from(config.samples_per_pixel);
        let mean = total / samples;
        let variance = if samples > 1.0 {
            (squared_luminance / samples - denoise::luminance(mean).powi(2)).max(0.0)
                / (samples - 1.0)
        } else {
            0.0
        };
        let mut guide = denoise::Guide::default();
        if config.denoise_passes > 0 {
            let sx = ((x as f64 + 0.5) / f64::from(config.width) * 2.0 - 1.0)
                * scene.camera.aspect
                * scene.camera.tangent;
            let sy =
                (1.0 - (y as f64 + 0.5) / f64::from(config.height) * 2.0) * scene.camera.tangent;
            let direction = (scene.camera.forward + scene.camera.right * sx + scene.camera.up * sy)
                .normalized();
            let ray = Ray { origin: scene.camera.origin, direction };
            if let Some(hit) = scene.geometry.intersect(ray, 1e-6, f64::INFINITY) {
                let surface = scene.geometry.surface(hit, ray);
                guide = denoise::Guide {
                    normal: surface.normal,
                    depth: hit.distance,
                    albedo: material(&surface, config).color,
                    surface: true,
                };
            }
        }
        *slot = (mean, variance, guide);
    });
    let colors: Vec<_> = output.iter().map(|p| p.0).collect();
    if config.denoise_passes == 0 {
        return Ok(colors);
    }
    let variance: Vec<_> = output.iter().map(|p| p.1).collect();
    let guides: Vec<_> = output.iter().map(|p| p.2).collect();
    Ok(denoise::filter(
        &colors,
        &guides,
        &variance,
        config.width as usize,
        config.height as usize,
        config.denoise_passes,
    ))
}

fn make_light(center: V3, width: f64, height: f64, radiance: V3, up: V3) -> Light {
    let normal = (-center).normalized();
    let across = up.cross(normal).normalized();
    let vertical = normal.cross(across);
    Light {
        center,
        across: across * width,
        vertical: vertical * height,
        normal,
        area: width * height,
        radiance,
    }
}

fn radiance(scene: &Scene<'_>, ray: Ray, rng: &mut Random, depth: u32) -> V3 {
    let Some(hit) = scene.geometry.intersect(ray, 1e-6, f64::INFINITY) else {
        return if depth == 0 {
            background(ray.direction, &scene.camera)
        } else {
            environment(ray.direction)
        };
    };
    let surface = scene.geometry.surface(hit, ray);
    let view = -ray.direction;
    let material = material(&surface, scene.config);
    let mut result = V3::ZERO;
    for light in &scene.lights {
        for _ in 0..scene.config.light_samples {
            let location = light.center
                + light.across * (rng.next() - 0.5)
                + light.vertical * (rng.next() - 0.5);
            let offset = location - surface.point;
            let distance = offset.length();
            let direction = offset / distance;
            let cos_light = light.normal.dot(-direction).max(0.0);
            if cos_light <= 0.0 {
                continue;
            }
            let origin = surface.point
                + surface.geometric
                    * (if direction.dot(surface.geometric) > 0.0 { 2e-5 } else { -2e-5 });
            let visibility = transmittance(
                &scene.geometry,
                origin,
                direction,
                distance - 4e-5,
                scene.config.transmission,
            );
            if visibility.x + visibility.y + visibility.z < 1e-5 {
                continue;
            }
            let light_pdf = distance * distance / (cos_light * light.area);
            let specular = sampling::evaluate(&surface, &material, view, direction);
            let response = diffuse_bsdf(&surface, &material, view, direction, scene.config)
                + specular.response * sampling::balance(light_pdf, specular.pdf);
            result += response.hadamard(light.radiance).hadamard(visibility)
                / (light_pdf * f64::from(scene.config.light_samples));
        }
    }
    // The shared GGX estimator uses the same sample count as each rectangle.
    // Therefore the N factors cancel in the paired balance weights. An invalid
    // reflected sample remains a zero outcome, never a retry or renormalization.
    for _ in 0..scene.config.light_samples {
        let Some(direction) = sampling::sample(&surface, &material, view, rng.next(), rng.next())
        else {
            continue;
        };
        let specular = sampling::evaluate(&surface, &material, view, direction);
        if specular.pdf <= 0.0 {
            continue;
        }
        let origin = surface.point
            + surface.geometric
                * (if direction.dot(surface.geometric) > 0.0 { 2e-5 } else { -2e-5 });
        // Rectangles are explicit light sources, absent from the cloth BVH.
        // Evaluate each emitter separately, matching the existing light sums.
        for light in &scene.lights {
            let Some((distance, light_pdf)) = sampling::rectangle(light, surface.point, direction)
            else {
                continue;
            };
            let visibility = transmittance(
                &scene.geometry,
                origin,
                direction,
                distance - 4e-5,
                scene.config.transmission,
            );
            let weight = sampling::balance(specular.pdf, light_pdf);
            result += specular.response.hadamard(light.radiance).hadamard(visibility)
                * (weight / (specular.pdf * f64::from(scene.config.light_samples)));
        }
    }
    if depth < scene.config.max_bounces {
        let transmit = rng.next() < scene.config.transmission;
        let hemisphere = if transmit { -surface.normal } else { surface.normal };
        let direction = cosine_direction(hemisphere, rng.next(), rng.next());
        let origin = surface.point
            + surface.geometric
                * (if direction.dot(surface.geometric) > 0.0 { 2e-5 } else { -2e-5 });
        let bounce = radiance(scene, Ray { origin, direction }, rng, depth + 1);
        result += material.color.hadamard(bounce)
            * (if transmit { 0.68 } else { 0.82 })
            * (1.0 - material.metallic);
    } else {
        // Final-bounce environment visibility keeps cavities grounded even at zero bounces.
        let direction = cosine_direction(surface.normal, rng.next(), rng.next());
        let origin = surface.point + surface.geometric * 2e-5;
        let visibility =
            transmittance(&scene.geometry, origin, direction, 4.0, scene.config.transmission);
        result += material.color.hadamard(environment(direction)).hadamard(visibility) * 0.72;
    }
    result
}

struct Material {
    color: V3,
    metallic: f64,
    roughness: f64,
}
fn material(s: &Surface, c: &RenderConfig) -> Material {
    let base = match c.palette.as_str() {
        "indigo" => V3::new(0.045, 0.075, 0.19),
        "rose" => V3::new(0.48, 0.16, 0.19),
        _ => V3::new(0.66, 0.61, 0.53),
    };
    let thread = (s.uv[0] * TAU * 180.0).sin() * (s.uv[1] * TAU * 180.0).sin();
    let weave = thread * c.weave_strength;
    let hem = (1.0 - s.hem_distance / 0.0035).clamp(0.0, 1.0);
    let copper = V3::new(0.55, 0.25, 0.09);
    Material {
        color: base.lerp(copper, hem * 0.85) * (1.0 + weave * 0.045),
        metallic: hem * 0.82,
        roughness: (c.roughness + weave * 0.025 - hem * 0.11).clamp(0.08, 0.85),
    }
}

fn diffuse_bsdf(s: &Surface, m: &Material, view: V3, light: V3, c: &RenderConfig) -> V3 {
    let nl = s.normal.dot(light);
    let nv = s.normal.dot(view).max(0.001);
    let white = V3::new(1.0, 1.0, 1.0);
    if nl < 0.0 {
        // Rough, thin-sheet diffuse transmission: overlapping layers retain volume.
        return m.color.lerp(white, 0.12) * (c.transmission * (1.0 - m.metallic) * (-nl) / PI);
    }
    let half = (view + light).normalized();
    let nh = s.normal.dot(half).max(0.0);
    let vh = view.dot(half).max(0.0);
    let fd90 = 0.5 + 2.0 * m.roughness * vh * vh;
    let burley =
        (1.0 + (fd90 - 1.0) * (1.0 - nl).powi(5)) * (1.0 + (fd90 - 1.0) * (1.0 - nv).powi(5));
    let diffuse = m.color * ((1.0 - c.transmission) * (1.0 - m.metallic) * burley * nl / PI * 0.95);
    // Broad microfiber distribution creates a soft luminous roll at grazing angles,
    // distinct from the narrow anisotropic highlights of the underlying thread.
    let inverse_roughness = 1.0 / (m.roughness * 0.7 + 0.25);
    let sin_half = (1.0 - nh * nh).max(0.0).sqrt();
    let fiber_distribution =
        (2.0 + inverse_roughness) * sin_half.powf(inverse_roughness) / (2.0 * PI);
    let fiber_visibility = 1.0 / (4.0 * (nl + nv - nl * nv).max(0.02));
    let fiber = m.color.lerp(white, 0.68)
        * (c.sheen * (1.0 - m.metallic) * fiber_distribution * fiber_visibility * nl);
    diffuse + fiber
}

fn transmittance(
    geometry: &Geometry,
    origin: V3,
    direction: V3,
    max_distance: f64,
    transmission: f64,
) -> V3 {
    let mut ray = Ray { origin, direction };
    let mut remaining = max_distance;
    let mut value = V3::new(1.0, 1.0, 1.0);
    for _ in 0..8 {
        let Some(hit) = geometry.intersect(ray, 1e-6, remaining) else {
            return value;
        };
        // Each intervening fold attenuates rather than creating an opaque cutout.
        let attenuation = (transmission * 0.55).clamp(0.0, 0.55);
        value = value.hadamard(V3::new(attenuation, attenuation * 0.94, attenuation * 0.88));
        if value.x < 0.001 {
            return V3::ZERO;
        }
        let advance = hit.distance + 3e-5;
        ray.origin += direction * advance;
        remaining -= advance;
        if remaining <= 0.0 {
            return value;
        }
    }
    V3::ZERO
}

fn cosine_direction(normal: V3, a: f64, b: f64) -> V3 {
    let tangent = basis(normal);
    let bitangent = normal.cross(tangent);
    let radius = a.sqrt();
    tangent * (radius * (TAU * b).cos())
        + bitangent * (radius * (TAU * b).sin())
        + normal * (1.0 - a).sqrt()
}

fn environment(direction: V3) -> V3 {
    let upper = (direction.y * 0.5 + 0.5).clamp(0.0, 1.0);
    V3::new(0.028, 0.033, 0.055).lerp(V3::new(0.065, 0.073, 0.11), upper)
}

fn background(direction: V3, camera: &Camera) -> V3 {
    let x = direction.dot(camera.right);
    let y = direction.dot(camera.up);
    let halo = (-((x + 0.08).powi(2) + (y - 0.06).powi(2)) * 22.0).exp();
    V3::new(0.0015, 0.0020, 0.0040) + V3::new(0.0030, 0.0040, 0.0080) * halo
}

fn display(color: V3, exposure: f64) -> [f64; 3] {
    let color = color * exposure.exp2();
    let channel = |x: f64| {
        let x = x.max(0.0);
        let mapped = ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0);
        if mapped <= 0.0031308 { mapped * 12.92 } else { 1.055 * mapped.powf(1.0 / 2.4) - 0.055 }
    };
    [channel(color.x), channel(color.y), channel(color.z)]
}

struct Random {
    seed: u64,
    sample: u32,
    dimension: usize,
}
impl Random {
    fn new(seed: u64, sample: u32) -> Self {
        Self { seed, sample, dimension: 0 }
    }
    fn next(&mut self) -> f64 {
        // Each dimension has a fixed pixel scramble, while the sample index
        // traverses a low-discrepancy sequence. No frame index enters the key.
        const PRIMES: [u32; 32] = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97, 101, 103, 107, 109, 113, 127, 131,
        ];
        let base = PRIMES[self.dimension % PRIMES.len()];
        let mut z =
            self.seed.wrapping_add((self.dimension as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        self.dimension += 1;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let shift = (z >> 11) as f64 * (1.0 / 9007199254740992.0);
        let mut index = self.sample + 1;
        let mut factor = 1.0 / f64::from(base);
        let mut value = 0.0;
        while index > 0 {
            value += f64::from(index % base) * factor;
            index /= base;
            factor /= f64::from(base);
        }
        (value + shift).fract()
    }
}

#[cfg(test)]
mod tests {
    use super::super::Mesh;
    use super::*;
    #[test]
    fn projection_uses_render_camera_axes_aspect_and_fit() {
        let positions = vec![
            V3::new(-1.0, -1.0, 0.0),
            V3::new(1.0, -1.0, 0.0),
            V3::new(1.0, 1.0, 0.0),
            V3::new(-1.0, 1.0, 0.0),
        ];
        let bake = ClothBake {
            mesh: Mesh {
                positions: positions.clone(),
                triangles: vec![[0, 1, 2], [0, 2, 3]],
                uv: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            },
            frames: vec![positions],
            fps: 24,
            stats: vec![],
            recipe: serde_json::json!({}),
        };
        let config = RenderConfig {
            width: 400,
            height: 200,
            azimuth_degrees: 0.0,
            elevation_degrees: 0.0,
            roll_degrees: 0.0,
            distance_scale: 1.2,
            ..RenderConfig::default()
        };
        let points = project_points(
            &bake,
            0,
            &config,
            &[V3::ZERO, V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0)],
        )
        .unwrap();
        assert_eq!(points[0], [0.5, 0.5]);
        assert!((points[1][0] - (0.5 + 0.25 / 1.2)).abs() < 1e-12);
        assert!((points[2][1] - (0.5 - 0.5 / 1.2)).abs() < 1e-12);
    }
    fn bake() -> ClothBake {
        let positions =
            vec![V3::new(-1.0, -0.7, 0.0), V3::new(1.0, -0.7, 0.0), V3::new(0.0, 1.0, 0.15)];
        ClothBake {
            mesh: Mesh {
                positions: positions.clone(),
                triangles: vec![[0, 1, 2]],
                uv: vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            },
            frames: vec![positions],
            fps: 24,
            stats: vec![],
            recipe: serde_json::json!({}),
        }
    }
    #[test]
    fn pixel_sampling_is_bit_identical_across_thread_counts() {
        let bake = bake();
        let config = RenderConfig {
            width: 24,
            height: 24,
            samples_per_pixel: 4,
            light_samples: 1,
            max_bounces: 1,
            ..RenderConfig::default()
        };
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| render_frame(&bake, 0, &config).unwrap());
        let many = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap()
            .install(|| render_frame(&bake, 0, &config).unwrap());
        assert_eq!(one.as_raw(), many.as_raw());
        assert!(one.pixels().any(|p| p[0] > 120), "lit fabric must appear in the image");
    }
    #[test]
    fn invalid_geometry_fails_before_bvh_construction() {
        let mut bake = bake();
        bake.mesh.triangles[0][2] = 999;
        assert!(render_frame(&bake, 0, &RenderConfig::default()).is_err());
    }
    #[test]
    fn repeated_fold_layers_attenuate_backlighting() {
        let bake = bake();
        let geom = Geometry::new(&bake.mesh, bake.mesh.positions.clone());
        let clear =
            transmittance(&geom, V3::new(3.0, 0.0, 1.0), V3::new(0.0, 0.0, -1.0), 2.0, 0.24);
        let blocked =
            transmittance(&geom, V3::new(0.0, 0.0, 1.0), V3::new(0.0, 0.0, -1.0), 2.0, 0.24);
        assert_eq!(clear, V3::new(1.0, 1.0, 1.0));
        assert!(blocked.x > 0.0 && blocked.x < clear.x * 0.2);
    }
}
