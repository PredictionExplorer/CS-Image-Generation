//! V62 `terra` -- Terra Trium Corporum.
//!
//! The shape sphere as a planet: a lattice of shape-space configurations is
//! probed with short capped simulations (basin-map on the shape sphere), the
//! fates paint landmasses and oceans, the seed's actual route is inked as an
//! expedition track, collision points become dragon sigils and the Lagrange
//! poles star capitals. Rendered as (a) a Winkel-tripel atlas plate on
//! archival cream and (b) a rotating globe with soft terminator lighting.
//! Cartouche and label typography are deferred until `common/text.rs`.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::render::context::PixelBuffer;
use crate::sim::Body;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::display::{auto_levels, encode_linear_rec2020_png16};
use crate::viz::common::raster::{Rgb64, draw_line};
use crate::viz::common::resim::{CellOutcome, DEFAULT_ESCAPE_THRESHOLD, GridParams, capped_fate};
use crate::viz::common::tube_render::{
    RenderParams, Scene, TexturedSphere, Vec3, fit_distance, orbit_camera, render,
};
use crate::viz::context::VizContext;
use crate::viz::modes::shape_sphere::shape_point;
use crate::viz::modes::worldtube::fill_rgba;
use crate::viz::sink::ArtifactSink;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use tracing::{info, warn};

/// Fate lattice dimensions (equirectangular, lon x lat).
const GRID: (usize, usize) = (512, 256);
/// Capped steps per probe as a fraction of the full integration (2x steps).
const CAP_FRACTION: f64 = 0.06;
/// Ejection check cadence inside probes.
const CHECK_INTERVAL: usize = 800;
/// Route samples inked on the map.
const ROUTE_SAMPLES: usize = 3_000;
/// Globe video length in seconds at 30 fps.
const VIDEO_SECONDS: usize = 30;

/// Representative configuration for a shape-sphere point: the inverse
/// Hopf-style Jacobi map at the seed's scale, started from rest.
fn configuration_for(n: Vector3<f64>, scale_sq: f64, masses: [f64; 3]) -> Vec<Body> {
    let scale = scale_sq.sqrt();
    let rho_len = (scale_sq * (1.0 - n.y).max(1e-8) / 2.0).sqrt().max(1e-4 * scale);
    let rho = (rho_len, 0.0);
    let lambda = (scale_sq * n.x / 2.0 / rho_len, scale_sq * n.z / 2.0 / rho_len);
    let total_mass: f64 = masses.iter().sum();
    let sqrt3_half = 3.0_f64.sqrt() / 2.0;
    let center = (
        (rho.0 * (masses[0] - masses[1]) / 2.0 - sqrt3_half * masses[2] * lambda.0) / total_mass,
        (rho.1 * (masses[0] - masses[1]) / 2.0 - sqrt3_half * masses[2] * lambda.1) / total_mass,
    );
    let r1 = (center.0 - rho.0 / 2.0, center.1 - rho.1 / 2.0);
    let r2 = (center.0 + rho.0 / 2.0, center.1 + rho.1 / 2.0);
    let r3 = (center.0 + sqrt3_half * lambda.0, center.1 + sqrt3_half * lambda.1);
    vec![
        Body::new(masses[0], Vector3::new(r1.0, r1.1, 0.0), Vector3::zeros()),
        Body::new(masses[1], Vector3::new(r2.0, r2.1, 0.0), Vector3::zeros()),
        Body::new(masses[2], Vector3::new(r3.0, r3.1, 0.0), Vector3::zeros()),
    ]
}

/// Winkel tripel forward projection (unit sphere radians -> map units).
fn winkel_tripel(lon: f64, lat: f64) -> (f64, f64) {
    let phi1_cos = 2.0 / std::f64::consts::PI;
    let alpha = (lat.cos() * (lon / 2.0).cos()).clamp(-1.0, 1.0).acos();
    let sinc = if alpha.abs() < 1e-9 { 1.0 } else { alpha.sin() / alpha };
    let x = 0.5 * (lon * phi1_cos + 2.0 * lat.cos() * (lon / 2.0).sin() / sinc);
    let y = 0.5 * (lat + lat.sin() / sinc);
    (x, y)
}

/// Fill a triangle with a flat color (bounding-box edge-function raster).
fn fill_triangle(
    buffer: &mut [Rgb64],
    width: usize,
    height: usize,
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
    color: Rgb64,
) {
    let min_x = a.0.min(b.0).min(c.0).floor().max(0.0) as usize;
    let max_x = (a.0.max(b.0).max(c.0).ceil() as usize).min(width.saturating_sub(1));
    let min_y = a.1.min(b.1).min(c.1).floor().max(0.0) as usize;
    let max_y = (a.1.max(b.1).max(c.1).ceil() as usize).min(height.saturating_sub(1));
    let edge = |p: (f64, f64), q: (f64, f64), x: f64, y: f64| -> f64 {
        (q.0 - p.0) * (y - p.1) - (q.1 - p.1) * (x - p.0)
    };
    let area = edge(a, b, c.0, c.1);
    if area.abs() < 1e-12 {
        return;
    }
    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let x = px as f64 + 0.5;
            let y = py as f64 + 0.5;
            let w0 = edge(a, b, x, y) / area;
            let w1 = edge(b, c, x, y) / area;
            let w2 = edge(c, a, x, y) / area;
            if w0 >= -1e-9 && w1 >= -1e-9 && w2 >= -1e-9 {
                buffer[py * width + px] = color;
            }
        }
    }
}

/// The terra mode.
pub struct Terra;

impl VizMode for Terra {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("terra").expect("terra is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 || ctx.bodies.len() != 3 {
            warn!("terra skipped: trajectory too short or bodies unavailable");
            return Ok(());
        }
        let mut rng = ctx.fork_rng("terra");
        let grain_seed = rng.next_u64();
        let jitter_seed = rng.next_u64();

        // --- Seed hyper-radius scale from the run itself.
        let scale_sq = {
            let mut sum = 0.0f64;
            let mut count = 0.0f64;
            let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
            for step in (0..steps).step_by((steps / 2_000).max(1)) {
                let p = |body: usize| ctx.positions[body][step];
                let rho = p(1) - p(0);
                let lambda = (p(2) * 2.0 - p(0) - p(1)) * inv_sqrt3;
                sum += rho.x * rho.x + rho.y * rho.y + lambda.x * lambda.x + lambda.y * lambda.y;
                count += 1.0;
            }
            (sum / count.max(1.0)).max(1e-9)
        };
        let masses = ctx.kinematics().masses;

        // --- Fate lattice over the shape sphere.
        let grid_w = ctx.quality.scale_count(GRID.0).max(64);
        let grid_h = ctx.quality.scale_count(GRID.1).max(32);
        let params = GridParams {
            n: 2,
            epsilon: 0.0,
            warmup: 0,
            cap: ((2 * steps) as f64 * CAP_FRACTION) as usize,
            check_interval: CHECK_INTERVAL,
            escape_threshold: DEFAULT_ESCAPE_THRESHOLD,
        };
        info!(
            "   terra: probing {grid_w}x{grid_h} shape-sphere lattice (cap {} steps)",
            params.cap
        );
        let scan_started = std::time::Instant::now();
        let fates: Vec<CellOutcome> = (0..grid_w * grid_h)
            .into_par_iter()
            .map(|cell| {
                let col = cell % grid_w;
                let row = cell / grid_w;
                let lon = (col as f64 + 0.5) / grid_w as f64 * std::f64::consts::TAU
                    - std::f64::consts::PI;
                let lat = std::f64::consts::FRAC_PI_2
                    - (row as f64 + 0.5) / grid_h as f64 * std::f64::consts::PI;
                let n = Vector3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
                let bodies = configuration_for(n, scale_sq, masses);
                capped_fate(&bodies, &params)
            })
            .collect();
        let land_count = fates.iter().filter(|fate| fate.escaper.is_none()).count();
        info!(
            "   terra: scan in {:.1}s ({} land / {} ocean cells)",
            scan_started.elapsed().as_secs_f64(),
            land_count,
            grid_w * grid_h - land_count
        );

        // --- Terrain palette: palette-derived ochre land, cyanotype ocean.
        let (_, _, palette_hue) = {
            let mean = ctx.mean_color(0);
            oklab_to_oklch(mean.0, mean.1, mean.2)
        };
        let land_hue = 70.0 + (palette_hue % 40.0);
        let land_color = |lat_fraction: f64| -> Rgb64 {
            let (l, a, b) = oklch_to_oklab(0.52 + 0.10 * lat_fraction, 0.055, land_hue);
            let (r, g, blue) = oklab_to_linear_rec2020(l, a, b);
            (r.max(0.0), g.max(0.0), blue.max(0.0))
        };
        let ocean_color = |depth: f64| -> Rgb64 {
            let (l, a, b) = oklch_to_oklab(0.34 - 0.16 * depth, 0.07, 250.0);
            let (r, g, blue) = oklab_to_linear_rec2020(l, a, b);
            (r.max(0.0), g.max(0.0), blue.max(0.0))
        };
        let terrain: Vec<Rgb64> = fates
            .iter()
            .enumerate()
            .map(|(cell, fate)| {
                let row = cell / grid_w;
                let lat_fraction = ((row as f64 + 0.5) / grid_h as f64 - 0.5).abs() * 2.0;
                if fate.escaper.is_none() {
                    land_color(lat_fraction)
                } else {
                    let depth = 1.0
                        - f64::from(fate.ejection_step.unwrap_or(params.cap as u32))
                            / params.cap.max(1) as f64;
                    ocean_color(depth)
                }
            })
            .collect();

        // --- Route on the sphere (shape-space expedition track).
        let route: Vec<(f64, f64)> = (0..steps)
            .step_by((steps / ROUTE_SAMPLES).max(1))
            .map(|step| {
                let p = |body: usize| (ctx.positions[body][step].x, ctx.positions[body][step].y);
                let n = shape_point(p(0), p(1), p(2));
                (n.y.atan2(n.x), n.z.clamp(-1.0, 1.0).asin())
            })
            .collect();

        // --- terra_data.png: raw colored equirect with the route burned in.
        {
            let out_w = grid_w * 4;
            let out_h = grid_h * 4;
            let mut data = vec![(0.0, 0.0, 0.0); out_w * out_h];
            for (index, slot) in data.iter_mut().enumerate() {
                let col = (index % out_w) * grid_w / out_w;
                let row = (index / out_w) * grid_h / out_h;
                *slot = terrain[row * grid_w + col];
            }
            let track: Rgb64 = (0.85, 0.78, 0.55);
            for &(lon, lat) in &route {
                let x = ((lon + std::f64::consts::PI) / std::f64::consts::TAU * out_w as f64)
                    .rem_euclid(out_w as f64) as f32;
                let y = ((std::f64::consts::FRAC_PI_2 - lat) / std::f64::consts::PI * out_h as f64)
                    .clamp(0.0, out_h as f64 - 1.0) as f32;
                draw_line(&mut data, out_w, out_h, (x, y), (x, y), track, 2.0, 0.9);
            }
            let image = encode_linear_rec2020_png16(&data, out_w as u32, out_h as u32);
            sink.save_png16(&image, "terra_data.png")?;
        }

        // --- Atlas plate (Winkel tripel on archival cream).
        let plate_h = ctx.quality.scale_dim(2234).max(300) as usize;
        let plate_w = plate_h * 3 / 2;
        let margin = plate_h / 16;
        let cream: Rgb64 = {
            let (l, a, b) = oklch_to_oklab(0.94, 0.02, 95.0);
            let (r, g, blue) = oklab_to_linear_rec2020(l, a, b);
            (r.max(0.0), g.max(0.0), blue.max(0.0))
        };
        let ink: Rgb64 = {
            let (l, a, b) = oklch_to_oklab(0.30, 0.035, 65.0);
            let (r, g, blue) = oklab_to_linear_rec2020(l, a, b);
            (r.max(0.0), g.max(0.0), blue.max(0.0))
        };
        let mut plate = vec![cream; plate_w * plate_h];
        // Paper grain: deterministic 3% multiplicative hash noise.
        for (index, pixel) in plate.iter_mut().enumerate() {
            let mut state =
                (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(grain_seed);
            state ^= state >> 33;
            state = state.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            let grain = 1.0 + 0.03 * (((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5);
            pixel.0 *= grain;
            pixel.1 *= grain;
            pixel.2 *= grain;
        }
        // Projection scale: Winkel tripel spans x in [-(2+pi)/2, (2+pi)/2]
        // and y in [-pi/2, pi/2].
        let data_w = (plate_w - 2 * margin) as f64;
        let data_h = (plate_h - 2 * margin) as f64;
        let scale = (data_w / 5.15).min(data_h / 3.15);
        let to_plate = |lon: f64, lat: f64| -> (f64, f64) {
            let (x, y) = winkel_tripel(lon, lat);
            (plate_w as f64 / 2.0 + x * scale, plate_h as f64 / 2.0 - y * scale)
        };
        // Terrain fill via projected quads.
        for row in 0..grid_h {
            for col in 0..grid_w {
                let lon0 =
                    col as f64 / grid_w as f64 * std::f64::consts::TAU - std::f64::consts::PI;
                let lon1 =
                    (col + 1) as f64 / grid_w as f64 * std::f64::consts::TAU - std::f64::consts::PI;
                let lat0 =
                    std::f64::consts::FRAC_PI_2 - row as f64 / grid_h as f64 * std::f64::consts::PI;
                let lat1 = std::f64::consts::FRAC_PI_2
                    - (row + 1) as f64 / grid_h as f64 * std::f64::consts::PI;
                let color = terrain[row * grid_w + col];
                // Tone the fill toward paper for the antique wash look.
                let washed = (
                    color.0 * 0.82 + cream.0 * 0.18,
                    color.1 * 0.82 + cream.1 * 0.18,
                    color.2 * 0.82 + cream.2 * 0.18,
                );
                let quad = [
                    to_plate(lon0, lat0),
                    to_plate(lon1, lat0),
                    to_plate(lon1, lat1),
                    to_plate(lon0, lat1),
                ];
                fill_triangle(&mut plate, plate_w, plate_h, quad[0], quad[1], quad[2], washed);
                fill_triangle(&mut plate, plate_w, plate_h, quad[0], quad[2], quad[3], washed);
            }
        }
        // Coastlines: boundary edges between land and ocean, double-inked.
        let is_land = |row: usize, col: usize| fates[row * grid_w + col].escaper.is_none();
        let draw_coast = |from: (f64, f64), to: (f64, f64), plate: &mut Vec<Rgb64>| {
            draw_line(
                plate,
                plate_w,
                plate_h,
                (from.0 as f32, from.1 as f32),
                (to.0 as f32, to.1 as f32),
                ink,
                1.4,
                0.95,
            );
            draw_line(
                plate,
                plate_w,
                plate_h,
                (from.0 as f32 + 0.5, from.1 as f32 + 0.5),
                (to.0 as f32 + 0.5, to.1 as f32 + 0.5),
                ink,
                0.5,
                0.6,
            );
        };
        for row in 0..grid_h {
            for col in 0..grid_w {
                let lon0 =
                    col as f64 / grid_w as f64 * std::f64::consts::TAU - std::f64::consts::PI;
                let lon1 =
                    (col + 1) as f64 / grid_w as f64 * std::f64::consts::TAU - std::f64::consts::PI;
                let lat0 =
                    std::f64::consts::FRAC_PI_2 - row as f64 / grid_h as f64 * std::f64::consts::PI;
                let lat1 = std::f64::consts::FRAC_PI_2
                    - (row + 1) as f64 / grid_h as f64 * std::f64::consts::PI;
                if col + 1 < grid_w && is_land(row, col) != is_land(row, col + 1) {
                    draw_coast(to_plate(lon1, lat0), to_plate(lon1, lat1), &mut plate);
                }
                if row + 1 < grid_h && is_land(row, col) != is_land(row + 1, col) {
                    draw_coast(to_plate(lon0, lat1), to_plate(lon1, lat1), &mut plate);
                }
            }
        }
        // Graticule every 30 degrees.
        let graticule =
            (ink.0 * 0.5 + cream.0 * 0.5, ink.1 * 0.5 + cream.1 * 0.5, ink.2 * 0.5 + cream.2 * 0.5);
        for meridian in 0..=12 {
            let lon = f64::from(meridian) / 12.0 * std::f64::consts::TAU - std::f64::consts::PI;
            let mut previous: Option<(f64, f64)> = None;
            for sample in 0..=64 {
                let lat =
                    std::f64::consts::FRAC_PI_2 - f64::from(sample) / 64.0 * std::f64::consts::PI;
                let point = to_plate(lon, lat);
                if let Some(prev) = previous {
                    draw_line(
                        &mut plate,
                        plate_w,
                        plate_h,
                        (prev.0 as f32, prev.1 as f32),
                        (point.0 as f32, point.1 as f32),
                        graticule,
                        0.7,
                        0.5,
                    );
                }
                previous = Some(point);
            }
        }
        for parallel in 1..6 {
            let lat =
                std::f64::consts::FRAC_PI_2 - f64::from(parallel) / 6.0 * std::f64::consts::PI;
            let mut previous: Option<(f64, f64)> = None;
            for sample in 0..=96 {
                let lon = f64::from(sample) / 96.0 * std::f64::consts::TAU - std::f64::consts::PI;
                let point = to_plate(lon, lat);
                if let Some(prev) = previous {
                    draw_line(
                        &mut plate,
                        plate_w,
                        plate_h,
                        (prev.0 as f32, prev.1 as f32),
                        (point.0 as f32, point.1 as f32),
                        graticule,
                        0.7,
                        0.5,
                    );
                }
                previous = Some(point);
            }
        }
        // Expedition route: dotted track with decile ticks.
        for (index, &(lon, lat)) in route.iter().enumerate() {
            if !index.is_multiple_of(3) {
                continue;
            }
            let point = to_plate(lon, lat);
            let decile = index % (route.len() / 10).max(1) == 0;
            let size = if decile { 2.6 } else { 1.3 };
            draw_line(
                &mut plate,
                plate_w,
                plate_h,
                (point.0 as f32, point.1 as f32),
                (point.0 as f32, point.1 as f32),
                ink,
                size,
                0.9,
            );
        }
        // Dragon sigils at the binary-collision points, stars at the poles.
        for lon_deg in [90.0f64, 210.0, 330.0] {
            let lon = (lon_deg.to_radians() + std::f64::consts::PI)
                .rem_euclid(std::f64::consts::TAU)
                - std::f64::consts::PI;
            let center = to_plate(lon, 0.0);
            let radius = plate_h as f64 / 60.0;
            let mut previous: Option<(f64, f64)> = None;
            for sample in 0..=20 {
                let theta = f64::from(sample) / 20.0 * std::f64::consts::TAU;
                let point = (center.0 + radius * theta.cos(), center.1 + radius * theta.sin());
                if let Some(prev) = previous {
                    draw_line(
                        &mut plate,
                        plate_w,
                        plate_h,
                        (prev.0 as f32, prev.1 as f32),
                        (point.0 as f32, point.1 as f32),
                        ink,
                        1.6,
                        1.0,
                    );
                }
                previous = Some(point);
            }
            draw_line(
                &mut plate,
                plate_w,
                plate_h,
                ((center.0 - radius * 0.7) as f32, center.1 as f32),
                ((center.0 + radius * 0.7) as f32, center.1 as f32),
                ink,
                1.6,
                1.0,
            );
        }
        for pole in [std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2] {
            let center = to_plate(0.0, pole);
            let radius = plate_h as f64 / 70.0;
            for ray in 0..6 {
                let theta = f64::from(ray) / 6.0 * std::f64::consts::TAU;
                draw_line(
                    &mut plate,
                    plate_w,
                    plate_h,
                    (center.0 as f32, center.1 as f32),
                    (
                        (center.0 + radius * theta.cos()) as f32,
                        (center.1 + radius * theta.sin()) as f32,
                    ),
                    ink,
                    1.6,
                    1.0,
                );
            }
        }
        // Border frame with meridian ticks along the equator's x positions.
        let frame = [
            (margin as f32, margin as f32, (plate_w - margin) as f32, margin as f32),
            (
                (plate_w - margin) as f32,
                margin as f32,
                (plate_w - margin) as f32,
                (plate_h - margin) as f32,
            ),
            (
                (plate_w - margin) as f32,
                (plate_h - margin) as f32,
                margin as f32,
                (plate_h - margin) as f32,
            ),
            (margin as f32, (plate_h - margin) as f32, margin as f32, margin as f32),
        ];
        for &(x0, y0, x1, y1) in &frame {
            draw_line(&mut plate, plate_w, plate_h, (x0, y0), (x1, y1), ink, 2.2, 1.0);
        }
        for meridian in 0..=12 {
            let lon = f64::from(meridian) / 12.0 * std::f64::consts::TAU - std::f64::consts::PI;
            let (x, _) = to_plate(lon, 0.0);
            let tick = plate_h as f64 / 90.0;
            for y_edge in [margin as f64, (plate_h - margin) as f64] {
                draw_line(
                    &mut plate,
                    plate_w,
                    plate_h,
                    (x as f32, y_edge as f32),
                    (
                        x as f32,
                        (y_edge + tick * if y_edge < plate_h as f64 / 2.0 { 1.0 } else { -1.0 })
                            as f32,
                    ),
                    ink,
                    1.4,
                    1.0,
                );
            }
        }
        let plate_image = encode_linear_rec2020_png16(&plate, plate_w as u32, plate_h as u32);
        sink.save_png16(&plate_image, "terra_plate.png")?;

        // --- Globe: emissive terrain texture with route glow, lit softly.
        let tex_w = grid_w * 2;
        let tex_h = grid_h * 2;
        let mut texture = vec![(0.0, 0.0, 0.0); tex_w * tex_h];
        for (index, slot) in texture.iter_mut().enumerate() {
            let col = (index % tex_w) * grid_w / tex_w;
            let row = (index / tex_w) * grid_h / tex_h;
            let base = terrain[row * grid_w + col];
            *slot = (base.0 * 0.9, base.1 * 0.9, base.2 * 0.9);
        }
        let glow: Rgb64 = (1.4, 1.25, 0.85);
        for &(lon, lat) in &route {
            let x = ((lon + std::f64::consts::PI) / std::f64::consts::TAU * tex_w as f64)
                .rem_euclid(tex_w as f64) as f32;
            let y = ((std::f64::consts::FRAC_PI_2 - lat) / std::f64::consts::PI * tex_h as f64)
                .clamp(0.0, tex_h as f64 - 1.0) as f32;
            draw_line(&mut texture, tex_w, tex_h, (x, y), (x, y), glow, 1.8, 0.8);
        }
        // Shape axes (n1, n2, n3) -> shader basis (x, z, y), as in V08.
        let shape_to_tex = Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
        let orientation_at = |spin: f64| -> Matrix3<f64> {
            let (s, c) = spin.sin_cos();
            let spin_matrix = Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
            shape_to_tex * spin_matrix
        };
        let mut scene = Scene::new(Vec::new(), 0);
        scene.spheres.push(TexturedSphere {
            center: Vec3::zeros(),
            radius: 1.0,
            texture,
            tex_w,
            tex_h,
            orientation: orientation_at(0.0),
            rim_color: (0.05, 0.08, 0.14),
            rim_strength: 0.8,
            sun: Some((Vec3::new(-0.45, 0.30, 0.84).normalize(), 0.30)),
        });
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(VIDEO_SECONDS * 30);
        let camera = orbit_camera(Vec3::zeros(), fit_distance(1.0, 30.0) * 1.1, 0.4, 0.24, 30.0);
        let video_params = RenderParams {
            width: video_w,
            height: video_h,
            spp: 1,
            max_steps: 32,
            shadows: false,
            shadow_emitters: 0,
            jitter_seed,
        };
        let frame0 = render(&scene, &camera, &video_params);
        let mut frame0_rgba: PixelBuffer = Vec::new();
        fill_rgba(&frame0, &mut frame0_rgba);
        let levels = auto_levels(
            &frame0_rgba,
            ctx.settings.resolved_config.clip_black,
            ctx.settings.resolved_config.clip_white,
            1.0,
        );
        let started = std::time::Instant::now();
        let mut logged = false;
        stream_video(
            video_w,
            video_h,
            30,
            &sink.path("terra_globe.mp4"),
            &sink.path("terra_globe_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let spin = frame as f64 / frame_count.max(1) as f64 * std::f64::consts::TAU;
                scene.spheres[0].orientation = orientation_at(spin);
                let pixels = render(&scene, &camera, &video_params);
                fill_rgba(&pixels, rgba);
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = started.elapsed().as_secs_f64();
                    info!(
                        "   terra globe: {per_frame:.2}s/frame, projected {:.1} min",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;
        sink.record("terra_globe.mp4", "video");
        sink.record("terra_globe_hq.mp4", "video");

        let meta = serde_json::json!({
            "grid": [grid_w, grid_h],
            "cap_steps": params.cap,
            "scale_sq": scale_sq,
            "land_cells": land_count,
            "route_samples": route.len(),
            "note": "cartouche and label typography deferred until text.rs",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("terra_params.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_shape_map_roundtrips_through_the_forward_map() {
        let masses = [1.0, 1.3, 0.8];
        for &(lon, lat) in &[(0.3f64, 0.2f64), (-1.2, 0.7), (2.4, -0.5), (0.0, 1.35), (-2.9, -1.2)]
        {
            let n = Vector3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
            let bodies = configuration_for(n, 4.0, masses);
            let p = |index: usize| (bodies[index].position.x, bodies[index].position.y);
            let roundtrip = shape_point(p(0), p(1), p(2));
            assert!(
                (roundtrip - n).norm() < 1e-9,
                "shape roundtrip failed at ({lon}, {lat}): {roundtrip:?} vs {n:?}"
            );
        }
    }

    #[test]
    fn winkel_tripel_maps_the_origin_and_stays_bounded() {
        let (x0, y0) = winkel_tripel(0.0, 0.0);
        assert!(x0.abs() < 1e-12 && y0.abs() < 1e-12);
        for &(lon, lat) in &[
            (std::f64::consts::PI, 0.0),
            (-std::f64::consts::PI, 0.5),
            (1.5, std::f64::consts::FRAC_PI_2),
            (-2.0, -std::f64::consts::FRAC_PI_2),
        ] {
            let (x, y) = winkel_tripel(lon, lat);
            assert!(x.is_finite() && y.is_finite());
            assert!(x.abs() <= 2.58 && y.abs() <= 1.58, "({lon},{lat}) -> ({x},{y})");
        }
    }

    #[test]
    fn triangle_fill_covers_interior_pixels() {
        let mut buffer = vec![(0.0, 0.0, 0.0); 32 * 32];
        fill_triangle(&mut buffer, 32, 32, (2.0, 2.0), (28.0, 4.0), (14.0, 28.0), (1.0, 0.5, 0.25));
        assert!(buffer[14 * 32 + 14].0 > 0.9, "centroid pixel must be filled");
        assert!(buffer[0].0 < 1e-9, "corner outside stays empty");
    }
}
