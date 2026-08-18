//! V50 `sculpture-export` -- The Printable Object.
//!
//! Watertight 3D-print files: the worldtube sculpture (V43 geometry with a
//! printability radius floor) voxel-remeshed over its SDF union with
//! marching tetrahedra, a relief plaque extruded from the master energy
//! field, a colored PLY point cloud for laser-in-glass shops, an honest
//! rendered preview of the floored geometry, and print notes carrying the
//! scales, orientation, and the watertightness audit results.

use crate::error::Result;
use crate::oklab::oklab_to_xyz;
use crate::render::context::PixelBuffer;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::grade_auto_levels;
use crate::viz::common::tube_render::{
    Capsule, CapsuleSet, Mesh, Plane, PlaneFinish, RenderParams, Scene, Vec3, fit_distance,
    mesh_from_sdf, orbit_camera, render,
};
use crate::viz::common::vector_export::{PlyPoint, write_ply_ascii, write_stl_binary};
use crate::viz::context::VizContext;
use crate::viz::modes::worldtube::{fill_rgba, worldtube_geometry};
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Declared sculpture height in millimeters.
const SCULPTURE_HEIGHT_MM: f64 = 180.0;
/// Printability radius floor in millimeters at the declared scale.
const RADIUS_FLOOR_MM: f64 = 1.2;
/// Voxel remesh cell in millimeters.
const VOXEL_MM: f64 = 0.4;
/// Plaque footprint in millimeters.
const PLAQUE_MM: (f64, f64) = (100.0, 64.0);
/// Plaque base slab thickness in millimeters.
const PLAQUE_BASE_MM: f64 = 3.0;
/// Relief height range above the base, in millimeters.
const RELIEF_MM: (f64, f64) = (1.5, 4.0);
/// Plaque heightfield grid resolution.
const PLAQUE_GRID: (usize, usize) = (250, 160);
/// Point cloud size.
const CLOUD_POINTS: usize = 200_000;

/// Watertight box mesh from a heightfield: top surface at `heights`,
/// flat bottom at z = 0, and side walls.
fn heightfield_mesh(heights: &[f64], nx: usize, ny: usize, dx: f64, dy: f64) -> Mesh {
    let mut vertices = Vec::with_capacity(nx * ny * 2);
    for y in 0..ny {
        for x in 0..nx {
            vertices.push(Vec3::new(x as f64 * dx, y as f64 * dy, heights[y * nx + x]));
        }
    }
    for y in 0..ny {
        for x in 0..nx {
            vertices.push(Vec3::new(x as f64 * dx, y as f64 * dy, 0.0));
        }
    }
    let top = |x: usize, y: usize| (y * nx + x) as u32;
    let bottom = |x: usize, y: usize| (nx * ny + y * nx + x) as u32;

    let mut triangles = Vec::new();
    for y in 0..ny - 1 {
        for x in 0..nx - 1 {
            // Top faces wind counter-clockwise seen from +z.
            triangles.push([top(x, y), top(x + 1, y), top(x + 1, y + 1)]);
            triangles.push([top(x, y), top(x + 1, y + 1), top(x, y + 1)]);
            // Bottom faces wind the other way.
            triangles.push([bottom(x, y), bottom(x + 1, y + 1), bottom(x + 1, y)]);
            triangles.push([bottom(x, y), bottom(x, y + 1), bottom(x + 1, y + 1)]);
        }
    }
    for x in 0..nx - 1 {
        triangles.push([top(x, 0), bottom(x, 0), bottom(x + 1, 0)]);
        triangles.push([top(x, 0), bottom(x + 1, 0), top(x + 1, 0)]);
        triangles.push([top(x, ny - 1), top(x + 1, ny - 1), bottom(x + 1, ny - 1)]);
        triangles.push([top(x, ny - 1), bottom(x + 1, ny - 1), bottom(x, ny - 1)]);
    }
    for y in 0..ny - 1 {
        triangles.push([top(0, y), top(0, y + 1), bottom(0, y + 1)]);
        triangles.push([top(0, y), bottom(0, y + 1), bottom(0, y)]);
        triangles.push([top(nx - 1, y), bottom(nx - 1, y), bottom(nx - 1, y + 1)]);
        triangles.push([top(nx - 1, y), bottom(nx - 1, y + 1), top(nx - 1, y + 1)]);
    }
    Mesh { vertices, triangles }
}

/// The sculpture-export mode.
pub struct SculptureExport;

impl VizMode for SculptureExport {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("sculpture-export").expect("sculpture-export is in the catalog")
    }

    fn needs_energy_field(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 2 {
            warn!("sculpture-export skipped: trajectory too short");
            return Ok(());
        }

        // --- Worldtube geometry in millimeters with the printability floor.
        let geometry = worldtube_geometry(ctx, 1.0);
        let mm_scale = SCULPTURE_HEIGHT_MM / geometry.height.max(1e-9);
        let floored: Vec<Capsule> = geometry
            .capsules
            .iter()
            .map(|capsule| Capsule {
                a: capsule.a * mm_scale,
                b: capsule.b * mm_scale,
                radius: (capsule.radius * mm_scale).max(RADIUS_FLOOR_MM),
                emission_a: capsule.emission_a,
                emission_b: capsule.emission_b,
                core_darkening: 0.0,
            })
            .collect();
        let capsule_count = floored.len();
        let set = CapsuleSet::new(floored);

        // --- Voxel remesh (marching tetrahedra over the SDF union).
        let (lo, hi) = set.bounds();
        let pad = Vec3::new(VOXEL_MM * 2.0, VOXEL_MM * 2.0, VOXEL_MM * 2.0);
        let started = std::time::Instant::now();
        let sdf = |p: Vec3| set.mesh_distance(p);
        let mesh = mesh_from_sdf(&sdf, lo - pad, hi + pad, VOXEL_MM);
        let watertight = mesh.is_watertight();
        let parity_ok = mesh.ray_parity_ok(24);
        info!(
            "   sculpture-export: remeshed {} capsules -> {} tris in {:.1}s \
             (watertight {watertight}, parity {parity_ok})",
            capsule_count,
            mesh.triangles.len(),
            started.elapsed().as_secs_f64()
        );
        if !watertight || !parity_ok {
            warn!("sculpture-export: worldtube mesh failed an audit; noted in print_notes.txt");
        }
        write_stl_binary(
            &sink.path("worldtube.stl"),
            &mesh,
            &format!("three-body worldtube seed 0x{}", ctx.seed_hex),
        )?;
        sink.record("worldtube.stl", "vector");

        // --- Relief plaque from the master energy field.
        let mut plaque_report = String::from("skipped (no energy field; image-only run)");
        if let Some(field) = ctx.energy_field() {
            let (nx, ny) = PLAQUE_GRID;
            let width = ctx.width as usize;
            let height = ctx.height as usize;
            let mut sample: Vec<f32> =
                field.iter().copied().filter(|&e| e > 0.0).step_by(13).collect();
            sample.sort_by(f32::total_cmp);
            let reference = sample
                .get(((sample.len().saturating_sub(1)) as f64 * 0.99) as usize)
                .copied()
                .unwrap_or(1.0)
                .max(1e-9);
            let mut heights = vec![PLAQUE_BASE_MM; nx * ny];
            for gy in 0..ny {
                for gx in 0..nx {
                    // Box-average the energy under this plaque cell.
                    let x0 = gx * width / nx;
                    let x1 = ((gx + 1) * width / nx).max(x0 + 1);
                    let y0 = gy * height / ny;
                    let y1 = ((gy + 1) * height / ny).max(y0 + 1);
                    let mut sum = 0.0f64;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            sum += f64::from(field[y * width + x]);
                        }
                    }
                    let mean = sum / ((x1 - x0) * (y1 - y0)) as f64;
                    let normalized = (mean / f64::from(reference)).clamp(0.0, 4.0);
                    let relief = (RELIEF_MM.0
                        + (RELIEF_MM.1 - RELIEF_MM.0) * (1.0 + 9.0 * normalized).ln()
                            / 10.0f64.ln())
                    .clamp(RELIEF_MM.0, RELIEF_MM.1);
                    heights[gy * nx + gx] = PLAQUE_BASE_MM + relief;
                }
            }
            let plaque = heightfield_mesh(
                &heights,
                nx,
                ny,
                PLAQUE_MM.0 / (nx - 1) as f64,
                PLAQUE_MM.1 / (ny - 1) as f64,
            );
            let plaque_watertight = plaque.is_watertight();
            write_stl_binary(
                &sink.path("relief_plaque.stl"),
                &plaque,
                &format!("three-body relief plaque seed 0x{}", ctx.seed_hex),
            )?;
            sink.record("relief_plaque.stl", "vector");
            plaque_report =
                format!("{} triangles, watertight {plaque_watertight}", plaque.triangles.len());
        } else {
            warn!("sculpture-export: relief plaque skipped (no energy field)");
        }

        // --- Point cloud for laser-in-glass engraving.
        let per_body = CLOUD_POINTS / 3;
        let stride = (steps / per_body).max(1);
        let speed_window = ctx.kinematics().speed_window();
        let time_scale = SCULPTURE_HEIGHT_MM / (steps.max(2) - 1) as f64;
        let mut points = Vec::with_capacity(CLOUD_POINTS + 8);
        for body in 0..3 {
            for step in (0..steps).step_by(stride) {
                let position = ctx.positions[body][step];
                let world = Vec3::new(
                    (position.x - geometry.center_xy.0) * mm_scale,
                    step as f64 * time_scale - 0.5 * SCULPTURE_HEIGHT_MM,
                    (position.y - geometry.center_xy.1) * mm_scale,
                );
                let (l, a, b) = ctx.colors[body][step.min(ctx.colors[body].len() - 1)];
                let (xyz_x, xyz_y, xyz_z) = oklab_to_xyz(l, a, b);
                // XYZ (D65) -> linear sRGB, IEC 61966-2-1.
                let lr = 3.240_969_941_904_521 * xyz_x
                    - 1.537_383_177_570_093 * xyz_y
                    - 0.498_610_760_293_003 * xyz_z;
                let lg = -0.969_243_636_280_87 * xyz_x
                    + 1.875_967_501_507_72 * xyz_y
                    + 0.041_555_057_407_175 * xyz_z;
                let lb = 0.055_630_079_696_993 * xyz_x - 0.203_976_958_888_976 * xyz_y
                    + 1.056_971_514_242_878 * xyz_z;
                let encode = |channel: f64| -> u8 {
                    let clamped = channel.clamp(0.0, 1.0);
                    let gamma = if clamped <= 0.003_130_8 {
                        12.92 * clamped
                    } else {
                        1.055 * clamped.powf(1.0 / 2.4) - 0.055
                    };
                    (gamma * 255.0).round() as u8
                };
                let speed = ctx.kinematics().speeds[body][step];
                let dwell = 1.0 - ctx.kinematics().normalized_speed(speed_window, speed);
                points.push(PlyPoint {
                    position: world,
                    color: (encode(lr), encode(lg), encode(lb)),
                    intensity: 0.15 + 0.85 * dwell,
                });
            }
        }
        write_ply_ascii(&sink.path("pointcloud.ply"), &points)?;
        sink.record("pointcloud.ply", "vector");

        // --- Honest preview: the floored capsule geometry in neutral studio
        // light (silhouette identical to the meshed SDF source).
        let preview_capsules: Vec<Capsule> = set
            .capsules()
            .iter()
            .map(|capsule| Capsule {
                emission_a: (0.66, 0.66, 0.68),
                emission_b: (0.66, 0.66, 0.68),
                core_darkening: 0.30,
                ..capsule.clone()
            })
            .collect();
        let mut scene = Scene::new(preview_capsules, 16);
        scene.planes.push(Plane {
            point: Vec3::new(0.0, -0.62 * SCULPTURE_HEIGHT_MM, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            u_axis: Vec3::new(1.0, 0.0, 0.0),
            albedo: (0.30, 0.30, 0.31),
            gloss: 0.05,
            extent: None,
            finish: PlaneFinish::Plain,
        });
        let mut rng = ctx.fork_rng("sculpture-export");
        let jitter_seed = rng.next_u64();
        let preview_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let preview_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let camera =
            orbit_camera(Vec3::zeros(), fit_distance(set.bounds().1.norm(), 34.0), 0.7, 0.22, 34.0);
        let params = RenderParams {
            width: preview_w,
            height: preview_h,
            spp: 2,
            max_steps: 160,
            shadows: true,
            shadow_emitters: 8,
            jitter_seed,
        };
        let pixels = render(&scene, &camera, &params);
        let mut rgba: PixelBuffer = Vec::new();
        fill_rgba(&pixels, &mut rgba);
        let image = grade_auto_levels(
            &rgba,
            preview_w,
            preview_h,
            ctx.settings.resolved_config.clip_black,
            ctx.settings.resolved_config.clip_white,
            1.0,
        );
        sink.save_png16(&image, "print_preview.png")?;

        // --- Print notes.
        let notes = format!(
            "three-body sculpture export -- seed 0x{seed}\n\
             \n\
             worldtube.stl\n\
             - units: millimeters; height {height:.0} mm (time axis = +Y)\n\
             - voxel remesh at {voxel:.1} mm over the capsule SDF union\n\
             - tube radius floored at {floor:.1} mm for printability\n\
             - {tris} triangles; watertight: {watertight}; ray parity: {parity}\n\
             - print upright (time axis vertical); supports: tree, from the\n\
               build plate only; material: SLA clear or MJF PA12\n\
             \n\
             relief_plaque.stl\n\
             - {plaque_w:.0} x {plaque_h:.0} mm footprint, {base:.0} mm base,\n\
               {relief_lo:.1}-{relief_hi:.1} mm log-mapped relief (energy field)\n\
             - {plaque_report}\n\
             - print flat, no supports; material: any\n\
             \n\
             pointcloud.ply\n\
             - {cloud} points, sRGB colors + dwell intensity, millimeters\n\
             - intended for subsurface laser engraving (laser-in-glass)\n",
            seed = ctx.seed_hex,
            height = SCULPTURE_HEIGHT_MM,
            voxel = VOXEL_MM,
            floor = RADIUS_FLOOR_MM,
            tris = mesh.triangles.len(),
            watertight = watertight,
            parity = parity_ok,
            plaque_w = PLAQUE_MM.0,
            plaque_h = PLAQUE_MM.1,
            base = PLAQUE_BASE_MM,
            relief_lo = RELIEF_MM.0,
            relief_hi = RELIEF_MM.1,
            plaque_report = plaque_report,
            cloud = points.len(),
        );
        sink.write_text("print_notes.txt", &notes, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heightfield_box_is_watertight() {
        let nx = 12;
        let ny = 9;
        let heights: Vec<f64> = (0..nx * ny)
            .map(|index| 3.0 + 1.5 * f64::from(u32::try_from(index % 7).expect("small")))
            .collect();
        let mesh = heightfield_mesh(&heights, nx, ny, 1.0, 1.0);
        assert!(mesh.is_watertight(), "heightfield box must be watertight");
        assert!(mesh.ray_parity_ok(12));
    }
}
