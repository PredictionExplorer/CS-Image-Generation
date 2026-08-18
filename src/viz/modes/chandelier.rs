//! V45 `chandelier` -- The Room Lit by the Orbit.
//!
//! The worldtube capsule set (time axis vertical) hung in a fogged dark
//! gallery: per-capsule emission is scaled by the master render's energy
//! field along the path, the six-plane room catches direct tube light with
//! colored shadows, the glossy floor pools the glow, and equiangular fog
//! scattering fills the beams. Hero still through the doorway; 20 s slow
//! push-in at half resolution.

use crate::error::Result;
use crate::render::context::{PixelBuffer, RenderContext};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::stream_video;
use crate::viz::common::display::auto_levels;
use crate::viz::common::tube_render::{Fog, Plane, PlaneFinish, RenderParams, Scene, Vec3, render};
use crate::viz::context::VizContext;
use crate::viz::modes::worldtube::{fill_rgba, worldtube_geometry};
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Emitter cluster count (spec constant K).
const EMITTER_COUNT: usize = 48;
/// Fog scattering per scene meter (spec: 0.012 / m).
const FOG_SIGMA_S_PER_M: f64 = 0.012;
/// Sculpture height in scene meters (sets the meter scale).
const SCULPTURE_METERS: f64 = 2.4;
/// Floor gloss.
const FLOOR_GLOSS: f64 = 0.25;
/// Video length in seconds at 30 fps.
const VIDEO_SECONDS: usize = 20;
/// Emission gain before energy-field scaling.
const EMISSION_GAIN: f64 = 2.0;

/// The chandelier mode.
pub struct Chandelier;

impl VizMode for Chandelier {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("chandelier").expect("chandelier is in the catalog")
    }

    fn needs_energy_field(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 2 {
            warn!("chandelier skipped: trajectory too short");
            return Ok(());
        }
        let geometry = worldtube_geometry(ctx, EMISSION_GAIN);
        let height = geometry.height;

        // --- Scale emission by the master energy field along the path.
        let energy = ctx.energy_field();
        let mut capsules = geometry.capsules;
        if let Some(field) = energy {
            let render_ctx = RenderContext::new(
                ctx.width,
                ctx.height,
                ctx.positions,
                ctx.settings.aspect_correction,
            );
            let width = ctx.width as usize;
            let height_px = ctx.height as usize;
            // Normalize by a high percentile so hot cores do not blow out.
            let mut sample: Vec<f32> =
                field.iter().copied().filter(|&e| e > 0.0).step_by(17).collect();
            sample.sort_by(f32::total_cmp);
            let reference = sample
                .get(((sample.len().saturating_sub(1)) as f64 * 0.98) as usize)
                .copied()
                .unwrap_or(1.0)
                .max(1e-9);
            for capsule in &mut capsules {
                let world_x = capsule.a.x + geometry.center_xy.0;
                let world_y = capsule.a.z + geometry.center_xy.1;
                let (px, py) = render_ctx.to_pixel(world_x, world_y);
                let x = (px.max(0.0) as usize).min(width - 1);
                let y = (py.max(0.0) as usize).min(height_px - 1);
                let local = f64::from(field[y * width + x] / reference).clamp(0.05, 2.5);
                let scale = 0.35 + 0.9 * local.powf(0.7);
                for emission in [&mut capsule.emission_a, &mut capsule.emission_b] {
                    emission.0 *= scale;
                    emission.1 *= scale;
                    emission.2 *= scale;
                }
            }
        } else {
            warn!("chandelier: no energy field (image-only run); using flat emission");
        }

        // --- The room: six planes around the hung sculpture.
        let meter = height / SCULPTURE_METERS;
        let sigma_s = FOG_SIGMA_S_PER_M / meter;
        let floor_y = -0.85 * height;
        let ceiling_y = 0.85 * height;
        let half_room = 1.5 * height;
        let wall_albedo = (0.042, 0.040, 0.044);
        let mut scene = Scene::new(capsules, EMITTER_COUNT);
        scene.fog = Some(Fog { sigma_s, sigma_t: sigma_s * 1.15 });
        scene.planes.push(Plane {
            point: Vec3::new(0.0, floor_y, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            u_axis: Vec3::new(1.0, 0.0, 0.0),
            albedo: (0.055, 0.051, 0.048),
            gloss: FLOOR_GLOSS,
            extent: Some((half_room, half_room)),
            finish: PlaneFinish::Plain,
        });
        scene.planes.push(Plane {
            point: Vec3::new(0.0, ceiling_y, 0.0),
            normal: Vec3::new(0.0, -1.0, 0.0),
            u_axis: Vec3::new(1.0, 0.0, 0.0),
            albedo: wall_albedo,
            gloss: 0.0,
            extent: Some((half_room, half_room)),
            finish: PlaneFinish::Plain,
        });
        for (point, normal, u_axis) in [
            (Vec3::new(-half_room, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
            (Vec3::new(half_room, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
            (Vec3::new(0.0, 0.0, -half_room), Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0)),
            (Vec3::new(0.0, 0.0, half_room), Vec3::new(0.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 0.0)),
        ] {
            scene.planes.push(Plane {
                point,
                normal,
                u_axis,
                albedo: wall_albedo,
                gloss: 0.0,
                extent: Some((half_room, half_room)),
                finish: PlaneFinish::Plain,
            });
        }

        let mut rng = ctx.fork_rng("chandelier");
        let jitter_seed = rng.next_u64();
        let clip_black = ctx.settings.resolved_config.clip_black;
        let clip_white = ctx.settings.resolved_config.clip_white;

        // --- Hero: doorway view, sculpture on the off-center third.
        let look_at = Vec3::new(0.12 * height, -0.06 * height, 0.0);
        let hero_camera = crate::viz::common::tube_render::Camera {
            position: Vec3::new(-0.95 * height, -0.12 * height, 1.30 * height),
            look_at,
            up: Vec3::new(0.0, 1.0, 0.0),
            vfov_deg: 46.0,
        };
        let hero_w = ctx.quality.scale_dim(ctx.width);
        let hero_h = ctx.quality.scale_dim(ctx.height);
        let (hero_spp, hero_shadow) = match ctx.quality {
            crate::viz::context::VizQuality::Final => (16u32, 12usize),
            crate::viz::context::VizQuality::Draft => (2u32, 4usize),
        };
        let hero_params = RenderParams {
            width: hero_w,
            height: hero_h,
            spp: hero_spp,
            max_steps: 200,
            shadows: true,
            shadow_emitters: hero_shadow,
            jitter_seed,
        };
        let started = std::time::Instant::now();
        let pixels = render(&scene, &hero_camera, &hero_params);
        info!(
            "   chandelier hero: {hero_w}x{hero_h} spp {hero_spp} in {:.1}s ({} emitters)",
            started.elapsed().as_secs_f64(),
            scene.emitters.len()
        );
        let mut rgba: PixelBuffer = Vec::new();
        fill_rgba(&pixels, &mut rgba);
        let mut hero_levels = auto_levels(&rgba, clip_black, clip_white, 1.0);
        hero_levels.exposure_scale *= 1.18;
        let image =
            crate::viz::common::display::grade_with_levels(&rgba, hero_w, hero_h, &hero_levels);
        sink.save_png16(&image, "chandelier.png")?;

        // --- Push-in video at half resolution.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let frame_count = ctx.quality.scale_count(VIDEO_SECONDS * 30);
        let video_params = RenderParams {
            width: video_w,
            height: video_h,
            spp: 2,
            max_steps: 160,
            shadows: false,
            shadow_emitters: 0,
            jitter_seed,
        };
        let camera_at = |frame: usize| {
            let t = frame as f64 / (frame_count - 1).max(1) as f64;
            let ease = t * t * (3.0 - 2.0 * t);
            crate::viz::common::tube_render::Camera {
                position: Vec3::new(
                    (-0.95 + 0.28 * ease) * height,
                    (-0.12 + 0.03 * ease) * height,
                    (1.30 - 0.42 * ease) * height,
                ),
                look_at,
                up: Vec3::new(0.0, 1.0, 0.0),
                vfov_deg: 46.0,
            }
        };
        let frame0 = render(&scene, &camera_at(0), &video_params);
        let mut frame0_rgba: PixelBuffer = Vec::new();
        fill_rgba(&frame0, &mut frame0_rgba);
        let mut levels = auto_levels(&frame0_rgba, clip_black, clip_white, 1.0);
        levels.exposure_scale *= 1.18;
        let video_started = std::time::Instant::now();
        let mut logged = false;
        stream_video(
            video_w,
            video_h,
            30,
            &sink.path("chandelier.mp4"),
            &sink.path("chandelier_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            &levels,
            |frame, rgba| {
                let pixels = render(&scene, &camera_at(frame), &video_params);
                fill_rgba(&pixels, rgba);
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = video_started.elapsed().as_secs_f64();
                    info!(
                        "   chandelier video: {per_frame:.2}s/frame, projected {:.1} min",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;
        sink.record("chandelier.mp4", "video");
        sink.record("chandelier_hq.mp4", "video");

        let meta = serde_json::json!({
            "emitters": scene.emitters.len(),
            "fog_sigma_s_per_m": FOG_SIGMA_S_PER_M,
            "sculpture_meters": SCULPTURE_METERS,
            "floor_gloss": FLOOR_GLOSS,
            "energy_field_used": energy.is_some(),
            "frames": frame_count,
            "fps": 30,
            "note": "VPL diffuse bounce approximated (direct + fog only); see Wave 6 addendum",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("chandelier_params.json", &json, "data")?;
        Ok(())
    }
}
