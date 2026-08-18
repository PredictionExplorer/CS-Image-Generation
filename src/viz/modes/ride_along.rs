//! V27 `ride-along` -- What Body Three Sees.
//!
//! Translates the frame to the steadiest body: the other two loop, lunge,
//! and retreat around a fixed hero point, re-accumulated with the full
//! production vocabulary so rosette geometries emerge. A seeded 30% gate
//! additionally rotates the frame to the hero's velocity heading. Rendered
//! with the public pass-2 pipeline: full video plus final-frame still.

use crate::error::Result;
use crate::render::{
    ImageBuffer, Pass2Params, Rgb, SpectralRenderSettings, SpectralScene, VideoEncodingOptions,
    VideoOutputSpec, create_videos_from_frames_singlepass, pass_2_write_frames_spectral,
};
use crate::spectrum::NUM_BINS;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::{resized_config, scene_levels};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;
use tracing::info;

/// Probability of the velocity-heading rotation variant.
const ROTATION_GATE: f64 = 0.30;
/// Target frame count for the ride-along video.
const TARGET_FRAMES: usize = 1200;

/// The ride-along (hero frame) mode.
pub struct RideAlong;

impl VizMode for RideAlong {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("ride-along").expect("ride-along is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        let kinematics = ctx.kinematics();

        // Hero: the body with the lowest speed variance (steadiest camera).
        let hero = (0..3)
            .min_by(|&a, &b| {
                let variance = |body: usize| {
                    let speeds = &kinematics.speeds[body];
                    let mean = speeds.iter().step_by(97).sum::<f64>()
                        / speeds.iter().step_by(97).count().max(1) as f64;
                    speeds.iter().step_by(97).map(|&speed| (speed - mean).powi(2)).sum::<f64>()
                };
                variance(a).total_cmp(&variance(b))
            })
            .unwrap_or(2);

        let mut rng = ctx.fork_rng(self.entry().flag);
        let rotate_to_heading = rng.next_f64() < ROTATION_GATE;

        // Transform: hero to origin; optional rotation to velocity heading.
        let transformed: Vec<Vec<Vector3<f64>>> = (0..3)
            .map(|body| {
                (0..steps)
                    .map(|step| {
                        let relative = ctx.positions[body][step] - ctx.positions[hero][step];
                        if rotate_to_heading {
                            let velocity = kinematics.velocities[hero][step];
                            let heading = velocity.y.atan2(velocity.x);
                            let (sin, cos) = (-heading).sin_cos();
                            Vector3::new(
                                relative.x * cos - relative.y * sin,
                                relative.x * sin + relative.y * cos,
                                relative.z,
                            )
                        } else {
                            relative
                        }
                    })
                    .collect()
            })
            .collect();

        info!(
            "   ride-along: hero body {hero}, rotation-to-heading {}",
            if rotate_to_heading { "on" } else { "off" }
        );

        // Render at half resolution with the production pass-2 pipeline.
        let width = (ctx.quality.scale_dim(ctx.width) / 2) & !1;
        let height = (ctx.quality.scale_dim(ctx.height) / 2) & !1;
        let resized = resized_config(ctx.settings.resolved_config, width, height);
        let render_config = *ctx.settings.render_config;
        let settings =
            SpectralRenderSettings::new(&resized, &render_config, ctx.settings.aspect_correction)
                .with_traits(ctx.settings.traits);
        let scene = SpectralScene::new(&transformed, ctx.colors, ctx.body_alphas);
        let levels = scene_levels(scene, settings, 1.0);

        let frame_interval = (steps / TARGET_FRAMES).max(1);
        let mut last_frame: Option<ImageBuffer<Rgb<u16>, Vec<u16>>> = None;
        let mut accum_spd: Vec<[f64; NUM_BINS]> = Vec::new();
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("ride_along.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("ride_along_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        create_videos_from_frames_singlepass(
            width,
            height,
            60,
            |out| {
                pass_2_write_frames_spectral(
                    Pass2Params {
                        scene,
                        frame_interval,
                        levels: &levels,
                        settings,
                        last_frame_out: &mut last_frame,
                        accum_spd: &mut accum_spd,
                    },
                    |frame_bytes| {
                        out.write_all(frame_bytes)
                            .map_err(crate::render::error::RenderError::VideoEncoding)?;
                        Ok(())
                    },
                )?;
                Ok(())
            },
            &outputs,
        )?;
        sink.record("ride_along.mp4", "video");
        sink.record("ride_along_hq.mp4", "video");

        if let Some(frame) = last_frame {
            sink.save_png16(&frame, "ride_along.png")?;
        }

        let meta = serde_json::json!({
            "hero_body": hero,
            "rotation_to_heading": rotate_to_heading,
            "render_size": [width, height],
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("frame.json", &json, "data")?;
        Ok(())
    }
}
