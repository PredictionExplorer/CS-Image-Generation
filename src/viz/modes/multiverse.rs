//! V24 `multiverse` -- The Garden of Forking Orbits.
//!
//! Eight sibling universes with initial conditions perturbed by one part in
//! ten to the ninth, re-simulated deterministically and rendered as a 3x3
//! grid around the original: visually identical at first, blossoming apart.
//! All nine trajectories share the master's recovered view rotation and the
//! center cell's framing; each cell keeps its own histogram levels for
//! fairness. Divergence times are measured on the raw trajectories and
//! flashed as one-frame hairline borders in the video.

use crate::error::Result;
use crate::render::constants::DEFAULT_DT;
use crate::render::context::RenderContext;
use crate::render::effects::convert_spd_buffer_to_rgba;
use crate::render::velocity_hdr::VelocityHdrCalculator;
use crate::render::{
    AccumulationParams, ChannelLevels, ImageBuffer, Rgb, SpectralRenderSettings, SpectralScene,
    VideoEncodingOptions, VideoOutputSpec, accumulate_spectral_steps,
    create_videos_from_frames_singlepass, default_accumulation_backend,
    quantize_display_buffer_to_16bit, tonemap_to_display_buffer,
};
use crate::spectrum::NUM_BINS;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::{resized_config, scene_levels};
use crate::viz::common::resim::{
    perturb_ensemble, recover_orientation, rerun, rotate_trajectories,
};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use nalgebra::Vector3;
use tracing::{info, warn};

/// Number of sibling universes.
const SIBLINGS: usize = 8;
/// Relative perturbation size.
const EPSILON: f64 = 1e-9;
/// Video frames at final quality (30 s at 30 fps).
const TOTAL_FRAMES: usize = 900;
/// Divergence threshold as a fraction of the scene scale.
const DIVERGENCE_FRACTION: f64 = 0.01;
/// Poster gutter as a fraction of the cell width.
const GUTTER_FRACTION: f64 = 0.02;

/// Grid order: siblings fill around the central original.
const GRID_ORDER: [usize; 9] = [1, 2, 3, 4, 0, 5, 6, 7, 8];

/// The multiverse mode.
pub struct Multiverse;

impl VizMode for Multiverse {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("multiverse").expect("multiverse is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 || ctx.bodies.len() != 3 {
            warn!("multiverse skipped: trajectory too short or bodies unavailable");
            return Ok(());
        }
        let mut rng = ctx.fork_rng("multiverse");
        let sibling_bodies = perturb_ensemble(ctx.bodies, SIBLINGS, EPSILON, &mut rng);

        // --- Nine deterministic simulations (index 0 = the original).
        let sim_started = std::time::Instant::now();
        let mut universes: Vec<Vec<Vec<Vector3<f64>>>> = Vec::with_capacity(SIBLINGS + 1);
        universes.push(rerun(ctx.bodies, steps));
        {
            use rayon::prelude::*;
            let mut siblings: Vec<Vec<Vec<Vector3<f64>>>> =
                sibling_bodies.par_iter().map(|bodies| rerun(bodies, steps)).collect();
            universes.append(&mut siblings);
        }
        let rotation = recover_orientation(&universes[0], ctx.positions);
        for universe in &mut universes {
            *universe = rotate_trajectories(universe, &rotation);
        }
        info!(
            "   multiverse: 9 universes simulated + dressed in {:.1}s",
            sim_started.elapsed().as_secs_f64()
        );

        // --- Divergence times on the raw trajectories.
        let scene_scale = {
            let bounds = RenderContext::new(64, 64, &universes[0], false);
            bounds.bounds().width.hypot(bounds.bounds().height)
        };
        let threshold = scene_scale * DIVERGENCE_FRACTION;
        let divergence: Vec<Option<usize>> = (1..=SIBLINGS)
            .map(|sibling| {
                (0..steps).find(|&step| {
                    (0..3).any(|body| {
                        (universes[sibling][body][step] - universes[0][body][step]).norm()
                            > threshold
                    })
                })
            })
            .collect();

        // --- Shared framing from the original; per-cell levels.
        let cell_settings = |width: u32,
                             height: u32|
         -> (RenderContext, SpectralRenderSettings<'_>) {
            let render_ctx =
                RenderContext::new(width, height, &universes[0], ctx.settings.aspect_correction);
            (render_ctx, ctx.settings)
        };
        let hdr_scale = ctx.settings.render_config.hdr_scale;
        let traits = ctx.settings.traits;
        let accumulate_full = |spd: &mut Vec<[f64; NUM_BINS]>,
                               render_ctx: &RenderContext,
                               universe: &[Vec<Vector3<f64>>],
                               range: std::ops::Range<usize>| {
            if range.is_empty() {
                return;
            }
            let velocity_calc = VelocityHdrCalculator::new(universe, DEFAULT_DT);
            accumulate_spectral_steps(
                spd,
                &AccumulationParams {
                    scene: SpectralScene::new(universe, ctx.colors, ctx.body_alphas),
                    ctx: render_ctx,
                    velocity_calc: &velocity_calc,
                    step_start: range.start,
                    step_end: range.end,
                    hdr_scale,
                    traits,
                },
                default_accumulation_backend(),
            );
        };

        // Per-universe levels at a common analysis size (fairness: each
        // cell gets its own histogram of its own scene).
        let levels_resized = resized_config(ctx.settings.resolved_config, 640, 414);
        let render_config = *ctx.settings.render_config;
        let levels_settings = SpectralRenderSettings::new(
            &levels_resized,
            &render_config,
            ctx.settings.aspect_correction,
        )
        .with_traits(traits);
        let cell_levels: Vec<ChannelLevels> = universes
            .iter()
            .map(|universe| {
                scene_levels(
                    SpectralScene::new(universe, ctx.colors, ctx.body_alphas),
                    levels_settings,
                    1.0,
                )
            })
            .collect();

        // --- Poster: third-res cells, serial SPD reuse.
        let poster_cell_w = ((ctx.quality.scale_dim(ctx.width) / 3) & !1).max(16);
        let poster_cell_h = ((ctx.quality.scale_dim(ctx.height) / 3) & !1).max(16);
        let gutter = ((f64::from(poster_cell_w) * GUTTER_FRACTION) as u32).max(2);
        let poster_w = poster_cell_w * 3 + gutter * 4;
        let poster_h = poster_cell_h * 3 + gutter * 4;
        let mut poster = vec![0u16; poster_w as usize * poster_h as usize * 3];
        let (poster_ctx, _) = cell_settings(poster_cell_w, poster_cell_h);
        let poster_started = std::time::Instant::now();
        {
            let mut spd: Vec<[f64; NUM_BINS]> =
                vec![[0.0; NUM_BINS]; poster_cell_w as usize * poster_cell_h as usize];
            let mut rgba = Vec::new();
            for (slot, &universe_index) in GRID_ORDER.iter().enumerate() {
                spd.fill([0.0; NUM_BINS]);
                accumulate_full(&mut spd, &poster_ctx, &universes[universe_index], 0..steps);
                rgba.clear();
                rgba.resize(poster_cell_w as usize * poster_cell_h as usize, (0.0, 0.0, 0.0, 0.0));
                convert_spd_buffer_to_rgba(
                    &spd,
                    &mut rgba,
                    poster_cell_w as usize,
                    poster_cell_h as usize,
                );
                let display = tonemap_to_display_buffer(&rgba, &cell_levels[universe_index]);
                let cell = quantize_display_buffer_to_16bit(&display);
                let origin_x = gutter as usize + (slot % 3) * (poster_cell_w + gutter) as usize;
                let origin_y = gutter as usize + (slot / 3) * (poster_cell_h + gutter) as usize;
                for row in 0..poster_cell_h as usize {
                    let src = row * poster_cell_w as usize * 3;
                    let dst = ((origin_y + row) * poster_w as usize + origin_x) * 3;
                    poster[dst..dst + poster_cell_w as usize * 3]
                        .copy_from_slice(&cell[src..src + poster_cell_w as usize * 3]);
                }
            }
        }
        info!("   multiverse poster: 9 cells in {:.1}s", poster_started.elapsed().as_secs_f64());
        let poster_image = ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(poster_w, poster_h, poster)
            .expect("poster buffer sized to poster dims");
        sink.save_png16(&poster_image, "multiverse_grid.png")?;

        // --- Video: quarter-res cells, nine concurrent SPDs, mosaic frames.
        let video_cell_w = ((ctx.quality.scale_dim(ctx.width) / 6) & !1).max(16);
        let video_cell_h = ((ctx.quality.scale_dim(ctx.height) / 6) & !1).max(16);
        let video_w = video_cell_w * 3;
        let video_h = video_cell_h * 3;
        let frame_count = ctx.quality.scale_count(TOTAL_FRAMES);
        let (video_ctx, _) = cell_settings(video_cell_w, video_cell_h);
        let flash_frames: Vec<Option<usize>> = divergence
            .iter()
            .map(|time| time.map(|step| step * frame_count / steps.max(1)))
            .collect();

        let mut cell_spds: Vec<Vec<[f64; NUM_BINS]>> = (0..9)
            .map(|_| vec![[0.0; NUM_BINS]; video_cell_w as usize * video_cell_h as usize])
            .collect();
        let mut cursor = 0usize;
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("multiverse.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("multiverse_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        let video_started = std::time::Instant::now();
        let mut logged = false;
        create_videos_from_frames_singlepass(
            video_w,
            video_h,
            30,
            |out| {
                let mut rgba = Vec::new();
                let mut mosaic = vec![0u16; video_w as usize * video_h as usize * 3];
                for frame in 0..frame_count {
                    let target = ((frame + 1) * steps / frame_count).min(steps);
                    for (universe_index, spd) in cell_spds.iter_mut().enumerate() {
                        accumulate_full(
                            spd,
                            &video_ctx,
                            &universes[universe_index],
                            cursor..target,
                        );
                    }
                    cursor = target;
                    for (slot, &universe_index) in GRID_ORDER.iter().enumerate() {
                        rgba.clear();
                        rgba.resize(
                            video_cell_w as usize * video_cell_h as usize,
                            (0.0, 0.0, 0.0, 0.0),
                        );
                        convert_spd_buffer_to_rgba(
                            &cell_spds[universe_index],
                            &mut rgba,
                            video_cell_w as usize,
                            video_cell_h as usize,
                        );
                        let display =
                            tonemap_to_display_buffer(&rgba, &cell_levels[universe_index]);
                        let mut cell = quantize_display_buffer_to_16bit(&display);
                        // One-frame hairline flash at this cell's divergence.
                        if universe_index > 0 && flash_frames[universe_index - 1] == Some(frame) {
                            let w = video_cell_w as usize;
                            let h = video_cell_h as usize;
                            for x in 0..w {
                                for y in [0, h - 1] {
                                    let index = (y * w + x) * 3;
                                    cell[index..index + 3].fill(u16::MAX);
                                }
                            }
                            for y in 0..h {
                                for x in [0, w - 1] {
                                    let index = (y * w + x) * 3;
                                    cell[index..index + 3].fill(u16::MAX);
                                }
                            }
                        }
                        let origin_x = (slot % 3) * video_cell_w as usize;
                        let origin_y = (slot / 3) * video_cell_h as usize;
                        for row in 0..video_cell_h as usize {
                            let src = row * video_cell_w as usize * 3;
                            let dst = ((origin_y + row) * video_w as usize + origin_x) * 3;
                            mosaic[dst..dst + video_cell_w as usize * 3]
                                .copy_from_slice(&cell[src..src + video_cell_w as usize * 3]);
                        }
                    }
                    out.write_all(bytemuck::cast_slice(&mosaic))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                    if frame == 0 && !logged {
                        logged = true;
                        let per_frame = video_started.elapsed().as_secs_f64();
                        info!(
                            "   multiverse video: {per_frame:.2}s/frame, projected {:.1} min",
                            per_frame * frame_count as f64 / 60.0
                        );
                    }
                }
                Ok(())
            },
            &outputs,
        )?;
        sink.record("multiverse.mp4", "video");
        sink.record("multiverse_hq.mp4", "video");

        // --- Divergence sidecar (feeds V68).
        let perturbations: Vec<serde_json::Value> = sibling_bodies
            .iter()
            .map(|bodies| {
                let deltas: Vec<Vec<f64>> = bodies
                    .iter()
                    .zip(ctx.bodies.iter())
                    .map(|(perturbed, original)| {
                        let delta = perturbed.position - original.position;
                        vec![delta.x, delta.y, delta.z]
                    })
                    .collect();
                serde_json::json!(deltas)
            })
            .collect();
        let meta = serde_json::json!({
            "siblings": SIBLINGS,
            "epsilon_relative": EPSILON,
            "divergence_threshold": threshold,
            "divergence_steps": divergence,
            "perturbation_vectors": perturbations,
            "frames": frame_count,
            "fps": 30,
            "note": "cell labels deferred until text.rs; center cell re-rendered \
                     (not the master downsample) for framing consistency",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("divergence.json", &json, "data")?;
        Ok(())
    }
}
