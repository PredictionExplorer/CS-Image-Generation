//! 360-degree orbit ("turntable") video rendering.
//!
//! Re-renders the fully accumulated sculpture from a camera sweeping one full
//! turn around it, as if walking around a hologram. The camera is implemented
//! as a rigid world rotation applied ahead of the fixed orthographic
//! projection, so every stroke vocabulary is re-projected exactly as the 3D
//! geometry it derives from. Three parts of the still pipeline are inherently
//! screen-space and are replaced by world-space analogues here:
//!
//! - **Framing**: instead of refitting the bounding box per view (zoom
//!   breathing) or reusing the canonical fit (clipping), the exact union of
//!   projected extents over the whole sweep is computed analytically once.
//! - **Symmetry**: the screen-space kaleidoscope replication becomes k
//!   world-space copies rotated about the canonical view axis, turning
//!   mandala seeds into true 3D rosettes.
//! - **Stardust**: the screen-space dot field becomes a seeded sphere shell
//!   that rotates with the world, giving the dust real parallax.
//!
//! Tonemap levels are frozen across the sweep and frame 0 sits at yaw 0, so
//! the encoded video loops seamlessly with no exposure pumping.

use super::context::{BoundingBox, PixelBuffer, RenderContext};
use super::drawing::{LineVertex, SpectralLineSegment, draw_line_segment_aa_spectral_rows};
use super::effects::{FinishEffectPipeline, FrameParams, convert_spd_buffer_to_rgba};
use super::error::{RenderError, Result};
use super::velocity_hdr::VelocityHdrCalculator;
use super::visual_profile::{StardustTraits, SymmetryOp};
use super::{
    AccumulationParams, ChannelLevels, FinishOutputMode, OklabColor, SpectralRenderSettings,
    SpectralScene, VideoOutputSpec, accumulate_spectral_steps, apply_spike_finish,
    build_effect_config_from_resolved, constants, create_videos_from_frames_singlepass,
    default_accumulation_backend, quantize_display_buffer_to_16bit, splitmix_unit,
    tonemap_to_display_buffer,
};
use crate::spectrum::NUM_BINS;
use nalgebra::{Matrix3, Vector3};
use tracing::info;

/// Configuration for one 360-degree orbit (turntable) video render.
#[derive(Clone, Debug)]
pub struct OrbitVideoConfig {
    /// Output frame width in pixels (keep even for broad codec support).
    pub width: u32,
    /// Output frame height in pixels (keep even for broad codec support).
    pub height: u32,
    /// Output frame rate in frames per second.
    pub fps: u32,
    /// Duration of the full 360-degree sweep in seconds.
    pub seconds: f64,
    /// Camera elevation above the sculpture's equator, in degrees.
    pub tilt_deg: f64,
    /// Keep every Nth simulation step; deposited energy is compensated so
    /// exposure matches the un-strided sculpture.
    pub step_stride: usize,
}

impl OrbitVideoConfig {
    /// Total frames in one seamless 360-degree loop.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        // f64→usize: frame counts are small positive integers by construction.
        ((self.seconds.max(0.0) * f64::from(self.fps)).round() as usize).max(1)
    }
}

/// Rotation about the +y (image-vertical) axis: the turntable yaw.
fn yaw_rotation(theta: f64) -> Matrix3<f64> {
    let (s, c) = theta.sin_cos();
    Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
}

/// Rotation about the +x (image-horizontal) axis: the camera tilt.
fn tilt_rotation(tilt: f64) -> Matrix3<f64> {
    let (s, c) = tilt.sin_cos();
    Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
}

/// Rotation about the +z (canonical view) axis, used for symmetry copies.
fn z_rotation(theta: f64) -> Matrix3<f64> {
    let (s, c) = theta.sin_cos();
    Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
}

/// Mirror across the x = 0 plane: the world-space analogue of `MirrorX`.
fn mirror_x() -> Matrix3<f64> {
    Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
}

/// World-space symmetry copies replacing the screen-space kaleidoscope.
///
/// The canonical camera looks along +z, so replication about the world z axis
/// projects to the same composition as the 2D stroke replication while
/// remaining a rigid 3D object that can be orbited.
fn world_symmetry_transforms(symmetry: SymmetryOp) -> Vec<Matrix3<f64>> {
    match symmetry {
        SymmetryOp::None => vec![Matrix3::identity()],
        SymmetryOp::MirrorX => vec![Matrix3::identity(), mirror_x()],
        SymmetryOp::Rotational { k } => {
            let k = usize::from(k.max(1));
            // usize→f64: fold counts are at most 12.
            (0..k).map(|i| z_rotation(std::f64::consts::TAU * i as f64 / k as f64)).collect()
        }
        SymmetryOp::Dihedral { k } => {
            let k = usize::from(k.max(1));
            let mut transforms = Vec::with_capacity(k * 2);
            for mirrored in [false, true] {
                for i in 0..k {
                    // usize→f64: fold counts are at most 12.
                    let rotation = z_rotation(std::f64::consts::TAU * i as f64 / k as f64);
                    transforms.push(if mirrored { rotation * mirror_x() } else { rotation });
                }
            }
            transforms
        }
    }
}

/// Strided copies of the trajectory and colour sequences.
struct StridedScene {
    positions: Vec<Vec<Vector3<f64>>>,
    colors: Vec<Vec<OklabColor>>,
}

/// Keep every `stride`-th step of the scene (positions and colours).
fn stride_scene(scene: SpectralScene<'_>, stride: usize) -> StridedScene {
    let stride = stride.max(1);
    StridedScene {
        positions: scene
            .positions
            .iter()
            .map(|body| body.iter().copied().step_by(stride).collect())
            .collect(),
        colors: scene
            .colors
            .iter()
            .map(|body| body.iter().copied().step_by(stride).collect())
            .collect(),
    }
}

/// View-invariant framing: the exact union of projected x/y extents over a
/// full 360-degree yaw sweep at `tilt_rad`, for every symmetry copy of every
/// trajectory point.
///
/// For a point `q`, the yaw circle projects to `x` in `[-r, r]` and `y` in
/// `[cos(t)·q_y − |sin(t)|·r, cos(t)·q_y + |sin(t)|·r]` with
/// `r = sqrt(q_x² + q_z²)`, so the union over points is exact — no angle
/// sampling, no per-frame zoom breathing, no clipping at any yaw.
fn orbit_bounds(
    positions: &[Vec<Vector3<f64>>],
    copies: &[Matrix3<f64>],
    tilt_rad: f64,
    width: u32,
    height: u32,
) -> BoundingBox {
    let cos_t = tilt_rad.cos();
    let sin_t = tilt_rad.sin().abs();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for copy in copies {
        for body in positions {
            for point in body {
                let q = copy * *point;
                if !(q.x.is_finite() && q.y.is_finite() && q.z.is_finite()) {
                    continue;
                }
                let radius = (q.x * q.x + q.z * q.z).sqrt();
                min_x = min_x.min(-radius);
                max_x = max_x.max(radius);
                min_y = min_y.min(cos_t * q.y - sin_t * radius);
                max_y = max_y.max(cos_t * q.y + sin_t * radius);
            }
        }
    }

    if !(min_x.is_finite() && max_x.is_finite() && min_y.is_finite() && max_y.is_finite()) {
        (min_x, max_x, min_y, max_y) = (-1.0, 1.0, -1.0, 1.0);
    }

    let pad = 0.02 * (max_x - min_x).max(max_y - min_y).max(1e-9);
    let mut bounds = BoundingBox {
        min_x: min_x - pad,
        max_x: max_x + pad,
        min_y: min_y - pad,
        max_y: max_y + pad,
        width: (max_x - min_x + 2.0 * pad).max(1e-12),
        height: (max_y - min_y + 2.0 * pad).max(1e-12),
    };
    bounds.apply_aspect_correction(width, height);
    bounds
}

/// One pre-generated stardust mote on the 3D shell around the sculpture.
struct DustMote {
    position: Vector3<f64>,
    thickness: f64,
    alpha: f64,
    color: OklabColor,
}

/// Sample the seeded stardust field on a sphere shell around the sculpture.
///
/// The still's screen-space stardust would be glued to the lens under a
/// moving camera; the shell rotates with the world instead, giving the dust
/// true parallax while keeping the same seeded visual language (sizes, glow,
/// occasional twinkle).
fn stardust_shell(dust: &StardustTraits, shell_radius: f64, base_energy: f64) -> Vec<DustMote> {
    if !dust.enabled() || base_energy <= 0.0 || shell_radius <= 0.0 {
        return Vec::new();
    }
    let mut state = dust.seed | 1;
    let mut motes = Vec::with_capacity(dust.count as usize);
    for _ in 0..dust.count {
        let z_unit = 2.0 * splitmix_unit(&mut state) - 1.0;
        let azimuth = std::f64::consts::TAU * splitmix_unit(&mut state);
        let ring = (1.0 - z_unit * z_unit).max(0.0).sqrt();
        let position =
            Vector3::new(ring * azimuth.cos(), ring * azimuth.sin(), z_unit) * shell_radius;

        let size = 0.45 + splitmix_unit(&mut state) * 1.15;
        let glow = splitmix_unit(&mut state);
        let hue = splitmix_unit(&mut state) * 360.0;
        let twinkle = if splitmix_unit(&mut state) < 0.06 { 3.0 } else { 1.0 };
        let lightness = (dust.lightness + (glow - 0.5) * 0.12).clamp(0.30, 0.92);
        let color = crate::oklab::oklch_to_oklab(lightness, dust.chroma, hue);
        motes.push(DustMote {
            position,
            thickness: size,
            alpha: base_energy * (0.4 + 0.6 * glow) * twinkle,
            color,
        });
    }
    motes
}

/// Splat the rotated dust shell into the spectral buffer for one frame.
fn splat_dust(
    accum_spd: &mut [[f64; NUM_BINS]],
    ctx: &RenderContext,
    motes: &[DustMote],
    view: &Matrix3<f64>,
) {
    for mote in motes {
        let q = view * mote.position;
        let (x, y) = ctx.to_pixel(q.x, q.y);
        let vertex = LineVertex {
            x,
            y,
            // f64→f32 precision loss is irrelevant at raster scale.
            z: q.z as f32,
            color: mote.color,
            alpha: mote.alpha,
        };
        draw_line_segment_aa_spectral_rows(
            accum_spd,
            ctx.width,
            ctx.height,
            0,
            ctx.height_usize,
            SpectralLineSegment {
                start: vertex,
                end: vertex,
                hdr_scale: 1.0,
                thickness_factor: mote.thickness,
            },
        );
    }
}

/// Largest distance of any trajectory point from the centre of mass.
fn max_point_radius(positions: &[Vec<Vector3<f64>>]) -> f64 {
    positions
        .iter()
        .flat_map(|body| body.iter())
        .map(nalgebra::Vector3::norm)
        .filter(|radius| radius.is_finite())
        .fold(0.0, f64::max)
}

/// Bounding-box centre of the trajectory.
///
/// The turntable rotates about the vertical axis through this point rather
/// than the centre of mass: lopsided sculptures then spin in place instead of
/// swinging around an off-centre axis, and the view-invariant framing tightens
/// because every point's sweep radius shrinks.
fn scene_center(positions: &[Vec<Vector3<f64>>]) -> Vector3<f64> {
    let mut min = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut max = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for point in positions.iter().flat_map(|body| body.iter()) {
        if !(point.x.is_finite() && point.y.is_finite() && point.z.is_finite()) {
            continue;
        }
        min = min.inf(point);
        max = max.sup(point);
    }
    if !(min.x.is_finite() && max.x.is_finite()) {
        return Vector3::zeros();
    }
    (min + max) * 0.5
}

/// Render every orbit frame and hand the raw 16-bit RGB bytes to `frame_sink`.
fn render_orbit_frames(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    config: &OrbitVideoConfig,
    mut frame_sink: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    if config.width == 0 || config.height == 0 {
        return Err(RenderError::InvalidDimensions { width: config.width, height: config.height });
    }

    let stride = config.step_stride.max(1);
    let mut strided = stride_scene(scene, stride);
    let steps = strided.positions.first().map_or(0, Vec::len);
    if steps < 2 {
        return Err(RenderError::InvalidConfig {
            parameter: "orbit_step_stride".into(),
            reason: "strided trajectory needs at least two steps".into(),
        });
    }

    // Centre the sculpture on the rotation axis so it spins in place.
    let center = scene_center(&strided.positions);
    for body in &mut strided.positions {
        for point in body.iter_mut() {
            *point -= center;
        }
    }

    // Energy compensation keeps exposure stable when steps are strided, and
    // symmetry copies share the total energy exactly like the screen-space
    // replication they replace.
    // usize→f64: stride and fold counts are tiny integers.
    let hdr_scale = settings.render_config.hdr_scale * stride as f64;
    let copies = world_symmetry_transforms(settings.traits.symmetry);
    let copy_hdr_scale = hdr_scale / copies.len() as f64;

    // Screen-space symmetry and stardust are replaced by their world-space
    // analogues; every other trait carries over unchanged.
    let mut orbit_traits = settings.traits;
    orbit_traits.symmetry = SymmetryOp::None;
    orbit_traits.stardust = StardustTraits::disabled();

    let tilt_rad = config.tilt_deg.to_radians();
    let bounds = orbit_bounds(&strided.positions, &copies, tilt_rad, config.width, config.height);
    let ctx = RenderContext::with_bounds(config.width, config.height, bounds);

    // Rebuild the finish chain at the orbit resolution so bloom radii scale.
    let mut orbit_resolved = settings.resolved_config.clone();
    orbit_resolved.width = config.width;
    orbit_resolved.height = config.height;
    let effect_config = build_effect_config_from_resolved(
        &orbit_resolved,
        settings.render_config,
        FinishOutputMode::Video,
    );
    let finish_pipeline = FinishEffectPipeline::new(effect_config);

    // Speeds are norms of world-space differences, so one calculator built
    // from the un-rotated base serves every viewing angle and symmetry copy.
    let dt = constants::DEFAULT_DT * stride as f64;
    let velocity_calc = VelocityHdrCalculator::new(&strided.positions, dt);

    let dust = settings.traits.stardust;
    // usize→f64: body counts and step counts are well within f64 precision.
    let mean_alpha = scene.body_alphas.iter().sum::<f64>() / scene.body_alphas.len().max(1) as f64;
    let dust_energy =
        dust.brightness * mean_alpha * steps as f64 * hdr_scale * constants::STARDUST_ENERGY_FACTOR;
    let motes = stardust_shell(&dust, max_point_radius(&strided.positions) * 1.15, dust_energy);

    let frames = config.frame_count();
    let backend = default_accumulation_backend();
    let mut accum_spd = vec![[0.0f64; NUM_BINS]; ctx.pixel_count()];
    let mut accum_rgba: PixelBuffer = vec![(0.0, 0.0, 0.0, 0.0); ctx.pixel_count()];
    let mut rotated = strided.positions.clone();
    let report_every = (frames / 10).max(1);

    for frame in 0..frames {
        // usize→f64: frame counts are small positive integers.
        let yaw = std::f64::consts::TAU * frame as f64 / frames as f64;
        let view = tilt_rotation(tilt_rad) * yaw_rotation(yaw);

        accum_spd.fill([0.0; NUM_BINS]);
        for copy in &copies {
            let transform = view * *copy;
            for (rotated_body, base_body) in rotated.iter_mut().zip(&strided.positions) {
                for (out, point) in rotated_body.iter_mut().zip(base_body) {
                    *out = transform * *point;
                }
            }
            accumulate_spectral_steps(
                &mut accum_spd,
                &AccumulationParams {
                    scene: SpectralScene::new(&rotated, &strided.colors, scene.body_alphas),
                    ctx: &ctx,
                    velocity_calc: &velocity_calc,
                    step_start: 0,
                    step_end: steps,
                    hdr_scale: copy_hdr_scale,
                    traits: orbit_traits,
                },
                backend,
            );
        }
        splat_dust(&mut accum_spd, &ctx, &motes, &view);

        convert_spd_buffer_to_rgba(&accum_spd, &mut accum_rgba, ctx.width_usize, ctx.height_usize);

        let frame_params = FrameParams { frame_number: frame, density: None };
        let rgba_buffer = std::mem::take(&mut accum_rgba);
        let mut trajectory_pixels = finish_pipeline
            .process_trajectory(rgba_buffer, ctx.width_usize, ctx.height_usize, &frame_params)
            .map_err(|e| RenderError::EffectChain {
                effect_name: "trajectory_chain".into(),
                reason: e.to_string(),
            })?;
        apply_spike_finish(
            &mut trajectory_pixels,
            ctx.width_usize,
            ctx.height_usize,
            &settings.traits,
        );

        let display_buffer = tonemap_to_display_buffer(&trajectory_pixels, levels);

        // Reclaim the trajectory buffer's allocation for the next frame; it is
        // fully overwritten by convert_spd_buffer_to_rgba each iteration.
        trajectory_pixels.resize(ctx.pixel_count(), (0.0, 0.0, 0.0, 0.0));
        accum_rgba = trajectory_pixels;

        let final_display = finish_pipeline
            .process_image(display_buffer, ctx.width_usize, ctx.height_usize, &frame_params)
            .map_err(|e| RenderError::EffectChain {
                effect_name: "image_chain".into(),
                reason: e.to_string(),
            })?;
        let buf_16bit = quantize_display_buffer_to_16bit(&final_display);
        frame_sink(bytemuck::cast_slice(&buf_16bit))?;

        if (frame + 1).is_multiple_of(report_every) || frame + 1 == frames {
            info!("   orbit render: frame {}/{frames} done", frame + 1);
        }
    }

    Ok(())
}

/// Render the full 360-degree orbit video and encode it to `outputs`.
///
/// Framing and tonemap `levels` are fixed across the sweep and frame 0 sits
/// at yaw 0, so the encoded video is a seamless loop.
pub fn render_orbit_video(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    config: &OrbitVideoConfig,
    outputs: &[VideoOutputSpec],
) -> Result<()> {
    info!(
        "   Orbit sweep: {} frames @ {} fps, {}x{}, tilt {:.1} deg, step stride {}, symmetry {}",
        config.frame_count(),
        config.fps,
        config.width,
        config.height,
        config.tilt_deg,
        config.step_stride.max(1),
        settings.traits.symmetry.label(),
    );

    create_videos_from_frames_singlepass(
        config.width,
        config.height,
        config.fps,
        |out| {
            render_orbit_frames(scene, levels, settings, config, |bytes| {
                out.write_all(bytes).map_err(RenderError::VideoEncoding)?;
                Ok(())
            })?;
            Ok(())
        },
        outputs,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::randomizable_config::ResolvedEffectConfig;
    use crate::render::{RenderConfig, SceneTraits};

    fn helix_positions(steps: usize) -> Vec<Vec<Vector3<f64>>> {
        (0..3u32)
            .map(|body| {
                let phase = f64::from(body) * 2.1;
                (0..steps)
                    .map(|step| {
                        let t = step as f64 * 0.045;
                        Vector3::new(
                            (t + phase).cos() * (60.0 + 12.0 * f64::from(body)),
                            (t * 0.7 + phase).sin() * 42.0,
                            (t * 1.3 + phase).sin() * 55.0,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    fn flat_colors(steps: usize) -> Vec<Vec<OklabColor>> {
        let palette = [(0.78, 0.09, 0.03), (0.66, -0.05, 0.11), (0.72, 0.02, -0.12)];
        palette.iter().map(|color| vec![*color; steps]).collect()
    }

    fn collect_frames(
        positions: &[Vec<Vector3<f64>>],
        colors: &[Vec<OklabColor>],
        config: &OrbitVideoConfig,
        traits: SceneTraits,
    ) -> Vec<Vec<u8>> {
        let body_alphas = [0.002, 0.002, 0.002];
        let scene = SpectralScene::new(positions, colors, &body_alphas);
        let resolved = ResolvedEffectConfig {
            width: config.width,
            height: config.height,
            hdr_scale: 1.0,
            ..ResolvedEffectConfig::default()
        };
        let render_config = RenderConfig { hdr_scale: 1.0, ..RenderConfig::default() };
        let settings =
            SpectralRenderSettings::new(&resolved, &render_config, true).with_traits(traits);
        let levels = ChannelLevels::new(0.0, 0.35, 0.0, 0.35, 0.0, 0.35);

        let mut frames = Vec::new();
        render_orbit_frames(scene, &levels, settings, config, |bytes| {
            frames.push(bytes.to_vec());
            Ok(())
        })
        .expect("orbit frame rendering should succeed");
        frames
    }

    fn small_config() -> OrbitVideoConfig {
        OrbitVideoConfig {
            width: 64,
            height: 48,
            fps: 4,
            seconds: 1.0,
            tilt_deg: 18.0,
            step_stride: 2,
        }
    }

    #[test]
    fn frame_count_rounds_and_clamps() {
        let config = OrbitVideoConfig { fps: 24, seconds: 8.0, ..small_config() };
        assert_eq!(config.frame_count(), 192);
        let degenerate = OrbitVideoConfig { fps: 30, seconds: 0.0, ..small_config() };
        assert_eq!(degenerate.frame_count(), 1);
    }

    #[test]
    fn world_symmetry_transform_counts_match_fold_counts() {
        for (symmetry, expected) in [
            (SymmetryOp::None, 1),
            (SymmetryOp::MirrorX, 2),
            (SymmetryOp::Rotational { k: 4 }, 4),
            (SymmetryOp::Dihedral { k: 3 }, 6),
        ] {
            let transforms = world_symmetry_transforms(symmetry);
            assert_eq!(transforms.len(), expected, "copy count for {symmetry:?}");
            for transform in &transforms {
                assert!(
                    (transform.determinant().abs() - 1.0).abs() < 1e-12,
                    "symmetry copies must be rigid (|det| = 1) for {symmetry:?}"
                );
            }
        }
    }

    #[test]
    fn orbit_bounds_contain_every_sampled_view() {
        let positions = helix_positions(600);
        for symmetry in [SymmetryOp::None, SymmetryOp::Rotational { k: 5 }] {
            let copies = world_symmetry_transforms(symmetry);
            for tilt_deg in [0.0, 18.0, 35.0] {
                let tilt_rad = f64::to_radians(tilt_deg);
                let bounds = orbit_bounds(&positions, &copies, tilt_rad, 640, 480);
                for yaw_step in 0..48 {
                    let yaw = std::f64::consts::TAU * f64::from(yaw_step) / 48.0;
                    let view = tilt_rotation(tilt_rad) * yaw_rotation(yaw);
                    for copy in &copies {
                        let transform = view * *copy;
                        for body in &positions {
                            for point in body {
                                let q = transform * *point;
                                assert!(
                                    q.x >= bounds.min_x && q.x <= bounds.max_x,
                                    "x {} outside [{}, {}] (tilt {tilt_deg}, {symmetry:?})",
                                    q.x,
                                    bounds.min_x,
                                    bounds.max_x
                                );
                                assert!(
                                    q.y >= bounds.min_y && q.y <= bounds.max_y,
                                    "y {} outside [{}, {}] (tilt {tilt_deg}, {symmetry:?})",
                                    q.y,
                                    bounds.min_y,
                                    bounds.max_y
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scene_center_returns_bbox_midpoint_and_shrinks_sweep_radius() {
        let offset = Vector3::new(120.0, -40.0, 65.0);
        let positions: Vec<Vec<Vector3<f64>>> = helix_positions(300)
            .into_iter()
            .map(|body| body.into_iter().map(|p| p + offset).collect())
            .collect();

        let center = scene_center(&positions);
        assert!((center - offset).norm() < 15.0, "center {center:?} should sit near {offset:?}");

        let recentered: Vec<Vec<Vector3<f64>>> =
            positions.iter().map(|body| body.iter().map(|p| p - center).collect()).collect();
        assert!(
            max_point_radius(&recentered) < max_point_radius(&positions) * 0.8,
            "recentering must shrink the turntable sweep radius"
        );
    }

    #[test]
    fn stride_scene_keeps_every_nth_step() {
        let positions = helix_positions(101);
        let colors = flat_colors(101);
        let alphas = [0.5, 0.5, 0.5];
        let scene = SpectralScene::new(&positions, &colors, &alphas);
        let strided = stride_scene(scene, 4);
        assert_eq!(strided.positions[0].len(), 26);
        assert_eq!(strided.colors[0].len(), 26);
        assert_eq!(strided.positions[1][1], positions[1][4]);
    }

    #[test]
    fn orbit_frames_are_deterministic_and_vary_with_yaw() {
        let positions = helix_positions(400);
        let colors = flat_colors(400);
        let config = small_config();

        let first = collect_frames(&positions, &colors, &config, SceneTraits::default());
        let second = collect_frames(&positions, &colors, &config, SceneTraits::default());

        assert_eq!(first.len(), config.frame_count());
        assert_eq!(first, second, "orbit frames must be bit-deterministic");

        let energy = |frame: &[u8]| frame.iter().map(|&b| u64::from(b)).sum::<u64>();
        assert!(energy(&first[0]) > 0, "frame 0 must contain visible strokes");
        assert_ne!(first[0], first[2], "a quarter turn of a 3D sculpture must change the image");
    }

    #[test]
    fn orbit_frames_render_symmetry_copies_without_screen_replication() {
        let positions = helix_positions(240);
        let colors = flat_colors(240);
        let config = small_config();
        let traits =
            SceneTraits { symmetry: SymmetryOp::Rotational { k: 3 }, ..SceneTraits::default() };

        let frames = collect_frames(&positions, &colors, &config, traits);
        assert_eq!(frames.len(), config.frame_count());
        let energy = |frame: &[u8]| frame.iter().map(|&b| u64::from(b)).sum::<u64>();
        assert!(energy(&frames[0]) > 0, "rosette frame must contain visible strokes");
    }
}
