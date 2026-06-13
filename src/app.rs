//! Application orchestration and workflow management
//!
//! This module breaks down the main application flow into clean, focused functions,
//! each with a single responsibility. This improves testability, readability, and
//! maintainability.

use crate::drift::parse_drift_mode;
use crate::drift_config::{ResolvedDriftConfig, resolve_drift_config};
use crate::error::{ConfigError, Result};
use crate::generation_log::{
    DriftConfig, GenerationLogger, GenerationRecord, LoggedRenderConfig, OrbitInfo,
    SimulationConfig,
};
use crate::render::{
    self, ChannelLevels, RenderConfig, SpectralRenderSettings, SpectralScene, ToneMappingControls,
    VideoEncodingOptions, constants, create_video_from_frames_singlepass,
    generate_body_color_sequences, pass_1_build_histogram_spectral, pass_2_write_frames_spectral,
    save_image_as_png_16bit,
};
use crate::sim::{self, Body, Sha3RandomByteStream, TrajectoryResult};
use image::{ImageBuffer, Rgb};
use nalgebra::{Matrix3, Vector3};
use std::fs;
use tracing::{info, warn};

/// RNG fork domain for the seeded viewing orientation.
const VIEW_RNG_DOMAIN: &[u8] = b"cosmic-view/v1";

/// Core `CosmicSignature` enhancement flags.
#[derive(Clone, Debug)]
pub struct Enhancements {
    /// Enable chroma boosting for richer color saturation.
    pub chroma_boost: bool,
    /// Enable perceptual saturation boost.
    pub sat_boost: bool,
    /// Enable ACES-inspired tone-mapping tweak.
    pub aces_tweak: bool,
    /// Enable per-body alpha variation for visual depth.
    pub alpha_variation: bool,
    /// Enable aspect-ratio correction for non-square outputs.
    pub aspect_correction: bool,
    /// Enable spectral dispersion boost.
    pub dispersion_boost: bool,
}

impl Default for Enhancements {
    fn default() -> Self {
        Self {
            chroma_boost: true,
            sat_boost: true,
            aces_tweak: true,
            alpha_variation: true,
            aspect_correction: true,
            dispersion_boost: false,
        }
    }
}

/// Configuration recorded for generation logging.
pub struct GenerationLogConfig {
    /// Number of simulation time-steps.
    pub num_steps_sim: usize,
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Black-point clipping percentile.
    pub clip_black: f64,
    /// White-point clipping percentile.
    pub clip_white: f64,
    /// Alpha denominator controlling base trail opacity.
    pub alpha_denom: usize,
    /// Alpha compression factor.
    pub alpha_compress: f64,
    /// Escape threshold for orbit rejection.
    pub escape_threshold: f64,
    /// Drift mode identifier (e.g. `"elliptical"`, `"none"`).
    pub drift_mode: String,
    /// Visual profile identifier.
    pub visual_profile: String,
    /// Whether any legacy post-processing effect was enabled.
    pub post_effects_enabled: bool,
    /// Bloom post-processing mode, kept for compatibility with older logs.
    pub bloom_mode: String,
    /// HDR tone-mapping mode.
    pub hdr_mode: String,
    /// HDR intensity scale factor.
    pub hdr_scale: f64,
    /// Active radial spectral dispersion strength.
    pub dispersion_strength: f64,
    /// Active spectral dispersion mode.
    pub dispersion_mode: String,
    /// Continuous palette genome fingerprint.
    pub palette_fingerprint: String,
    /// Palette beauty-gate descriptor (attempts / repair flag).
    pub palette_gate: String,
    /// Minimum body mass for simulation.
    pub min_mass: f64,
    /// Maximum body mass for simulation.
    pub max_mass: f64,
    /// Initial location spread parameter.
    pub location: f64,
    /// Initial velocity spread parameter.
    pub velocity: f64,
    /// Weight for chaos metric in Borda scoring.
    pub chaos_weight: f64,
    /// Weight for equilibrium metric in Borda scoring.
    pub equil_weight: f64,
    /// Whether Borda weights were randomized.
    pub weights_randomized: bool,
}

/// Initialize per-seed output directory structure:
///   output/{seed}/
///   output/{seed}/spectral/
///
/// Rejects output names containing path separators or `..` to prevent directory traversal.
pub fn setup_seed_directory(seed: &str) -> Result<String> {
    if seed.contains("..") || seed.contains('/') || seed.contains('\\') {
        return Err(ConfigError::InvalidResolution {
            reason: format!("Output name '{seed}' must not contain path separators or '..'"),
        }
        .into());
    }

    let seed_dir = format!("output/{seed}");
    let spectral_dir = format!("{seed_dir}/spectral");

    fs::create_dir_all(&seed_dir).map_err(|e| ConfigError::FileSystem {
        operation: "create directory".to_string(),
        path: seed_dir.clone(),
        error: e,
    })?;

    fs::create_dir_all(&spectral_dir).map_err(|e| ConfigError::FileSystem {
        operation: "create directory".to_string(),
        path: spectral_dir,
        error: e,
    })?;

    Ok(seed_dir)
}

/// Parse and validate hex seed
pub fn parse_seed(seed: &str) -> Result<Vec<u8>> {
    let hex_seed = seed.strip_prefix("0x").unwrap_or(seed);

    hex::decode(hex_seed)
        .map_err(|e| ConfigError::InvalidSeed { seed: seed.to_string(), error: e }.into())
}

/// Run Borda selection to find the best orbit
pub fn run_borda_selection(
    rng: &mut Sha3RandomByteStream,
    num_sims: usize,
    num_steps_sim: usize,
    chaos_weight: f64,
    equil_weight: f64,
    escape_threshold: f64,
) -> Result<(Vec<Body>, TrajectoryResult)> {
    info!("STAGE 1/7: Borda search over {} random orbits...", num_sims);

    sim::select_best_trajectory(
        rng,
        num_sims,
        num_steps_sim,
        chaos_weight,
        equil_weight,
        escape_threshold,
    )
}

/// Winning orbit of the combined Borda + aesthetic selection.
pub struct AestheticSelection {
    /// Initial body states of the winning candidate.
    pub bodies: Vec<Body>,
    /// Borda/physics metrics of the winning candidate.
    pub result: TrajectoryResult,
    /// Full trajectory of the winning candidate, already transformed into the
    /// seed's projection space (no re-simulation needed).
    pub positions: Vec<Vec<Vector3<f64>>>,
    /// Proxy-render aesthetic score of the winning candidate.
    pub aesthetic: render::aesthetic_score::AestheticScore,
    /// Aesthetic score plus the small prior bonus used for tie-breaking.
    pub selection_score: f64,
    /// Layer stack chosen by adaptive orbit × stack scoring.
    pub stack: render::LayerStack,
    /// Layer stack originally rolled by the seed.
    pub preferred_stack: render::LayerStack,
    /// Number of bounded retry searches used before accepting this candidate.
    pub retry_count: usize,
}

/// Number of top Borda candidates proxy-rendered for aesthetic selection.
pub const AESTHETIC_SHORTLIST_LEN: usize = 24;
/// Minimum proxy aesthetic score accepted before bounded retry searches.
pub const AESTHETIC_QUALITY_FLOOR: f64 = 0.82;
/// Maximum deterministic retry searches when the best proxy score is below the floor.
pub const AESTHETIC_MAX_RETRIES: usize = 3;
/// RNG fork domain prefix for bounded aesthetic retry searches.
const AESTHETIC_RETRY_RNG_DOMAIN_PREFIX: &str = "cosmic-retry/v1/";

/// Transform a trajectory into the seed's projection space.
///
/// Phase-space projections plot mixed position/velocity coordinates, producing
/// Lissajous-like curve families impossible in plain position space. Velocity
/// axes are rescaled to the position extent so mixed-axis projections stay
/// well-proportioned; the render context refits the bounding box afterwards.
#[must_use]
pub fn apply_projection(
    positions: &[Vec<Vector3<f64>>],
    projection: render::ProjectionMode,
) -> Vec<Vec<Vector3<f64>>> {
    use render::ProjectionMode;

    if projection == ProjectionMode::Position {
        return positions.to_vec();
    }
    let steps = positions.first().map_or(0, Vec::len);
    if positions.len() < 3 || steps < 2 {
        return positions.to_vec();
    }

    // Per-body velocities by forward difference (last step repeats).
    let dt = constants::DEFAULT_DT;
    let velocities: Vec<Vec<Vector3<f64>>> = positions
        .iter()
        .map(|body| {
            (0..steps)
                .map(|step| {
                    let next = (step + 1).min(steps - 1);
                    let prev = next.saturating_sub(1);
                    (body[next] - body[prev]) / dt
                })
                .collect()
        })
        .collect();

    let extent = |data: &[Vec<Vector3<f64>>], axis: usize| -> f64 {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for body in data {
            for p in body {
                if p[axis].is_finite() {
                    min = min.min(p[axis]);
                    max = max.max(p[axis]);
                }
            }
        }
        (max - min).max(1e-12)
    };
    let pos_extent = extent(positions, 0).max(extent(positions, 1));
    let vel_extent = extent(&velocities, 0).max(extent(&velocities, 1));
    let velocity_scale = if vel_extent > 1e-12 { pos_extent / vel_extent } else { 1.0 };

    (0..positions.len())
        .map(|body| {
            (0..steps)
                .map(|step| {
                    let p = positions[body][step];
                    let v = velocities[body][step] * velocity_scale;
                    match projection {
                        ProjectionMode::Position => p,
                        ProjectionMode::PhasePortrait => Vector3::new(p.x, v.x, p.y),
                        ProjectionMode::CrossBraid => {
                            let partner = positions[(body + 1) % 3][step];
                            Vector3::new(p.x, partner.y, p.z)
                        }
                        ProjectionMode::Hodograph => Vector3::new(v.x, v.y, p.z),
                    }
                })
                .collect()
        })
        .collect()
}

/// Curated fallback stacks evaluated alongside the seed's own stack.
///
/// Keeps adaptive selection bounded while ensuring that if the seed's stack
/// genuinely does not suit the orbit, a reliable alternative exists.
fn adaptive_structure_stacks(preferred: render::LayerStack) -> Vec<render::LayerStack> {
    let mut stacks = vec![preferred];
    for fallback in [
        render::LayerStack::solo(preferred.primary),
        render::LayerStack::solo(render::StructureMode::OrbitRibbons),
        render::LayerStack::solo(render::StructureMode::TimeChords),
        render::LayerStack::with_underlay(
            render::StructureMode::OrbitRibbons,
            render::StructureMode::TriangleWeb,
            0.30,
        ),
    ] {
        if !stacks.contains(&fallback) {
            stacks.push(fallback);
        }
    }
    stacks
}

fn retry_borda_weights(chaos_weight: f64, rng: &mut Sha3RandomByteStream) -> (f64, f64) {
    let descriptor = render::parameter_descriptors::EQUIL_CHAOS_RATIO;
    let log_min = descriptor.min.ln();
    let log_max = descriptor.max.ln();
    let ratio = (log_min + rng.next_f64() * (log_max - log_min)).exp();
    (chaos_weight, chaos_weight * ratio)
}

/// Prior bonus protecting seed identity during adaptive stack selection.
///
/// The seed's own stack gets the strongest bonus (larger for the newer
/// vocabularies so adaptive scoring cannot systematically homogenize the
/// population back to webs); fallbacks get small nudges. Ribbons carry no
/// fallback bonus at all: they proxy-score so well that even a small nudge
/// collapsed most of the population to `orbit_ribbons`, so they may only
/// override the preferred stack when genuinely better.
fn mode_selection_prior(stack: render::LayerStack, preferred: render::LayerStack) -> f64 {
    if stack == preferred {
        return match preferred.primary {
            render::StructureMode::OrbitRibbons
            | render::StructureMode::TimeChords
            | render::StructureMode::NebulaVeil
            | render::StructureMode::HarmonicWeave
            | render::StructureMode::StippleConstellation
            | render::StructureMode::TangentCaustics => 0.24,
            render::StructureMode::TriangleWeb
            | render::StructureMode::Duet { .. }
            | render::StructureMode::Spokes => 0.12,
        };
    }
    if stack.primary == preferred.primary {
        // The seed's primary without its secondary layers.
        return 0.04;
    }
    match stack.primary {
        // Chord sails carry the family's calligraphic signature, so the
        // fallback gets a slightly stronger nudge when scores are comparable.
        render::StructureMode::TimeChords => 0.035,
        render::StructureMode::OrbitRibbons => {
            if stack.underlay.is_some() {
                -0.07
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

fn best_aesthetic_selection_from_shortlist(
    shortlist: Vec<sim::ShortlistedTrajectory>,
    num_steps_sim: usize,
    preferred_stack: render::LayerStack,
    projection: render::ProjectionMode,
    retry_count: usize,
) -> AestheticSelection {
    let stacks = adaptive_structure_stacks(preferred_stack);
    let mut best: Option<AestheticSelection> = None;

    for (rank, candidate) in shortlist.into_iter().enumerate() {
        let raw_positions = sim::get_positions(candidate.bodies.clone(), num_steps_sim).positions;
        // Score candidates in the projection space they will be rendered in,
        // so the aesthetic gate curates phase-space seeds too.
        let positions = apply_projection(&raw_positions, projection);
        for &stack in &stacks {
            let proxy_params = render::aesthetic_score::ProxyRenderParams::for_candidates(stack);
            let score = render::aesthetic_score::score_trajectory(&positions, proxy_params);
            let prior_bonus = mode_selection_prior(stack, preferred_stack);
            // No upper clamp: clamping at 1.0 used to collapse every strong
            // preferred-stack candidate to the same score, so the first tie in
            // iteration order won instead of the genuinely best orbit.
            let selection_score = (score.total + prior_bonus).max(0.0);
            info!(
                "   candidate {rank}: orbit idx {} stack={} borda {:.1} aesthetic {:.4} select {:.4} \
                 (coverage {:.3}, balance {:.3}, contrast {:.3}, mix {:.3}, mush {:.3}, veil {:.3}, crisp {:.3}, void {:.3}, full {:.3})",
                candidate.result.selected_index,
                stack.label(),
                candidate.result.total_score_weighted,
                score.total,
                selection_score,
                score.coverage,
                score.balance,
                score.contrast,
                score.body_mix,
                score.mush_fraction,
                score.veil_fraction,
                score.crispness,
                score.negative_space,
                score.fullness,
            );

            let improves = best.as_ref().is_none_or(|cur| selection_score > cur.selection_score);
            if improves {
                best = Some(AestheticSelection {
                    bodies: candidate.bodies.clone(),
                    result: candidate.result.clone(),
                    positions: positions.clone(),
                    aesthetic: score,
                    selection_score,
                    stack,
                    preferred_stack,
                    retry_count,
                });
            }
        }
    }

    best.expect("shortlist is non-empty by construction")
}

/// Run Borda selection, then pick the most paintable candidate and layer stack by proxy render.
///
/// The Borda search ranks orbits on physics proxies only; this pass re-simulates
/// the top [`AESTHETIC_SHORTLIST_LEN`] candidates (a negligible cost next to the
/// search itself), scores each one in image space under the seed's preferred
/// stack plus a curated set of high-yield alternatives, and returns the
/// highest-scoring `(orbit, stack)` pair together with its full trajectory in
/// the seed's projection space.
pub fn run_borda_selection_with_aesthetics(
    rng: &mut Sha3RandomByteStream,
    num_sims: usize,
    num_steps_sim: usize,
    chaos_weight: f64,
    equil_weight: f64,
    escape_threshold: f64,
    preferred_stack: render::LayerStack,
    projection: render::ProjectionMode,
) -> Result<AestheticSelection> {
    let stacks = adaptive_structure_stacks(preferred_stack);
    let mut global_best: Option<AestheticSelection> = None;

    for attempt in 0..=AESTHETIC_MAX_RETRIES {
        let (attempt_cw, attempt_ew, shortlist) = if attempt == 0 {
            info!("STAGE 1/7: Borda search over {} random orbits...", num_sims);
            (
                chaos_weight,
                equil_weight,
                sim::select_best_trajectory_shortlist(
                    rng,
                    num_sims,
                    num_steps_sim,
                    chaos_weight,
                    equil_weight,
                    escape_threshold,
                    AESTHETIC_SHORTLIST_LEN,
                )?,
            )
        } else {
            let domain = format!("{AESTHETIC_RETRY_RNG_DOMAIN_PREFIX}{attempt}");
            let mut retry_rng = rng.fork(domain.as_bytes());
            let (retry_cw, retry_ew) = retry_borda_weights(chaos_weight, &mut retry_rng);
            info!(
                "STAGE 1/7: Retry {attempt}/{AESTHETIC_MAX_RETRIES} over {num_sims} random orbits \
                 (chaos={retry_cw:.3}, equil={retry_ew:.3})..."
            );
            let shortlist = match sim::select_best_trajectory_shortlist(
                &mut retry_rng,
                num_sims,
                num_steps_sim,
                retry_cw,
                retry_ew,
                escape_threshold,
                AESTHETIC_SHORTLIST_LEN,
            ) {
                Ok(shortlist) => shortlist,
                Err(e) if global_best.is_some() => {
                    warn!("Aesthetic retry {attempt} produced no valid orbits: {e}");
                    continue;
                }
                Err(e) => return Err(e),
            };
            (retry_cw, retry_ew, shortlist)
        };

        info!(
            "STAGE 1.5/7: Aesthetic scoring of {} shortlisted candidate(s) across {} stack(s) [projection={}]: {}",
            shortlist.len(),
            stacks.len(),
            projection.label(),
            stacks.iter().map(render::LayerStack::label).collect::<Vec<_>>().join(", ")
        );
        let attempt_best = best_aesthetic_selection_from_shortlist(
            shortlist,
            num_steps_sim,
            preferred_stack,
            projection,
            attempt,
        );
        info!(
            "   => Attempt {attempt} winner: orbit idx {} stack={} score {:.4} (select {:.4}, chaos={attempt_cw:.3}, equil={attempt_ew:.3})",
            attempt_best.result.selected_index,
            attempt_best.stack.label(),
            attempt_best.aesthetic.total,
            attempt_best.selection_score,
        );

        let improves = global_best
            .as_ref()
            .is_none_or(|cur| attempt_best.selection_score > cur.selection_score);
        if improves {
            global_best = Some(attempt_best);
        }
        if global_best.as_ref().is_some_and(|best| best.aesthetic.total >= AESTHETIC_QUALITY_FLOOR)
        {
            break;
        }
    }

    let selection = global_best.expect("at least one Borda attempt must return candidates");
    info!(
        "   => Aesthetic winner: orbit idx {} stack={} preferred={} score {:.4} retries={}",
        selection.result.selected_index,
        selection.stack.label(),
        selection.preferred_stack.label(),
        selection.aesthetic.total,
        selection.retry_count,
    );
    Ok(selection)
}

/// Re-run the best orbit to get full trajectory
pub fn simulate_best_orbit(best_bodies: Vec<Body>, num_steps_sim: usize) -> Vec<Vec<Vector3<f64>>> {
    info!("STAGE 2/7: Re-running best orbit for {} steps...", num_steps_sim);
    let sim_result = sim::get_positions(best_bodies, num_steps_sim);
    info!("   => Done.");
    sim_result.positions
}

/// Number of seeded candidate viewing orientations evaluated per seed.
pub const VIEW_CANDIDATE_COUNT: usize = 4;

/// Build the rotation matrix for one Shoemake-uniform quaternion triple.
fn shoemake_rotation(u1: f64, u2: f64, u3: f64) -> Matrix3<f64> {
    let two_pi = crate::render::constants::TWO_PI;
    let (qx, qy) =
        ((1.0 - u1).sqrt() * (two_pi * u2).sin(), (1.0 - u1).sqrt() * (two_pi * u2).cos());
    let (qz, qw) = (u1.sqrt() * (two_pi * u3).sin(), u1.sqrt() * (two_pi * u3).cos());
    quaternion_to_matrix(qw, qx, qy, qz)
}

/// Rotate a strided sample of the trajectory (cheap copy for view scoring).
fn rotated_sample(
    positions: &[Vec<Vector3<f64>>],
    rotation: &Matrix3<f64>,
    samples_per_body: usize,
) -> Vec<Vec<Vector3<f64>>> {
    positions
        .iter()
        .map(|body| {
            let stride = (body.len() / samples_per_body.max(1)).max(1);
            body.iter().step_by(stride).map(|position| rotation * *position).collect()
        })
        .collect()
}

/// Rotate the whole trajectory by the best of several seeded 3D orientations.
///
/// The simulation produces fully three-dimensional structures, but the
/// renderer projects onto the fixed x/y plane. Without this step every seed
/// is photographed from the same axis. Instead of committing to a single
/// random angle, [`VIEW_CANDIDATE_COUNT`] Shoemake-uniform rotations are drawn
/// from the forked view RNG and each is scored on its projected 2D occupancy
/// (coverage / balance via [`render::aesthetic_score`]); the best-composed
/// viewpoint wins. Deterministic per seed.
///
/// Positions are already expressed in the centre-of-mass frame, so rotating
/// about the origin is rotation about the COM. Uses a forked RNG domain so it
/// does not perturb the main seed stream consumed by drift and colors.
pub fn apply_view_orientation(
    positions: &mut [Vec<Vector3<f64>>],
    rng: &Sha3RandomByteStream,
    stack: render::LayerStack,
) -> (f64, f64, f64) {
    info!(
        "STAGE 2.25/7: Selecting best of {} seeded viewing orientations...",
        VIEW_CANDIDATE_COUNT
    );
    let mut view_rng = rng.fork(VIEW_RNG_DOMAIN);
    let proxy_params = render::aesthetic_score::ProxyRenderParams::for_view_selection(stack);

    let mut best_triple = (0.0, 0.0, 0.0);
    let mut best_score = f64::NEG_INFINITY;
    for candidate in 0..VIEW_CANDIDATE_COUNT {
        let triple = (view_rng.next_f64(), view_rng.next_f64(), view_rng.next_f64());
        let rotation = shoemake_rotation(triple.0, triple.1, triple.2);
        let sample = rotated_sample(positions, &rotation, proxy_params.samples_per_body);
        let score = render::aesthetic_score::score_trajectory(&sample, proxy_params).total;
        info!(
            "   view candidate {candidate}: ({:.3}, {:.3}, {:.3}) occupancy score {score:.4}",
            triple.0, triple.1, triple.2
        );
        if score > best_score {
            best_score = score;
            best_triple = triple;
        }
    }

    let rotation = shoemake_rotation(best_triple.0, best_triple.1, best_triple.2);
    for body_positions in positions.iter_mut() {
        for position in body_positions.iter_mut() {
            *position = rotation * *position;
        }
    }

    info!(
        "   => View quaternion components: ({:.3}, {:.3}, {:.3}) score {best_score:.4}",
        best_triple.0, best_triple.1, best_triple.2
    );
    best_triple
}

fn quaternion_to_matrix(w: f64, x: f64, y: f64, z: f64) -> Matrix3<f64> {
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);

    Matrix3::new(
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy - wz),
        2.0 * (xz + wy),
        2.0 * (xy + wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz - wx),
        2.0 * (xz - wy),
        2.0 * (yz + wx),
        1.0 - 2.0 * (xx + yy),
    )
}

/// Apply drift transformation to positions
pub fn apply_drift_transformation(
    positions: &mut [Vec<Vector3<f64>>],
    drift_mode: &str,
    drift_scale: Option<f64>,
    drift_arc_fraction: Option<f64>,
    drift_orbit_eccentricity: Option<f64>,
    rng: &mut Sha3RandomByteStream,
) -> Result<Option<ResolvedDriftConfig>> {
    info!("STAGE 2.5/7: Resolving drift configuration...");

    let resolved =
        resolve_drift_config(drift_scale, drift_arc_fraction, drift_orbit_eccentricity, rng)?;

    info!("Applying {} drift...", drift_mode);
    let num_steps = positions[0].len();
    let drift_params = resolved.to_drift_parameters();

    if crate::utils::is_zero(drift_params.arc_fraction)
        && drift_mode.to_lowercase().starts_with("ell")
    {
        warn!("Elliptical drift requested with zero arc fraction; skipping motion");
    }

    let mut drift_transform = parse_drift_mode(drift_mode, rng, drift_params, num_steps)?;
    drift_transform.apply(positions, constants::DEFAULT_DT);

    info!("   => Drift applied successfully");
    Ok(Some(resolved))
}

/// Generate color sequences and alpha values for bodies
pub fn generate_colors(
    rng: &mut Sha3RandomByteStream,
    num_steps_sim: usize,
    alpha_denom: usize,
    enhancements: &Enhancements,
    palette_phase: f64,
) -> (Vec<Vec<render::OklabColor>>, Vec<f64>) {
    info!("STAGE 3/7: Generating color sequences + alpha...");
    generate_body_color_sequences(
        rng,
        num_steps_sim,
        alpha_denom,
        enhancements.chroma_boost,
        enhancements.alpha_variation,
        palette_phase,
    )
}

/// Build histogram and determine color levels.
///
/// `profile` supplies the seed's scene traits (the histogram must sample the
/// same structure mode that pass 2 renders) and the display exposure key:
/// values below 1 produce darker, ember-like seeds and values above 1
/// brighter, airier seeds.
pub fn build_histogram_and_levels(
    positions: &[Vec<Vector3<f64>>],
    colors: &[Vec<render::OklabColor>],
    body_alphas: &[f64],
    resolved_config: &render::randomizable_config::ResolvedEffectConfig,
    render_config: &RenderConfig,
    aspect_correction: bool,
    profile: &render::visual_profile::CosmicSignatureParameters,
) -> Result<ChannelLevels> {
    info!("STAGE 5/7: PASS 1 => building global histogram...");

    let target_frames = constants::DEFAULT_HISTOGRAM_SAMPLE_FRAMES;
    let frame_interval = (positions[0].len() / target_frames as usize).max(1);

    let histogram = pass_1_build_histogram_spectral(
        SpectralScene::new(positions, colors, body_alphas),
        frame_interval,
        SpectralRenderSettings::new(resolved_config, render_config, aspect_correction)
            .with_traits(profile.scene_traits()),
    );

    info!("STAGE 6/7: Determine global black/white/gamma...");
    let analysis = render::histogram::analyze_tonemapping(
        histogram.data(),
        resolved_config.clip_black,
        resolved_config.clip_white,
    );

    let exposure_key = profile.exposure_key;
    let keyed_exposure = analysis.exposure_scale * exposure_key.clamp(0.5, 1.5);
    info!(
        "   => R:[{:.3e},{:.3e}] G:[{:.3e},{:.3e}] B:[{:.3e},{:.3e}] exposure={:.3} key={:.3} near_clip={:.3}%",
        analysis.black_r,
        analysis.white_r,
        analysis.black_g,
        analysis.white_g,
        analysis.black_b,
        analysis.white_b,
        keyed_exposure,
        exposure_key,
        analysis.near_clip_ratio * constants::PERCENT_FACTOR
    );

    Ok(ChannelLevels::with_tone_mapping(
        analysis.black_r,
        analysis.white_r,
        analysis.black_g,
        analysis.white_g,
        analysis.black_b,
        analysis.white_b,
        ToneMappingControls {
            exposure_scale: keyed_exposure,
            paper_white: constants::DEFAULT_TONEMAP_PAPER_WHITE,
            highlight_rolloff: constants::DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF,
        },
    ))
}

/// Render full video, returning the fully accumulated SPD buffer for spectral outputs.
pub fn render_video(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_vid: &str,
    output_png: &str,
    fast_encode: bool,
    enable_temporal_smoothing: bool,
) -> Result<Vec<[f64; crate::spectrum::NUM_BINS]>> {
    if fast_encode {
        info!("STAGE 7/7: PASS 2 => final frames => video (FAST ENCODE MODE)...");
    } else {
        info!("STAGE 7/7: PASS 2 => final frames => video (HIGH QUALITY MODE)...");
    }

    let frame_rate = constants::DEFAULT_VIDEO_FPS;
    let target_frames = constants::DEFAULT_TARGET_FRAMES;
    let frame_interval = (scene.step_count() / target_frames as usize).max(1);

    let mut last_frame_png: Option<ImageBuffer<Rgb<u16>, Vec<u16>>> = None;
    let video_options = if fast_encode {
        VideoEncodingOptions::fast_encode()
    } else {
        VideoEncodingOptions::default()
    };

    let mut accum_spd = Vec::new();

    create_video_from_frames_singlepass(
        settings.resolved_config.width,
        settings.resolved_config.height,
        frame_rate,
        |out| {
            pass_2_write_frames_spectral(
                render::Pass2Params {
                    scene,
                    frame_interval,
                    levels,
                    settings,
                    last_frame_out: &mut last_frame_png,
                    enable_temporal_smoothing,
                    accum_spd: &mut accum_spd,
                },
                |buf_8bit| {
                    out.write_all(buf_8bit).map_err(render::error::RenderError::VideoEncoding)?;
                    Ok(())
                },
            )?;
            Ok(())
        },
        output_vid,
        &video_options,
    )?;

    if let Some(frame) = last_frame_png {
        info!("Saving still image from final video frame: {}", output_png);
        save_image_as_png_16bit(&frame, output_png)?;
    } else {
        warn!("Warning: No final frame was generated to save as PNG.");
    }

    Ok(accum_spd)
}

/// Render only the fully accumulated still image, skipping all video outputs.
///
/// This is the fast-iteration path for parameter-tuning batches: the still is
/// identical in composition to the final video frame but avoids encoding the
/// trajectory video and the spectral sweep.
pub fn render_still_image(
    scene: SpectralScene<'_>,
    levels: &ChannelLevels,
    settings: SpectralRenderSettings<'_>,
    output_png: &str,
) -> Result<()> {
    info!("STAGE 7/7: PASS 2 => final still only (IMAGE-ONLY MODE)...");
    let frame = render::render_final_frame_spectral(scene, levels, settings)?;
    info!("Saving still image: {}", output_png);
    Ok(save_image_as_png_16bit(&frame, output_png)?)
}

/// Generate the spectral gallery: 64 per-bin 16-bit PNGs in `spectral_dir`.
pub fn generate_spectral_gallery(
    accum_spd: &[[f64; crate::spectrum::NUM_BINS]],
    width: u32,
    height: u32,
    spectral_dir: &str,
) -> Result<()> {
    Ok(render::spectral_output::generate_spectral_gallery(accum_spd, width, height, spectral_dir)?)
}

/// Generate the spectral sweep video (violet-to-red-to-violet cycle) at `output_path`.
pub fn generate_spectral_sweep_video(
    accum_spd: &[[f64; crate::spectrum::NUM_BINS]],
    width: u32,
    height: u32,
    output_path: &str,
    fast_encode: bool,
) -> Result<()> {
    Ok(render::spectral_output::generate_spectral_sweep_video(
        accum_spd,
        width,
        height,
        output_path,
        fast_encode,
    )?)
}

/// Log generation parameters for reproducibility.
///
/// Returns `Ok(())` on success, or an `Err` describing the I/O or serialisation failure.
pub fn log_generation(
    config: &GenerationLogConfig,
    file_name: &str,
    seed: &str,
    drift_config: &Option<ResolvedDriftConfig>,
    num_sims: usize,
    selection: &AestheticSelection,
    randomization_log: Option<&render::effect_randomizer::RandomizationLog>,
) -> Result<()> {
    let logger = GenerationLogger::new();

    let mut record = GenerationRecord::new(file_name.to_string(), format!("0x{seed}"));

    record.render_config = LoggedRenderConfig {
        width: config.width,
        height: config.height,
        clip_black: config.clip_black,
        clip_white: config.clip_white,
        alpha_denom: config.alpha_denom,
        alpha_compress: config.alpha_compress,
        visual_profile: config.visual_profile.clone(),
        post_effects_enabled: config.post_effects_enabled,
        bloom_mode: config.bloom_mode.clone(),
        hdr_mode: config.hdr_mode.clone(),
        hdr_scale: config.hdr_scale,
        dispersion_strength: config.dispersion_strength,
        dispersion_mode: config.dispersion_mode.clone(),
        palette_fingerprint: config.palette_fingerprint.clone(),
        palette_gate: config.palette_gate.clone(),
    };

    record.drift_config = if let Some(drift) = drift_config {
        DriftConfig {
            enabled: true,
            mode: config.drift_mode.clone(),
            scale: drift.scale,
            arc_fraction: drift.arc_fraction,
            orbit_eccentricity: drift.orbit_eccentricity,
            randomized: drift.was_randomized,
        }
    } else {
        DriftConfig {
            enabled: false,
            mode: "none".to_string(),
            scale: 0.0,
            arc_fraction: 0.0,
            orbit_eccentricity: 0.0,
            randomized: false,
        }
    };

    record.simulation_config = SimulationConfig {
        num_sims,
        num_steps_sim: config.num_steps_sim,
        location: config.location,
        velocity: config.velocity,
        min_mass: config.min_mass,
        max_mass: config.max_mass,
        chaos_weight: config.chaos_weight,
        equil_weight: config.equil_weight,
        escape_threshold: config.escape_threshold,
        weights_randomized: config.weights_randomized,
    };

    let best_info = &selection.result;
    let score = selection.aesthetic;
    record.orbit_info = OrbitInfo {
        selected_index: best_info.selected_index,
        weighted_score: best_info.total_score_weighted,
        total_candidates: num_sims * (selection.retry_count + 1),
        discarded_count: best_info.discarded_count,
        preferred_structure: selection.preferred_stack.label(),
        chosen_structure: selection.stack.label(),
        retry_count: selection.retry_count,
        aesthetic_score: score.total,
        selection_score: selection.selection_score,
        coverage: score.coverage,
        balance: score.balance,
        contrast: score.contrast,
        body_mix: score.body_mix,
        mush_fraction: score.mush_fraction,
        veil_fraction: score.veil_fraction,
        crispness: score.crispness,
        negative_space: score.negative_space,
        fullness: score.fullness,
    };

    // Include randomization log if provided
    record.randomization_log = randomization_log.cloned();

    logger.log_generation(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_seed_valid() {
        let result = parse_seed("0x100033");
        assert!(result.is_ok());

        let bytes = result.expect("hex bytes should parse");
        assert_eq!(bytes, vec![0x10, 0x00, 0x33]);
    }

    #[test]
    fn test_parse_seed_no_prefix() {
        let result = parse_seed("100033");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_seed_invalid() {
        let result = parse_seed("0xZZZ");
        assert!(result.is_err());
    }

    #[test]
    fn test_enhancements_default_quality_profile() {
        let e = Enhancements::default();
        assert!(e.chroma_boost);
        assert!(e.sat_boost);
        assert!(e.aces_tweak);
        assert!(e.alpha_variation);
        assert!(e.aspect_correction);
        assert!(!e.dispersion_boost);
    }

    #[test]
    fn test_enhancements_selective_disable() {
        let e = Enhancements {
            chroma_boost: false,
            sat_boost: true,
            aces_tweak: false,
            alpha_variation: true,
            aspect_correction: false,
            dispersion_boost: true,
        };
        assert!(!e.chroma_boost);
        assert!(e.sat_boost);
        assert!(!e.aces_tweak);
        assert!(e.alpha_variation);
        assert!(!e.aspect_correction);
        assert!(e.dispersion_boost);
    }

    #[test]
    fn test_generate_colors_with_enhancements() {
        use crate::sim::Sha3RandomByteStream;
        let mut rng = Sha3RandomByteStream::new(&[1, 2, 3, 4], 100.0, 300.0, 300.0, 1.0);
        let enhancements = Enhancements::default();
        let (colors, alphas) = generate_colors(&mut rng, 100, 15_000_000, &enhancements, 0.5);

        assert_eq!(colors.len(), 3);
        assert_eq!(alphas.len(), 3);
        for body_colors in &colors {
            assert_eq!(body_colors.len(), 100);
        }
        let unique: std::collections::HashSet<u64> = alphas.iter().map(|a| a.to_bits()).collect();
        assert!(unique.len() > 1, "default enhancements should enable alpha variation");
    }

    #[test]
    fn test_generate_colors_no_enhancements() {
        use crate::sim::Sha3RandomByteStream;
        let mut rng = Sha3RandomByteStream::new(&[1, 2, 3, 4], 100.0, 300.0, 300.0, 1.0);
        let enhancements =
            Enhancements { alpha_variation: false, chroma_boost: false, ..Enhancements::default() };
        let (colors, alphas) = generate_colors(&mut rng, 100, 15_000_000, &enhancements, 0.5);

        assert_eq!(colors.len(), 3);
        assert_eq!(alphas[0], alphas[1]);
        assert_eq!(alphas[1], alphas[2]);
    }

    /// Run the full seed-to-pixels pipeline at minimal scale and return the
    /// raw 16-bit pixel buffer.  Two calls with the same seed MUST return
    /// bitwise-identical buffers on the same architecture.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum PipelineRenderMode {
        DefaultParallel,
        SerialReference,
    }

    fn assert_pixel_buffers_eq(actual: &[u16], expected: &[u16], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: pixel buffer lengths differ");

        if actual != expected {
            let diff_count = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
            let first = actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .expect("expected differing pixel position");
            panic!(
                "{label}: pixel buffers differ: {diff_count} of {} values, \
                 first at index {first} ({} vs {})",
                actual.len(),
                actual[first],
                expected[first],
            );
        }
    }

    fn run_full_pipeline(seed: &[u8], mode: PipelineRenderMode) -> Vec<u16> {
        use crate::sim::Sha3RandomByteStream;

        let width = 64u32;
        let height = 36u32;
        let num_sims = 20;
        let num_steps = 5_000;

        let mut rng = Sha3RandomByteStream::new(seed, 100.0, 300.0, 300.0, 1.0);

        let config = render::randomizable_config::RandomizableEffectConfig {
            enable_bloom: Some(false),
            enable_glow: Some(false),
            enable_chromatic_bloom: Some(false),
            enable_perceptual_blur: Some(false),
            enable_micro_contrast: Some(false),
            enable_gradient_map: Some(false),
            enable_color_grade: Some(false),
            enable_champleve: Some(false),
            enable_aether: Some(false),
            enable_opalescence: Some(false),
            enable_edge_luminance: Some(false),
            enable_atmospheric_depth: Some(false),
            enable_fine_texture: Some(false),
            ..Default::default()
        };
        let (resolved, _) = config.resolve(&mut rng, width, height);

        let (best_bodies, _) =
            crate::sim::select_best_trajectory(&mut rng, num_sims, num_steps, 0.75, 11.0, -0.3)
                .expect("Borda search should find at least one valid orbit");

        let mut positions = simulate_best_orbit(best_bodies, num_steps);

        apply_drift_transformation(&mut positions, "elliptical", None, None, None, &mut rng)
            .expect("drift config resolution should succeed with all-None args");

        let enhancements = Enhancements {
            chroma_boost: false,
            sat_boost: false,
            aces_tweak: false,
            alpha_variation: false,
            aspect_correction: false,
            dispersion_boost: false,
        };
        let (colors, body_alphas) =
            generate_colors(&mut rng, num_steps, 15_000_000, &enhancements, 0.5);

        let render_config =
            render::RenderConfig { hdr_scale: resolved.hdr_scale, ..Default::default() };
        let scene = SpectralScene::new(&positions, &colors, &body_alphas);
        let settings = SpectralRenderSettings::new(&resolved, &render_config, false);
        let frame_interval = (scene.step_count()
            / render::constants::DEFAULT_HISTOGRAM_SAMPLE_FRAMES as usize)
            .max(1);
        let histogram = match mode {
            PipelineRenderMode::DefaultParallel => {
                render::pass_1_build_histogram_spectral(scene, frame_interval, settings)
            }
            PipelineRenderMode::SerialReference => {
                render::pass_1_build_histogram_spectral_serial_reference(
                    scene,
                    frame_interval,
                    settings,
                )
            }
        };
        let analysis = render::histogram::analyze_tonemapping(
            histogram.data(),
            resolved.clip_black,
            resolved.clip_white,
        );
        let levels = render::ChannelLevels::with_tone_mapping(
            analysis.black_r,
            analysis.white_r,
            analysis.black_g,
            analysis.white_g,
            analysis.black_b,
            analysis.white_b,
            render::ToneMappingControls {
                exposure_scale: analysis.exposure_scale,
                paper_white: render::constants::DEFAULT_TONEMAP_PAPER_WHITE,
                highlight_rolloff: render::constants::DEFAULT_TONEMAP_HIGHLIGHT_ROLLOFF,
            },
        );

        let image = match mode {
            PipelineRenderMode::DefaultParallel => {
                render::render_single_frame_spectral(scene, &levels, settings)
            }
            PipelineRenderMode::SerialReference => {
                render::render_single_frame_spectral_serial_reference(scene, &levels, settings)
            }
        }
        .expect("render should succeed");

        image.into_raw()
    }

    #[test]
    fn test_end_to_end_pipeline_determinism() {
        for seed in [[0xCA, 0xFE], [0xBE, 0xEF], [0x12, 0x34]] {
            let pixels_a = run_full_pipeline(&seed, PipelineRenderMode::DefaultParallel);
            let pixels_b = run_full_pipeline(&seed, PipelineRenderMode::DefaultParallel);
            assert_pixel_buffers_eq(&pixels_a, &pixels_b, &format!("default_parallel/{seed:02X?}"));
        }
    }

    #[test]
    fn test_setup_seed_directory_returns_correct_path() {
        let result = setup_seed_directory("test_seed_42");
        assert!(result.is_ok());
        let seed_dir = result.expect("seed directory setup should succeed");
        assert_eq!(seed_dir, "output/test_seed_42");
        assert!(std::path::Path::new("output/test_seed_42").is_dir());
        assert!(std::path::Path::new("output/test_seed_42/spectral").is_dir());
        let _ = fs::remove_dir_all("output/test_seed_42");
    }

    #[test]
    fn test_setup_seed_directory_idempotent() {
        let r1 = setup_seed_directory("seed_idem");
        let r2 = setup_seed_directory("seed_idem");
        assert!(r1.is_ok());
        assert!(r2.is_ok());
        let _ = fs::remove_dir_all("output/seed_idem");
    }

    proptest::proptest! {
        #[test]
        fn proptest_parse_seed_never_panics(input in "\\PC*") {
            let _ = parse_seed(&input);
        }
    }

    fn projection_fixture() -> Vec<Vec<Vector3<f64>>> {
        (0..3)
            .map(|body| {
                let phase = f64::from(body as u32) * 2.1;
                (0..256)
                    .map(|step| {
                        let t = f64::from(step) * 0.05;
                        Vector3::new(
                            (t + phase).cos() * (1.0 + 0.3 * f64::from(body as u32)),
                            (t * 1.3 + phase).sin(),
                            0.2 * (t * 0.7).sin(),
                        )
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_position_projection_is_identity() {
        let positions = projection_fixture();
        let projected = apply_projection(&positions, render::ProjectionMode::Position);
        assert_eq!(projected, positions);
    }

    #[test]
    fn test_phase_projections_are_finite_deterministic_and_distinct() {
        let positions = projection_fixture();
        for projection in [
            render::ProjectionMode::PhasePortrait,
            render::ProjectionMode::CrossBraid,
            render::ProjectionMode::Hodograph,
        ] {
            let a = apply_projection(&positions, projection);
            let b = apply_projection(&positions, projection);
            assert_eq!(a, b, "{projection:?} must be deterministic");

            assert_eq!(a.len(), positions.len());
            let mut differs = false;
            for (body_idx, body) in a.iter().enumerate() {
                assert_eq!(body.len(), positions[body_idx].len());
                for (step, p) in body.iter().enumerate() {
                    assert!(
                        p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
                        "{projection:?} body {body_idx} step {step} not finite: {p:?}"
                    );
                    if (p - positions[body_idx][step]).norm() > 1e-9 {
                        differs = true;
                    }
                }
            }
            assert!(differs, "{projection:?} must differ from position space");
        }
    }

    #[test]
    fn test_phase_projection_velocity_axes_match_position_scale() {
        // The velocity axis of the phase portrait must be rescaled into the
        // same magnitude band as the position axes, or the composition would
        // collapse onto a line.
        let positions = projection_fixture();
        let projected = apply_projection(&positions, render::ProjectionMode::PhasePortrait);

        let extent = |data: &[Vec<Vector3<f64>>], axis: usize| {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for body in data {
                for p in body {
                    min = min.min(p[axis]);
                    max = max.max(p[axis]);
                }
            }
            max - min
        };
        let x_extent = extent(&projected, 0);
        let v_extent = extent(&projected, 1);
        assert!(
            v_extent > x_extent * 0.05 && v_extent < x_extent * 20.0,
            "velocity axis badly scaled: x={x_extent} v={v_extent}"
        );
    }

    #[test]
    fn test_projection_handles_degenerate_inputs() {
        let empty: Vec<Vec<Vector3<f64>>> = vec![Vec::new(), Vec::new(), Vec::new()];
        let projected = apply_projection(&empty, render::ProjectionMode::Hodograph);
        assert_eq!(projected.len(), 3);

        let single = vec![
            vec![Vector3::new(1.0, 2.0, 3.0)],
            vec![Vector3::new(-1.0, 0.0, 1.0)],
            vec![Vector3::new(0.0, 1.0, -1.0)],
        ];
        let projected = apply_projection(&single, render::ProjectionMode::PhasePortrait);
        assert_eq!(projected, single, "single-step trajectories pass through unchanged");
    }

    #[test]
    fn test_adaptive_stacks_include_seed_stack_and_fallbacks() {
        let preferred = render::LayerStack::with_underlay(
            render::StructureMode::NebulaVeil,
            render::StructureMode::StippleConstellation,
            0.25,
        );
        let stacks = adaptive_structure_stacks(preferred);

        assert_eq!(stacks[0], preferred, "seed stack must be evaluated first");
        assert!(
            stacks.contains(&render::LayerStack::solo(render::StructureMode::NebulaVeil)),
            "primary-only variant must be a fallback"
        );
        assert!(
            stacks.contains(&render::LayerStack::solo(render::StructureMode::OrbitRibbons)),
            "ribbons must be a reliable fallback"
        );
        assert!(stacks.len() <= 5, "adaptive evaluation must stay bounded: {}", stacks.len());

        // No duplicates.
        for (i, stack) in stacks.iter().enumerate() {
            for other in &stacks[i + 1..] {
                assert_ne!(stack, other, "duplicate stack in adaptive list");
            }
        }
    }

    #[test]
    fn test_mode_selection_prior_protects_new_vocabularies() {
        let veil = render::LayerStack::solo(render::StructureMode::NebulaVeil);
        let web = render::LayerStack::solo(render::StructureMode::TriangleWeb);
        let ribbons = render::LayerStack::solo(render::StructureMode::OrbitRibbons);

        assert!(
            mode_selection_prior(veil, veil) > mode_selection_prior(web, web),
            "novel vocabularies need a stronger identity prior than the classic web"
        );
        assert!(
            mode_selection_prior(veil, veil) > mode_selection_prior(ribbons, veil),
            "the seed's own stack must outrank fallbacks at equal aesthetic score"
        );
    }

    #[test]
    fn test_view_orientation_is_deterministic_and_rigid() {
        let make_positions = || -> Vec<Vec<Vector3<f64>>> {
            vec![
                vec![Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0)],
                vec![Vector3::new(-1.0, 0.5, 2.0), Vector3::new(0.0, -3.0, 1.0)],
                vec![Vector3::new(7.0, -2.0, 0.0), Vector3::new(-4.0, 1.0, -1.0)],
            ]
        };

        let rng = Sha3RandomByteStream::new(&[0xAB, 0xCD], 100.0, 300.0, 300.0, 1.0);
        let mut a = make_positions();
        let mut b = make_positions();
        let original = make_positions();

        let stack = render::LayerStack::solo(render::StructureMode::TriangleWeb);
        apply_view_orientation(&mut a, &rng, stack);
        apply_view_orientation(&mut b, &rng, stack);

        for body in 0..3 {
            for step in 0..2 {
                // Deterministic: identical results across calls with the same seed.
                assert_eq!(a[body][step], b[body][step], "body {body} step {step} diverged");
                // Rigid: rotation preserves distance from the origin (COM).
                let norm_before = original[body][step].norm();
                let norm_after = a[body][step].norm();
                assert!(
                    (norm_before - norm_after).abs() < 1e-9,
                    "rotation must preserve norms: {norm_before} vs {norm_after}"
                );
            }
        }

        // The orientation should actually rotate (not be the identity).
        let moved = (0..3).any(|body| (a[body][0] - original[body][0]).norm() > 1e-6);
        assert!(moved, "seeded view orientation should differ from identity");

        // A different seed must produce a different orientation.
        let rng2 = Sha3RandomByteStream::new(&[0x11, 0x22], 100.0, 300.0, 300.0, 1.0);
        let mut c = make_positions();
        apply_view_orientation(&mut c, &rng2, stack);
        let differs = (0..3).any(|body| (a[body][0] - c[body][0]).norm() > 1e-6);
        assert!(differs, "different seeds should view from different angles");
    }

    #[test]
    fn test_end_to_end_pipeline_parallel_matches_serial_reference() {
        for seed in [[0xCA, 0xFE], [0xBE, 0xEF], [0x12, 0x34]] {
            let parallel_pixels = run_full_pipeline(&seed, PipelineRenderMode::DefaultParallel);
            let serial_pixels = run_full_pipeline(&seed, PipelineRenderMode::SerialReference);
            assert_pixel_buffers_eq(
                &parallel_pixels,
                &serial_pixels,
                &format!("parallel_vs_serial_reference/{seed:02X?}"),
            );
        }
    }
}
