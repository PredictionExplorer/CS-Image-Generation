//! V67 `vanitas` -- The Life and Death of an Artwork.
//!
//! Four seasons and a death in one continuous 120 s film, the acts handing
//! off *organically* inside one shared frame buffer (no cuts to hide the
//! seams -- the spec's compositor assembly is unnecessary because every act
//! inherits the previous act's pixels):
//!
//! - SPRING: the artwork accumulates under V22's drama-retimed schedule.
//! - SUMMER: accumulation continues at full brilliance; V35's dielectric
//!   bolts strike at the close approaches crossed by the schedule, their
//!   afterglow burned into the artwork; exposure swells +8%.
//! - AUTUMN: V33's Physarum feeds on the *final* energy field, its scouts
//!   seeded at the last bolt's endpoints (handoff one); the artwork ghost
//!   dims 100 -> 45% and loses 20% chroma.
//! - WINTER: V34's DLA frost grows from the vein junctions (handoff two),
//!   revealed in stick order while the whole frame drifts cold (`OkLab` b
//!   -0.03, L -10%).
//! - DEATH: a single Ken-Burns pull on the frosted composite, a fade to
//!   black, three seconds of silence, then the small end card.
//!
//! The score arcs bright -> sparse -> shimmer -> the strain's last whisper,
//! then nothing.

use crate::error::Result;
use crate::oklab::{
    linear_rec2020_to_oklab, oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab,
};
use crate::render::constants::DEFAULT_DT;
use crate::render::context::{PixelBuffer, RenderContext};
use crate::render::effects::convert_spd_buffer_to_rgba;
use crate::render::velocity_hdr::VelocityHdrCalculator;
use crate::render::{
    AccumulationParams, SpectralScene, accumulate_spectral_steps, default_accumulation_backend,
};
use crate::spectrum::NUM_BINS;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::agents::{
    DlaGrid, DlaParams, PhysarumParams, PhysarumSwarm, diffuse_decay,
};
use crate::viz::common::audio::{
    self, SAMPLE_RATE, fade_ends, fm_bell, mux, normalize_to_lufs, saw_bandlimited, soft_limit,
    write_wav_stereo_24bit,
};
use crate::viz::common::display::SpdCanvas;
use crate::viz::common::style::{Paper, ink_color, paper_color, type_px};
use crate::viz::common::text::{Align, Face, TextStyle, draw_text};
use crate::viz::context::VizContext;
use crate::viz::modes::editorial_retime::build_schedule;
use crate::viz::modes::gw_chirp::strain_series;
use crate::viz::modes::lightning::{Bolt, draw_bolt, grow_bolts};
use crate::viz::modes::physarum::energy_density_grid;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Film length in seconds at 30 fps.
const FILM_SECONDS: f64 = 120.0;
/// Frame rate.
const FPS: u32 = 30;
/// Act boundaries in seconds: spring, summer, autumn, winter ends.
const ACT_ENDS: [f64; 4] = [30.0, 55.0, 85.0, 110.0];
/// Summer exposure swell.
const SUMMER_SWELL: f64 = 1.08;
/// Ghost floor by the end of autumn.
const GHOST_FLOOR: f64 = 0.45;
/// Autumn chroma loss.
const CHROMA_LOSS: f64 = 0.20;
/// Winter cold shift: `OkLab` b drift and lightness factor at full winter.
const COLD_B_DRIFT: f64 = -0.03;
const COLD_L_FACTOR: f64 = 0.90;
/// Bolts considered during summer.
const MAX_BOLTS: usize = 24;
/// Physarum agents at final quality.
const AGENT_COUNT: usize = 900_000;
/// Physarum ticks per frame.
const TICKS_PER_FRAME: usize = 2;
/// DLA walker budget at final quality.
const WALKER_BUDGET: usize = 500_000;
/// Vein junction cap (winter nucleation sites).
const MAX_JUNCTIONS: usize = 64;

/// The vanitas mode.
pub struct Vanitas;

/// The winter cold shift in `OkLab`: `b` drifts by `progress * COLD_B_DRIFT`
/// and lightness scales toward `COLD_L_FACTOR`. Exact and monotone.
pub(crate) fn cold_shift(color: (f64, f64, f64), progress: f64) -> (f64, f64, f64) {
    let (l, a, b) = linear_rec2020_to_oklab(color.0.max(0.0), color.1.max(0.0), color.2.max(0.0));
    let factor = 1.0 - (1.0 - COLD_L_FACTOR) * progress;
    let shifted = oklab_to_linear_rec2020(l * factor, a, b + COLD_B_DRIFT * progress);
    (shifted.0.max(0.0), shifted.1.max(0.0), shifted.2.max(0.0))
}

/// Mean `OkLab` `b` of a linear RGBA buffer (the color-temperature audit).
pub(crate) fn mean_oklab_b(rgba: &[(f64, f64, f64, f64)]) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0.0f64;
    for &(r, g, b, _) in rgba.iter().step_by(97) {
        let (_, _, ob) = linear_rec2020_to_oklab(r.max(0.0), g.max(0.0), b.max(0.0));
        sum += ob;
        count += 1.0;
    }
    sum / count.max(1.0)
}

/// Extract up to `cap` spaced vein junctions from a Physarum trail grid:
/// the strongest trail cells, greedily separated (handoff two's nuclei).
pub(crate) fn vein_junctions(
    trail: &[f32],
    width: usize,
    height: usize,
    spacing: f64,
    cap: usize,
) -> Vec<(usize, usize)> {
    let mut ranked: Vec<(f32, usize)> = trail
        .iter()
        .enumerate()
        .filter(|&(_, &value)| value > 0.0)
        .map(|(index, &value)| (value, index))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut junctions: Vec<(usize, usize)> = Vec::new();
    for &(_, index) in &ranked {
        if junctions.len() >= cap {
            break;
        }
        let col = index % width;
        let row = index / width;
        let spaced = junctions.iter().all(|&(jc, jr)| {
            let dx = jc as f64 - col as f64;
            let dy = jr as f64 - row as f64;
            dx.hypot(dy) >= spacing
        });
        if spaced {
            junctions.push((col, row));
        }
    }
    if junctions.is_empty() {
        junctions.push((width / 2, height / 2));
    }
    junctions
}

impl VizMode for Vanitas {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("vanitas").expect("vanitas is in the catalog")
    }

    fn needs_energy_field(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 1_000 {
            warn!("vanitas skipped: trajectory too short");
            return Ok(());
        }
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(16);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(16);
        let cells = video_w as usize * video_h as usize;
        let frame_count = ctx.quality.scale_count((FILM_SECONDS * f64::from(FPS)) as usize);
        let frame_of =
            |seconds: f64| -> usize { ((seconds / FILM_SECONDS) * frame_count as f64) as usize };
        let spring_end = frame_of(ACT_ENDS[0]);
        let summer_end = frame_of(ACT_ENDS[1]);
        let autumn_end = frame_of(ACT_ENDS[2]);
        let winter_end = frame_of(ACT_ENDS[3]);

        // --- SPRING + SUMMER: drama-retimed accumulation (V22's schedule
        // over the two acts) with the master framing and levels.
        let drama = ctx.events().drama(ctx.kinematics());
        let schedule = build_schedule(&drama, summer_end.max(2));
        let render_ctx =
            RenderContext::new(video_w, video_h, ctx.positions, ctx.settings.aspect_correction);
        let mut spd: Vec<[f64; NUM_BINS]> = vec![[0.0; NUM_BINS]; cells];
        let velocity_calc = VelocityHdrCalculator::new(ctx.positions, DEFAULT_DT);
        let traits = ctx.settings.traits;
        let hdr_scale = ctx.settings.render_config.hdr_scale;

        // --- SUMMER's storms: upstream RNG domain, endpoints kept for
        // AUTUMN's scouts (handoff one).
        let mut bolt_rng = ctx.fork_rng("lightning");
        let mut bolts: Vec<Bolt> = grow_bolts(ctx, &mut bolt_rng, MAX_BOLTS);
        bolts.sort_by_key(|bolt| bolt.step);
        let summer_start_step = schedule[spring_end.min(schedule.len() - 1)];
        let summer_bolts: Vec<&Bolt> =
            bolts.iter().filter(|bolt| bolt.step >= summer_start_step).collect();
        let last_endpoints: Vec<(f32, f32)> =
            summer_bolts.last().copied().or(bolts.last()).map(Bolt::endpoints).unwrap_or_default();
        info!("   vanitas: {} bolts total, {} strike in summer", bolts.len(), summer_bolts.len());

        // Bolt strike frames: where the schedule crosses each bolt's step.
        let strike_frame_of = |step: usize| -> usize {
            schedule
                .iter()
                .position(|&cursor| cursor >= step)
                .unwrap_or(summer_end.saturating_sub(1))
                .max(spring_end)
        };
        let video_scale = f64::from(video_w) / f64::from(ctx.width);
        let strikes: Vec<(usize, usize)> = summer_bolts
            .iter()
            .enumerate()
            .map(|(index, bolt)| (strike_frame_of(bolt.step), index))
            .collect();

        // --- Persistent state across acts.
        let mut flash = SpdCanvas::new(video_w, video_h);
        let mut flash_rgba: PixelBuffer = Vec::new();
        let mut cursor = 0usize;
        let mut summer_final: PixelBuffer = Vec::new();
        let mut autumn_final: PixelBuffer = Vec::new();
        let mut winter_final: PixelBuffer = Vec::new();
        // Overlay exposure reference: the artwork's own p95 luminance, so
        // veins and frost stay in scale with the master grading.
        let mut overlay_ref = 1.0f64;
        // Physarum state (built at the autumn boundary).
        let mut swarm: Option<PhysarumSwarm> = None;
        let mut trail = vec![0.0f32; cells];
        let mut trail_blurred = vec![0.0f32; cells];
        let mut trail_scratch: Vec<f32> = Vec::new();
        let mut deposits: Vec<(u32, f32)> = Vec::new();
        let mut food: Vec<f32> = Vec::new();
        let mut trail_norm = 1.0f32;
        let mut agent_rng = ctx.fork_rng("physarum");
        let physarum_params = PhysarumParams {
            sense_distance: (9.0 * video_w.min(video_h) as f32 / 1117.0).max(3.0),
            sense_angle: std::f32::consts::FRAC_PI_8,
            turn_angle: std::f32::consts::FRAC_PI_8,
            step_length: 1.0,
            deposit: 0.06,
            decay: 0.94,
            jitter: 0.12,
            food_weight: 0.35,
        };
        let vein_ink = {
            let (l, a, b) = ctx.mean_color(0);
            let (_, _, hue) = oklab_to_oklch(l, a, b);
            let (ll, aa, bb) = oklch_to_oklab(0.78, 0.06, hue);
            let rgb = oklab_to_linear_rec2020(ll, aa, bb);
            (rgb.0.max(0.0), rgb.1.max(0.0), rgb.2.max(0.0))
        };
        // Frost state (built at the winter boundary).
        let mut frost: Option<DlaGrid> = None;
        let mut junctions: Vec<(usize, usize)> = Vec::new();
        let mut frost_rng = ctx.fork_rng("frost");
        let mut mean_b_track: Vec<(usize, f64)> = Vec::new();

        let started = std::time::Instant::now();
        let mut logged = false;
        crate::viz::common::accum::stream_video(
            video_w,
            video_h,
            FPS,
            &sink.path("vanitas_silent.mp4"),
            &sink.path("vanitas_silent_hq.mp4"),
            ctx.fast_encode,
            frame_count,
            ctx.levels,
            |frame, rgba| {
                if frame < summer_end {
                    // --- SPRING / SUMMER: the artwork accumulates.
                    let target = schedule[frame.min(schedule.len() - 1)];
                    if target > cursor {
                        accumulate_spectral_steps(
                            &mut spd,
                            &AccumulationParams {
                                scene: SpectralScene::new(
                                    ctx.positions,
                                    ctx.colors,
                                    ctx.body_alphas,
                                ),
                                ctx: &render_ctx,
                                velocity_calc: &velocity_calc,
                                step_start: cursor,
                                step_end: target,
                                hdr_scale,
                                traits,
                            },
                            default_accumulation_backend(),
                        );
                        cursor = target;
                    }
                    convert_spd_buffer_to_rgba(&spd, rgba, video_w as usize, video_h as usize);
                    if frame >= spring_end {
                        // SUMMER: bolts on the live stream + the swell.
                        flash.clear();
                        let mut any = false;
                        for &(strike, bolt_index) in &strikes {
                            if frame < strike {
                                continue;
                            }
                            let age = frame - strike;
                            let bolt = summer_bolts[bolt_index];
                            if age <= 2 {
                                draw_bolt(&mut flash, bolt, 3.0, video_scale as f32);
                                any = true;
                            } else if age <= 14 {
                                let envelope = (-((age - 2) as f64) / 4.0).exp();
                                draw_bolt(&mut flash, bolt, 3.0 * envelope, video_scale as f32);
                                any = true;
                            } else if age == 15 {
                                // The afterglow scars the artwork itself
                                // (spectral deposit into the shared SPD).
                                let mut scar = SpdCanvas::new(video_w, video_h);
                                draw_bolt(&mut scar, bolt, 0.10, video_scale as f32);
                                scar.add_into(&mut spd);
                            }
                        }
                        if any {
                            flash.convert_into(&mut flash_rgba);
                            for (pixel, extra) in rgba.iter_mut().zip(flash_rgba.iter()) {
                                pixel.0 += extra.0;
                                pixel.1 += extra.1;
                                pixel.2 += extra.2;
                                pixel.3 = (pixel.3 + extra.3).min(1.0);
                            }
                        }
                        // Exposure swell ramps in over summer.
                        let ramp =
                            (frame - spring_end) as f64 / (summer_end - spring_end).max(1) as f64;
                        let swell = 1.0 + (SUMMER_SWELL - 1.0) * ramp.min(1.0);
                        for pixel in rgba.iter_mut() {
                            pixel.0 *= swell;
                            pixel.1 *= swell;
                            pixel.2 *= swell;
                        }
                    }
                    if frame + 1 == summer_end {
                        summer_final.clone_from(rgba);
                        let mut sample: Vec<f64> = summer_final
                            .iter()
                            .step_by(7)
                            .map(|&(r, g, b, _)| 0.2627 * r + 0.678 * g + 0.0593 * b)
                            .filter(|&value| value > 0.0)
                            .collect();
                        if !sample.is_empty() {
                            sample.sort_by(f64::total_cmp);
                            overlay_ref =
                                sample[((sample.len() - 1) as f64 * 0.95) as usize].max(1e-6);
                        }
                    }
                } else if frame < autumn_end {
                    // --- AUTUMN: the organism feeds on the artwork.
                    let progress =
                        (frame - summer_end) as f64 / (autumn_end - summer_end).max(1) as f64;
                    if swarm.is_none() {
                        let grid_w = video_w as usize;
                        let grid_h = video_h as usize;
                        food = energy_density_grid(ctx, grid_w, grid_h);
                        for value in &mut food {
                            *value = value.sqrt();
                        }
                        // Handoff one: scouts spawn at the last bolt's
                        // endpoints (weight bumps on an otherwise dead grid).
                        let mut seeds = vec![1e-6f32; cells];
                        for &(x, y) in &last_endpoints {
                            let cx = f64::from(x) * video_scale;
                            let cy = f64::from(y) * video_scale;
                            for dy in -6i64..=6 {
                                for dx in -6i64..=6 {
                                    let col = (cx as i64 + dx).clamp(0, grid_w as i64 - 1);
                                    let row = (cy as i64 + dy).clamp(0, grid_h as i64 - 1);
                                    let weight = (-((dx * dx + dy * dy) as f32) / 10.0).exp();
                                    seeds[row as usize * grid_w + col as usize] += weight;
                                }
                            }
                        }
                        // Cap agent density at ~4 per cell so tiny draft
                        // grids grow filaments instead of saturating.
                        let agents = ctx.quality.scale_count(AGENT_COUNT).min(cells * 4);
                        swarm = Some(PhysarumSwarm::seed_weighted(
                            agents,
                            grid_w,
                            grid_h,
                            &seeds,
                            &mut agent_rng,
                        ));
                        // High-pass normalization: a filament core stands a
                        // steady-state's worth above its neighborhood.
                        let density = agents as f32 / cells as f32;
                        trail_norm =
                            physarum_params.deposit * density / (1.0 - physarum_params.decay);
                        info!("   vanitas: autumn begins, {agents} scouts at the last bolt");
                    }
                    if let Some(swarm) = swarm.as_mut() {
                        for tick in 0..TICKS_PER_FRAME {
                            let tick_id = (frame * TICKS_PER_FRAME + tick) as u64;
                            swarm.tick(&trail, &food, &physarum_params, tick_id, &mut deposits);
                            PhysarumSwarm::apply_deposits(&mut trail, &deposits);
                            diffuse_decay(
                                &mut trail,
                                &mut trail_scratch,
                                video_w as usize,
                                video_h as usize,
                                physarum_params.decay,
                            );
                        }
                    }
                    // Vein structure from the trail's high-pass: the
                    // cluster interior cancels against its own blur, so
                    // only filament cores and edges glow (V33's unsharp
                    // lesson -- raw density plateaus wherever agents crowd).
                    trail_blurred.clone_from(&trail);
                    for _ in 0..4 {
                        diffuse_decay(
                            &mut trail_blurred,
                            &mut trail_scratch,
                            video_w as usize,
                            video_h as usize,
                            1.0,
                        );
                    }
                    // Ghost dims and desaturates; veins glaze over it as
                    // filaments, easing in over the act's first 15%.
                    let ghost = 1.0 - (1.0 - GHOST_FLOOR) * progress;
                    let desat = CHROMA_LOSS * progress;
                    let vein_ramp = (progress / 0.15).clamp(0.0, 1.0);
                    rgba.resize(cells, (0.0, 0.0, 0.0, 0.0));
                    for (index, pixel) in rgba.iter_mut().enumerate() {
                        let base = summer_final[index];
                        let luma = 0.2627 * base.0 + 0.678 * base.1 + 0.0593 * base.2;
                        // Filament response: steep curve, exposure-matched.
                        let high_pass = (trail[index] - trail_blurred[index]).max(0.0);
                        let vein = f64::from((high_pass / trail_norm).clamp(0.0, 1.0)).powf(1.6)
                            * overlay_ref
                            * vein_ramp;
                        pixel.0 =
                            (base.0 + (luma - base.0) * desat) * ghost + vein_ink.0 * vein * 1.1;
                        pixel.1 =
                            (base.1 + (luma - base.1) * desat) * ghost + vein_ink.1 * vein * 1.1;
                        pixel.2 =
                            (base.2 + (luma - base.2) * desat) * ghost + vein_ink.2 * vein * 1.1;
                        pixel.3 = base.3.max(vein.min(1.0));
                    }
                    if frame + 1 == autumn_end {
                        autumn_final.clone_from(rgba);
                        let spacing = 14.0 * f64::from(video_w.min(video_h)) / 1117.0;
                        junctions = vein_junctions(
                            &trail,
                            video_w as usize,
                            video_h as usize,
                            spacing.max(6.0),
                            MAX_JUNCTIONS,
                        );
                        info!(
                            "   vanitas: winter begins, {} vein junctions nucleate the frost",
                            junctions.len()
                        );
                    }
                } else if frame < winter_end {
                    // --- WINTER: frost claims the veins.
                    let progress =
                        (frame - autumn_end) as f64 / (winter_end - autumn_end).max(1) as f64;
                    if frost.is_none() {
                        let grid_w = video_w as usize;
                        let grid_h = video_h as usize;
                        let orientation = vec![0.0f32; cells];
                        let mut grid = DlaGrid::new(grid_w, grid_h, &junctions);
                        let budget = ctx.quality.scale_count(WALKER_BUDGET);
                        let grow_started = std::time::Instant::now();
                        grid.grow(
                            budget,
                            &food,
                            &orientation,
                            &DlaParams {
                                stick_base: 0.35,
                                stick_span: 0.65,
                                bias_strength: 0.55,
                                batch_steps: 64,
                                concurrent: 4_096,
                            },
                            &mut frost_rng,
                        );
                        info!(
                            "   vanitas: frost grew {} cells in {:.1}s",
                            grid.stuck_count(),
                            grow_started.elapsed().as_secs_f64()
                        );
                        frost = Some(grid);
                    }
                    let grid = frost.as_ref().expect("frost grown at act start");
                    let reveal = (f64::from(grid.stuck_count()) * progress) as u32;
                    rgba.resize(cells, (0.0, 0.0, 0.0, 0.0));
                    for (index, pixel) in rgba.iter_mut().enumerate() {
                        let base = autumn_final[index];
                        let mut color = (base.0, base.1, base.2);
                        let age = grid.age[index];
                        if age != 0 && age <= reveal {
                            // Ice in the artwork's own exposure range,
                            // newest crystal the brightest.
                            let recency = 1.0 - f64::from(reveal - age) / f64::from(reveal.max(1));
                            let sparkle = (0.35 + 0.65 * recency) * overlay_ref;
                            color.0 += 0.68 * sparkle;
                            color.1 += 0.78 * sparkle;
                            color.2 += 1.0 * sparkle;
                        }
                        let cold = cold_shift(color, progress);
                        pixel.0 = cold.0;
                        pixel.1 = cold.1;
                        pixel.2 = cold.2;
                        pixel.3 = base.3;
                    }
                    if frame + 1 == winter_end {
                        winter_final.clone_from(rgba);
                    }
                } else {
                    // --- DEATH: the pull, the fade, the card.
                    let progress =
                        (frame - winter_end) as f64 / (frame_count - winter_end).max(1) as f64;
                    let zoom = 1.0 / (1.0 + 1.4 * progress); // pull back
                    let fade = (1.0 - (progress - 0.55).max(0.0) / 0.30).clamp(0.0, 1.0);
                    rgba.resize(cells, (0.0, 0.0, 0.0, 0.0));
                    let w = video_w as usize;
                    let h = video_h as usize;
                    for row in 0..h {
                        for col in 0..w {
                            let sx = (col as f64 - w as f64 / 2.0) / zoom + w as f64 / 2.0;
                            let sy = (row as f64 - h as f64 / 2.0) / zoom + h as f64 / 2.0;
                            let pixel = &mut rgba[row * w + col];
                            if sx < 0.0 || sy < 0.0 || sx >= w as f64 || sy >= h as f64 {
                                *pixel = (0.0, 0.0, 0.0, 1.0);
                            } else {
                                let source = winter_final[(sy as usize) * w + sx as usize];
                                *pixel =
                                    (source.0 * fade, source.1 * fade, source.2 * fade, source.3);
                            }
                        }
                    }
                    // The end card in the final 1.5 s (after the silence).
                    if progress > 1.0 - (1.5 / (FILM_SECONDS - ACT_ENDS[3])) {
                        let mut card = vec![paper_color(Paper::DeepBlack); cells];
                        let ink = ink_color(Paper::DeepBlack);
                        let style = TextStyle {
                            align: Align::Center,
                            opacity: 0.55,
                            tabular: true,
                            ..TextStyle::caption(Face::Mono, type_px(h, 1), ink)
                        };
                        draw_text(
                            &mut card,
                            w,
                            h,
                            w as f64 / 2.0,
                            h as f64 * 0.52,
                            &style,
                            &format!("0x{}", ctx.seed_hex.to_uppercase()),
                        );
                        for (pixel, over) in rgba.iter_mut().zip(card.iter()) {
                            pixel.0 += over.0;
                            pixel.1 += over.1;
                            pixel.2 += over.2;
                            pixel.3 = 1.0;
                        }
                    }
                }
                // The color-temperature audit: sampled mean OkLab b.
                if frame.is_multiple_of((frame_count / 48).max(1)) {
                    mean_b_track.push((frame, mean_oklab_b(rgba)));
                }
                if frame == 0 && !logged {
                    logged = true;
                    let per_frame = started.elapsed().as_secs_f64();
                    info!(
                        "   vanitas: {per_frame:.2}s first frame, projected {:.1} min",
                        per_frame * frame_count as f64 / 60.0
                    );
                }
            },
        )?;

        // --- The score: bright bed, bolt crashes, shimmer, last chirp.
        {
            let duration = frame_count as f64 / f64::from(FPS);
            let sample_rate = f64::from(SAMPLE_RATE);
            let total_samples = (duration * sample_rate) as usize;
            let act_t = |frame: usize| frame as f64 / f64::from(FPS);
            let spring_t = act_t(spring_end);
            let summer_t = act_t(summer_end);
            let autumn_t = act_t(autumn_end);
            let winter_t = act_t(winter_end);
            let crash_times: Vec<f64> = strikes.iter().map(|&(strike, _)| act_t(strike)).collect();
            let (h_plus, _) = strain_series(ctx.positions, &ctx.kinematics().masses);
            // The last chirp: the strain's final 3 s, resampled and quiet.
            let chirp_span = (3.0 * sample_rate) as usize;
            let chirp_source = &h_plus[h_plus.len().saturating_sub(h_plus.len() / 8)..];
            let chirp = audio::resample_linear(chirp_source, chirp_span.max(2));
            let chirp_peak = chirp.iter().fold(1e-12f64, |acc, &v| acc.max(v.abs()));
            let chirp_start = winter_t + 1.0;

            let mut left = vec![0.0f64; total_samples];
            let mut right = vec![0.0f64; total_samples];
            let root = 65.4; // C2
            let mut phase = (0.0f64, 0.0f64);
            let mut bed_filter = audio::Biquad::default();
            let mut shimmer_phase = [0.0f64; 4];
            for sample in 0..total_samples {
                let t = sample as f64 / sample_rate;
                // Bed level arcs bright -> sparse -> silent.
                let bed_level = if t < spring_t {
                    0.16 + 0.08 * (t / spring_t)
                } else if t < summer_t {
                    0.26
                } else if t < autumn_t {
                    0.20 * (1.0 - 0.6 * (t - summer_t) / (autumn_t - summer_t))
                } else if t < winter_t {
                    0.05 * (1.0 - (t - autumn_t) / (winter_t - autumn_t))
                } else {
                    0.0
                };
                phase.0 = (phase.0 + root / sample_rate) % 1.0;
                phase.1 = (phase.1 + root * 1.5 / sample_rate) % 1.0;
                if sample.is_multiple_of(64) {
                    let brightness = if t < summer_t {
                        420.0 + 700.0 * (t / summer_t)
                    } else {
                        (1100.0 - 900.0 * ((t - summer_t) / 30.0)).max(200.0)
                    };
                    bed_filter.set_lowpass(brightness, 0.8, sample_rate);
                }
                let mut value = bed_filter.process(
                    saw_bandlimited(phase.0, root / sample_rate) * 0.6
                        + saw_bandlimited(phase.1, root * 1.5 / sample_rate) * 0.35,
                ) * bed_level;
                // Bolt crashes.
                for &crash in &crash_times {
                    if t >= crash && t < crash + 2.0 {
                        value += fm_bell(t - crash, root * 1.5, 2.6) * 0.28;
                    }
                }
                // Winter: high shimmer only.
                if t >= autumn_t && t < winter_t {
                    let winter_progress = (t - autumn_t) / (winter_t - autumn_t);
                    let shimmer_level = 0.035 * (1.0 - winter_progress * 0.5);
                    for (index, phase) in shimmer_phase.iter_mut().enumerate() {
                        let frequency = 1568.0 * (1.0 + 0.26 * index as f64);
                        *phase = (*phase + frequency / sample_rate) % 1.0;
                        let tremble = 0.6 + 0.4 * (t * (2.1 + index as f64 * 0.7)).sin();
                        value +=
                            (std::f64::consts::TAU * *phase).sin() * shimmer_level * tremble / 4.0;
                    }
                }
                // Death: the last chirp, then nothing.
                let mut frame_lr = (value, value);
                if t >= chirp_start {
                    let index = ((t - chirp_start) * sample_rate) as usize;
                    if index < chirp.len() {
                        let whisper = chirp[index] / chirp_peak
                            * 0.12
                            * (1.0 - (t - chirp_start) / 3.0).clamp(0.0, 1.0);
                        frame_lr.0 += whisper;
                        frame_lr.1 += whisper;
                    }
                }
                left[sample] = frame_lr.0;
                right[sample] = frame_lr.1 * 0.97;
            }
            soft_limit(&mut left, &mut right, 1.2);
            normalize_to_lufs(&mut left, &mut right, -17.0, sample_rate);
            fade_ends(&mut left, (sample_rate * 0.3) as usize);
            fade_ends(&mut right, (sample_rate * 0.3) as usize);
            let mix_path = sink.path("vanitas_mix.wav");
            write_wav_stereo_24bit(&mix_path, &left, &right)?;
            for (silent, scored) in
                [("vanitas_silent.mp4", "vanitas.mp4"), ("vanitas_silent_hq.mp4", "vanitas_hq.mp4")]
            {
                let silent_path = sink.path(silent);
                match mux(&silent_path, &mix_path, &sink.path(scored)) {
                    Ok(()) => {
                        sink.record(scored, "video");
                        let _ = std::fs::remove_file(&silent_path);
                    }
                    Err(error) => {
                        warn!("vanitas mux failed for {scored}: {error}; keeping silent cut");
                        let _ = std::fs::rename(&silent_path, sink.path(scored));
                        sink.record(scored, "video");
                    }
                }
            }
            let _ = std::fs::remove_file(&mix_path);
        }

        // --- The audit sidecar: handoffs + the color-temperature arc.
        let post_summer: Vec<f64> = mean_b_track
            .iter()
            .filter(|&&(frame, _)| frame >= summer_end)
            .map(|&(_, b)| b)
            .collect();
        let arc_monotone = post_summer.windows(2).all(|pair| pair[1] <= pair[0] + 0.004);
        let meta = serde_json::json!({
            "act_ends_s": ACT_ENDS,
            "summer_bolts": summer_bolts.len(),
            "handoff_bolt_endpoints_px": last_endpoints
                .iter()
                .map(|&(x, y)| [f64::from(x) * video_scale, f64::from(y) * video_scale])
                .collect::<Vec<_>>(),
            "handoff_vein_junctions": junctions.len(),
            "ghost_floor": GHOST_FLOOR,
            "cold_b_drift": COLD_B_DRIFT,
            "mean_oklab_b_samples": mean_b_track
                .iter()
                .map(|&(frame, b)| serde_json::json!([frame, (b * 1e4).round() / 1e4]))
                .collect::<Vec<_>>(),
            "post_summer_b_monotone_decline": arc_monotone,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("vanitas.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_shift_drives_b_down_and_darkens_monotonically() {
        let warm = (0.5, 0.32, 0.12);
        let mut previous_b = f64::INFINITY;
        let mut previous_l = f64::INFINITY;
        for step in 0..=10 {
            let progress = f64::from(step) / 10.0;
            let shifted = cold_shift(warm, progress);
            let (l, _, b) = linear_rec2020_to_oklab(shifted.0, shifted.1, shifted.2);
            assert!(b <= previous_b + 1e-9, "b must decline monotonically");
            assert!(l <= previous_l + 1e-9, "L must decline monotonically");
            previous_b = b;
            previous_l = l;
        }
        // The full shift lands close to the specified drift.
        let base = linear_rec2020_to_oklab(warm.0, warm.1, warm.2);
        let cold = cold_shift(warm, 1.0);
        let cold_lab = linear_rec2020_to_oklab(cold.0, cold.1, cold.2);
        assert!(
            (cold_lab.2 - (base.2 + COLD_B_DRIFT)).abs() < 0.012,
            "full-winter b should sit near the specified drift"
        );
    }

    #[test]
    fn vein_junctions_are_spaced_and_capped() {
        let (width, height) = (64usize, 48usize);
        let mut trail = vec![0.0f32; width * height];
        // A bright blob cluster: junctions must not stack inside it.
        for row in 10..20 {
            for col in 10..30 {
                trail[row * width + col] = 1.0 + (row + col) as f32 * 0.01;
            }
        }
        let junctions = vein_junctions(&trail, width, height, 8.0, 5);
        assert!(!junctions.is_empty() && junctions.len() <= 5);
        for (index, &(ac, ar)) in junctions.iter().enumerate() {
            for &(bc, br) in junctions.iter().skip(index + 1) {
                let distance =
                    ((ac as f64 - bc as f64).powi(2) + (ar as f64 - br as f64).powi(2)).sqrt();
                assert!(distance >= 8.0, "junctions must respect the spacing");
            }
        }
    }

    #[test]
    fn empty_trail_still_yields_a_central_nucleus() {
        let junctions = vein_junctions(&vec![0.0f32; 32 * 32], 32, 32, 8.0, 4);
        assert_eq!(junctions, vec![(16, 16)]);
    }

    #[test]
    fn mean_oklab_b_reads_warm_versus_cold_fields() {
        let warm: Vec<(f64, f64, f64, f64)> = vec![(0.5, 0.3, 0.05, 1.0); 512];
        let cold: Vec<(f64, f64, f64, f64)> = vec![(0.1, 0.2, 0.55, 1.0); 512];
        assert!(mean_oklab_b(&warm) > mean_oklab_b(&cold));
    }
}
