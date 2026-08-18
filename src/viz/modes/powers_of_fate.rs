//! V64 `powers-of-fate` -- The Dive.
//!
//! One continuous shot: open on the basin-map fractal, dive toward the
//! seed's crosshair at a constant perceptual rate (one octave per 5.5 s);
//! as lattice cells grow past ~40 px they crossfade from fate colors into
//! *actual micro-renders* of their would-be artworks (a 64x64 apron of 96
//! px renders, an 8x8 core of 512 px renders -- each an honest capped sim
//! through the production accumulator with per-cell auto-levels); the dive
//! lands frame-exactly on this seed's cell, which blooms into the master
//! render. Sound: a Shepard-adjacent falling glide, and the V16 opening
//! triad as the impact chord at landing.
//!
//! L0 comes from V42's `basin_raw.png` + `basin.json` when present (the
//! planner runs lower ids first); without them a reduced lattice is
//! recomputed with V42's exact parameters and colors.

use crate::error::Result;
use crate::oklab::oklab_to_oklch;
use crate::render::{VideoEncodingOptions, VideoOutputSpec, create_videos_from_frames_singlepass};
use crate::spectrum::linear_rec2020_to_display_p3;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::accum::Accumulator;
use crate::viz::common::audio::{
    Adsr, SAMPLE_RATE, fade_ends, mux, normalize_to_lufs, saw_bandlimited, soft_limit,
    write_wav_stereo_24bit,
};
use crate::viz::common::display::auto_levels;
use crate::viz::common::resim::{grid_cell_bodies, perturb_grid, rerun};
use crate::viz::context::VizContext;
use crate::viz::modes::basin_map::{fate_color, grid_params};
use crate::viz::modes::chord_progression::{note_frequency, quantized_score};
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::{info, warn};

/// Dive length in seconds at final quality.
const DIVE_SECONDS: f64 = 60.0;
/// Zoom rate: one octave per this many seconds (constant perceptual rate).
const OCTAVE_SECONDS: f64 = 5.5;
/// Frame rate.
const FPS: u32 = 30;
/// L1 apron edge in cells (final quality).
const L1_EDGE: usize = 64;
/// L1 thumbnail resolution.
const L1_PX: u32 = 96;
/// L2 core edge in cells.
const L2_EDGE: usize = 8;
/// L2 thumbnail resolution.
const L2_PX: u32 = 512;
/// Steps-cap fractions for L1 / L2 micro-sims (of the production window).
const CAP_FRACTIONS: (f64, f64) = (0.06, 0.25);
/// On-screen cell span where fate color starts crossfading to thumbnails.
const BLEND_START_PX: f64 = 40.0;
/// Fallback L0 lattice edge when V42's artifacts are absent.
const FALLBACK_N: usize = 192;

/// The powers-of-fate mode.
pub struct PowersOfFate;

/// Encode one linear Rec.2020 color to the display-space unit range used
/// for compositing (the exact PNG transfer, so thumbs match V42's pixels).
fn encode_display(color: (f64, f64, f64)) -> [f32; 3] {
    let (r, g, b) = linear_rec2020_to_display_p3(color.0, color.1, color.2);
    [
        (r.clamp(0.0, 1.0).powf(1.0 / 2.2)) as f32,
        (g.clamp(0.0, 1.0).powf(1.0 / 2.2)) as f32,
        (b.clamp(0.0, 1.0).powf(1.0 / 2.2)) as f32,
    ]
}

/// An encoded display-space texture (unit-range f32, row-major RGB).
struct Texture {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl Texture {
    /// Bilinear sample at unit coordinates.
    fn sample(&self, u: f64, v: f64) -> [f32; 3] {
        let x = (u.clamp(0.0, 1.0) * (self.width - 1) as f64).max(0.0);
        let y = (v.clamp(0.0, 1.0) * (self.height - 1) as f64).max(0.0);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = (x - x0 as f64) as f32;
        let fy = (y - y0 as f64) as f32;
        let at =
            |px: usize, py: usize, channel: usize| self.data[(py * self.width + px) * 3 + channel];
        std::array::from_fn(|channel| {
            let top = at(x0, y0, channel) * (1.0 - fx) + at(x1, y0, channel) * fx;
            let bottom = at(x0, y1, channel) * (1.0 - fx) + at(x1, y1, channel) * fx;
            top * (1.0 - fy) + bottom * fy
        })
    }

    /// Nearest sample at integer texel coordinates.
    fn texel(&self, x: usize, y: usize) -> [f32; 3] {
        let index = (y.min(self.height - 1) * self.width + x.min(self.width - 1)) * 3;
        [self.data[index], self.data[index + 1], self.data[index + 2]]
    }
}

/// Load a 16-bit PNG as an encoded display-space texture.
fn load_png_texture(path: &str) -> Option<Texture> {
    let decoded = image::ImageReader::open(path).ok()?.decode().ok()?;
    let rgb = decoded.into_rgb16();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let data: Vec<f32> = rgb.as_raw().iter().map(|&value| f32::from(value) / 65535.0).collect();
    Some(Texture { width, height, data })
}

/// Micro-render one lattice cell: capped replay through the production
/// accumulator at `size` px with per-cell auto-levels; returns the encoded
/// display-space thumbnail.
#[allow(clippy::too_many_arguments)]
fn micro_render(
    ctx: &VizContext<'_>,
    n: usize,
    epsilon: f64,
    row: usize,
    col: usize,
    size: u32,
    cap_steps: usize,
) -> Vec<f32> {
    let bodies = grid_cell_bodies(ctx.bodies, n, epsilon, row, col);
    let positions = rerun(&bodies, cap_steps.max(64));
    let recorded = positions[0].len();
    // The palette journey compressed onto the capped window.
    let steps = ctx.step_count();
    let colors: Vec<Vec<crate::render::OklabColor>> = (0..3)
        .map(|body| {
            (0..recorded)
                .map(|index| {
                    let source = (index * steps / recorded.max(1)).min(steps - 1);
                    ctx.colors[body][source.min(ctx.colors[body].len() - 1)]
                })
                .collect()
        })
        .collect();
    let mut accumulator = Accumulator::new(
        positions,
        colors,
        ctx.body_alphas.to_vec(),
        size,
        size,
        false,
        ctx.settings.traits,
        ctx.settings.render_config.hdr_scale,
    );
    accumulator.accumulate(0..recorded);
    let rgba = accumulator.convert();
    let clip_black = ctx.settings.resolved_config.clip_black;
    let clip_white = ctx.settings.resolved_config.clip_white;
    let levels = auto_levels(&rgba, clip_black, clip_white, 1.0);
    let display = crate::render::tonemap_to_display_buffer(&rgba, &levels);
    let quantized = crate::render::quantize_display_buffer_to_16bit(&display);
    quantized.iter().map(|&value| f32::from(value) / 65535.0).collect()
}

/// Smoothstep in [0, 1].
fn smooth(t: f64) -> f64 {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

impl VizMode for PowersOfFate {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("powers-of-fate").expect("powers-of-fate is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 || ctx.bodies.len() != 3 {
            warn!("powers-of-fate skipped: trajectory too short or bodies unavailable");
            return Ok(());
        }
        let draft = ctx.quality == crate::viz::context::VizQuality::Draft;

        // --- L0: V42's fate image + lattice parameters, or the fallback.
        let basin_dir = format!("{}/viz/basin-map", ctx.seed_dir);
        let sidecar: Option<serde_json::Value> =
            std::fs::read_to_string(format!("{basin_dir}/basin.json"))
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok());
        let artifact_texture = load_png_texture(&format!("{basin_dir}/basin_raw.png"));
        let fallback_n = if draft { 96 } else { FALLBACK_N };
        let (l0, n, epsilon) = if let (Some(texture), Some(meta)) = (artifact_texture, &sidecar) {
            let n = meta.get("n").and_then(serde_json::Value::as_u64).unwrap_or(0) as usize;
            let epsilon = meta.get("epsilon").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            if n >= 32 && epsilon > 0.0 && texture.width == n {
                info!("   powers-of-fate: L0 from basin-map artifacts ({n}x{n})");
                (texture, n, epsilon)
            } else {
                warn!("powers-of-fate: basin sidecar inconsistent; recomputing L0");
                build_fallback_l0(ctx, fallback_n)
            }
        } else {
            warn!("powers-of-fate: basin-map artifacts missing; recomputing a reduced L0");
            build_fallback_l0(ctx, fallback_n)
        };

        // --- L1/L2 micro-render pyramids around the crosshair cell.
        let l1_edge = if draft { 24 } else { L1_EDGE.min(n) };
        let l2_edge = if draft { 4 } else { L2_EDGE.min(n) };
        let center_cell = n / 2;
        let cap_l1 = ((steps as f64 * CAP_FRACTIONS.0) as usize).max(64);
        let cap_l2 = ((steps as f64 * CAP_FRACTIONS.1) as usize).max(128);
        let started = std::time::Instant::now();
        let render_level = |edge: usize, size: u32, cap: usize| -> Vec<Vec<f32>> {
            let base = center_cell - edge / 2;
            (0..edge * edge)
                .into_par_iter()
                .map(|cell| {
                    let row = base + cell / edge;
                    let col = base + cell % edge;
                    micro_render(ctx, n, epsilon, row, col, size, cap)
                })
                .collect()
        };
        let l1 = render_level(l1_edge, L1_PX, cap_l1);
        info!(
            "   powers-of-fate: L1 {}x{} micro-renders in {:.1}s",
            l1_edge,
            l1_edge,
            started.elapsed().as_secs_f64()
        );
        let l2_started = std::time::Instant::now();
        let l2 = render_level(l2_edge, L2_PX, cap_l2);
        info!(
            "   powers-of-fate: L2 {}x{} renders in {:.1}s",
            l2_edge,
            l2_edge,
            l2_started.elapsed().as_secs_f64()
        );

        // --- L3: the master render.
        let master = load_png_texture(&format!("{}/images/source/master.png", ctx.seed_dir));
        if master.is_none() {
            warn!("powers-of-fate: master.png missing; the dive lands on the L2 render");
        }

        // --- Camera schedule: hold, constant-rate dive, landing hold.
        let video_w = ((ctx.quality.scale_dim(ctx.width) / 2) & !1).max(64);
        let video_h = ((ctx.quality.scale_dim(ctx.height) / 2) & !1).max(64);
        let total_octaves = (n as f64).log2();
        let frame_count = ctx.quality.scale_count((DIVE_SECONDS * f64::from(FPS)) as usize);
        let duration = frame_count as f64 / f64::from(FPS);
        let hold_in = duration * (4.0 / 60.0);
        let hold_out = duration * (3.0 / 60.0);
        let dive_span = (duration - hold_in - hold_out).max(1.0);
        let rate = total_octaves / dive_span; // octaves per second
        info!(
            "   powers-of-fate: {total_octaves:.2} octaves over {dive_span:.1}s \
             ({:.2}s/octave vs spec {OCTAVE_SECONDS})",
            1.0 / rate
        );

        let octaves_at = |t: f64| -> f64 { ((t - hold_in).max(0.0) * rate).min(total_octaves) };
        // Center easing: a fixed off-center opening drifting onto the
        // crosshair over the first two octaves (cell coordinates).
        let center = (center_cell as f64 + 0.5, center_cell as f64 + 0.5);
        let opening_offset = (-0.11 * n as f64, 0.07 * n as f64);

        let l1_base = center_cell - l1_edge / 2;
        let l2_base = center_cell - l2_edge / 2;
        let aspect = f64::from(video_w) / f64::from(video_h);

        // --- Render the dive.
        let outputs = [
            VideoOutputSpec {
                output_file: sink.path("powers_of_fate_silent.mp4"),
                options: VideoEncodingOptions::web_compatible(),
            },
            VideoOutputSpec {
                output_file: sink.path("powers_of_fate_silent_hq.mp4"),
                options: if ctx.fast_encode {
                    VideoEncodingOptions::fast_encode()
                } else {
                    VideoEncodingOptions::high_quality()
                },
            },
        ];
        let render_started = std::time::Instant::now();
        let mut logged = false;
        create_videos_from_frames_singlepass(
            video_w,
            video_h,
            FPS,
            |out| {
                let mut frame_u16 = vec![0u16; video_w as usize * video_h as usize * 3];
                for frame in 0..frame_count {
                    let t = frame as f64 / f64::from(FPS);
                    let octaves = octaves_at(t);
                    // Viewport extent in cells (height); the landing frame
                    // shows exactly one cell.
                    let extent_cells = n as f64 / 2.0f64.powf(octaves);
                    let ease = smooth(octaves / 2.0);
                    let view_center = (
                        center.0 + opening_offset.0 * (1.0 - ease),
                        center.1 + opening_offset.1 * (1.0 - ease),
                    );
                    let cell_px = f64::from(video_h) / extent_cells;
                    // L2 -> L3 handoff on the center cell over the last octave.
                    let l3_blend = smooth((octaves - (total_octaves - 1.0)).max(0.0));

                    frame_u16.par_chunks_mut(video_w as usize * 3).enumerate().for_each(
                        |(py, row_out)| {
                            for px in 0..video_w as usize {
                                let u = (px as f64 + 0.5) / f64::from(video_w) - 0.5;
                                let v = (py as f64 + 0.5) / f64::from(video_h) - 0.5;
                                let cell_x = view_center.0 + u * extent_cells * aspect;
                                let cell_y = view_center.1 + v * extent_cells;
                                let mut color = [0.0f32; 3];
                                if cell_x >= 0.0
                                    && cell_y >= 0.0
                                    && cell_x < n as f64
                                    && cell_y < n as f64
                                {
                                    let col = cell_x as usize;
                                    let row = cell_y as usize;
                                    // Base: the fate color (L0 texel).
                                    color = l0.texel(col, row);
                                    // L1 apron: crossfade color -> thumbnail
                                    // across the 40..80 px cell span.
                                    let in_l1 = col >= l1_base
                                        && col < l1_base + l1_edge
                                        && row >= l1_base
                                        && row < l1_base + l1_edge;
                                    if in_l1 && cell_px >= BLEND_START_PX {
                                        let local_u = cell_x - col as f64;
                                        let local_v = cell_y - row as f64;
                                        let thumb =
                                            &l1[(row - l1_base) * l1_edge + (col - l1_base)];
                                        let sampled =
                                            sample_thumb(thumb, L1_PX as usize, local_u, local_v);
                                        let blend =
                                            smooth((cell_px - BLEND_START_PX) / BLEND_START_PX)
                                                as f32;
                                        for channel in 0..3 {
                                            color[channel] = color[channel] * (1.0 - blend)
                                                + sampled[channel] * blend;
                                        }
                                    }
                                    // L2 core: takes over as cells pass the
                                    // L1 thumb's native resolution.
                                    let in_l2 = col >= l2_base
                                        && col < l2_base + l2_edge
                                        && row >= l2_base
                                        && row < l2_base + l2_edge;
                                    if in_l2 && cell_px >= f64::from(L1_PX) {
                                        let local_u = cell_x - col as f64;
                                        let local_v = cell_y - row as f64;
                                        let thumb =
                                            &l2[(row - l2_base) * l2_edge + (col - l2_base)];
                                        let sampled =
                                            sample_thumb(thumb, L2_PX as usize, local_u, local_v);
                                        let blend =
                                            smooth((cell_px - f64::from(L1_PX)) / f64::from(L1_PX))
                                                as f32;
                                        for channel in 0..3 {
                                            color[channel] = color[channel] * (1.0 - blend)
                                                + sampled[channel] * blend;
                                        }
                                    }
                                    // L3: the real master blooms on the
                                    // center cell only (frame-exact landing).
                                    if l3_blend > 0.0
                                        && col == center_cell
                                        && row == center_cell
                                        && let Some(master) = &master
                                    {
                                        let local_u = cell_x - col as f64;
                                        let local_v = cell_y - row as f64;
                                        let sampled = master.sample(local_u, local_v);
                                        let blend = l3_blend as f32;
                                        for channel in 0..3 {
                                            color[channel] = color[channel] * (1.0 - blend)
                                                + sampled[channel] * blend;
                                        }
                                    }
                                }
                                let base = px * 3;
                                for channel in 0..3 {
                                    row_out[base + channel] =
                                        (f64::from(color[channel]) * 65535.0) as u16;
                                }
                            }
                        },
                    );
                    out.write_all(bytemuck::cast_slice(&frame_u16))
                        .map_err(crate::render::error::RenderError::VideoEncoding)?;
                    if frame == 0 && !logged {
                        logged = true;
                        let per_frame = render_started.elapsed().as_secs_f64();
                        info!(
                            "   powers-of-fate: {per_frame:.2}s first frame, projected {:.1} min",
                            per_frame * frame_count as f64 / 60.0
                        );
                    }
                }
                Ok(())
            },
            &outputs,
        )?;

        // --- Audio: the falling glide + the V16 impact triad at landing.
        let sample_rate = f64::from(SAMPLE_RATE);
        let total_samples = (duration * sample_rate) as usize;
        let landing_t = hold_in + total_octaves / rate;
        let score = quantized_score(ctx);
        let triad: Vec<f64> = (0..3)
            .map(|voice| {
                score.voices[voice]
                    .first()
                    .map_or(score.root_hz, |note| note_frequency(score.root_hz, note))
            })
            .collect();
        let chord_adsr = Adsr { attack: 0.01, decay: 2.4, sustain: 0.18, release: 1.2 };
        let mut left = vec![0.0f64; total_samples];
        let mut right = vec![0.0f64; total_samples];
        // Shepard bank: six voices an octave apart under a raised-cosine
        // spectral window; all descend at the dive rate.
        let voices = 6usize;
        let low_hz = 55.0f64;
        let span_octaves = voices as f64;
        let mut phases = vec![0.0f64; voices];
        for sample in 0..total_samples {
            let t = sample as f64 / sample_rate;
            let fall = octaves_at(t); // octaves fallen so far
            let mut value = 0.0;
            for (voice, phase) in phases.iter_mut().enumerate() {
                // Log-frequency position wraps inside the bank's span.
                let position = (voice as f64 - fall).rem_euclid(span_octaves);
                let frequency = low_hz * 2.0f64.powf(position);
                *phase = (*phase + frequency / sample_rate) % 1.0;
                // Raised-cosine loudness window over the bank.
                let window = 0.5 - 0.5 * (std::f64::consts::TAU * (position / span_octaves)).cos();
                value += saw_bandlimited(*phase, frequency / sample_rate) * window * 0.10;
            }
            // Impact chord: the seed's opening V16 triad at the landing.
            let mut frame = (value, value);
            if t >= landing_t {
                let held = t - landing_t;
                let envelope = chord_adsr.amplitude(held, 2.0);
                for (voice, &frequency) in triad.iter().enumerate() {
                    let tone = (std::f64::consts::TAU * frequency * held).sin()
                        + 0.25 * (std::f64::consts::TAU * frequency * 2.0 * held).sin()
                        + 0.12 * (std::f64::consts::TAU * frequency * 3.0 * held).sin();
                    let (l, r) = crate::viz::common::audio::equal_power_pan(
                        tone * envelope * 0.16,
                        (voice as f64 - 1.0) * 0.5,
                    );
                    frame.0 += l;
                    frame.1 += r;
                }
            }
            left[sample] = frame.0;
            right[sample] = frame.1;
        }
        soft_limit(&mut left, &mut right, 1.2);
        normalize_to_lufs(&mut left, &mut right, -16.0, sample_rate);
        fade_ends(&mut left, (sample_rate * 0.3) as usize);
        fade_ends(&mut right, (sample_rate * 0.3) as usize);
        let mix_path = sink.path("dive_mix.wav");
        write_wav_stereo_24bit(&mix_path, &left, &right)?;

        // --- Mux (the compositor's only role here).
        for (silent, scored) in [
            ("powers_of_fate_silent.mp4", "powers_of_fate.mp4"),
            ("powers_of_fate_silent_hq.mp4", "powers_of_fate_hq.mp4"),
        ] {
            let silent_path = sink.path(silent);
            match mux(&silent_path, &mix_path, &sink.path(scored)) {
                Ok(()) => {
                    sink.record(scored, "video");
                    let _ = std::fs::remove_file(&silent_path);
                }
                Err(error) => {
                    warn!("powers-of-fate mux failed for {scored}: {error}; keeping silent cut");
                    let _ = std::fs::rename(&silent_path, sink.path(scored));
                    sink.record(scored, "video");
                }
            }
        }
        let _ = std::fs::remove_file(&mix_path);

        let meta = serde_json::json!({
            "lattice_n": n,
            "epsilon": epsilon,
            "l1_edge": l1_edge,
            "l2_edge": l2_edge,
            "cap_fractions": [CAP_FRACTIONS.0, CAP_FRACTIONS.1],
            "total_octaves": total_octaves,
            "octave_seconds": 1.0 / rate,
            "landing_seconds": landing_t,
            "l0_source": if sidecar.is_some() { "basin-map artifacts" } else { "fallback lattice" },
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("dive.json", &json, "data")?;
        Ok(())
    }
}

/// Bilinear sample of a size x size encoded thumbnail at cell-local UV.
fn sample_thumb(thumb: &[f32], size: usize, u: f64, v: f64) -> [f32; 3] {
    let x = (u.clamp(0.0, 1.0) * (size - 1) as f64).max(0.0);
    let y = (v.clamp(0.0, 1.0) * (size - 1) as f64).max(0.0);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(size - 1);
    let y1 = (y0 + 1).min(size - 1);
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;
    let at = |px: usize, py: usize, channel: usize| thumb[(py * size + px) * 3 + channel];
    std::array::from_fn(|channel| {
        let top = at(x0, y0, channel) * (1.0 - fx) + at(x1, y0, channel) * fx;
        let bottom = at(x0, y1, channel) * (1.0 - fx) + at(x1, y1, channel) * fx;
        top * (1.0 - fy) + bottom * fy
    })
}

/// Recompute a reduced L0 with V42's exact parameters and colors.
fn build_fallback_l0(ctx: &VizContext<'_>, n: usize) -> (Texture, usize, f64) {
    let params = grid_params(ctx, n);
    let outcomes = perturb_grid(ctx.bodies, &params);
    let hues: [f64; 3] = std::array::from_fn(|body| {
        let (l, a, b) = ctx.mean_color(body);
        oklab_to_oklch(l, a, b).2
    });
    let mut data = Vec::with_capacity(n * n * 3);
    for outcome in &outcomes {
        let color = fate_color(outcome, params.cap, &hues);
        data.extend_from_slice(&encode_display(color));
    }
    (Texture { width: n, height: n, data }, n, params.epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_schedule_lands_exactly_on_one_cell() {
        // The dive covers log2(n) octaves; at landing the viewport height
        // equals exactly one lattice cell.
        let n = 768usize;
        let total_octaves = (n as f64).log2();
        let extent_at = |octaves: f64| n as f64 / 2.0f64.powf(octaves);
        assert!((extent_at(0.0) - 768.0).abs() < 1e-9);
        assert!((extent_at(total_octaves) - 1.0).abs() < 1e-9);
        // Constant perceptual rate: equal octave steps shrink by equal factors.
        let ratio_a = extent_at(2.0) / extent_at(3.0);
        let ratio_b = extent_at(7.5) / extent_at(8.5);
        assert!((ratio_a - 2.0).abs() < 1e-12 && (ratio_b - 2.0).abs() < 1e-12);
    }

    #[test]
    fn crossfade_window_spans_exactly_one_octave() {
        // 40 px -> 80 px is one doubling: the spec's "crossfade over a
        // scale octave".
        let blend_at = |cell_px: f64| smooth((cell_px - BLEND_START_PX) / BLEND_START_PX);
        assert_eq!(blend_at(BLEND_START_PX), 0.0);
        assert_eq!(blend_at(BLEND_START_PX * 2.0), 1.0);
        let mid = blend_at(BLEND_START_PX * 1.5);
        assert!(mid > 0.4 && mid < 0.6, "midpoint blend should be near half, got {mid}");
    }

    #[test]
    fn thumb_sampling_interpolates_bilinearly() {
        // 2x2 thumb: corners black/white; center is the mean.
        let mut thumb = vec![0.0f32; 2 * 2 * 3];
        for channel in 0..3 {
            thumb[3 + channel] = 1.0; // (1, 0) white
            thumb[6 + channel] = 1.0; // (0, 1) white
        }
        let center = sample_thumb(&thumb, 2, 0.5, 0.5);
        for value in &center {
            assert!((value - 0.5).abs() < 1e-6);
        }
        let corner = sample_thumb(&thumb, 2, 0.0, 0.0);
        assert!(corner[0] < 1e-6);
    }
}
