//! V42 `basin-map` -- Where Your Artwork Lives in Chaos.
//!
//! A 2D scan of initial-condition space around this seed: body 1's initial
//! position is displaced over a small lattice, every cell runs a short
//! capped simulation, and the fates (which body ejects, how early) paint
//! the Wada-basin fractal of the three-body problem with a crosshair at
//! this artwork's exact coordinates. Poster annotation is hairline-only
//! until `common/text.rs` lands; grid parameters ship in `basin.json`.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklab_to_oklch, oklch_to_oklab};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::encode_linear_rec2020_png16;
use crate::viz::common::raster::{Rgb64, draw_line};
use crate::viz::common::resim::{
    CellOutcome, DEFAULT_ESCAPE_THRESHOLD, GridParams, capped_fate, perturb_grid,
};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Lattice edge at final quality (the default of the spec's budget ladder).
const GRID_N: usize = 768;
/// Displacement half-extent as a fraction of the mean pair separation.
const DELTA_FRACTION: f64 = 0.005;
/// Capped steps per cell as a fraction of the full integration (2x steps).
const CAP_FRACTION: f64 = 0.25;
/// Ejection check cadence inside capped runs.
const CHECK_INTERVAL: usize = 2_500;
/// Zoom inset magnification.
const INSET_ZOOM: usize = 8;

/// Fate cell color: escaper hue at an earliness-driven value; bound cells
/// stay deep neutral.
fn fate_color(outcome: &CellOutcome, cap: usize, hues: &[f64; 3]) -> Rgb64 {
    match outcome.escaper {
        Some(escaper) => {
            let earliness =
                1.0 - f64::from(outcome.ejection_step.unwrap_or(cap as u32)) / cap.max(1) as f64;
            let value = 0.25 + 0.55 * earliness;
            let (l, a, b) = oklch_to_oklab(value, 0.11, hues[usize::from(escaper) % 3]);
            let (r, g, blue) = oklab_to_linear_rec2020(l, a, b);
            (r.max(0.0), g.max(0.0), blue.max(0.0))
        }
        None => (0.010, 0.011, 0.016),
    }
}

/// The basin-map mode.
pub struct BasinMap;

impl VizMode for BasinMap {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("basin-map").expect("basin-map is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 || ctx.bodies.len() != 3 {
            warn!("basin-map skipped: trajectory too short or bodies unavailable");
            return Ok(());
        }
        let n = ctx.quality.scale_count(GRID_N).max(32);
        let mean_separation = {
            let mut sum = 0.0f64;
            for i in 0..3 {
                for j in i + 1..3 {
                    sum += (ctx.bodies[i].position - ctx.bodies[j].position).norm();
                }
            }
            (sum / 3.0).max(1e-9)
        };
        let params = GridParams {
            n,
            epsilon: DELTA_FRACTION * mean_separation,
            warmup: 0,
            cap: ((2 * steps) as f64 * CAP_FRACTION) as usize,
            check_interval: CHECK_INTERVAL,
            escape_threshold: DEFAULT_ESCAPE_THRESHOLD,
        };
        info!(
            "   basin-map: {n}x{n} lattice, epsilon {:.3e}, cap {} steps",
            params.epsilon, params.cap
        );
        let scan_started = std::time::Instant::now();
        let outcomes = perturb_grid(ctx.bodies, &params);
        let seed_fate = capped_fate(ctx.bodies, &params);
        info!(
            "   basin-map: scan finished in {:.1}s ({} ejections)",
            scan_started.elapsed().as_secs_f64(),
            outcomes.iter().filter(|outcome| outcome.escaper.is_some()).count()
        );
        let center_cell = &outcomes[(n / 2) * n + n / 2];
        let consistent = center_cell.escaper == seed_fate.escaper;
        if !consistent {
            warn!(
                "basin-map: center cell fate {:?} differs from the unperturbed fate {:?}",
                center_cell.escaper, seed_fate.escaper
            );
        }

        // --- Color the lattice with fate-aware bilateral smoothing.
        let hues: [f64; 3] = std::array::from_fn(|body| {
            let (l, a, b) = ctx.mean_color(body);
            oklab_to_oklch(l, a, b).2
        });
        let colored: Vec<Rgb64> =
            outcomes.iter().map(|outcome| fate_color(outcome, params.cap, &hues)).collect();
        let mut smoothed = colored.clone();
        for row in 0..n {
            for col in 0..n {
                let fate = outcomes[row * n + col].escaper;
                let mut sum = (0.0f64, 0.0f64, 0.0f64);
                let mut count = 0.0f64;
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let y = row as i64 + dy;
                        let x = col as i64 + dx;
                        if y < 0 || x < 0 || y >= n as i64 || x >= n as i64 {
                            continue;
                        }
                        let neighbor = (y as usize) * n + x as usize;
                        if outcomes[neighbor].escaper == fate {
                            sum.0 += colored[neighbor].0;
                            sum.1 += colored[neighbor].1;
                            sum.2 += colored[neighbor].2;
                            count += 1.0;
                        }
                    }
                }
                smoothed[row * n + col] =
                    (sum.0 / count.max(1.0), sum.1 / count.max(1.0), sum.2 / count.max(1.0));
            }
        }

        // --- Raw data image.
        let raw_image = encode_linear_rec2020_png16(&smoothed, n as u32, n as u32);
        sink.save_png16(&raw_image, "basin_raw.png")?;

        // --- Poster: margins, nearest upscale, crosshair, ticks, inset.
        let poster_size = ctx.quality.scale_dim(2048).max(256) as usize;
        let margin = poster_size / 24;
        let data_size = poster_size - 2 * margin;
        let paper: Rgb64 = (0.004, 0.0045, 0.006);
        let ink: Rgb64 = (0.55, 0.56, 0.60);
        let mut poster = vec![paper; poster_size * poster_size];
        let sample_data = |px: usize, py: usize| -> Rgb64 {
            let col = (px * n / data_size).min(n - 1);
            let row = (py * n / data_size).min(n - 1);
            smoothed[row * n + col]
        };
        for py in 0..data_size {
            for px in 0..data_size {
                poster[(margin + py) * poster_size + margin + px] = sample_data(px, py);
            }
        }
        // Crosshair through the seed's cell (the exact center).
        let center = margin as f32 + data_size as f32 / 2.0;
        let full = (margin + data_size) as f32;
        draw_line(
            &mut poster,
            poster_size,
            poster_size,
            (center, margin as f32),
            (center, full),
            ink,
            1.2,
            0.8,
        );
        draw_line(
            &mut poster,
            poster_size,
            poster_size,
            (margin as f32, center),
            (full, center),
            ink,
            1.2,
            0.8,
        );
        // Margin ticks every eighth of the axis.
        for tick in 0..=8 {
            let offset = margin as f32 + data_size as f32 * tick as f32 / 8.0;
            let tick_len = margin as f32 * 0.35;
            draw_line(
                &mut poster,
                poster_size,
                poster_size,
                (offset, margin as f32),
                (offset, margin as f32 - tick_len),
                ink,
                1.2,
                0.9,
            );
            draw_line(
                &mut poster,
                poster_size,
                poster_size,
                (margin as f32, offset),
                (margin as f32 - tick_len, offset),
                ink,
                1.2,
                0.9,
            );
        }
        // Inset: 8x nearest zoom of the crosshair neighborhood, bottom-right.
        let inset_size = data_size / 4;
        let inset_cells = (n / INSET_ZOOM / 2).max(4);
        let inset_origin = (
            margin + data_size - inset_size - margin / 2,
            margin + data_size - inset_size - margin / 2,
        );
        for py in 0..inset_size {
            for px in 0..inset_size {
                let cell_col = n / 2 - inset_cells + px * (2 * inset_cells) / inset_size;
                let cell_row = n / 2 - inset_cells + py * (2 * inset_cells) / inset_size;
                let color = smoothed[cell_row.min(n - 1) * n + cell_col.min(n - 1)];
                poster[(inset_origin.1 + py) * poster_size + inset_origin.0 + px] = color;
            }
        }
        for edge in 0..4 {
            let (from, to) = match edge {
                0 => (
                    (inset_origin.0 as f32, inset_origin.1 as f32),
                    ((inset_origin.0 + inset_size) as f32, inset_origin.1 as f32),
                ),
                1 => (
                    ((inset_origin.0 + inset_size) as f32, inset_origin.1 as f32),
                    ((inset_origin.0 + inset_size) as f32, (inset_origin.1 + inset_size) as f32),
                ),
                2 => (
                    ((inset_origin.0 + inset_size) as f32, (inset_origin.1 + inset_size) as f32),
                    (inset_origin.0 as f32, (inset_origin.1 + inset_size) as f32),
                ),
                _ => (
                    (inset_origin.0 as f32, (inset_origin.1 + inset_size) as f32),
                    (inset_origin.0 as f32, inset_origin.1 as f32),
                ),
            };
            draw_line(&mut poster, poster_size, poster_size, from, to, ink, 1.4, 1.0);
        }
        let poster_image =
            encode_linear_rec2020_png16(&poster, poster_size as u32, poster_size as u32);
        sink.save_png16(&poster_image, "basin.png")?;

        // --- Data sidecar (consumed by V64): one row string per lattice row.
        let rows: Vec<String> = (0..n)
            .map(|row| {
                (0..n)
                    .map(|col| match outcomes[row * n + col].escaper {
                        None => '.',
                        Some(0) => '1',
                        Some(1) => '2',
                        Some(_) => '3',
                    })
                    .collect()
            })
            .collect();
        let meta = serde_json::json!({
            "n": n,
            "epsilon": params.epsilon,
            "delta_fraction": DELTA_FRACTION,
            "cap_steps": params.cap,
            "check_interval": CHECK_INTERVAL,
            "outcome_codes": { ".": "bound", "1": "body 1 ejects", "2": "body 2 ejects", "3": "body 3 ejects" },
            "rows": rows,
            "seed_fate_escaper": seed_fate.escaper,
            "center_cell_escaper": center_cell.escaper,
            "center_consistent": consistent,
            "note": "axis labels and cartography text deferred until text.rs",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("basin.json", &json, "data")?;
        Ok(())
    }
}
