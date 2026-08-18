//! V11 `recurrence` -- Fingerprint of Chaos.
//!
//! The classic recurrence plot as a monumental textile: pixel (i, j) is
//! bright where the system's 12-dimensional state at time i nearly repeats
//! at time j. Quasi-periodic orbits weave plaids; chaos storms. Rendered
//! with the palette's ink over deep black, plus a central 4x zoom crop.

use crate::error::Result;
use crate::oklab::oklab_to_linear_rec2020;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::encode_linear_rec2020_png16;
use crate::viz::common::raster::Rgb64;
use crate::viz::common::style::accent_color;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use rayon::prelude::*;
use tracing::{info, warn};

/// Decimated state count (the recurrence lattice edge).
const STATES: usize = 2_048;
/// Poster edge at final quality.
const POSTER_EDGE: u32 = 3_456;

/// The recurrence mode.
pub struct Recurrence;

impl VizMode for Recurrence {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("recurrence").expect("recurrence is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 {
            warn!("recurrence skipped: trajectory too short");
            return Ok(());
        }
        let states = ctx.quality.scale_count(STATES).max(128);
        let kinematics = ctx.kinematics();

        // 12-d normalized state series: xy positions + xy velocities.
        let mut scale_pos = 0.0f64;
        let mut scale_vel = 0.0f64;
        for body in 0..3 {
            for step in (0..steps).step_by(97) {
                scale_pos = scale_pos.max(ctx.positions[body][step].xy().norm());
                scale_vel = scale_vel.max(kinematics.velocities[body][step].xy().norm());
            }
        }
        let (scale_pos, scale_vel) = (scale_pos.max(1e-9), scale_vel.max(1e-9));
        let series: Vec<[f64; 12]> = (0..states)
            .map(|index| {
                let step = (index * steps / states).min(steps - 1);
                let mut state = [0.0f64; 12];
                for body in 0..3 {
                    state[body * 2] = ctx.positions[body][step].x / scale_pos;
                    state[body * 2 + 1] = ctx.positions[body][step].y / scale_pos;
                    state[6 + body * 2] = kinematics.velocities[body][step].x / scale_vel;
                    state[6 + body * 2 + 1] = kinematics.velocities[body][step].y / scale_vel;
                }
                state
            })
            .collect();

        // Distance normalization: the 10th percentile of sampled distances.
        let mut probe: Vec<f64> = Vec::with_capacity(4_096);
        let mut hash_state = 0x9E37_79B9u64;
        for _ in 0..4_096 {
            hash_state = hash_state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let i = (hash_state >> 20) as usize % states;
            let j = (hash_state >> 40) as usize % states;
            let d: f64 = series[i]
                .iter()
                .zip(series[j].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            probe.push(d);
        }
        probe.sort_by(f64::total_cmp);
        let epsilon = probe[probe.len() / 10].max(1e-6);
        info!("   recurrence: {states} states, epsilon {epsilon:.4}");

        // Recurrence brightness lattice.
        let mut lattice = vec![0.0f32; states * states];
        lattice.par_chunks_mut(states).enumerate().for_each(|(i, row)| {
            for (j, slot) in row.iter_mut().enumerate() {
                let d: f64 = series[i]
                    .iter()
                    .zip(series[j].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                *slot = (-d / epsilon).exp() as f32;
            }
        });

        // Colorize: deep black -> palette accent -> paper white.
        let accent = accent_color(ctx);
        let colorize = |value: f32| -> Rgb64 {
            let t = f64::from(value).powf(0.8);
            if t < 0.7 {
                let mix = t / 0.7;
                (accent.0 * mix * 0.85, accent.1 * mix * 0.85, accent.2 * mix * 0.85)
            } else {
                let mix = (t - 0.7) / 0.3;
                let (r, g, b) = oklab_to_linear_rec2020(0.93, 0.0, 0.01);
                (
                    accent.0 * 0.85 + (r - accent.0 * 0.85) * mix,
                    accent.1 * 0.85 + (g - accent.1 * 0.85) * mix,
                    accent.2 * 0.85 + (b - accent.2 * 0.85) * mix,
                )
            }
        };

        // Poster: nearest sampling of the lattice keeps the plaid crisp.
        let edge = ctx.quality.scale_dim(POSTER_EDGE) as usize;
        let mut poster = vec![(0.0, 0.0, 0.0); edge * edge];
        poster.par_chunks_mut(edge).enumerate().for_each(|(y, row)| {
            let j = y * states / edge;
            for (x, slot) in row.iter_mut().enumerate() {
                let i = x * states / edge;
                *slot = colorize(lattice[j.min(states - 1) * states + i.min(states - 1)]);
            }
        });
        let image = encode_linear_rec2020_png16(&poster, edge as u32, edge as u32);
        sink.save_png16(&image, "recurrence.png")?;

        // Central 4x zoom crop at the same output size.
        let quarter = states / 8;
        let mut zoom = vec![(0.0, 0.0, 0.0); edge * edge];
        zoom.par_chunks_mut(edge).enumerate().for_each(|(y, row)| {
            let j = states / 2 - quarter + y * (2 * quarter) / edge;
            for (x, slot) in row.iter_mut().enumerate() {
                let i = states / 2 - quarter + x * (2 * quarter) / edge;
                *slot = colorize(lattice[j.min(states - 1) * states + i.min(states - 1)]);
            }
        });
        let zoom_image = encode_linear_rec2020_png16(&zoom, edge as u32, edge as u32);
        sink.save_png16(&zoom_image, "recurrence_zoom.png")?;

        let meta = serde_json::json!({
            "states": states,
            "epsilon": epsilon,
            "note": "state = (xy positions, xy velocities) / global scales",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("recurrence_params.json", &json, "data")?;
        Ok(())
    }
}
