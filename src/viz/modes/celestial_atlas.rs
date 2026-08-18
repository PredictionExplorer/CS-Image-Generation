//! V63 `celestial-atlas` -- The Collection as a Sky.
//!
//! Every seed package found in `--viz-seeds-dir` charted as a star on an
//! engraved celestial atlas: position from a deterministic PCA embedding
//! of its recorded metrics (relaxed by 200 repulsion iterations),
//! brightness from the weighted Borda score, hue from the palette
//! fingerprint; kindred seeds joined into constellations named by a
//! curated fact-to-epithet grammar. This seed's star wears a coronet.
//! Without a seeds dir the chart degenerates gracefully to one star.

use crate::error::Result;
use crate::oklab::{oklab_to_linear_rec2020, oklch_to_oklab};
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::display::encode_linear_rec2020_png16;
use crate::viz::common::raster::{Rgb64, draw_line};
use crate::viz::common::style::{Paper, draw_footer, ink_color, paper_color, type_px};
use crate::viz::common::text::{Align, Face, TextStyle, draw_text};
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use std::fmt::Write as _;
use tracing::{info, warn};

/// Repulsion relaxation iterations.
const RELAX_ITERATIONS: usize = 200;
/// Minimum constellation size.
const MIN_CLUSTER: usize = 3;
/// Star magnitude levels.
const MAGNITUDES: usize = 5;

/// One charted seed.
#[derive(Clone, Debug)]
struct Star {
    seed: String,
    features: Vec<f64>,
    score: f64,
    hue: f64,
    position: (f64, f64),
    is_current: bool,
}

/// Parse a package's `generation.json` into a feature record.
fn ingest(path: &std::path::Path) -> Option<(String, Vec<f64>, f64, f64)> {
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let seed = json.get("seed")?.as_str()?.to_string();
    let get = |pointer: &str| json.pointer(pointer).and_then(serde_json::Value::as_f64);
    let score = get("/orbit_info/weighted_score").unwrap_or(0.0);
    // Palette fingerprint leads with the anchor hue: "h093.8_...".
    let hue = json
        .pointer("/render_config/palette_fingerprint")
        .and_then(serde_json::Value::as_str)
        .and_then(|fp| fp.strip_prefix('h'))
        .and_then(|rest| rest.split('_').next())
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    let features = vec![
        (score / 1_000.0).tanh(),
        (hue.to_radians()).sin(),
        (hue.to_radians()).cos(),
        get("/render_config/hdr_scale").unwrap_or(0.5),
        get("/drift_config/orbit_eccentricity").unwrap_or(0.5),
        get("/simulation_config/equil_weight").unwrap_or(1.0).ln_1p() / 3.0,
    ];
    Some((seed, features, score, hue))
}

/// Deterministic 2D PCA embedding of feature rows.
fn embed(rows: &[Vec<f64>]) -> Vec<(f64, f64)> {
    let count = rows.len();
    let dims = rows.first().map_or(0, Vec::len);
    if count == 0 || dims == 0 {
        return Vec::new();
    }
    let mean: Vec<f64> =
        (0..dims).map(|d| rows.iter().map(|r| r[d]).sum::<f64>() / count as f64).collect();
    // Covariance matrix.
    let mut cov = vec![vec![0.0f64; dims]; dims];
    for row in rows {
        for i in 0..dims {
            for j in 0..dims {
                cov[i][j] += (row[i] - mean[i]) * (row[j] - mean[j]);
            }
        }
    }
    // Top-2 eigenvectors by deterministic power iteration + deflation.
    let mut axes: Vec<Vec<f64>> = Vec::new();
    let mut work = cov.clone();
    for component in 0..2 {
        let mut v: Vec<f64> = (0..dims).map(|d| if d == component { 1.0 } else { 0.3 }).collect();
        for _ in 0..64 {
            let mut next = vec![0.0f64; dims];
            for i in 0..dims {
                for j in 0..dims {
                    next[i] += work[i][j] * v[j];
                }
            }
            let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            v = next.into_iter().map(|x| x / norm).collect();
        }
        // Deflate.
        let lambda: f64 =
            (0..dims).map(|i| v[i] * (0..dims).map(|j| work[i][j] * v[j]).sum::<f64>()).sum();
        for i in 0..dims {
            for j in 0..dims {
                work[i][j] -= lambda * v[i] * v[j];
            }
        }
        axes.push(v);
    }
    rows.iter()
        .map(|row| {
            let project = |axis: &[f64]| {
                row.iter().zip(axis.iter()).zip(mean.iter()).map(|((r, a), m)| (r - m) * a).sum()
            };
            (project(&axes[0]), project(&axes[1]))
        })
        .collect()
}

/// Curated epithet grammar over a cluster's dominant trait.
fn constellation_name(stars: &[&Star], index: usize) -> (String, String) {
    let mean_score = stars.iter().map(|s| s.score).sum::<f64>() / stars.len() as f64;
    let mean_ecc = stars.iter().map(|s| s.features[4]).sum::<f64>() / stars.len() as f64;
    let mean_hue = stars.iter().map(|s| s.hue).sum::<f64>() / stars.len() as f64;
    let (epithet, clause) = if mean_ecc > 0.55 {
        ("THE LONG EMBRACE", "wide swings, no parting")
    } else if mean_score > 2_000.0 {
        ("THE CROWNED HOST", "high scorers, close-ranked")
    } else if mean_hue < 90.0 {
        ("THE EMBER FIELD", "warm anchors, slow fires")
    } else if mean_hue < 210.0 {
        ("THE VERDANT WHEEL", "green anchors in mid-turn")
    } else {
        ("THE COLD PROCESSION", "blue anchors, even paces")
    };
    (format!("{epithet} {}", ["I", "II", "III", "IV", "V", "VI"][index % 6]), clause.to_string())
}

/// The celestial-atlas mode.
pub struct CelestialAtlas;

impl VizMode for CelestialAtlas {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("celestial-atlas").expect("celestial-atlas is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        // --- Ingest the seed packages.
        let mut records: Vec<(String, Vec<f64>, f64, f64)> = Vec::new();
        if let Some(dir) = ctx.seeds_dir {
            let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
                .map(|reader| {
                    reader.filter_map(std::result::Result::ok).map(|entry| entry.path()).collect()
                })
                .unwrap_or_default();
            entries.sort();
            for package in entries {
                let candidate = package.join("metadata/generation.json");
                if candidate.exists()
                    && let Some(record) = ingest(&candidate)
                {
                    records.push(record);
                }
            }
            info!("   celestial-atlas: {} packages ingested from {dir}", records.len());
        }
        let current_meta =
            std::path::PathBuf::from(format!("{}/metadata/generation.json", ctx.seed_dir));
        let current_seed = format!("0x{}", ctx.seed_hex.to_uppercase());
        if !records.iter().any(|(seed, ..)| seed.eq_ignore_ascii_case(&current_seed)) {
            if let Some(record) = ingest(&current_meta) {
                records.push(record);
            } else {
                warn!("celestial-atlas: current generation.json unreadable; charting a lone star");
                records.push((current_seed.clone(), vec![0.5; 6], 1_000.0, 180.0));
            }
        }

        // --- Embed and relax into the sky disc.
        let features: Vec<Vec<f64>> =
            records.iter().map(|(_, features, ..)| features.clone()).collect();
        let mut positions = embed(&features);
        if positions.len() == 1 {
            positions[0] = (0.0, 0.0);
        }
        // Normalize into the unit disc.
        let max_radius = positions.iter().map(|&(x, y)| x.hypot(y)).fold(1e-9f64, f64::max);
        for position in &mut positions {
            position.0 /= max_radius * 1.25;
            position.1 /= max_radius * 1.25;
        }
        // Repulsion relaxation (fixed order, deterministic).
        for _ in 0..RELAX_ITERATIONS {
            for i in 0..positions.len() {
                let mut push = (0.0f64, 0.0f64);
                for j in 0..positions.len() {
                    if i == j {
                        continue;
                    }
                    let dx = positions[i].0 - positions[j].0;
                    let dy = positions[i].1 - positions[j].1;
                    let dist_sq = (dx * dx + dy * dy).max(1e-6);
                    if dist_sq < 0.012 {
                        push.0 += dx / dist_sq * 3e-5;
                        push.1 += dy / dist_sq * 3e-5;
                    }
                }
                positions[i].0 = (positions[i].0 + push.0).clamp(-0.98, 0.98);
                positions[i].1 = (positions[i].1 + push.1).clamp(-0.98, 0.98);
            }
        }

        let stars: Vec<Star> = records
            .iter()
            .zip(positions.iter())
            .map(|((seed, features, score, hue), &position)| Star {
                seed: seed.clone(),
                features: features.clone(),
                score: *score,
                hue: *hue,
                position,
                is_current: seed.eq_ignore_ascii_case(&current_seed),
            })
            .collect();

        // --- Constellations: single-linkage under the 12th-percentile gap.
        let mut links: Vec<(usize, usize)> = Vec::new();
        let mut cluster_of: Vec<usize> = (0..stars.len()).collect();
        if stars.len() >= MIN_CLUSTER {
            let mut gaps: Vec<f64> = Vec::new();
            for i in 0..stars.len() {
                for j in i + 1..stars.len() {
                    let dx = stars[i].position.0 - stars[j].position.0;
                    let dy = stars[i].position.1 - stars[j].position.1;
                    gaps.push(dx.hypot(dy));
                }
            }
            gaps.sort_by(f64::total_cmp);
            let threshold = gaps[((gaps.len() - 1) as f64 * 0.12) as usize];
            let find = |mut node: usize, parents: &[usize]| -> usize {
                while parents[node] != node {
                    node = parents[node];
                }
                node
            };
            for i in 0..stars.len() {
                for j in i + 1..stars.len() {
                    let dx = stars[i].position.0 - stars[j].position.0;
                    let dy = stars[i].position.1 - stars[j].position.1;
                    if dx.hypot(dy) <= threshold {
                        links.push((i, j));
                        let (ri, rj) = (find(i, &cluster_of), find(j, &cluster_of));
                        if ri != rj {
                            cluster_of[ri.max(rj)] = ri.min(rj);
                        }
                    }
                }
            }
            // Path-compress.
            for index in 0..cluster_of.len() {
                cluster_of[index] = find(index, &cluster_of);
            }
        }

        // --- Chart: engraved deep-black square poster.
        let edge = ctx.quality.scale_dim(2_800).max(280) as usize;
        let paper = Paper::DeepBlack;
        let mut chart = vec![paper_color(paper); edge * edge];
        let ink = ink_color(paper);
        let faint = (ink.0 * 0.16, ink.1 * 0.16, ink.2 * 0.16);
        let center = edge as f64 / 2.0;
        let sky_radius = edge as f64 * 0.42;
        let to_px = |position: (f64, f64)| -> (f64, f64) {
            (center + position.0 * sky_radius, center + position.1 * sky_radius)
        };
        // Graticule: rings and spokes.
        for ring in 1..=4 {
            let radius = sky_radius * f64::from(ring) / 4.0;
            let mut previous: Option<(f32, f32)> = None;
            for sample in 0..=128 {
                let theta = f64::from(sample) / 128.0 * std::f64::consts::TAU;
                let point = (
                    (center + radius * theta.cos()) as f32,
                    (center + radius * theta.sin()) as f32,
                );
                if let Some(prev) = previous {
                    draw_line(&mut chart, edge, edge, prev, point, faint, 1.0, 0.8);
                }
                previous = Some(point);
            }
        }
        for spoke in 0..12 {
            let theta = f64::from(spoke) / 12.0 * std::f64::consts::TAU;
            draw_line(
                &mut chart,
                edge,
                edge,
                (center as f32, center as f32),
                (
                    (center + sky_radius * theta.cos()) as f32,
                    (center + sky_radius * theta.sin()) as f32,
                ),
                faint,
                0.8,
                0.6,
            );
        }
        // Constellation hairlines.
        for &(i, j) in &links {
            let a = to_px(stars[i].position);
            let b = to_px(stars[j].position);
            draw_line(
                &mut chart,
                edge,
                edge,
                (a.0 as f32, a.1 as f32),
                (b.0 as f32, b.1 as f32),
                (ink.0 * 0.35, ink.1 * 0.35, ink.2 * 0.35),
                1.0,
                0.8,
            );
        }
        // Stars: magnitude glyphs with diffraction-spike sigils.
        let score_max = stars.iter().map(|s| s.score).fold(1e-9, f64::max);
        for star in &stars {
            let (x, y) = to_px(star.position);
            let magnitude = ((star.score / score_max) * MAGNITUDES as f64).ceil().max(1.0);
            let size = 2.0 + magnitude * (edge as f64 / 700.0);
            let (l, a, b) = oklch_to_oklab(0.86, 0.07, star.hue);
            let (r, g, blue) = oklab_to_linear_rec2020(l, a, b);
            let color: Rgb64 = (r.max(0.0) * 1.6, g.max(0.0) * 1.6, blue.max(0.0) * 1.6);
            for (dx, dy) in [(1.0f64, 0.0f64), (0.0, 1.0), (0.7, 0.7), (0.7, -0.7)] {
                draw_line(
                    &mut chart,
                    edge,
                    edge,
                    ((x - dx * size) as f32, (y - dy * size) as f32),
                    ((x + dx * size) as f32, (y + dy * size) as f32),
                    color,
                    if dx == 1.0 || dy == 1.0 { 1.4 } else { 0.8 },
                    0.9,
                );
            }
            if star.is_current {
                let ring = size * 1.9;
                let mut previous: Option<(f32, f32)> = None;
                for sample in 0..=48 {
                    let theta = f64::from(sample) / 48.0 * std::f64::consts::TAU;
                    let point = ((x + ring * theta.cos()) as f32, (y + ring * theta.sin()) as f32);
                    if let Some(prev) = previous {
                        draw_line(&mut chart, edge, edge, prev, point, color, 1.2, 0.9);
                    }
                    previous = Some(point);
                }
            }
        }
        // Legend + mythology lines.
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        {
            let mut by_root: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();
            for (index, &root) in cluster_of.iter().enumerate() {
                by_root.entry(root).or_default().push(index);
            }
            let mut roots: Vec<usize> = by_root.keys().copied().collect();
            roots.sort_unstable();
            for root in roots {
                let members = by_root.remove(&root).unwrap_or_default();
                if members.len() >= MIN_CLUSTER {
                    clusters.push(members);
                }
            }
        }
        let mut mythologies = String::new();
        let label_px = type_px(edge, -1);
        for (index, members) in clusters.iter().enumerate() {
            let cluster_stars: Vec<&Star> = members.iter().map(|&m| &stars[m]).collect();
            let (name, clause) = constellation_name(&cluster_stars, index);
            let _ = writeln!(mythologies, "{name} \u{2014} {clause}");
            // Label at the cluster centroid.
            let cx =
                members.iter().map(|&m| stars[m].position.0).sum::<f64>() / members.len() as f64;
            let cy =
                members.iter().map(|&m| stars[m].position.1).sum::<f64>() / members.len() as f64;
            let (lx, ly) = to_px((cx, cy));
            let style = TextStyle {
                align: Align::Center,
                opacity: 0.7,
                ..TextStyle::caption(Face::SansItalic, label_px, ink)
            };
            draw_text(&mut chart, edge, edge, lx, ly - label_px * 2.0, &style, &name);
        }
        if mythologies.is_empty() {
            mythologies.push_str("A LONE LANTERN \u{2014} one seed, charted while the sky fills\n");
        }
        // Title and legend.
        let title_style = TextStyle {
            align: Align::Center,
            tracking: 0.18,
            ..TextStyle::caption(Face::Sans, type_px(edge, 2), ink)
        };
        draw_text(
            &mut chart,
            edge,
            edge,
            center,
            edge as f64 * 0.055,
            &title_style,
            &format!("CELESTIAL ATLAS \u{b7} {} SEEDS", stars.len()),
        );
        let legend_style = TextStyle {
            align: Align::Center,
            opacity: 0.65,
            ..TextStyle::caption(Face::Mono, type_px(edge, -1), ink)
        };
        draw_text(
            &mut chart,
            edge,
            edge,
            center,
            edge as f64 * 0.945,
            &legend_style,
            "brightness: borda score \u{b7} hue: palette anchor \u{b7} ring: this seed",
        );
        draw_footer(&mut chart, edge, edge, ctx, "V63 CELESTIAL-ATLAS", paper);
        let image = encode_linear_rec2020_png16(&chart, edge as u32, edge as u32);
        sink.save_png16(&image, "atlas.png")?;
        sink.write_text("mythologies.txt", &mythologies, "data")?;

        let atlas: Vec<serde_json::Value> = stars
            .iter()
            .enumerate()
            .map(|(index, star)| {
                serde_json::json!({
                    "seed": star.seed,
                    "position": [star.position.0, star.position.1],
                    "score": star.score,
                    "hue": star.hue,
                    "cluster": cluster_of[index],
                    "current": star.is_current,
                })
            })
            .collect();
        let meta = serde_json::json!({
            "stars": atlas,
            "relax_iterations": RELAX_ITERATIONS,
            "min_cluster": MIN_CLUSTER,
            "note": "invoke with --viz-seeds-dir <dir of seed packages> to chart the collection",
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("atlas.json", &json, "data")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_is_deterministic_and_separates_distinct_rows() {
        let rows = vec![
            vec![0.0, 0.0, 1.0, 0.2, 0.3, 0.5],
            vec![1.0, 0.5, 0.0, 0.8, 0.9, 0.1],
            vec![0.1, 0.05, 0.9, 0.25, 0.35, 0.45],
            vec![0.9, 0.55, 0.1, 0.75, 0.85, 0.15],
        ];
        let a = embed(&rows);
        let b = embed(&rows);
        assert_eq!(a, b, "embedding must be deterministic");
        let near = |p: (f64, f64), q: (f64, f64)| (p.0 - q.0).hypot(p.1 - q.1);
        assert!(near(a[0], a[2]) < near(a[0], a[1]), "kindred rows must sit closer");
    }
}
