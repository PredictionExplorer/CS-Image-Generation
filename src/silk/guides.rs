//! Small identity markers for a separate, guided comparison of matching films.
//!
//! The source images remain unchanged. These markers locate the same three
//! source bodies in each presentation; they do not suggest the two projections
//! have the same shape, or conceal one presentation's artistic transformations.
use super::{
    ClothBake, OrbitData, SilkResult, V3,
    render::{RenderConfig, project_points},
};
use image::RgbImage;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::Path};

const COLORS: [[u8; 3]; 3] = [[248, 113, 113], [251, 191, 36], [34, 211, 238]];
const GLYPHS: [[u8; 7]; 3] = [
    [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
];

/// Export source-body positions in the silk film's exact rendered camera.
///
/// The schema uses `frames[].bodies` as three normalized `(x, y)` pairs ordered
/// A/B/C. Frame numbers start at zero; off-screen coordinates stay unclamped.
/// Visible source endpoints come from the bake, including its pre-roll choice.
pub fn generate_silk_markers(
    orbit: &OrbitData,
    bake: &ClothBake,
    config: &RenderConfig,
) -> SilkResult<Value> {
    if orbit.samples.len() < 2 || bake.frames.is_empty() || bake.fps == 0 {
        return Err("markers require a nonempty bake and at least two orbit samples".into());
    }
    if let Some(seed) = bake.recipe["seed"].as_str()
        && seed != orbit.seed
    {
        return Err("orbit seed does not match the cloth recipe".into());
    }
    if let Some(samples) = bake.recipe["source_samples"].as_u64()
        && samples != orbit.samples.len() as u64
    {
        return Err("orbit sample count does not match the cloth recipe".into());
    }
    let center: V3 = serde_json::from_value(bake.recipe["world_center"].clone())?;
    let scale = bake.recipe["world_scale"].as_f64().ok_or("cloth recipe has no world_scale")?;
    let first = bake.recipe["first_visible_source_fraction"]
        .as_f64()
        .ok_or("cloth recipe has no visible source start")?;
    let last = bake.recipe["last_visible_source_fraction"]
        .as_f64()
        .or_else(|| bake.recipe["source_end_fraction"].as_f64())
        .ok_or("cloth recipe has no visible source end")?;
    if !center.is_finite()
        || !scale.is_finite()
        || scale <= 0.0
        || !(0.0..=1.0).contains(&first)
        || !(first..=1.0).contains(&last)
    {
        return Err("invalid cloth source normalization or visible source interval".into());
    }
    let count = bake.frames.len();
    let fractions: Vec<f64> = (0..count)
        .map(|frame| {
            if frame == 0 {
                first
            } else if frame + 1 == count {
                last
            } else {
                first + (last - first) * frame as f64 / (count - 1) as f64
            }
        })
        .collect();
    let world: Vec<V3> = fractions
        .iter()
        .flat_map(|&fraction| {
            let index = fraction * (orbit.samples.len() - 1) as f64;
            let left = index.floor() as usize;
            let right = (left + 1).min(orbit.samples.len() - 1);
            (0..3).map(move |body| {
                (orbit.samples[left][body].lerp(orbit.samples[right][body], index - left as f64)
                    - center)
                    * scale
            })
        })
        .collect();
    // A film normally has one fixed fit. Project the complete marker sequence
    // together so computing its global bounds is linear in the bake size.
    let projected = if config.fit_all_frames {
        project_points(bake, 0, config, &world)?
    } else {
        let mut projected = Vec::with_capacity(world.len());
        for (frame, points) in world.chunks_exact(3).enumerate() {
            projected.extend(project_points(bake, frame, config, points)?);
        }
        projected
    };
    let frames: Vec<Value> = projected
        .chunks_exact(3)
        .enumerate()
        .map(|(frame, positions)| {
            json!({
                "frame":frame,"source_fraction":fractions[frame],"bodies":positions,
            })
        })
        .collect();
    Ok(json!({
        "schema_version":1,"seed":orbit.seed,"fps":bake.fps,
        "width":config.width,"height":config.height,
        "coordinate_system":"normalized_xy_top_left","body_ids":["A","B","C"],
        "body_colors":COLORS,"connectors":false,
        "projection":"raw-physics-through-silk-camera",
        "first_visible_source_fraction":first,"last_visible_source_fraction":last,
        "frames":frames,
    }))
}

/// Save a separate RGB8 PNG with three colored A/B/C body markers.
pub fn overlay_image(input: &Path, output: &Path, positions: [[f64; 2]; 3]) -> SilkResult<()> {
    overlay_image_with_connectors(input, output, positions, false)
}

/// Save identity markers, optionally joining them with a faint triangle.
pub fn overlay_image_with_connectors(
    input: &Path,
    output: &Path,
    positions: [[f64; 2]; 3],
    connectors: bool,
) -> SilkResult<()> {
    if input == output || (output.exists() && fs::canonicalize(input)? == fs::canonicalize(output)?)
    {
        return Err("guide output must be separate from the original image".into());
    }
    if !positions.iter().flatten().all(|value| value.is_finite()) {
        return Err("marker coordinates must be finite".into());
    }
    let mut image = image::open(input)?.to_rgb8();
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err("cannot annotate an empty image".into());
    }
    let pixels = positions.map(|p| [p[0] * f64::from(width), p[1] * f64::from(height)]);
    if connectors {
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            line(&mut image, pixels[a], pixels[b]);
        }
    }
    let radius = 5.0 * (f64::from(width.min(height)) / 720.0).sqrt().clamp(0.8, 2.0);
    let glyph_scale = if width.min(height) >= 1000 { 3 } else { 2 };
    for (body, &position) in positions.iter().enumerate() {
        if !(0.0..=1.0).contains(&position[0]) || !(0.0..=1.0).contains(&position[1]) {
            continue;
        }
        let [x, y] = pixels[body];
        disc(&mut image, [x, y], radius + 2.0, [2, 5, 10], 0.90);
        disc(&mut image, [x, y], radius, COLORS[body], 0.98);
        let label_width = 5 * glyph_scale;
        let label_height = 7 * glyph_scale;
        let mut label_x = (x + radius + 5.0).round() as i32;
        if label_x + label_width >= width as i32 {
            label_x = (x - radius - 5.0).round() as i32 - label_width;
        }
        let label_y = (y - f64::from(label_height) * 0.5)
            .round()
            .clamp(1.0, f64::from(height.saturating_sub(label_height as u32 + 1)).max(1.0))
            as i32;
        glyph(&mut image, label_x, label_y, glyph_scale, GLYPHS[body], COLORS[body]);
    }
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    image.save_with_format(output, image::ImageFormat::Png)?;
    Ok(())
}

/// Overlay an exported marker sequence on zero-indexed `frame_XXXXXX.png` files.
///
/// `connectors` is optional in the JSON and defaults to false. Original frames
/// remain untouched; duplicate frame numbers and mismatched dimensions fail.
pub fn overlay_frames(input_dir: &Path, output_dir: &Path, markers: &Value) -> SilkResult<()> {
    if input_dir == output_dir
        || (output_dir.exists() && fs::canonicalize(input_dir)? == fs::canonicalize(output_dir)?)
    {
        return Err("guide directory must differ from the original directory".into());
    }
    if markers["coordinate_system"] != "normalized_xy_top_left" {
        return Err("unsupported marker coordinate system".into());
    }
    let frames = markers["frames"].as_array().ok_or("markers have no frames array")?;
    if frames.is_empty() {
        return Err("marker sequence is empty".into());
    }
    let connectors = markers["connectors"].as_bool().unwrap_or(false);
    let width = markers["width"].as_u64().ok_or("markers have no image width")?;
    let height = markers["height"].as_u64().ok_or("markers have no image height")?;
    let mut seen = BTreeSet::new();
    let mut work = Vec::with_capacity(frames.len());
    for frame in frames {
        let number = frame["frame"].as_u64().ok_or("marker frame number is not an integer")?;
        if !seen.insert(number) {
            return Err("duplicate marker frame number".into());
        }
        let positions: [[f64; 2]; 3] = serde_json::from_value(frame["bodies"].clone())?;
        let name = format!("frame_{number:06}.png");
        let input = input_dir.join(&name);
        let dimensions = image::image_dimensions(&input)?;
        if u64::from(dimensions.0) != width || u64::from(dimensions.1) != height {
            return Err(format!("marker dimensions do not match {}", input.display()).into());
        }
        work.push((input, output_dir.join(name), positions));
    }
    fs::create_dir_all(output_dir)?;
    work.par_iter().try_for_each(|(input, output, positions)| {
        overlay_image_with_connectors(input, output, *positions, connectors)
    })
}

fn blend(image: &mut RgbImage, x: i32, y: i32, color: [u8; 3], alpha: f64) {
    if x < 0 || y < 0 || x >= image.width() as i32 || y >= image.height() as i32 {
        return;
    }
    let pixel = image.get_pixel_mut(x as u32, y as u32);
    for channel in 0..3 {
        pixel[channel] = (f64::from(pixel[channel]) * (1.0 - alpha)
            + f64::from(color[channel]) * alpha)
            .round() as u8;
    }
}

fn disc(image: &mut RgbImage, center: [f64; 2], radius: f64, color: [u8; 3], alpha: f64) {
    for y in (center[1] - radius - 1.0).floor() as i32..=(center[1] + radius + 1.0).ceil() as i32 {
        for x in
            (center[0] - radius - 1.0).floor() as i32..=(center[0] + radius + 1.0).ceil() as i32
        {
            let distance = (f64::from(x) + 0.5 - center[0]).hypot(f64::from(y) + 0.5 - center[1]);
            let coverage = (radius + 0.5 - distance).clamp(0.0, 1.0);
            if coverage > 0.0 {
                blend(image, x, y, color, alpha * coverage);
            }
        }
    }
}

fn glyph(image: &mut RgbImage, x: i32, y: i32, scale: i32, rows: [u8; 7], color: [u8; 3]) {
    for outline in [true, false] {
        for (row, &bits) in rows.iter().enumerate() {
            for column in 0..5 {
                if bits & (1 << (4 - column)) == 0 {
                    continue;
                }
                let pad = i32::from(outline);
                for dy in -pad..scale + pad {
                    for dx in -pad..scale + pad {
                        blend(
                            image,
                            x + column * scale + dx,
                            y + row as i32 * scale + dy,
                            if outline { [2, 5, 10] } else { color },
                            if outline { 0.85 } else { 1.0 },
                        );
                    }
                }
            }
        }
    }
}

fn line(image: &mut RgbImage, a: [f64; 2], b: [f64; 2]) {
    let distance = (a[0] - b[0]).hypot(a[1] - b[1]);
    // Very distant off-screen points must not turn annotation into a long loop.
    let steps = distance.ceil().min(f64::from(image.width().max(image.height())) * 4.0) as usize;
    for step in 0..=steps {
        let t = step as f64 / steps.max(1) as f64;
        blend(
            image,
            (a[0] + (b[0] - a[0]) * t).round() as i32,
            (a[1] + (b[1] - a[1]) * t).round() as i32,
            [174, 194, 214],
            0.16,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::Mesh;
    use super::*;
    use image::Rgb;
    #[test]
    fn source_interval_and_world_transform_locate_the_same_body() {
        let points = vec![
            V3::new(-1.0, -1.0, 0.0),
            V3::new(1.0, -1.0, 0.0),
            V3::new(1.0, 1.0, 0.0),
            V3::new(-1.0, 1.0, 0.0),
        ];
        let bake = ClothBake {
            mesh: Mesh {
                positions: points.clone(),
                triangles: vec![[0, 1, 2], [0, 2, 3]],
                uv: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            },
            frames: vec![points; 3],
            fps: 30,
            stats: vec![],
            recipe: json!({"seed":"0x01","source_samples":2,"world_center":{"x":10.0,"y":0.0,"z":0.0},"world_scale":2.0,"first_visible_source_fraction":0.25,"last_visible_source_fraction":0.75}),
        };
        let orbit = OrbitData {
            seed: "0x01".into(),
            dt: 0.001,
            masses: [1.0; 3],
            samples: vec![
                [V3::new(9.8, 0.0, 0.0), V3::new(10.0, 0.0, 0.0), V3::new(10.2, 0.0, 0.0)],
                [V3::new(9.6, 0.0, 0.0), V3::new(10.1, 0.0, 0.0), V3::new(10.4, 0.0, 0.0)],
            ],
            provenance: json!({}),
        };
        let config = RenderConfig {
            azimuth_degrees: 0.0,
            elevation_degrees: 0.0,
            roll_degrees: 0.0,
            ..RenderConfig::default()
        };
        let markers = generate_silk_markers(&orbit, &bake, &config).unwrap();
        assert_eq!(markers["body_ids"], json!(["A", "B", "C"]));
        for (frame, fraction) in [0.25, 0.5, 0.75].into_iter().enumerate() {
            assert_eq!(markers["frames"][frame]["source_fraction"], fraction);
        }
        assert!(
            (markers["frames"][0]["bodies"][0][0].as_f64().unwrap()
                - (0.5 - 0.25 / config.distance_scale))
                .abs()
                < 1e-12
        );
        assert!(
            (markers["frames"][2]["bodies"][0][0].as_f64().unwrap()
                - (0.5 - 0.35 / config.distance_scale))
                .abs()
                < 1e-12
        );
    }
    #[test]
    fn overlay_keeps_original_and_uses_stable_identity_colors() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        let output = dir.path().join("guide.png");
        RgbImage::from_pixel(120, 80, Rgb([10, 10, 10])).save(&input).unwrap();
        let original = fs::read(&input).unwrap();
        overlay_image(&input, &output, [[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]]).unwrap();
        assert_eq!(fs::read(&input).unwrap(), original);
        let image = image::open(output).unwrap().to_rgb8();
        for (body, x) in [24, 60, 96].into_iter().enumerate() {
            let pixel = image.get_pixel(x, 40);
            for channel in 0..3 {
                assert!((i32::from(pixel[channel]) - i32::from(COLORS[body][channel])).abs() <= 5);
            }
        }
        assert_eq!(image.get_pixel(0, 0), &Rgb([10, 10, 10]));
    }
    #[test]
    fn off_screen_markers_are_not_clamped_onto_the_frame() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        let output = dir.path().join("guide.png");
        let original = RgbImage::from_pixel(32, 32, Rgb([5, 5, 5]));
        original.save(&input).unwrap();
        overlay_image(&input, &output, [[-0.1, 0.5], [1.1, 0.5], [0.5, -0.1]]).unwrap();
        assert_eq!(image::open(output).unwrap().to_rgb8(), original);
    }
}
