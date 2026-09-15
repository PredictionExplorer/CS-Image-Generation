//! Small real pipeline checks for cache interchange and resumable silk frames.
//!
//! Source motion is integrated from explicit three-dimensional physical states.
//! No candidate search or external media encoder is needed by these tests.

use std::path::Path;
use std::process::{Command, Output};

use serde::Serialize;
use three_body_problem::silk::orbit::{InitialBody, OrbitConfig};
use three_body_problem::silk::render::RenderConfig;
use three_body_problem::silk::simulation::SimulationConfig;
use three_body_problem::silk::{V3, cache};

fn run(args: &[&str], directory: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tidal_silk"))
        .args(args)
        .current_dir(directory)
        .output()
        .expect("tidal_silk should launch")
}

fn succeeded(output: &Output) {
    assert!(
        output.status.success(),
        "tidal_silk failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn write_json(path: &Path, value: &impl Serialize) {
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

fn source_config() -> OrbitConfig {
    OrbitConfig {
        seed: "0xcafe".into(),
        steps: 128,
        sample_stride: 4,
        initial_conditions: Some([
            InitialBody {
                mass: 140.0,
                position: V3::new(0.0, 160.0, 20.0),
                velocity: V3::new(0.1, 0.4, 0.2),
            },
            InitialBody {
                mass: 180.0,
                position: V3::new(-140.0, -80.0, -10.0),
                velocity: V3::new(-0.3, -0.2, 0.1),
            },
            InitialBody {
                mass: 200.0,
                position: V3::new(140.0, -80.0, 10.0),
                velocity: V3::new(0.2, -0.1, -0.2),
            },
        ]),
        ..OrbitConfig::default()
    }
}

#[test]
fn physical_orbit_to_cached_cloth_to_rgb16_preserves_compatible_resume() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write_json(&root.join("orbit-config.json"), &source_config());
    succeeded(&run(&["orbit", "--config", "orbit-config.json", "--output", "orbit.bin"], root));
    let orbit = cache::read_orbit(&root.join("orbit.bin")).unwrap();
    assert_eq!(orbit.seed, "0xcafe");
    assert_eq!(orbit.samples.len(), 32);
    assert_ne!(orbit.samples[0], orbit.samples[31]);
    assert_eq!(orbit.provenance["selection"]["source"], "explicit-initial-conditions");

    let config = SimulationConfig {
        subdivisions: 8,
        frames: 3,
        fps: 24,
        substeps: 3,
        iterations: 8,
        settle_seconds: 0.3,
        preroll_seconds: 0.1,
        self_collision: false,
        ..SimulationConfig::default()
    };
    write_json(&root.join("cloth-config.json"), &config);
    succeeded(&run(
        &["bake", "--orbit", "orbit.bin", "--config", "cloth-config.json", "--output", "cloth.bin"],
        root,
    ));
    let bake = cache::read_bake(&root.join("cloth.bin")).unwrap();
    assert_eq!(bake.frames.len(), 3);
    assert!(!bake.mesh.triangles.is_empty());
    assert_eq!(
        bake.recipe["orbit_cache_sha256"],
        cache::file_hash(&root.join("orbit.bin")).unwrap()
    );

    let config = RenderConfig {
        width: 40,
        height: 40,
        samples_per_pixel: 1,
        light_samples: 1,
        max_bounces: 0,
        ..RenderConfig::default()
    };
    write_json(&root.join("render-config.json"), &config);
    let arguments = [
        "--threads",
        "1",
        "render",
        "--bake",
        "cloth.bin",
        "--config",
        "render-config.json",
        "--output",
        "frames",
        "--frame",
        "1",
    ];
    succeeded(&run(&arguments, root));
    let frame_path = root.join("frames/frame_000001.png");
    let pixels = image::open(&frame_path).unwrap();
    assert!(pixels.as_rgb16().is_some(), "master frame must retain 16-bit channels");
    let original_hash = cache::file_hash(&frame_path).unwrap();
    let resumed = run(&arguments, root);
    succeeded(&resumed);
    assert!(String::from_utf8_lossy(&resumed.stderr).contains("Reusing frame 1"));
    assert_eq!(cache::file_hash(&frame_path).unwrap(), original_hash);

    let mut conflicting = arguments.to_vec();
    conflicting.extend_from_slice(&["--palette", "indigo"]);
    let refused = run(&conflicting, root);
    assert!(!refused.status.success(), "different material must not reuse old frames");
    assert_eq!(cache::file_hash(&frame_path).unwrap(), original_hash);

    // Emulate interruption immediately after --overwrite publishes the new
    // manifest, while its old PNG and receipt still belong to the old material.
    let manifest_path = root.join("frames/render.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["config"]["palette"] = "indigo".into();
    manifest["complete"] = false.into();
    write_json(&manifest_path, &manifest);
    let recovered = run(&conflicting, root);
    succeeded(&recovered);
    assert!(!String::from_utf8_lossy(&recovered.stderr).contains("Reusing frame 1"));
    let current_config = RenderConfig { palette: "indigo".into(), ..config };
    let expected =
        three_body_problem::silk::render::render_frame16(&bake, 1, &current_config).unwrap();
    assert_eq!(image::open(&frame_path).unwrap().to_rgb16(), expected);

    // A valid PNG with the right dimensions but altered pixels must not be
    // mistaken for completed work solely because its filename still exists.
    image::ImageBuffer::from_pixel(40, 40, image::Rgb([0_u16; 3])).save(&frame_path).unwrap();
    succeeded(&run(&conflicting, root));
    assert_eq!(image::open(&frame_path).unwrap().to_rgb16(), expected);
}

#[test]
fn metadata_sidecar_cannot_replace_orbit_payload() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write_json(&root.join("config.json"), &source_config());
    succeeded(&run(&["orbit", "--config", "config.json", "--output", "unusual-output.json"], root));
    let orbit = cache::read_orbit(&root.join("unusual-output.json")).unwrap();
    assert_eq!(orbit.samples.len(), 32);
}

#[test]
fn descending_video_frame_indices_are_rejected_before_encoder_start() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write_json(
        &root.join("render.json"),
        &serde_json::json!({
            "schema_version":1,
            "bake_sha256":"fixture",
            "binary_sha256":"fixture",
            "config":RenderConfig::default(),
            "fps":30,
            "frames":[2,1],
            "complete":true,
        }),
    );
    let refused = run(&["encode", "--input", ".", "--output", "film.mp4"], root);
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("Frame indices must increase"));
    assert!(!root.join("film.mp4").exists());
}
