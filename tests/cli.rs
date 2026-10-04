//! CLI binary integration tests.
//!
//! Validates that the binary accepts valid arguments, rejects invalid ones with the documented
//! exit statuses, and writes (or withholds) the ember edition's files as documented.
//!
//! The tests that render a package need `ffmpeg` on `PATH` (the stills' WebP derivatives). They
//! skip with a message when it is missing, except under CI (the `CI` environment variable is
//! set), where a missing `ffmpeg` fails them instead of letting them pass without running.

use std::path::Path;
use std::process::{Command, Output};

use three_body_problem::app::EMBER_OUTPUT_PATHS;

fn binary_path() -> std::path::PathBuf {
    let mut path = std::env::current_exe().expect("failed to get test binary path");
    path.pop();
    path.pop();
    path.push("three_body_problem");
    path
}

fn run_binary(args: &[&str]) -> Output {
    Command::new(binary_path()).args(args).output().expect("failed to execute binary")
}

/// Runs the binary with `args` in `dir` (packages are written to `dir/output/<name>`).
fn run_binary_in(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary_path())
        .current_dir(dir)
        .args(args)
        .output()
        .expect("failed to execute binary")
}

/// Standard output followed by standard error, for assertion messages.
fn log_of(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Whether `ffmpeg` runs here. When it does not, the calling test `test` should return early:
/// locally it is skipped with a message; under CI (the `CI` environment variable is set) this
/// panics instead, so that a runner without `ffmpeg` fails rather than passes without testing.
fn ffmpeg_available(test: &str) -> bool {
    let available =
        Command::new("ffmpeg").arg("-version").output().is_ok_and(|output| output.status.success());
    if !available {
        assert!(
            std::env::var_os("CI").is_none(),
            "{test} needs ffmpeg on PATH, and CI is set: install ffmpeg in this job"
        );
        eprintln!("skipping {test}: ffmpeg is not on PATH");
    }
    available
}

/// Writes a placeholder for every ember edition file into the package at `package`, as an
/// earlier full run into the same output directory would have left them.
fn plant_stale_ember_files(package: &Path) {
    for file in EMBER_OUTPUT_PATHS {
        let path = package.join(file);
        std::fs::create_dir_all(path.parent().expect("parent directory")).expect("package dir");
        std::fs::write(&path, b"stale ember output of an earlier run").expect("stale file");
    }
}

/// Asserts that no ember edition file is in the package and that its asset manifest lists none.
fn assert_no_ember_files(package: &Path) {
    for file in EMBER_OUTPUT_PATHS {
        assert!(!package.join(file).exists(), "{file} must not exist");
    }
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(package.join("metadata/assets.json")).expect("assets.json"),
    )
    .expect("assets.json is JSON");
    let assets = manifest["assets"].as_array().expect("assets array");
    assert!(!assets.is_empty(), "the main outputs are listed: {manifest}");
    for asset in assets {
        let role = asset["role"].as_str().expect("role");
        let path = asset["path"].as_str().expect("path");
        assert!(!role.starts_with("ember") && !path.contains("ember"), "{asset}");
    }
}

/// The core files every package written by a tiny `--image-only` run must contain.
const IMAGE_ONLY_CORE_FILES: [&str; 6] = [
    "images/source/master.png",
    "images/web/full.webp",
    "images/web/preview.webp",
    "metadata/assets.json",
    "metadata/generation.json",
    "metadata/nft_traits.json",
];

#[test]
fn help_flag_exits_successfully() {
    let output = run_binary(&["--help"]);
    assert!(output.status.success(), "--help should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage"), "help output should contain Usage");
}

#[test]
fn help_lists_the_no_ember_flag() {
    let output = run_binary(&["--help"]);
    assert!(output.status.success(), "--help should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.lines().any(|line| line.trim_start().starts_with("--no-ember")),
        "--help should list --no-ember:\n{stdout}"
    );
    assert!(stdout.contains("Skip the ember edition"), "--no-ember should be documented");
}

#[test]
fn an_orbit_the_ember_edition_rejects_still_yields_the_rest_of_the_package() {
    // One recorded step is too short for the ember edition: the preflight skips it, the rest of
    // the package is written, and the generator exits 3 ("complete except the ember edition").
    // The main still's WebP derivatives need FFmpeg.
    if !ffmpeg_available("an_orbit_the_ember_edition_rejects_still_yields_the_rest_of_the_package")
    {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let package = dir.path().join("output/one-step");
    // A complete ember edition of an earlier run into the same directory must not survive.
    plant_stale_ember_files(&package);
    let output = run_binary_in(
        dir.path(),
        &[
            "--seed",
            "0x01",
            "--sims",
            "2",
            "--steps",
            "1",
            "--resolution",
            "64x40",
            "--output",
            "one-step",
            "--image-only",
        ],
    );
    let log = log_of(&output);
    assert_eq!(output.status.code(), Some(3), "complete except the ember edition:\n{log}");
    assert!(log.contains("STAGE EMBER (preflight)"), "{log}");
    assert!(log.contains("Pass --no-ember"), "the warning must name the way out:\n{log}");
    for file in IMAGE_ONLY_CORE_FILES {
        assert!(package.join(file).is_file(), "{file} must be written");
    }
    assert_no_ember_files(&package);
}

#[test]
fn no_ember_removes_the_ember_files_of_an_earlier_run() {
    // A --no-ember run into a directory that holds an earlier run's ember edition must not ship
    // those files next to a manifest that does not list them.
    if !ffmpeg_available("no_ember_removes_the_ember_files_of_an_earlier_run") {
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let package = dir.path().join("output/stale");
    plant_stale_ember_files(&package);
    let output = run_binary_in(
        dir.path(),
        &[
            "--seed",
            "0x01",
            "--sims",
            "2",
            "--steps",
            "1",
            "--resolution",
            "64x40",
            "--output",
            "stale",
            "--image-only",
            "--no-ember",
        ],
    );
    let log = log_of(&output);
    assert_eq!(output.status.code(), Some(0), "--no-ember completes the package:\n{log}");
    assert!(log.contains("STAGE EMBER: skipped (--no-ember)"), "{log}");
    for file in IMAGE_ONLY_CORE_FILES {
        assert!(package.join(file).is_file(), "{file} must be written");
    }
    assert_no_ember_files(&package);
}

/// Size and side of the blocks the colour checks average over (4:2:0 chroma noise on thin lines
/// averages out; a colour conversion does not).
const WIRING_SIZE: (usize, usize) = (160, 104);
const WIRING_BLOCK: usize = 4;

/// Decodes `path` to 16-bit RGB frames. Videos are converted with the BT.709 matrix every encode
/// is tagged with; `select` (an `FFmpeg` frame-select expression) or `from_end` (seconds before
/// the end) limits the frames.
fn decode_rgb48(path: &Path, select: Option<&str>, from_end: Option<&str>) -> Vec<Vec<u16>> {
    let mut command = Command::new("ffmpeg");
    command.args(["-v", "error"]);
    if let Some(seconds) = from_end {
        command.args(["-sseof", seconds]);
    }
    command.arg("-i").arg(path);
    let is_video = path.extension().is_some_and(|ext| ext == "mp4");
    if is_video {
        let scale = "scale=in_color_matrix=bt709:in_range=tv";
        let filter = select.map_or(scale.to_string(), |s| format!("select='{s}',{scale}"));
        command.args(["-vf", &filter, "-fps_mode", "passthrough"]);
    }
    let output = command
        .args(["-f", "rawvideo", "-pix_fmt", "rgb48le", "-"])
        .output()
        .expect("ffmpeg decodes");
    assert!(
        output.status.success(),
        "decode {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let samples: Vec<u16> =
        output.stdout.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    samples.chunks_exact(WIRING_SIZE.0 * WIRING_SIZE.1 * 3).map(<[u16]>::to_vec).collect()
}

/// Block means of a frame, in 8-bit levels.
fn block_means(frame: &[u16]) -> Vec<[f64; 3]> {
    let (width, height) = WIRING_SIZE;
    let mut means = vec![[0.0; 3]; (width / WIRING_BLOCK) * (height / WIRING_BLOCK)];
    let scale = 257.0 * (WIRING_BLOCK * WIRING_BLOCK) as f64;
    for y in 0..height {
        for x in 0..width {
            let block = (y / WIRING_BLOCK) * (width / WIRING_BLOCK) + x / WIRING_BLOCK;
            for channel in 0..3 {
                means[block][channel] += f64::from(frame[(y * width + x) * 3 + channel]) / scale;
            }
        }
    }
    means
}

/// Squared error of `decoded` against `good` over `bad` (the other encoding), summed over the
/// blocks where the two encodings differ by more than 4 levels, and the number of such blocks.
fn colour_errors(decoded: &[u16], good: &[u16], bad: &[u16]) -> (f64, f64, usize) {
    let (decoded, good, bad) = (block_means(decoded), block_means(good), block_means(bad));
    let mut errors = (0.0, 0.0, 0);
    for ((d, g), b) in decoded.iter().zip(&good).zip(&bad) {
        if (0..3).any(|c| (g[c] - b[c]).abs() > 4.0) {
            errors.0 += (0..3).map(|c| (d[c] - g[c]).powi(2)).sum::<f64>();
            errors.1 += (0..3).map(|c| (d[c] - b[c]).powi(2)).sum::<f64>();
            errors.2 += 1;
        }
    }
    errors
}

/// Asserts that `decoded` frames are much closer to `good` than to `bad`.
fn assert_encoding(label: &str, frames: &[(Vec<u16>, Vec<u16>, Vec<u16>)]) {
    let (mut good, mut bad, mut blocks) = (0.0, 0.0, 0);
    for (decoded, want, other) in frames {
        let (g, b, n) = colour_errors(decoded, want, other);
        good += g;
        bad += b;
        blocks += n;
    }
    assert!(blocks >= 40, "{label}: only {blocks} blocks tell the encodings apart");
    assert!(
        good < 0.5 * bad,
        "{label}: error {good:.0} against the right encoding, {bad:.0} against the other"
    );
}

#[test]
fn web_files_are_srgb_and_archival_films_display_p3() {
    // The web films and WebPs show the sRGB conversion of the Display P3 frames; the archival
    // films keep the P3 frames. Checked on the main film's last frame (the still), the WebP and
    // frames across the spectral sweep. Swapping the streams fails every check.
    if !ffmpeg_available("web_files_are_srgb_and_archival_films_display_p3") {
        return;
    }
    use three_body_problem::render::display_p3::to_srgb_samples;
    let dir = tempfile::tempdir().expect("temp dir");
    let output = run_binary_in(
        dir.path(),
        &[
            "--seed",
            "0x01",
            "--sims",
            "2",
            "--steps",
            "3600",
            "--resolution",
            &format!("{}x{}", WIRING_SIZE.0, WIRING_SIZE.1),
            "--output",
            "wiring",
            "--no-ember",
            "--fast-encode",
        ],
    );
    assert_eq!(output.status.code(), Some(0), "{}", log_of(&output));
    let package = dir.path().join("output/wiring");
    let file = |path: &str| package.join(path);

    let master = decode_rgb48(&file("images/source/master.png"), None, None).remove(0);
    let srgb = to_srgb_samples(&master);
    let last = |path: &str| decode_rgb48(&file(path), None, Some("-0.2")).pop().expect("frames");
    assert_encoding("web main.mp4", &[(last("videos/web/main.mp4"), srgb.clone(), master.clone())]);
    assert_encoding("hq main.mp4", &[(last("videos/hq/main.mp4"), master.clone(), srgb.clone())]);
    let webp = decode_rgb48(&file("images/web/full.webp"), None, None).remove(0);
    assert_encoding("full.webp", &[(webp, srgb, master)]);

    let every_50th = Some("not(mod(n\\,50))");
    let sweep_hq = decode_rgb48(&file("videos/hq/spectral_sweep.mp4"), every_50th, None);
    let sweep_web = decode_rgb48(&file("videos/web/spectral_sweep.mp4"), every_50th, None);
    assert_eq!(sweep_hq.len(), sweep_web.len(), "same sweep frames");
    let sweep: Vec<_> = sweep_web
        .into_iter()
        .zip(sweep_hq)
        .map(|(web, hq)| (web, to_srgb_samples(&hq), hq))
        .collect();
    assert_encoding("web spectral_sweep.mp4", &sweep);
}

#[test]
fn version_flag_exits_successfully() {
    let output = run_binary(&["--version"]);
    assert!(output.status.success(), "--version should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // 1.1.0 is the first version with the ember edition: nft_traits.json's pipeline_version and
    // the certificate's build.crate_version tell ember-era packages apart by it.
    assert_eq!(stdout.trim(), "three_body_problem 1.1.0");
    assert!(stdout.contains(env!("CARGO_PKG_VERSION")), "{stdout}");
}

/// run.py's stale-edition probe: the id of the ember look, alone on stdout, and exit status 0,
/// without touching the output directory.
#[test]
fn ember_algorithm_flag_prints_the_look_id() {
    let dir = tempfile::tempdir().expect("temp dir");
    let output = run_binary_in(dir.path(), &["--ember-algorithm"]);
    assert!(output.status.success(), "--ember-algorithm should exit 0: {}", log_of(&output));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout, format!("{}\n", three_body_problem::ember::certificate::ALGORITHM_VERSION));
    assert!(stdout.starts_with("ember-v"), "{stdout}");
    assert_eq!(std::fs::read_dir(dir.path()).expect("readable").count(), 0, "nothing written");
}

#[test]
fn exit_statuses_follow_the_documented_contract() {
    let dir = tempfile::tempdir().expect("temp dir");
    // 2: rejected by the argument parser (an unknown flag or a malformed value).
    for args in [
        &["--nonexistent-flag"][..],
        &["--resolution", "notaresolution"],
        &["--resolution", "0x0"],
        &["--sims", "0"],
        &["--no-ember=maybe"],
    ] {
        let output = run_binary_in(dir.path(), args);
        assert_eq!(output.status.code(), Some(2), "{args:?}:\n{}", log_of(&output));
    }
    // 1: any other failure, here before any work: an invalid --seed (odd number of hex digits)
    // or a resolution above 16,384 pixels per side.
    for args in [&["--seed", "0x123"][..], &["--resolution", "20000x40"]] {
        let output = run_binary_in(dir.path(), args);
        assert_eq!(output.status.code(), Some(1), "{args:?}:\n{}", log_of(&output));
    }
    assert!(!dir.path().join("output").exists(), "the rejected runs wrote nothing");
}

#[test]
fn invalid_resolution_is_rejected() {
    let output = run_binary(&["--resolution", "notaresolution", "--seed", "0xdeadbeef"]);
    assert!(!output.status.success(), "invalid resolution should cause non-zero exit");
}

#[test]
fn zero_resolution_is_rejected() {
    let output = run_binary(&["--resolution", "0x0", "--seed", "0xdeadbeef"]);
    assert!(!output.status.success(), "zero resolution should cause non-zero exit");
}

#[test]
fn unknown_flags_are_rejected() {
    let output = run_binary(&["--nonexistent-flag"]);
    assert!(!output.status.success(), "unknown flags should cause non-zero exit");
}
