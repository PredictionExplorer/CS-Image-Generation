//! Build and archive a shell grown from one unchanged physical orbit.
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
use three_body_problem::{
    atelier::{OrbitSeries, SilkResult, V3},
    remaining::mesh,
    remembering_shell::{self, Recipe},
    silk::cache,
};

#[derive(Parser)]
#[command(about = "Deterministic three-dimensional shell growth from recorded motion")]
struct Args {
    /// Parallel workers; geometry and source chronology stay deterministic.
    #[arg(long, global = true, default_value_t = 4)]
    threads: usize,
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    /// Write the fully specified default shell recipe.
    Config {
        #[arg(long)]
        output: PathBuf,
    },
    /// Validate and expand a shell recipe without loading its source.
    Resolve {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Generate and archive a closed shell with its supplied surface normals.
    Build {
        #[arg(long)]
        orbit: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        time: Option<f64>,
        /// Reuse an unchanged complete shell after verifying identity and contents.
        #[arg(long)]
        resume: bool,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Identity {
    schema_version: u32,
    orbit_sha256: String,
    executable_sha256: String,
    recipe: Recipe,
}

#[derive(Deserialize)]
struct CompletedBuild {
    identity: Identity,
    identity_sha256: String,
    mesh_sha256: String,
}

#[derive(Debug, Serialize)]
struct NativePrecision {
    nonfinite_vertices: usize,
    collapsed_triangles: usize,
    reversed_triangles: usize,
}

fn native_precision(mesh: &mesh::Mesh) -> SilkResult<NativePrecision> {
    let points: Vec<V3> = mesh
        .vertices
        .iter()
        .map(|p| V3::new(f64::from(p.x as f32), f64::from(p.y as f32), f64::from(p.z as f32)))
        .collect();
    let mut audit = NativePrecision {
        nonfinite_vertices: points.iter().filter(|p| !p.is_finite()).count(),
        collapsed_triangles: 0,
        reversed_triangles: 0,
    };
    for triangle in &mesh.triangles {
        let [a, b, c] = triangle.map(|i| points[i as usize]);
        let n = (b - a).cross(c - a);
        if !n.is_finite() || n.x.hypot(n.y).hypot(n.z) == 0.0 {
            audit.collapsed_triangles += 1;
            continue;
        }
        let [a, b, c] = triangle.map(|i| mesh.vertices[i as usize]);
        if n.dot((b - a).cross(c - a)) <= 0.0 {
            audit.reversed_triangles += 1;
        }
    }
    if audit.nonfinite_vertices != 0
        || audit.collapsed_triangles != 0
        || audit.reversed_triangles != 0
    {
        return Err(format!("shell fails native f32 geometry precision: {audit:?}; no vertices or faces were repaired").into());
    }
    Ok(audit)
}

fn identity_hash(identity: &Identity) -> SilkResult<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(identity)?)))
}

fn json(path: &Path, value: &impl Serialize) -> SilkResult<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("{}.partial", std::process::id()));
    let mut file = OpenOptions::new().write(true).create_new(true).open(&temporary)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)?;
    Ok(())
}

fn completed(output: &Path, expected: &str) -> SilkResult<bool> {
    let receipt = output.join("build.json");
    if !receipt.exists() {
        return Ok(false);
    }
    let record: CompletedBuild = serde_json::from_slice(&fs::read(receipt)?)?;
    record.identity.recipe.validate()?;
    if record.identity.schema_version != 1
        || record.identity_sha256 != expected
        || identity_hash(&record.identity)? != record.identity_sha256
    {
        return Err("existing shell receipt has an inconsistent identity or a different source, recipe, or executable".into());
    }
    let archived: Recipe = serde_json::from_slice(&fs::read(output.join("recipe.json"))?)?;
    archived.validate()?;
    if serde_json::to_vec(&archived)? != serde_json::to_vec(&record.identity.recipe)? {
        return Err("archived shell recipe differs from its completed build identity".into());
    }
    if cache::file_hash(&output.join("mesh.ply"))? != record.mesh_sha256 {
        return Err("existing shell mesh differs from its completed build receipt".into());
    }
    Ok(true)
}

fn read_verified_source<T>(
    path: &Path,
    expected_sha256: &str,
    read: impl FnOnce(&Path) -> SilkResult<T>,
) -> SilkResult<T> {
    let before = cache::file_hash(path)?;
    if before != expected_sha256 {
        return Err("orbit source changed after the shell build identity was recorded".into());
    }
    let source = read(path)?;
    if cache::file_hash(path)? != before {
        return Err("orbit source changed while it was being read; no shell was built".into());
    }
    Ok(source)
}

fn build_archive(orbit: &Path, output: &Path, recipe: Recipe, resume: bool) -> SilkResult<()> {
    recipe.validate()?;
    let identity = Identity {
        schema_version: 1,
        orbit_sha256: cache::file_hash(orbit)?,
        executable_sha256: cache::file_hash(&std::env::current_exe()?)?,
        recipe,
    };
    let hash = identity_hash(&identity)?;
    fs::create_dir_all(output)?;
    // Retain the advisory-lock inode so concurrent writers cannot lock replacements.
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(output.join(".build.lock"))?;
    lock.try_lock().map_err(|error| {
        format!("another shell build owns this output, or locking failed: {error}")
    })?;
    if completed(output, &hash)? {
        if !resume {
            return Err("completed shell exists; use --resume or a new directory".into());
        }
        eprintln!("Verified and reused {}", output.display());
        return Ok(());
    }
    if output.join("mesh.ply").exists() {
        return Err(
            "shell mesh has no complete receipt; preserve it and use a new output directory".into(),
        );
    }
    let started = Instant::now();
    let data = read_verified_source(orbit, &identity.orbit_sha256, cache::read_orbit)?;
    let source = OrbitSeries::new(&data)?;
    eprintln!("Growing shell through source fraction {:.9}", identity.recipe.source_fraction);
    let built = remembering_shell::build(&source, &identity.recipe)?;
    let audit = built.mesh.validate()?;
    let native_precision = native_precision(&built.mesh)?;
    let mesh_path = output.join("mesh.ply");
    mesh::write_ply_with_normals(&mesh_path, &built.mesh, &built.normals)?;
    json(&output.join("recipe.json"), &identity.recipe)?;
    let record = serde_json::json!({
        "identity":identity,"identity_sha256":hash,"seed":data.seed,
        "mesh_sha256":cache::file_hash(&mesh_path)?,
        "vertices":built.mesh.vertices.len(),"triangles":built.mesh.triangles.len(),
        "normal_count":built.normals.len(),"mesh":audit,"shell":built.diagnostics,
        "native_precision":native_precision,
        "total_seconds":started.elapsed().as_secs_f64(),
    });
    json(&output.join("build.json"), &record)?;
    eprintln!(
        "Completed {} vertices, {} triangles in {:.2}s",
        built.mesh.vertices.len(),
        built.mesh.triangles.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> SilkResult<()> {
    let args = Args::parse();
    if !(1..=256).contains(&args.threads) {
        return Err("threads must be in 1..256".into());
    }
    rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global()?;
    match args.action {
        Action::Config { output } => {
            let recipe = Recipe::default();
            recipe.validate()?;
            json(&output, &recipe)
        }
        Action::Resolve { config, output } => {
            let recipe: Recipe = serde_json::from_slice(&fs::read(config)?)?;
            recipe.validate()?;
            json(&output, &recipe)
        }
        Action::Build { orbit, config, output, time, resume } => {
            let mut recipe: Recipe = serde_json::from_slice(&fs::read(config)?)?;
            if let Some(time) = time {
                recipe.source_fraction = time;
            }
            build_archive(&orbit, &output, recipe, resume)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Identity {
        Identity {
            schema_version: 1,
            orbit_sha256: "a".repeat(64),
            executable_sha256: "b".repeat(64),
            recipe: Recipe::default(),
        }
    }

    #[test]
    fn shell_receipt_checks_embedded_identity_recipe_and_mesh_before_reuse() {
        let directory = tempfile::tempdir().unwrap();
        let mesh = directory.path().join("mesh.ply");
        let recipe_path = directory.path().join("recipe.json");
        let receipt_path = directory.path().join("build.json");
        fs::write(&mesh, b"shell mesh fixture").unwrap();
        let identity = identity();
        let hash = identity_hash(&identity).unwrap();
        let record = serde_json::json!({"identity":identity,"identity_sha256":hash,
            "mesh_sha256":cache::file_hash(&mesh).unwrap()});
        json(&recipe_path, &identity.recipe).unwrap();
        json(&receipt_path, &record).unwrap();
        assert!(completed(directory.path(), &hash).unwrap());
        assert!(completed(directory.path(), "different").is_err());
        for key in ["identity", "identity_sha256", "mesh_sha256"] {
            let mut missing = record.clone();
            missing.as_object_mut().unwrap().remove(key);
            json(&receipt_path, &missing).unwrap();
            assert!(completed(directory.path(), &hash).is_err());
        }
        let mut changed = record.clone();
        changed["identity"]["orbit_sha256"] = serde_json::json!("c".repeat(64));
        json(&receipt_path, &changed).unwrap();
        assert!(completed(directory.path(), &hash).is_err());
        json(&receipt_path, &record).unwrap();
        let mut recipe = identity.recipe.clone();
        recipe.source_fraction = 0.5;
        json(&recipe_path, &recipe).unwrap();
        assert!(completed(directory.path(), &hash).is_err());
        fs::remove_file(&recipe_path).unwrap();
        assert!(completed(directory.path(), &hash).is_err());
        json(&recipe_path, &identity.recipe).unwrap();
        fs::write(&mesh, b"changed shell geometry").unwrap();
        assert!(completed(directory.path(), &hash).is_err());
    }

    #[test]
    fn source_changes_before_or_during_reading_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.orbit");
        fs::write(&source, b"original").unwrap();
        let hash = cache::file_hash(&source).unwrap();
        let value = read_verified_source(&source, &hash, |path| Ok(fs::read(path)?)).unwrap();
        assert_eq!(value.as_slice(), b"original");
        fs::write(&source, b"changed before").unwrap();
        let mut called = false;
        assert!(
            read_verified_source(&source, &hash, |_| {
                called = true;
                Ok(())
            })
            .is_err()
        );
        assert!(!called);
        fs::write(&source, b"original").unwrap();
        assert!(
            read_verified_source(&source, &hash, |path| {
                let bytes = fs::read(path)?;
                fs::write(path, b"changed during")?;
                Ok(bytes)
            })
            .is_err()
        );
    }

    #[test]
    fn cli_defaults_and_time_override_keep_the_film_runner_contract() {
        let args = Args::try_parse_from([
            "remembering_shell",
            "build",
            "--orbit",
            "source.orbit",
            "--config",
            "recipe.json",
            "--output",
            "result",
            "--time",
            "0.625",
            "--resume",
        ])
        .unwrap();
        assert_eq!(args.threads, 4);
        assert!(matches!(args.action, Action::Build { time, resume, .. }
            if time == Some(0.625) && resume));
        let recipe = Recipe { source_fraction: f64::NAN, ..Recipe::default() };
        assert!(recipe.validate().is_err());
    }

    #[test]
    fn native_precision_rejects_actual_f32_collapse_without_float_cross_cancellation() {
        let tetrahedron = |step| mesh::Mesh {
            vertices: vec![
                V3::new(1.0, 1.0, 1.0),
                V3::new(1.0 + step, 1.0, 1.0),
                V3::new(1.0, 2.0, 1.0),
                V3::new(1.0, 1.0, 2.0),
            ],
            triangles: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        };
        let regular = tetrahedron(1.0);
        regular.validate().unwrap();
        native_precision(&regular).unwrap();
        let thin = tetrahedron(1e-9);
        thin.validate().unwrap();
        assert!(native_precision(&thin).is_err());
    }
}
