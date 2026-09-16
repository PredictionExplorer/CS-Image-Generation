//! Build and archive a cumulative sculpture from one unchanged physical orbit.
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
    remaining::{
        Grid, OrbitSeries, SilkResult,
        field::{FieldConfig, PreparedField},
        mesh,
    },
    silk::cache,
};

#[derive(Parser)]
#[command(about = "Deterministic three-dimensional sculpture from recorded motion")]
struct Args {
    /// Parallel spatial workers; source integration order stays fixed.
    #[arg(long, global = true, default_value_t = 4)]
    threads: usize,
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    /// Write the complete initial geometry recipe.
    Config {
        #[arg(long)]
        output: PathBuf,
    },
    /// Validate and expand a geometry recipe without loading the source.
    Resolve {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Generate a closed mesh and measured geometry evidence.
    Build {
        #[arg(long)]
        orbit: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        time: Option<f64>,
        #[arg(long)]
        resolution: Option<usize>,
        /// Reuse an unchanged, complete mesh after checking its content hash.
        #[arg(long)]
        resume: bool,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SurfaceNormalSource {
    #[default]
    Field,
    Mesh,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Recipe {
    resolution: usize,
    source_fraction: f64,
    field: FieldConfig,
    surface_normals: SurfaceNormalSource,
}

impl Default for Recipe {
    fn default() -> Self {
        Self {
            resolution: 192,
            source_fraction: 1.0,
            field: FieldConfig::default(),
            surface_normals: SurfaceNormalSource::Field,
        }
    }
}

impl Recipe {
    fn validate(&self) -> SilkResult<()> {
        if !(16..=512).contains(&self.resolution)
            || !self.source_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.source_fraction)
        {
            return Err("require resolution 16..512 and source fraction 0..1".into());
        }
        self.field.validate()
    }
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
    normal_source: SurfaceNormalSource,
    normal_minimum_gradient: Option<f64>,
    normal_maximum_gradient: Option<f64>,
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
    let record = output.join("build.json");
    if !record.exists() {
        return Ok(false);
    }
    let value: CompletedBuild = serde_json::from_slice(&fs::read(record)?)?;
    value.identity.recipe.validate()?;
    if value.identity.schema_version != 1
        || value.identity_sha256 != expected
        || identity_hash(&value.identity)? != value.identity_sha256
    {
        return Err("existing sculpture receipt has an inconsistent identity or a different source, recipe, or executable".into());
    }
    if value.normal_source != value.identity.recipe.surface_normals {
        return Err("completed sculpture normal source differs from its recipe".into());
    }
    match (value.normal_source, value.normal_minimum_gradient, value.normal_maximum_gradient) {
        (SurfaceNormalSource::Field, Some(minimum), Some(maximum))
            if minimum.is_finite()
                && maximum.is_finite()
                && minimum > 0.0
                && maximum >= minimum => {}
        (SurfaceNormalSource::Mesh, None, None) => {}
        _ => return Err("completed sculpture has inconsistent normal-gradient evidence".into()),
    }
    let archived: Recipe = serde_json::from_slice(&fs::read(output.join("recipe.json"))?)?;
    archived.validate()?;
    if serde_json::to_vec(&archived)? != serde_json::to_vec(&value.identity.recipe)? {
        return Err("archived sculpture recipe does not match its completed build identity".into());
    }
    let actual = cache::file_hash(&output.join("mesh.ply"))?;
    if value.mesh_sha256 != actual {
        return Err("existing mesh does not match its completed build receipt".into());
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
        return Err("orbit source changed after the build identity was recorded".into());
    }
    let source = read(path)?;
    if cache::file_hash(path)? != before {
        return Err("orbit source changed while it was being read; no geometry was built".into());
    }
    Ok(source)
}

/// Connectivity of negative grid nodes under face adjacency; a resolution diagnostic.
/// This is distinct from the connectivity of boundary surfaces around cavities.
fn solid_components(grid: &Grid) -> Vec<usize> {
    let [nx, ny, nz] = grid.dims;
    let plane = nx * ny;
    let mut visited = vec![false; grid.values.len()];
    let mut stack = Vec::new();
    let mut counts = Vec::new();
    for start in 0..grid.values.len() {
        if visited[start] || grid.values[start] >= 0.0 {
            continue;
        }
        visited[start] = true;
        stack.push(start);
        let mut count = 0;
        while let Some(index) = stack.pop() {
            count += 1;
            let x = index % nx;
            let y = (index / nx) % ny;
            let z = index / plane;
            let candidates = [
                (x > 0).then(|| index - 1),
                (x + 1 < nx).then_some(index + 1),
                (y > 0).then(|| index - nx),
                (y + 1 < ny).then_some(index + nx),
                (z > 0).then(|| index - plane),
                (z + 1 < nz).then_some(index + plane),
            ];
            for neighbor in candidates.into_iter().flatten() {
                if !visited[neighbor] && grid.values[neighbor] < 0.0 {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        counts.push(count);
    }
    counts.sort_unstable_by(|a, b| b.cmp(a));
    counts
}

fn build(orbit: &Path, output: &Path, recipe: Recipe, resume: bool) -> SilkResult<()> {
    recipe.validate()?;
    let identity = Identity {
        schema_version: 1,
        orbit_sha256: cache::file_hash(orbit)?,
        executable_sha256: cache::file_hash(&std::env::current_exe()?)?,
        recipe,
    };
    let hash = identity_hash(&identity)?;
    fs::create_dir_all(output)?;
    // Advisory locks release on exit or interruption. Keep the inode in place
    // so separate processes cannot acquire locks on different replacement files.
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(output.join(".build.lock"))?;
    lock.try_lock()
        .map_err(|error| format!("another build owns this output, or locking failed: {error}"))?;
    if completed(output, &hash)? {
        if !resume {
            return Err("completed sculpture exists; use --resume or a new directory".into());
        }
        eprintln!("Verified and reused {}", output.display());
        return Ok(());
    }
    if output.join("mesh.ply").exists() {
        return Err(
            "mesh has no complete receipt; preserve it and use a new output directory".into()
        );
    }
    let started = Instant::now();
    let data = read_verified_source(orbit, &identity.orbit_sha256, cache::read_orbit)?;
    let source = OrbitSeries::new(&data)?;
    let prepared = PreparedField::new(&source, &identity.recipe.field)?;
    eprintln!("Integrating the cumulative material field");
    let (grid, field_diagnostics) =
        prepared.grid(identity.recipe.source_fraction, identity.recipe.resolution)?;
    let field_seconds = started.elapsed().as_secs_f64();
    eprintln!("Extracting {} x {} x {} surface", grid.dims[0], grid.dims[1], grid.dims[2]);
    let (object, extraction) = mesh::extract_with_diagnostics(&grid)?;
    let audit = object.validate()?;
    let components = solid_components(&grid);
    let component_total: usize = components.iter().sum();
    let largest_fraction = components.first().map_or(0.0, |n| *n as f64 / component_total as f64);
    let mesh_path = output.join("mesh.ply");
    let (
        normal_minimum_gradient,
        normal_maximum_gradient,
        normal_stock_difference_step,
        normal_finite_difference_fallbacks,
    ) = match identity.recipe.surface_normals {
        SurfaceNormalSource::Field => {
            eprintln!("Evaluating continuous material-field shading normals");
            let normals = prepared.surface_normals(
                &object.vertices,
                identity.recipe.source_fraction,
                &grid,
            )?;
            if !normals.minimum_gradient_length.is_finite()
                || !normals.maximum_gradient_length.is_finite()
                || normals.minimum_gradient_length <= 0.0
                || normals.maximum_gradient_length < normals.minimum_gradient_length
                || !normals.stock_difference_step.is_finite()
                || normals.stock_difference_step < 0.0
                || normals.finite_difference_fallbacks > object.vertices.len()
            {
                return Err("invalid material-field normal diagnostics".into());
            }
            mesh::write_ply_with_normals(&mesh_path, &object, &normals.normals)?;
            (
                Some(normals.minimum_gradient_length),
                Some(normals.maximum_gradient_length),
                Some(normals.stock_difference_step),
                Some(normals.finite_difference_fallbacks),
            )
        }
        SurfaceNormalSource::Mesh => {
            mesh::write_ply(&mesh_path, &object)?;
            (None, None, None, None)
        }
    };
    json(&output.join("recipe.json"), &identity.recipe)?;
    let record = serde_json::json!({
        "identity": identity, "identity_sha256": hash, "seed": data.seed,
        "mesh_sha256": cache::file_hash(&mesh_path)?,
        "vertices": object.vertices.len(), "triangles": object.triangles.len(),
        "grid": {"dimensions":grid.dims,"origin":grid.origin,"spacing":grid.spacing},
        "field":field_diagnostics, "mesh":audit, "extraction":extraction,
        "normal_source":identity.recipe.surface_normals,
        "normal_minimum_gradient":normal_minimum_gradient,
        "normal_maximum_gradient":normal_maximum_gradient,
        "normal_stock_difference_step":normal_stock_difference_step,
        "normal_finite_difference_fallbacks":normal_finite_difference_fallbacks,
        "solid_grid_components":components,"largest_solid_grid_fraction":largest_fraction,
        "connectivity_note":"Negative grid nodes with face adjacency; refine thin features before interpreting this as exact solid connectivity.",
        "field_seconds":field_seconds,"total_seconds":started.elapsed().as_secs_f64()
    });
    json(&output.join("build.json"), &record)?;
    eprintln!(
        "Completed {} vertices, {} triangles in {:.2}s; {} sampled solid components",
        object.vertices.len(),
        object.triangles.len(),
        started.elapsed().as_secs_f64(),
        components.len()
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
        Action::Config { output } => json(&output, &Recipe::default()),
        Action::Resolve { config, output } => {
            let recipe: Recipe = serde_json::from_slice(&fs::read(config)?)?;
            recipe.validate()?;
            json(&output, &recipe)
        }
        Action::Build { orbit, config, output, time, resolution, resume } => {
            let mut recipe: Recipe = serde_json::from_slice(&fs::read(config)?)?;
            if let Some(time) = time {
                recipe.source_fraction = time;
            }
            if let Some(resolution) = resolution {
                recipe.resolution = resolution;
            }
            build(&orbit, &output, recipe, resume)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use three_body_problem::remaining::V3;

    #[test]
    fn recipe_rejects_invalid_time_resolution_and_unknown_controls() {
        let mut recipe = Recipe::default();
        recipe.validate().unwrap();
        recipe.source_fraction = f64::NAN;
        assert!(recipe.validate().is_err());
        recipe.source_fraction = 1.0;
        recipe.resolution = usize::MAX;
        assert!(recipe.validate().is_err());
        assert!(serde_json::from_str::<Recipe>(r#"{"mistyped_resolution":192}"#).is_err());
    }

    #[test]
    fn normal_source_defaults_to_field_and_changes_the_build_identity() {
        let default: Recipe = serde_json::from_str("{}").unwrap();
        assert_eq!(default.surface_normals, SurfaceNormalSource::Field);
        let mesh: Recipe = serde_json::from_str(r#"{"surface_normals":"mesh"}"#).unwrap();
        assert_eq!(mesh.surface_normals, SurfaceNormalSource::Mesh);
        assert!(serde_json::from_str::<Recipe>(r#"{"surface_normals":"unknown"}"#).is_err());
        let identity = Identity {
            schema_version: 1,
            orbit_sha256: "a".repeat(64),
            executable_sha256: "b".repeat(64),
            recipe: default,
        };
        let mut changed = identity.clone();
        changed.recipe = mesh;
        assert_ne!(identity_hash(&identity).unwrap(), identity_hash(&changed).unwrap());
        assert_eq!(serde_json::to_value(&identity.recipe).unwrap()["surface_normals"], "field");
    }

    #[test]
    fn receipt_reuse_rejects_changed_identity_and_mesh_content() {
        let directory = tempfile::tempdir().unwrap();
        let mesh = directory.path().join("mesh.ply");
        fs::write(&mesh, b"verified mesh fixture").unwrap();
        let identity = Identity {
            schema_version: 1,
            orbit_sha256: "a".repeat(64),
            executable_sha256: "b".repeat(64),
            recipe: Recipe::default(),
        };
        let hash = identity_hash(&identity).unwrap();
        let record = serde_json::json!({
            "identity":identity,"identity_sha256":hash,"mesh_sha256":cache::file_hash(&mesh).unwrap(),
            "normal_source":"field","normal_minimum_gradient":1.0,"normal_maximum_gradient":2.0,
        });
        let receipt_path = directory.path().join("build.json");
        let recipe_path = directory.path().join("recipe.json");
        json(&recipe_path, &identity.recipe).unwrap();
        json(&receipt_path, &record).unwrap();
        assert!(completed(directory.path(), &hash).unwrap());
        assert!(completed(directory.path(), "another").is_err());

        for key in [
            "identity",
            "identity_sha256",
            "mesh_sha256",
            "normal_source",
            "normal_minimum_gradient",
        ] {
            let mut missing = record.clone();
            missing.as_object_mut().unwrap().remove(key);
            json(&receipt_path, &missing).unwrap();
            assert!(completed(directory.path(), &hash).is_err());
        }
        let mut changed = record.clone();
        changed["identity"]["recipe"]["source_fraction"] = serde_json::json!(0.25);
        json(&receipt_path, &changed).unwrap();
        assert!(completed(directory.path(), &hash).is_err());
        changed = record.clone();
        changed["identity"]["schema_version"] = serde_json::json!(2);
        json(&receipt_path, &changed).unwrap();
        assert!(completed(directory.path(), &hash).is_err());
        changed = record.clone();
        changed["normal_source"] = serde_json::json!("mesh");
        json(&receipt_path, &changed).unwrap();
        assert!(completed(directory.path(), &hash).is_err());
        changed = record.clone();
        changed["normal_minimum_gradient"] = serde_json::json!(0.0);
        json(&receipt_path, &changed).unwrap();
        assert!(completed(directory.path(), &hash).is_err());
        json(&receipt_path, &record).unwrap();

        fs::remove_file(&recipe_path).unwrap();
        assert!(completed(directory.path(), &hash).is_err());
        let mut changed_recipe = identity.recipe.clone();
        changed_recipe.source_fraction = 0.25;
        json(&recipe_path, &changed_recipe).unwrap();
        assert!(completed(directory.path(), &hash).is_err());
        json(&recipe_path, &identity.recipe).unwrap();
        assert!(completed(directory.path(), &hash).unwrap());

        fs::write(&mesh, b"changed mesh fixture").unwrap();
        assert!(completed(directory.path(), &hash).is_err());
    }

    #[test]
    fn source_is_verified_before_and_after_reading() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.orbit");
        let original = b"immutable source fixture";
        fs::write(&source, original).unwrap();
        let hash = cache::file_hash(&source).unwrap();
        let bytes = read_verified_source(&source, &hash, |path| Ok(fs::read(path)?)).unwrap();
        assert_eq!(bytes, original);

        fs::write(&source, b"changed before reading").unwrap();
        let mut reader_called = false;
        let result = read_verified_source(&source, &hash, |_| {
            reader_called = true;
            Ok(())
        });
        assert!(result.is_err());
        assert!(!reader_called);

        fs::write(&source, original).unwrap();
        let result = read_verified_source(&source, &hash, |path| {
            let bytes = fs::read(path)?;
            fs::write(path, b"changed during reading")?;
            Ok(bytes)
        });
        assert!(result.is_err());
    }

    #[test]
    fn solid_connectivity_counts_material_instead_of_boundary_shells() {
        let mut grid =
            Grid { dims: [5, 5, 5], origin: V3::ZERO, spacing: 1.0, values: vec![1.0; 125] };
        for z in 1..4 {
            for y in 1..4 {
                for x in 1..4 {
                    grid.values[x + 5 * (y + 5 * z)] = -1.0;
                }
            }
        }
        grid.values[2 + 5 * (2 + 5 * 2)] = 1.0;
        assert_eq!(solid_components(&grid), vec![26]);
        grid.values.fill(1.0);
        grid.values[31] = -1.0;
        grid.values[93] = -1.0;
        assert_eq!(solid_components(&grid), vec![1, 1]);
    }

    #[test]
    fn advisory_output_lock_releases_without_deleting_its_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lock");
        let a = OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
        let b = OpenOptions::new().read(true).write(true).open(&path).unwrap();
        a.try_lock().unwrap();
        assert!(b.try_lock().is_err());
        drop(a);
        b.try_lock().unwrap();
    }
}
