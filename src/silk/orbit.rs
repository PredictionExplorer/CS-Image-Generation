//! Import or select an original orbit without running the finished-art renderer.
//!
//! A generation record identifies the selected candidate, so replaying an existing
//! artwork does not repeat its expensive candidate search. This module deliberately
//! reuses the production integrator and records that numerical implementation; the
//! existing native selection path is not a promise of cross-architecture identity.

use super::{OrbitData, SilkResult, V3};
use crate::generation_log::GenerationRecord;
use crate::render::visual_profile::ResolvedVisualProfile;
use crate::sim::{self, Body, Sha3RandomByteStream};
use crate::{app, render};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io;
use std::path::PathBuf;
use tracing::info;

/// Serializable source conditions before the recording warm-up.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InitialBody {
    /// Positive gravitational mass.
    pub mass: f64,
    /// Initial position in physical simulation coordinates.
    pub position: V3,
    /// Initial velocity in physical simulation coordinates.
    pub velocity: V3,
}

/// Source selection, recording and sampling settings for a silk trajectory.
///
/// `generation_record` uses that record's seed, step count and selected candidate;
/// it never substitutes a newly selected orbit. `initial_conditions` bypasses all
/// candidate selection. The two import mechanisms are mutually exclusive.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OrbitConfig {
    /// Hexadecimal seed used when no generation record is supplied.
    pub seed: String,
    /// Number of candidates for a new selection; part of the artwork identity.
    pub sims: usize,
    /// Production integration steps for both warm-up and recorded motion.
    pub steps: usize,
    /// Keep one sample every this many integrator steps, without changing physics.
    pub sample_stride: usize,
    /// Width supplied to the original visual-profile selection stage.
    pub selection_width: u32,
    /// Height supplied to the original visual-profile selection stage.
    pub selection_height: u32,
    /// Explicit chaos preference; omission follows the production seeded draw.
    pub chaos_weight: Option<f64>,
    /// Explicit triangle-balance preference; omission follows the seeded draw.
    pub equil_weight: Option<f64>,
    /// Existing package's `metadata/generation.json`, replayed without searching.
    pub generation_record: Option<PathBuf>,
    /// Exact source states supplied explicitly, bypassing profile and selection.
    pub initial_conditions: Option<[InitialBody; 3]>,
}

impl Default for OrbitConfig {
    fn default() -> Self {
        Self {
            seed: "0x100033".into(),
            sims: 256,
            steps: 1_000_000,
            sample_stride: 1,
            selection_width: 3456,
            selection_height: 2234,
            chaos_weight: None,
            equil_weight: None,
            generation_record: None,
            initial_conditions: None,
        }
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn source_seed(seed: &str) -> SilkResult<(String, Vec<u8>)> {
    let bytes = app::parse_seed(seed)?;
    if bytes.is_empty() {
        return Err(invalid("orbit seed must contain at least one byte").into());
    }
    Ok((format!("0x{}", hex::encode(&bytes)), bytes))
}

fn validate_recording(steps: usize, sample_stride: usize) -> SilkResult<()> {
    if !(2..=100_000_000).contains(&steps) {
        return Err(invalid("orbit steps must be between 2 and 100000000").into());
    }
    if sample_stride == 0 || sample_stride >= steps {
        return Err(invalid("sample_stride must be positive and smaller than steps").into());
    }
    Ok(())
}

fn validate_states(states: &[InitialBody; 3]) -> SilkResult<()> {
    for (index, body) in states.iter().enumerate() {
        if !body.mass.is_finite() || body.mass <= 0.0 {
            return Err(invalid(format!("body {index} needs a finite positive mass")).into());
        }
        if !body.position.is_finite() || !body.velocity.is_finite() {
            return Err(invalid(format!("body {index} has non-finite source coordinates")).into());
        }
    }
    if !states.iter().map(|body| body.mass).sum::<f64>().is_finite() {
        return Err(invalid("total body mass is not finite").into());
    }
    Ok(())
}

fn to_body(body: InitialBody) -> Body {
    Body::new(
        body.mass,
        Vector3::new(body.position.x, body.position.y, body.position.z),
        Vector3::new(body.velocity.x, body.velocity.y, body.velocity.z),
    )
}

fn from_body(body: &Body) -> InitialBody {
    InitialBody {
        mass: body.mass,
        position: V3::new(body.position.x, body.position.y, body.position.z),
        velocity: V3::new(body.velocity.x, body.velocity.y, body.velocity.z),
    }
}

fn sample_body(rng: &mut Sha3RandomByteStream) -> Body {
    // Keep this order aligned with sim::random_body: mass, xyz, then velocity xyz.
    Body::new(
        rng.random_mass(),
        Vector3::new(rng.random_location(), rng.random_location(), rng.random_location()),
        Vector3::new(rng.random_velocity(), rng.random_velocity(), rng.random_velocity()),
    )
}

fn candidate_at(rng: &mut Sha3RandomByteStream, index: usize) -> Vec<Body> {
    for _ in 0..index {
        for _ in 0..21 {
            let _ = rng.next_f64();
        }
    }
    let mut bodies = (0..3).map(|_| sample_body(rng)).collect::<Vec<_>>();
    // Candidate generation recenters once, and production recording does so again.
    sim::shift_bodies_to_com(&mut bodies);
    bodies
}

fn resolve_weights(config: &OrbitConfig, rng: &mut Sha3RandomByteStream) -> SilkResult<(f64, f64)> {
    for weight in [config.chaos_weight, config.equil_weight].into_iter().flatten() {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(invalid("orbit weights must be finite and positive").into());
        }
    }
    if let (Some(chaos), Some(equil)) = (config.chaos_weight, config.equil_weight) {
        return Ok((chaos, equil));
    }
    let range = render::parameter_descriptors::EQUIL_CHAOS_RATIO;
    let ratio = (range.min.ln() + rng.next_f64() * (range.max.ln() - range.min.ln())).exp();
    Ok(match (config.chaos_weight, config.equil_weight) {
        (Some(chaos), _) => (chaos, chaos * ratio),
        (_, Some(equil)) => (equil / ratio, equil),
        _ => (1.0, ratio),
    })
}

struct Source {
    seed: String,
    bodies: Vec<Body>,
    steps: usize,
    details: Value,
}

fn from_record(record: &GenerationRecord) -> SilkResult<Source> {
    let simulation = &record.simulation_config;
    let orbit = &record.orbit_info;
    if record.render_config.visual_profile != "cosmic_signature" {
        return Err(invalid("generation record uses an unsupported visual profile").into());
    }
    if record.render_config.width == 0 || record.render_config.height == 0 {
        return Err(invalid("generation record has invalid selection dimensions").into());
    }
    if simulation.num_sims == 0
        || simulation.num_sims > 10_000_000
        || orbit.selected_index >= simulation.num_sims
        || orbit.retry_count > app::AESTHETIC_MAX_RETRIES
    {
        return Err(
            invalid("generation record has an invalid candidate index or retry count").into()
        );
    }
    if ![simulation.min_mass, simulation.max_mass, simulation.location, simulation.velocity]
        .iter()
        .all(|value| value.is_finite())
        || simulation.min_mass <= 0.0
        || simulation.max_mass < simulation.min_mass
        || simulation.location <= 0.0
        || simulation.velocity < 0.0
    {
        return Err(invalid("generation record has invalid physical sampling bounds").into());
    }
    let (seed, seed_bytes) = source_seed(&record.seed)?;
    let mut rng = Sha3RandomByteStream::new(
        &seed_bytes,
        simulation.min_mass,
        simulation.max_mass,
        simulation.location,
        simulation.velocity,
    );
    let profile = ResolvedVisualProfile::cosmic_signature(
        &mut rng,
        record.render_config.width,
        record.render_config.height,
    );
    if orbit.preferred_structure.is_empty()
        || profile.parameters.stack.label() != orbit.preferred_structure
    {
        return Err(invalid(
            "generation record's seeded profile differs from this generator; import original initial conditions instead",
        )
        .into());
    }
    if simulation.weights_randomized {
        let _ = rng.next_f64();
    }
    if orbit.retry_count > 0 {
        rng = rng.fork(format!("cosmic-retry/v1/{}", orbit.retry_count).as_bytes());
        let _ = rng.next_f64();
    }
    info!(
        "Reconstructing recorded candidate {} (attempt {}); candidate search skipped",
        orbit.selected_index, orbit.retry_count
    );
    let bodies = candidate_at(&mut rng, orbit.selected_index);
    Ok(Source {
        seed,
        bodies,
        steps: simulation.num_steps_sim,
        details: json!({
            "source": "existing-generation-record",
            "candidate_index": orbit.selected_index,
            "retry_count": orbit.retry_count,
            "simulation_config": simulation,
            "selection_resolution": [record.render_config.width, record.render_config.height],
            "preferred_structure": orbit.preferred_structure,
            "chosen_structure": orbit.chosen_structure,
            "compatibility": "cosmic-signature-v2-profile-and-candidate-draw-order",
            "note": "Recorded candidate is replayed; it is not reselected or projected",
        }),
    })
}

fn select_new(config: &OrbitConfig) -> SilkResult<Source> {
    if config.sims == 0 || config.sims > 10_000_000 {
        return Err(invalid("orbit sims must be between 1 and 10000000").into());
    }
    if config.selection_width == 0 || config.selection_height == 0 {
        return Err(invalid("selection dimensions must be positive").into());
    }
    let (seed, seed_bytes) = source_seed(&config.seed)?;
    let mut rng = Sha3RandomByteStream::new(&seed_bytes, 100.0, 300.0, 300.0, 1.0);
    let profile = ResolvedVisualProfile::cosmic_signature(
        &mut rng,
        config.selection_width,
        config.selection_height,
    );
    let (chaos, equil) = resolve_weights(config, &mut rng)?;
    let selection = app::run_borda_selection_with_aesthetics(
        &mut rng,
        config.sims,
        config.steps,
        chaos,
        equil,
        -0.3,
        profile.parameters.stack,
        profile.parameters.projection,
    )?;
    Ok(Source {
        seed,
        bodies: selection.bodies,
        steps: config.steps,
        details: json!({
            "source": "new-production-selection",
            "candidate_index": selection.result.selected_index,
            "retry_count": selection.retry_count,
            "candidate_count": config.sims,
            "chaos_weight": chaos,
            "equil_weight": equil,
            "escape_threshold": -0.3,
            "sampling_bounds": {"min_mass":100.0,"max_mass":300.0,"location":300.0,"velocity":1.0},
            "selection_resolution": [config.selection_width, config.selection_height],
            "preferred_structure": selection.preferred_stack.label(),
            "chosen_structure": selection.stack.label(),
            "projection_used_for_selection": profile.parameters.projection.label(),
            "note": "Candidate count and recording settings are part of source identity; reduced searches need not select an existing artwork's orbit",
        }),
    })
}

fn replay(mut source: Source, sample_stride: usize) -> SilkResult<OrbitData> {
    validate_recording(source.steps, sample_stride)?;
    let initial =
        [from_body(&source.bodies[0]), from_body(&source.bodies[1]), from_body(&source.bodies[2])];
    validate_states(&initial)?;
    let masses = initial.map(|body| body.mass);
    sim::shift_bodies_to_com(&mut source.bodies);
    let dt = render::constants::DEFAULT_DT;
    info!("Replaying {} warm-up and {} recorded integration steps", source.steps, source.steps);
    for _ in 0..source.steps {
        sim::symplectic_step(&mut source.bodies, dt);
    }
    let mut samples = Vec::with_capacity(source.steps.div_ceil(sample_stride));
    for step in 0..source.steps {
        if step % sample_stride == 0 {
            let sample = std::array::from_fn(|body| {
                let position = source.bodies[body].position;
                V3::new(position.x, position.y, position.z)
            });
            if sample.iter().any(|position| !position.is_finite()) {
                return Err(
                    invalid(format!("orbit became non-finite at recorded step {step}")).into()
                );
            }
            samples.push(sample);
        }
        sim::symplectic_step(&mut source.bodies, dt);
    }
    let initial_bits = initial.map(|body| {
        [
            body.mass.to_bits(),
            body.position.x.to_bits(),
            body.position.y.to_bits(),
            body.position.z.to_bits(),
            body.velocity.x.to_bits(),
            body.velocity.y.to_bits(),
            body.velocity.z.to_bits(),
        ]
    });
    Ok(OrbitData {
        seed: source.seed,
        dt: dt * sample_stride as f64,
        masses,
        samples,
        provenance: json!({
            "schema_version": 1,
            "integrator": "production-yoshida4-native-f64-v1",
            "gravitational_constant": sim::G,
            "integration_dt": dt,
            "warmup_steps": source.steps,
            "recording_steps": source.steps,
            "sample_stride": sample_stride,
            "initial_conditions": initial,
            "initial_condition_f64_bits": initial_bits,
            "selection": source.details,
            "coordinate_frame": "original-Newtonian-center-of-mass",
            "numeric_contract": "native-production-integrator; cross-architecture identity must be separately verified",
        }),
    })
}

/// Generate or reconstruct raw physical motion without rendering the original art.
pub fn generate(config: &OrbitConfig) -> SilkResult<OrbitData> {
    if config.generation_record.is_some() && config.initial_conditions.is_some() {
        return Err(invalid("choose either generation_record or initial_conditions").into());
    }
    let source = if let Some(path) = &config.generation_record {
        let record: GenerationRecord = serde_json::from_reader(std::fs::File::open(path)?)?;
        let source = from_record(&record)?;
        validate_recording(source.steps, config.sample_stride)?;
        source
    } else {
        validate_recording(config.steps, config.sample_stride)?;
        if let Some(states) = config.initial_conditions {
            validate_states(&states)?;
            Source {
                seed: source_seed(&config.seed)?.0,
                bodies: states.into_iter().map(to_body).collect(),
                steps: config.steps,
                details: json!({"source":"explicit-initial-conditions","candidate_search":false}),
            }
        } else {
            select_new(config)?
        }
    };
    replay(source, config.sample_stride)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_states() -> [InitialBody; 3] {
        [
            InitialBody {
                mass: 140.0,
                position: V3::new(-150.0, 30.0, 15.0),
                velocity: V3::new(0.1, 0.4, 0.2),
            },
            InitialBody {
                mass: 180.0,
                position: V3::new(130.0, -90.0, -10.0),
                velocity: V3::new(-0.3, -0.2, 0.1),
            },
            InitialBody {
                mass: 200.0,
                position: V3::new(10.0, 110.0, 20.0),
                velocity: V3::new(0.2, -0.1, -0.2),
            },
        ]
    }

    #[test]
    fn sampled_replay_matches_original_integrator_exactly() {
        let initial = source_states();
        let config = OrbitConfig {
            steps: 128,
            sample_stride: 7,
            initial_conditions: Some(initial),
            ..OrbitConfig::default()
        };
        let orbit = generate(&config).unwrap();
        let reference = sim::get_positions(initial.into_iter().map(to_body).collect(), 128);
        assert_eq!(orbit.samples.len(), 19);
        for (sample_index, sample) in orbit.samples.iter().enumerate() {
            for (body, position) in sample.iter().enumerate() {
                let expected = reference.positions[body][sample_index * 7];
                assert_eq!(
                    [position.x.to_bits(), position.y.to_bits(), position.z.to_bits()],
                    [expected.x.to_bits(), expected.y.to_bits(), expected.z.to_bits()]
                );
            }
        }
    }

    #[test]
    fn repeated_import_preserves_raw_bits_and_sampling_time() {
        let config = OrbitConfig {
            steps: 64,
            sample_stride: 4,
            initial_conditions: Some(source_states()),
            ..OrbitConfig::default()
        };
        let first = generate(&config).unwrap();
        let second = generate(&config).unwrap();
        assert_eq!(first.samples, second.samples);
        assert_eq!(first.provenance, second.provenance);
        assert_eq!(first.dt, 0.004);
        assert_eq!(first.samples.len(), 16);
    }

    #[test]
    fn candidate_index_skipping_matches_original_draw_order() {
        let mut sequential = Sha3RandomByteStream::new(&[0xCA, 0xFE], 100.0, 300.0, 300.0, 1.0);
        let mut skipped = Sha3RandomByteStream::new(&[0xCA, 0xFE], 100.0, 300.0, 300.0, 1.0);
        let mut expected = Vec::new();
        for _ in 0..8 {
            expected = (0..3).map(|_| sample_body(&mut sequential)).collect();
            sim::shift_bodies_to_com(&mut expected);
        }
        let actual = candidate_at(&mut skipped, 7);
        for (a, b) in actual.iter().zip(expected) {
            assert_eq!(a.mass.to_bits(), b.mass.to_bits());
            assert_eq!(a.position, b.position);
            assert_eq!(a.velocity, b.velocity);
        }
    }

    #[test]
    fn record_reconstruction_matches_production_selected_body_states() {
        // Compare to candidates actually sampled by the production search, so this
        // catches drift in its draw order, recentering, and retry-stream handling.
        for retry in [0, 2] {
            let seed = [0xCA, 0xFE];
            let mut rng = Sha3RandomByteStream::new(&seed, 100.0, 300.0, 300.0, 1.0);
            let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);
            let _ = resolve_weights(&OrbitConfig::default(), &mut rng).unwrap();
            if retry > 0 {
                rng = rng.fork(format!("cosmic-retry/v1/{retry}").as_bytes());
                let _ = rng.next_f64();
            }
            let candidates =
                sim::select_best_trajectory_shortlist(&mut rng, 200, 128, 1.0, 5.0, -0.3, 2)
                    .unwrap();
            let expected = &candidates[0];
            let mut record = GenerationRecord::new("fixture", "0xCAFE");
            record.render_config.width = 640;
            record.render_config.height = 360;
            record.render_config.visual_profile = "cosmic_signature".into();
            record.simulation_config.num_sims = 200;
            record.simulation_config.num_steps_sim = 128;
            record.simulation_config.weights_randomized = true;
            record.orbit_info.preferred_structure = profile.parameters.stack.label();
            record.orbit_info.selected_index = expected.result.selected_index;
            record.orbit_info.retry_count = retry;
            let actual = from_record(&record).unwrap();
            for (a, b) in actual.bodies.iter().zip(&expected.bodies) {
                assert_eq!(a.mass.to_bits(), b.mass.to_bits());
                assert_eq!(a.position, b.position);
                assert_eq!(a.velocity, b.velocity);
            }
        }
    }

    #[test]
    fn generation_record_controls_identity_and_recording_length() {
        let mut record = GenerationRecord::new("fixture", "0xCAFE");
        let mut rng = Sha3RandomByteStream::new(&[0xCA, 0xFE], 100.0, 300.0, 300.0, 1.0);
        let profile = ResolvedVisualProfile::cosmic_signature(&mut rng, 640, 360);
        record.render_config.width = 640;
        record.render_config.height = 360;
        record.render_config.visual_profile = "cosmic_signature".into();
        record.simulation_config.num_sims = 1;
        record.simulation_config.num_steps_sim = 128;
        record.simulation_config.weights_randomized = false;
        record.orbit_info.preferred_structure = profile.parameters.stack.label();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("generation.json");
        std::fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
        let orbit = generate(&OrbitConfig {
            seed: "not-used".into(),
            steps: 0,
            sims: 0,
            sample_stride: 4,
            generation_record: Some(path),
            ..OrbitConfig::default()
        })
        .unwrap();
        assert_eq!(orbit.seed, "0xcafe");
        assert_eq!(orbit.samples.len(), 32);
        assert_eq!(orbit.provenance["warmup_steps"], 128);
        assert_eq!(orbit.provenance["selection"]["candidate_index"], 0);
    }

    #[test]
    fn invalid_states_and_sample_intervals_fail_before_simulation() {
        let mut config = OrbitConfig {
            steps: 64,
            sample_stride: 0,
            initial_conditions: Some(source_states()),
            ..OrbitConfig::default()
        };
        assert!(generate(&config).is_err());
        config.sample_stride = 64;
        assert!(generate(&config).is_err());
        config.sample_stride = 4;
        config.initial_conditions.as_mut().unwrap()[1].mass = f64::NAN;
        assert!(generate(&config).is_err());
    }

    #[test]
    fn misspelled_source_settings_are_rejected() {
        assert!(serde_json::from_value::<OrbitConfig>(json!({"step":128})).is_err());
        assert!(
            serde_json::from_value::<InitialBody>(json!({
                "mass":1.0,"position":{"x":0.0,"y":0.0,"z":0.0},
                "velocity":{"x":0.0,"y":0.0,"z":0.0},"velocty":1.0,
            }))
            .is_err()
        );
    }
}
