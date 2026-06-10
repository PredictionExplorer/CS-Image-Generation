//! End-to-end determinism integration tests.
//!
//! Validates that identical seeds produce bit-identical simulation and
//! rendering output across multiple runs.

use three_body_problem::app;
use three_body_problem::sim::Sha3RandomByteStream;

const TEST_SEED: &str = "0xdeadbeef";
const MIN_MASS: f64 = 100.0;
const MAX_MASS: f64 = 300.0;
const LOCATION: f64 = 300.0;
const VELOCITY: f64 = 1.0;

fn make_rng(seed_bytes: &[u8]) -> Sha3RandomByteStream {
    Sha3RandomByteStream::new(seed_bytes, MIN_MASS, MAX_MASS, LOCATION, VELOCITY)
}

#[test]
fn same_seed_produces_identical_borda_selection() {
    let seed_bytes = app::parse_seed(TEST_SEED).expect("seed should parse");

    let mut rng1 = make_rng(&seed_bytes);
    let mut rng2 = make_rng(&seed_bytes);

    let (bodies1, traj1) = app::run_borda_selection(&mut rng1, 500, 5_000, 1.0, 5.0, -0.3)
        .expect("borda selection should succeed");
    let (bodies2, traj2) = app::run_borda_selection(&mut rng2, 500, 5_000, 1.0, 5.0, -0.3)
        .expect("borda selection should succeed");

    for (b1, b2) in bodies1.iter().zip(&bodies2) {
        assert_eq!(b1.mass.to_bits(), b2.mass.to_bits(), "mass diverged");
        assert_eq!(b1.position, b2.position, "position diverged");
        assert_eq!(b1.velocity, b2.velocity, "velocity diverged");
    }

    assert_eq!(traj1.total_score, traj2.total_score, "trajectory total_score diverged");
    assert_eq!(traj1.chaos.to_bits(), traj2.chaos.to_bits(), "trajectory chaos score diverged");
}

#[test]
fn same_seed_produces_identical_simulation() {
    let seed_bytes = app::parse_seed(TEST_SEED).expect("seed should parse");

    let mut rng = make_rng(&seed_bytes);
    let (bodies, _) = app::run_borda_selection(&mut rng, 200, 2_000, 1.0, 5.0, -0.3)
        .expect("borda selection should succeed");

    let positions1 = app::simulate_best_orbit(bodies.clone(), 2_000);
    let positions2 = app::simulate_best_orbit(bodies, 2_000);

    for (body_idx, (p1, p2)) in positions1.iter().zip(&positions2).enumerate() {
        for (step, (v1, v2)) in p1.iter().zip(p2).enumerate() {
            assert_eq!(v1, v2, "body {body_idx} step {step}: position diverged");
        }
    }
}

#[test]
fn different_seeds_produce_different_orbits() {
    let seed_a = app::parse_seed("0xdeadbeef").expect("seed should parse");
    let seed_b = app::parse_seed("0xcafebabe").expect("seed should parse");

    let mut rng_a = make_rng(&seed_a);
    let mut rng_b = make_rng(&seed_b);

    let (bodies_a, _) = app::run_borda_selection(&mut rng_a, 200, 2_000, 1.0, 5.0, -0.3)
        .expect("borda selection should succeed");
    let (bodies_b, _) = app::run_borda_selection(&mut rng_b, 200, 2_000, 1.0, 5.0, -0.3)
        .expect("borda selection should succeed");

    let all_same = bodies_a.iter().zip(&bodies_b).all(|(a, b)| {
        a.mass.to_bits() == b.mass.to_bits() && a.position == b.position && a.velocity == b.velocity
    });
    assert!(!all_same, "different seeds should produce different initial conditions");
}

#[test]
fn rng_stream_is_reproducible() {
    let seed = app::parse_seed("0x01020304").expect("seed should parse");
    let mut rng1 = make_rng(&seed);
    let mut rng2 = make_rng(&seed);

    for i in 0..100 {
        let v1 = rng1.next_f64();
        let v2 = rng2.next_f64();
        assert_eq!(v1.to_bits(), v2.to_bits(), "RNG diverged at step {i}");
    }
}
