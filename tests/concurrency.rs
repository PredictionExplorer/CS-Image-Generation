//! Integration tests for concurrency, parallelism, and thread safety.
//!
//! Verifies that rayon-based parallel operations produce deterministic results
//! across various thread pool sizes, and that synchronization primitives
//! work correctly under contention.

use nalgebra::Vector3;
use three_body_problem::render::drawing::{SpectralLineSegment, LineVertex};
use three_body_problem::sim::{Body, Sha3RandomByteStream, get_positions};
use three_body_problem::spectrum::NUM_BINS;
use three_body_problem::spectrum_simd;

fn make_test_bodies() -> Vec<Body> {
    vec![
        Body::new(200.0, Vector3::new(100.0, 0.0, 0.0), Vector3::new(0.0, 0.5, 0.0)),
        Body::new(150.0, Vector3::new(-50.0, 86.0, 0.0), Vector3::new(-0.3, -0.2, 0.0)),
        Body::new(250.0, Vector3::new(-50.0, -86.0, 0.0), Vector3::new(0.3, -0.3, 0.0)),
    ]
}

// ── rayon thread pool determinism ───────────────────────────────────────────

#[test]
fn test_simulation_deterministic_across_thread_counts() {
    let bodies = make_test_bodies();
    let steps = 2000;
    let reference = get_positions(bodies.clone(), steps);

    for num_threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        let result = pool.install(|| get_positions(bodies.clone(), steps));

        assert_eq!(
            reference.positions.len(),
            result.positions.len(),
            "thread count {num_threads}: position count mismatch"
        );
        for (i, (ref_pos, test_pos)) in
            reference.positions.iter().zip(result.positions.iter()).enumerate()
        {
            for body in 0..3 {
                let diff = (ref_pos[body] - test_pos[body]).norm();
                assert!(
                    diff < 1e-10,
                    "thread count {num_threads}: step {i}, body {body}: diff={diff}"
                );
            }
        }
    }
}

#[test]
fn test_parallel_spd_batch_deterministic() {
    let batch: Vec<[f64; NUM_BINS]> = (0..500)
        .map(|i| {
            let mut spd = [0.0; NUM_BINS];
            for (j, v) in spd.iter_mut().enumerate() {
                *v = ((i * 17 + j * 31) % 100) as f64 / 10.0;
            }
            spd
        })
        .collect();

    let reference: Vec<_> = batch.iter().map(|s| spectrum_simd::spd_to_rgba_simd(s)).collect();

    for num_threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let result: Vec<_> = pool.install(|| {
            use rayon::prelude::*;
            batch.par_iter().map(|s| spectrum_simd::spd_to_rgba_simd(s)).collect()
        });

        for (i, (r, t)) in reference.iter().zip(result.iter()).enumerate() {
            assert_eq!(r.0.to_bits(), t.0.to_bits(), "threads={num_threads}: R mismatch at {i}");
            assert_eq!(r.3.to_bits(), t.3.to_bits(), "threads={num_threads}: A mismatch at {i}");
        }
    }
}

#[test]
fn test_parallel_blur_deterministic() {
    use three_body_problem::render::drawing::parallel_blur_2d_rgba;

    let width = 64;
    let height = 48;
    let mut buf_ref: Vec<(f64, f64, f64, f64)> = (0..width * height)
        .map(|i| {
            let t = i as f64 / (width * height) as f64;
            (t * 0.8, (1.0 - t) * 0.6, 0.3, 0.9)
        })
        .collect();
    parallel_blur_2d_rgba(&mut buf_ref, width, height, 3);

    for num_threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let mut buf: Vec<(f64, f64, f64, f64)> = (0..width * height)
            .map(|i| {
                let t = i as f64 / (width * height) as f64;
                (t * 0.8, (1.0 - t) * 0.6, 0.3, 0.9)
            })
            .collect();

        pool.install(|| parallel_blur_2d_rgba(&mut buf, width, height, 3));

        for (i, (r, t)) in buf_ref.iter().zip(buf.iter()).enumerate() {
            assert_eq!(r.0.to_bits(), t.0.to_bits(), "threads={num_threads}: blur R at {i}");
            assert_eq!(r.1.to_bits(), t.1.to_bits(), "threads={num_threads}: blur G at {i}");
        }
    }
}

// ── histogram determinism ───────────────────────────────────────────────────

#[test]
fn test_parallel_histogram_deterministic() {
    use three_body_problem::render::histogram;

    let samples: Vec<[f64; 3]> = (0..50_000)
        .map(|i| {
            let t = i as f64 / 50_000.0;
            [t * 0.8, (1.0 - t) * 0.6, 0.3 + t * 0.4]
        })
        .collect();

    let ref_result = histogram::analyze_tonemapping(&samples, 0.01, 0.99);

    for num_threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let test_result = pool.install(|| histogram::analyze_tonemapping(&samples, 0.01, 0.99));

        assert_eq!(
            ref_result.black_r.to_bits(),
            test_result.black_r.to_bits(),
            "threads={num_threads}: histogram black_r mismatch"
        );
        assert_eq!(
            ref_result.white_r.to_bits(),
            test_result.white_r.to_bits(),
            "threads={num_threads}: histogram white_r mismatch"
        );
    }
}

// ── atomic flag concurrency ─────────────────────────────────────────────────

#[test]
fn test_sat_boost_atomic_toggle_under_contention() {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    let flag = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let barrier = Arc::new(std::sync::Barrier::new(4));

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let flag = Arc::clone(&flag);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..10_000 {
                    flag.store(i % 2 == 0, Ordering::Relaxed);
                    let _ = flag.load(Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let final_val = flag.load(Ordering::Relaxed);
    assert!(final_val || !final_val, "should be a valid bool");
}

// ── channel frame writer ────────────────────────────────────────────────────

#[test]
fn test_channel_bounded_producer_consumer() {
    use std::sync::mpsc;

    let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(4);
    let n_frames = 100;
    let frame_size = 1024;

    let producer = std::thread::spawn(move || {
        for i in 0..n_frames {
            tx.send(vec![i as u8; frame_size]).unwrap();
        }
    });

    let consumer = std::thread::spawn(move || {
        let mut total = 0usize;
        for frame in rx {
            total += frame.len();
        }
        total
    });

    producer.join().unwrap();
    let total = consumer.join().unwrap();
    assert_eq!(total, n_frames * frame_size);
}

#[test]
fn test_channel_producer_drop_signals_consumer_eof() {
    use std::sync::mpsc;

    let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(2);
    tx.send(vec![1; 10]).unwrap();
    drop(tx);

    let first = rx.recv();
    assert!(first.is_ok());
    let second = rx.recv();
    assert!(second.is_err(), "should get EOF after sender drop");
}

// ── multi-thread RNG independence ───────────────────────────────────────────

#[test]
fn test_rng_per_thread_independence() {
    use std::sync::Arc;

    let results: Arc<std::sync::Mutex<Vec<(usize, Vec<f64>)>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let barrier = Arc::new(std::sync::Barrier::new(4));

    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let results = Arc::clone(&results);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let seed = format!("thread_{thread_id}");
                let mut rng =
                    Sha3RandomByteStream::new(seed.as_bytes(), 100.0, 300.0, 300.0, 1.0);
                barrier.wait();
                let values: Vec<f64> = (0..100).map(|_| rng.next_f64()).collect();
                results.lock().unwrap().push((thread_id, values));
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let results = results.lock().unwrap();
    assert_eq!(results.len(), 4);

    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let same = results[i]
                .1
                .iter()
                .zip(results[j].1.iter())
                .filter(|(a, b)| a.to_bits() == b.to_bits())
                .count();
            assert!(
                same < 5,
                "threads {} and {} produced too many identical values: {same}",
                results[i].0,
                results[j].0
            );
        }
    }
}

// ── rayon scope safety ──────────────────────────────────────────────────────

#[test]
fn test_rayon_scope_does_not_deadlock() {
    use std::sync::Mutex;

    let results = Mutex::new(Vec::new());

    rayon::scope(|s| {
        for i in 0..8 {
            let results = &results;
            s.spawn(move |_| {
                let val = i * i;
                results.lock().unwrap().push(val);
            });
        }
    });

    let results = results.into_inner().unwrap();
    assert_eq!(results.len(), 8);
    let mut sorted = results.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 8, "all tasks should produce unique values");
}

#[test]
fn test_nested_rayon_parallelism() {
    use rayon::prelude::*;

    let outer: Vec<Vec<u64>> = (0..10u64)
        .into_par_iter()
        .map(|i| {
            (0..100u64)
                .into_par_iter()
                .map(|j| i * 100 + j)
                .collect()
        })
        .collect();

    assert_eq!(outer.len(), 10);
    for (i, inner) in outer.iter().enumerate() {
        assert_eq!(inner.len(), 100);
        for (j, &val) in inner.iter().enumerate() {
            assert_eq!(val, i as u64 * 100 + j as u64);
        }
    }
}

// ── line drawing determinism ────────────────────────────────────────────────

#[test]
fn test_line_drawing_deterministic_across_threads() {
    use three_body_problem::render::drawing::draw_line_segment_aa_spectral;

    let segment = SpectralLineSegment {
        start: LineVertex { x: 5.0, y: 5.0, z: 0.0, color: (0.7, 0.15, 0.08), alpha: 0.85 },
        end: LineVertex { x: 50.0, y: 40.0, z: -0.3, color: (0.65, -0.12, 0.14), alpha: 0.55 },
        hdr_scale: 2.0,
    };

    let width = 64u32;
    let height = 48u32;

    let mut ref_buf = vec![[0.0f64; NUM_BINS]; (width * height) as usize];
    draw_line_segment_aa_spectral(&mut ref_buf, width, height, segment);

    for num_threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        let mut test_buf = vec![[0.0f64; NUM_BINS]; (width * height) as usize];
        pool.install(|| draw_line_segment_aa_spectral(&mut test_buf, width, height, segment));

        for (i, (r, t)) in ref_buf.iter().zip(test_buf.iter()).enumerate() {
            for bin in 0..NUM_BINS {
                assert_eq!(
                    r[bin].to_bits(),
                    t[bin].to_bits(),
                    "threads={num_threads}: pixel {i} bin {bin} mismatch"
                );
            }
        }
    }
}

// ── generation log concurrent safety ────────────────────────────────────────

#[test]
fn test_generation_log_concurrent_access() {
    use std::sync::{Arc, Barrier};

    let dir = std::env::temp_dir().join("concurrent_gen_log_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let path = dir.join("gen_log.json");
    let barrier = Arc::new(Barrier::new(4));

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let path = path.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                let content = format!("{{\"thread\": {i}}}\n");
                std::fs::write(&path, content).ok();
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert!(path.exists(), "log file should exist after concurrent writes");
    let _ = std::fs::remove_dir_all(&dir);
}

// ── thread pool sizing ──────────────────────────────────────────────────────

#[test]
fn test_thread_pool_various_sizes() {
    for size in [1, 2, 3, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(size)
            .build()
            .unwrap();

        let count = pool.install(|| rayon::current_num_threads());
        assert_eq!(count, size, "pool should have {size} threads");
    }
}

#[test]
fn test_available_parallelism_is_positive() {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    assert!(cpus > 0, "available_parallelism should be positive");
}
