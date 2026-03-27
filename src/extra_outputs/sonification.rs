//! Sonification — ambient audio synthesis from three-body simulation data.
//!
//! Each body becomes a voice: velocity magnitude maps to pitch, inter-body
//! distances map to harmonic intervals, and the overall chaos drives texture.

use nalgebra::Vector3;
use std::io::Write;
use std::process::{Command, Stdio};
use tracing::info;

const SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 2;
const BITS_PER_SAMPLE: u16 = 16;
const MIN_FREQ: f64 = 80.0;
const MAX_FREQ: f64 = 600.0;

pub fn generate_sonification(
    positions: &[Vec<Vector3<f64>>],
    duration_seconds: f64,
    output_wav: &str,
    output_video: Option<&str>,
    input_video: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating sonification ({:.1}s)...", duration_seconds);

    let total_samples = (SAMPLE_RATE as f64 * duration_seconds) as usize;
    let step_count = positions[0].len();

    let vel_stats = compute_velocity_stats(positions);
    let samples = synthesize(positions, &vel_stats, total_samples, step_count);

    write_wav(&samples, output_wav)?;
    info!("   Saved audio => {}", output_wav);

    if let (Some(vid_out), Some(vid_in)) = (output_video, input_video) {
        mux_audio_video(output_wav, vid_in, vid_out)?;
        info!("   Muxed audio+video => {}", vid_out);
    }

    Ok(())
}

struct VelocityStats {
    max_speed: f64,
    max_distance: f64,
}

fn compute_velocity_stats(positions: &[Vec<Vector3<f64>>]) -> VelocityStats {
    let mut max_speed = 0.0f64;
    let mut max_distance = 0.0f64;
    let steps = positions[0].len();

    for step in 1..steps {
        for body_pos in positions.iter().take(3) {
            let vel = (body_pos[step] - body_pos[step - 1]).norm();
            max_speed = max_speed.max(vel);
        }
        for i in 0..3.min(positions.len()) {
            for j in (i + 1)..3.min(positions.len()) {
                let dist = (positions[i][step] - positions[j][step]).norm();
                max_distance = max_distance.max(dist);
            }
        }
    }

    VelocityStats {
        max_speed: max_speed.max(1e-10),
        max_distance: max_distance.max(1e-10),
    }
}

fn synthesize(
    positions: &[Vec<Vector3<f64>>],
    stats: &VelocityStats,
    total_samples: usize,
    step_count: usize,
) -> Vec<i16> {
    let num_bodies = 3.min(positions.len());
    let mut samples = vec![0i16; total_samples * CHANNELS as usize];
    let mut phases = vec![0.0f64; num_bodies];

    for sample_idx in 0..total_samples {
        let t = sample_idx as f64 / total_samples as f64;
        let step_f = t * (step_count - 2) as f64;
        let step = step_f as usize;
        let frac = step_f - step as f64;

        let mut left = 0.0f64;
        let mut right = 0.0f64;

        for body in 0..num_bodies {
            let vel_now = if step + 1 < step_count {
                (positions[body][step + 1] - positions[body][step]).norm()
            } else {
                0.0
            };
            let vel_next = if step + 2 < step_count {
                (positions[body][step + 2] - positions[body][step + 1]).norm()
            } else {
                vel_now
            };
            let speed = vel_now * (1.0 - frac) + vel_next * frac;
            let normalized_speed = (speed / stats.max_speed).clamp(0.0, 1.0);

            let freq = MIN_FREQ + (MAX_FREQ - MIN_FREQ) * normalized_speed;

            let mut amplitude = 0.0f64;
            for other in 0..num_bodies {
                if other != body {
                    let dist = (positions[body][step] - positions[other][step]).norm();
                    let closeness = 1.0 - (dist / stats.max_distance).clamp(0.0, 1.0);
                    amplitude += closeness;
                }
            }
            amplitude = (amplitude / (num_bodies - 1) as f64).clamp(0.0, 1.0) * 0.25;

            phases[body] += freq / SAMPLE_RATE as f64;
            let phase = phases[body] * std::f64::consts::TAU;

            let wave = phase.sin() * 0.6
                + (phase * 2.0).sin() * 0.2
                + (phase * 3.0).sin() * 0.1
                + (phase * 5.0).sin() * 0.05;

            let x_norm = if step < step_count {
                let min_x = positions[body].iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
                let max_x =
                    positions[body].iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
                let range = (max_x - min_x).max(1e-10);
                ((positions[body][step].x - min_x) / range).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let pan_l = (1.0 - x_norm).sqrt();
            let pan_r = x_norm.sqrt();

            left += wave * amplitude * pan_l;
            right += wave * amplitude * pan_r;
        }

        let fade = fade_envelope(t);
        left *= fade;
        right *= fade;

        let idx = sample_idx * 2;
        samples[idx] = (left.clamp(-1.0, 1.0) * 32000.0) as i16;
        samples[idx + 1] = (right.clamp(-1.0, 1.0) * 32000.0) as i16;
    }

    samples
}

fn fade_envelope(t: f64) -> f64 {
    let fade_in = (t * 20.0).clamp(0.0, 1.0);
    let fade_out = ((1.0 - t) * 20.0).clamp(0.0, 1.0);
    fade_in * fade_out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_positions() -> Vec<Vec<Vector3<f64>>> {
        vec![
            vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0), Vector3::new(2.0, 1.0, 0.0)],
            vec![Vector3::new(5.0, 0.0, 0.0), Vector3::new(5.0, 1.0, 0.0), Vector3::new(5.0, 2.0, 0.0)],
            vec![Vector3::new(0.0, 5.0, 0.0), Vector3::new(1.0, 5.0, 0.0), Vector3::new(2.0, 5.0, 0.0)],
        ]
    }

    #[test]
    fn test_fade_envelope_near_zero_at_start() {
        assert!(fade_envelope(0.0) < 0.01);
    }

    #[test]
    fn test_fade_envelope_near_one_at_middle() {
        assert!(fade_envelope(0.5) > 0.99);
    }

    #[test]
    fn test_fade_envelope_near_zero_at_end() {
        assert!(fade_envelope(1.0) < 0.01);
    }

    #[test]
    fn test_fade_envelope_always_non_negative() {
        for i in 0..=1000 {
            let t = i as f64 / 1000.0;
            assert!(fade_envelope(t) >= 0.0, "negative at t={t}");
        }
    }

    #[test]
    fn test_compute_velocity_stats_known_trajectory() {
        let positions = vec![
            vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 4.0, 0.0)],
            vec![Vector3::new(10.0, 0.0, 0.0), Vector3::new(10.0, 0.0, 0.0)],
            vec![Vector3::new(0.0, 10.0, 0.0), Vector3::new(0.0, 10.0, 0.0)],
        ];

        let stats = compute_velocity_stats(&positions);
        assert!((stats.max_speed - 5.0).abs() < 1e-10, "body 0 moves (3,4) => speed 5");

        let expected_dist = ((10.0f64).powi(2) + (10.0f64).powi(2)).sqrt();
        assert!(
            (stats.max_distance - expected_dist).abs() < 1e-10,
            "max distance between body 1 and 2 at step 0"
        );
    }

    #[test]
    fn test_synthesize_determinism() {
        let pos = sample_positions();
        let stats = compute_velocity_stats(&pos);
        let a = synthesize(&pos, &stats, 100, pos[0].len());
        let b = synthesize(&pos, &stats, 100, pos[0].len());
        assert_eq!(a, b, "same input must produce identical output");
    }

    #[test]
    fn test_synthesize_output_length() {
        let pos = sample_positions();
        let stats = compute_velocity_stats(&pos);
        let samples = synthesize(&pos, &stats, 441, pos[0].len());
        assert_eq!(samples.len(), 441 * CHANNELS as usize);
    }

    #[test]
    fn test_synthesize_samples_in_range() {
        let pos = sample_positions();
        let stats = compute_velocity_stats(&pos);
        let samples = synthesize(&pos, &stats, 4410, pos[0].len());
        for (i, &s) in samples.iter().enumerate() {
            assert!(
                (-32000..=32000).contains(&s),
                "sample {i} out of range: {s}"
            );
        }
    }
}

fn write_wav(samples: &[i16], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;

    let mut file = std::fs::File::create(path)?;

    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&CHANNELS.to_le_bytes())?;
    file.write_all(&SAMPLE_RATE.to_le_bytes())?;
    let byte_rate = SAMPLE_RATE * CHANNELS as u32 * (BITS_PER_SAMPLE / 8) as u32;
    file.write_all(&byte_rate.to_le_bytes())?;
    let block_align = CHANNELS * (BITS_PER_SAMPLE / 8);
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&BITS_PER_SAMPLE.to_le_bytes())?;

    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for &sample in samples {
        file.write_all(&sample.to_le_bytes())?;
    }

    Ok(())
}

fn mux_audio_video(
    audio_path: &str,
    video_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()?;

    if !status.success() {
        return Err(format!("ffmpeg mux failed with status {:?}", status).into());
    }
    Ok(())
}
