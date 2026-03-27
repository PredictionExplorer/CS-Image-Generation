//! Extended ambient soundtrack — a longer-form (2-3 minute) audio piece
//! derived from orbital mechanics with reverb, layered drones, and
//! polished fade structure suitable for gallery installations.

use nalgebra::Vector3;
use std::io::Write;
use std::process::{Command, Stdio};
use tracing::info;

const SAMPLE_RATE: u32 = 44100;
const CHANNELS: u16 = 2;
const BITS_PER_SAMPLE: u16 = 16;
const DURATION_SECONDS: f64 = 150.0;
const MIN_FREQ: f64 = 55.0;
const MAX_FREQ: f64 = 440.0;
const FADE_IN_SECONDS: f64 = 8.0;
const FADE_OUT_SECONDS: f64 = 15.0;

const REVERB_DELAY_SAMPLES: usize = 8820;
const REVERB_FEEDBACK: f64 = 0.45;
const REVERB_MIX: f64 = 0.35;

pub fn generate_extended_audio(
    positions: &[Vec<Vector3<f64>>],
    output_wav: &str,
    output_mp3: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating extended ambient soundtrack ({:.0}s)...", DURATION_SECONDS);

    let total_samples = (SAMPLE_RATE as f64 * DURATION_SECONDS) as usize;
    let step_count = positions[0].len();

    let vel_stats = compute_velocity_stats(positions);
    let dry = synthesize_ambient(positions, &vel_stats, total_samples, step_count);
    let wet = apply_reverb(&dry);

    write_wav(&wet, output_wav)?;
    info!("   Saved extended audio => {}", output_wav);

    if let Some(mp3_path) = output_mp3 {
        encode_mp3(output_wav, mp3_path)?;
        info!("   Saved MP3 => {}", mp3_path);
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

fn synthesize_ambient(
    positions: &[Vec<Vector3<f64>>],
    stats: &VelocityStats,
    total_samples: usize,
    step_count: usize,
) -> Vec<i16> {
    let num_bodies = 3.min(positions.len());
    let mut samples = vec![0i16; total_samples * CHANNELS as usize];
    let mut phases = vec![0.0f64; num_bodies];
    let mut drone_phase = 0.0f64;

    for sample_idx in 0..total_samples {
        let t = sample_idx as f64 / total_samples as f64;
        let step_f = t * (step_count - 2) as f64;
        let step = step_f as usize;
        let frac = step_f - step as f64;

        let mut left = 0.0f64;
        let mut right = 0.0f64;

        let mut avg_closeness = 0.0f64;

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
            let norm_speed = (speed / stats.max_speed).clamp(0.0, 1.0);

            let freq = MIN_FREQ + (MAX_FREQ - MIN_FREQ) * norm_speed * norm_speed;

            let mut amplitude = 0.0f64;
            for other in 0..num_bodies {
                if other != body {
                    let dist = (positions[body][step] - positions[other][step]).norm();
                    let closeness = 1.0 - (dist / stats.max_distance).clamp(0.0, 1.0);
                    amplitude += closeness;
                    avg_closeness += closeness;
                }
            }
            amplitude = (amplitude / (num_bodies - 1) as f64).clamp(0.0, 1.0) * 0.18;

            phases[body] += freq / SAMPLE_RATE as f64;
            let phase = phases[body] * std::f64::consts::TAU;

            let wave = phase.sin() * 0.5
                + (phase * 2.0).sin() * 0.15
                + (phase * 3.0).sin() * 0.08
                + (phase * 5.0).sin() * 0.03
                + (phase * 7.0).sin() * 0.02;

            let x_norm = if step < step_count {
                let min_x = positions[body].iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
                let max_x = positions[body].iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
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

        avg_closeness /= (num_bodies * (num_bodies - 1)) as f64;
        let drone_freq = MIN_FREQ * 0.5;
        drone_phase += drone_freq / SAMPLE_RATE as f64;
        let drone = (drone_phase * std::f64::consts::TAU).sin() * 0.06 * avg_closeness;
        left += drone;
        right += drone;

        let fade = extended_fade_envelope(t);
        left *= fade;
        right *= fade;

        let idx = sample_idx * 2;
        samples[idx] = (left.clamp(-1.0, 1.0) * 30000.0) as i16;
        samples[idx + 1] = (right.clamp(-1.0, 1.0) * 30000.0) as i16;
    }

    samples
}

fn extended_fade_envelope(t: f64) -> f64 {
    let total = DURATION_SECONDS;
    let fade_in = (t * total / FADE_IN_SECONDS).clamp(0.0, 1.0);
    let fade_in_smooth = fade_in * fade_in * (3.0 - 2.0 * fade_in);
    let fade_out = ((1.0 - t) * total / FADE_OUT_SECONDS).clamp(0.0, 1.0);
    let fade_out_smooth = fade_out * fade_out * (3.0 - 2.0 * fade_out);
    fade_in_smooth * fade_out_smooth
}

fn apply_reverb(samples: &[i16]) -> Vec<i16> {
    let n = samples.len();
    let mut output = vec![0i16; n];
    let mut delay_l = vec![0.0f64; REVERB_DELAY_SAMPLES];
    let mut delay_r = vec![0.0f64; REVERB_DELAY_SAMPLES];
    let mut dl_idx = 0usize;
    let mut dr_idx = 0usize;

    for i in (0..n).step_by(2) {
        let dry_l = samples[i] as f64 / 32768.0;
        let dry_r = samples[i + 1] as f64 / 32768.0;

        let wet_l = delay_l[dl_idx];
        let wet_r = delay_r[dr_idx];

        delay_l[dl_idx] = dry_l + wet_l * REVERB_FEEDBACK;
        delay_r[dr_idx] = dry_r + wet_r * REVERB_FEEDBACK;

        dl_idx = (dl_idx + 1) % REVERB_DELAY_SAMPLES;
        dr_idx = (dr_idx + 1) % REVERB_DELAY_SAMPLES;

        let mix_l = dry_l * (1.0 - REVERB_MIX) + wet_l * REVERB_MIX;
        let mix_r = dry_r * (1.0 - REVERB_MIX) + wet_r * REVERB_MIX;

        output[i] = (mix_l.clamp(-1.0, 1.0) * 32000.0) as i16;
        output[i + 1] = (mix_r.clamp(-1.0, 1.0) * 32000.0) as i16;
    }

    output
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

fn encode_mp3(wav_path: &str, mp3_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new("ffmpeg")
        .args([
            "-y", "-i", wav_path,
            "-codec:a", "libmp3lame", "-b:a", "192k",
            "-ar", "44100",
            mp3_path,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;

    if !status.success() {
        return Err(format!("ffmpeg MP3 encode failed with status {:?}", status).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fade_envelope_shape() {
        assert!(extended_fade_envelope(0.0) < 0.01);
        assert!(extended_fade_envelope(0.5) > 0.9);
        assert!(extended_fade_envelope(1.0) < 0.01);
    }

    #[test]
    fn test_reverb_preserves_length() {
        let input = vec![100i16; 200];
        let output = apply_reverb(&input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_reverb_does_not_clip() {
        let input: Vec<i16> = (0..2000).map(|i| ((i as f64 / 50.0).sin() * 30000.0) as i16).collect();
        let output = apply_reverb(&input);
        for &s in &output {
            assert!((-32000..=32000).contains(&s), "clipped: {}", s);
        }
    }
}
