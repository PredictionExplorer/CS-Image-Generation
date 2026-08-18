//! V10 `sonification` -- The Orbit's Score.
//!
//! A musical rendering of the run: each pair's distance drives a
//! band-limited saw voice (closer = higher), pair speeds drive the
//! amplitude envelopes, a drama-following resonant low-pass opens in the
//! violent stretches, and syzygies strike FM bells on a pentatonic lattice
//! rooted at the palette's anchor hue. Mastered to -14 LUFS and muxed onto
//! copies of the finished main and sweep videos.

use crate::error::Result;
use crate::oklab::oklab_to_oklch;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::common::audio::{
    Adsr, Biquad, SAMPLE_RATE, equal_power_pan, fade_ends, fm_bell, integrated_lufs, mux,
    normalize_to_lufs, saw_bandlimited, soft_limit, write_wav_stereo_24bit,
};
use crate::viz::common::kinematics::PAIRS;
use crate::viz::context::VizContext;
use crate::viz::sink::ArtifactSink;
use tracing::{info, warn};

/// Score length in seconds (matches the 30 s main video).
const SCORE_SECONDS: f64 = 30.0;
/// Pitch base in Hz.
const BASE_HZ: f64 = 55.0;
/// Pitch span in octaves.
const OCTAVES: f64 = 5.0;
/// Mastering target.
const TARGET_LUFS: f64 = -14.0;
/// Pentatonic degrees (semitones above the root).
const PENTATONIC: [f64; 5] = [0.0, 2.0, 4.0, 7.0, 9.0];

/// Percentile window of a series (strided sample).
fn percentile_window(series: &[f64], low: f64, high: f64) -> (f64, f64) {
    let mut sample: Vec<f64> =
        series.iter().step_by(37).copied().filter(|v| v.is_finite()).collect();
    if sample.is_empty() {
        return (0.0, 1.0);
    }
    sample.sort_by(f64::total_cmp);
    let at = |q: f64| sample[((sample.len() - 1) as f64 * q) as usize];
    let lo = at(low);
    (lo, at(high).max(lo + 1e-12))
}

/// The sonification mode.
pub struct Sonification;

impl VizMode for Sonification {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("sonification").expect("sonification is in the catalog")
    }

    #[allow(clippy::too_many_lines)]
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let steps = ctx.step_count();
        if steps < 100 {
            warn!("sonification skipped: trajectory too short");
            return Ok(());
        }
        let sample_rate = f64::from(SAMPLE_RATE);
        let total_samples = (SCORE_SECONDS * sample_rate) as usize;
        let kinematics = ctx.kinematics();
        let drama = ctx.events().drama(kinematics);
        let mut rng = ctx.fork_rng("sonification");

        // Per-pair normalization windows and pan sources.
        let windows: Vec<(f64, f64)> = kinematics
            .pairwise
            .iter()
            .map(|series| percentile_window(series, 0.05, 0.95))
            .collect();
        // Pair relative-speed windows for the amplitude envelopes.
        let pair_speed = |pair: usize, step: usize| -> f64 {
            let (a, b) = PAIRS[pair];
            (kinematics.velocities[a][step] - kinematics.velocities[b][step]).norm()
        };
        let speed_windows: Vec<(f64, f64)> = (0..3)
            .map(|pair| {
                let series: Vec<f64> =
                    (0..steps).step_by(37).map(|step| pair_speed(pair, step)).collect();
                percentile_window(&series, 0.10, 0.95)
            })
            .collect();

        // Root pitch class from the palette anchor hue.
        let root_class = {
            let mean = ctx.mean_color(0);
            let (_, _, hue) = oklab_to_oklch(mean.0, mean.1, mean.2);
            (hue / 360.0 * 12.0).floor().clamp(0.0, 11.0)
        };

        // Bell events from syzygies, humanized within +/- 20 ms.
        struct Bell {
            start: f64,
            frequency: f64,
            gain: f64,
            pan: f64,
        }
        let mut bells: Vec<Bell> = Vec::new();
        for syzygy in &ctx.events().syzygies {
            let time =
                syzygy.step as f64 / steps as f64 * SCORE_SECONDS + (rng.next_f64() - 0.5) * 0.04;
            let degree = PENTATONIC[syzygy.middle_body % PENTATONIC.len()];
            let octave = f64::from(u32::try_from(syzygy.middle_body).unwrap_or(0));
            let frequency =
                BASE_HZ * 4.0 * 2.0_f64.powf((root_class + degree) / 12.0 + octave * 0.5);
            bells.push(Bell {
                start: time.max(0.0),
                frequency,
                gain: 0.25 + 0.35 * syzygy.sharpness,
                pan: (syzygy.step as f64 / steps as f64) * 1.2 - 0.6,
            });
        }
        info!(
            "   sonification: {} bells, root class {root_class:.0}, {} samples",
            bells.len(),
            total_samples
        );

        // --- Synthesis.
        let mut left = vec![0.0f64; total_samples];
        let mut right = vec![0.0f64; total_samples];
        let mut phases = [0.0f64; 3];
        let mut filters = [Biquad::default(), Biquad::default(), Biquad::default()];
        let attack = Adsr { attack: 1.2, decay: 0.0, sustain: 1.0, release: 1.5 };
        for sample in 0..total_samples {
            let t = sample as f64 / sample_rate;
            let step = ((sample as f64 / total_samples as f64) * steps as f64) as usize % steps;
            let drama_now = drama[step.min(drama.len() - 1)];
            let mut frame = (0.0f64, 0.0f64);
            for pair in 0..3 {
                let (lo, hi) = windows[pair];
                let normalized =
                    ((kinematics.pairwise[pair][step] - lo) / (hi - lo)).clamp(0.0, 1.0);
                let frequency = BASE_HZ * 2.0_f64.powf(OCTAVES * (1.0 - normalized));
                let dphase = frequency / sample_rate;
                phases[pair] = (phases[pair] + dphase) % 1.0;
                let raw = saw_bandlimited(phases[pair], dphase);
                // Update the drama-following filter every 64 samples.
                if sample.is_multiple_of(64) {
                    filters[pair].set_lowpass(280.0 + 4200.0 * drama_now, 1.15, sample_rate);
                }
                let filtered = filters[pair].process(raw);
                let (slo, shi) = speed_windows[pair];
                let speed_norm = ((pair_speed(pair, step) - slo) / (shi - slo)).clamp(0.0, 1.0);
                let amplitude = (0.10 + 0.24 * speed_norm)
                    * attack.amplitude(t, SCORE_SECONDS - attack.release);
                // Pan by the pair centroid's normalized x position.
                let (a, b) = PAIRS[pair];
                let centroid_x = (ctx.positions[a][step].x + ctx.positions[b][step].x) * 0.5;
                let pan = (centroid_x / windows[pair].1).clamp(-0.7, 0.7);
                let (l, r) = equal_power_pan(filtered * amplitude, pan);
                frame.0 += l;
                frame.1 += r;
            }
            for bell in &bells {
                if t >= bell.start && t < bell.start + 3.0 {
                    let (l, r) = equal_power_pan(
                        fm_bell(t - bell.start, bell.frequency, 2.2) * bell.gain,
                        bell.pan,
                    );
                    frame.0 += l;
                    frame.1 += r;
                }
            }
            left[sample] = frame.0;
            right[sample] = frame.1;
        }
        soft_limit(&mut left, &mut right, 1.3);
        normalize_to_lufs(&mut left, &mut right, TARGET_LUFS, sample_rate);
        fade_ends(&mut left, (sample_rate * 0.05) as usize);
        fade_ends(&mut right, (sample_rate * 0.05) as usize);
        let measured = integrated_lufs(&left, &right, sample_rate);
        info!("   sonification: mastered to {measured:.1} LUFS (target {TARGET_LUFS})");

        write_wav_stereo_24bit(&sink.path("score.wav"), &left, &right)?;
        sink.record("score.wav", "audio");

        // --- Mux onto copies of the finished web videos.
        let main_video = format!("{}/videos/web/main.mp4", ctx.seed_dir);
        let sweep_video = format!("{}/videos/web/spectral_sweep.mp4", ctx.seed_dir);
        for (source, name) in [(main_video, "main_scored.mp4"), (sweep_video, "sweep_scored.mp4")] {
            if std::path::Path::new(&source).exists() {
                match mux(&source, &sink.path("score.wav"), &sink.path(name)) {
                    Ok(()) => sink.record(name, "video"),
                    Err(error) => warn!("sonification mux failed for {name}: {error}"),
                }
            } else {
                warn!("sonification: {source} missing (image-only run?); skipping {name}");
            }
        }

        let bell_events: Vec<serde_json::Value> = bells
            .iter()
            .map(|bell| {
                serde_json::json!({
                    "time_s": bell.start,
                    "frequency_hz": bell.frequency,
                    "gain": bell.gain,
                })
            })
            .collect();
        let meta = serde_json::json!({
            "seconds": SCORE_SECONDS,
            "base_hz": BASE_HZ,
            "octaves": OCTAVES,
            "root_pitch_class": root_class,
            "target_lufs": TARGET_LUFS,
            "measured_lufs": measured,
            "bells": bell_events,
        });
        let json = serde_json::to_string_pretty(&meta).map_err(std::io::Error::other)?;
        sink.write_text("score.json", &json, "data")?;
        Ok(())
    }
}
