//! Multi-act video pacing: non-uniform simulation checkpoints for richer cinematography.

use crate::render::constants::DEFAULT_TARGET_FRAMES;

/// Build cumulative simulation step indices for each emitted video frame.
///
/// Act I (slow reveal), Act II (ballet), Act III (long-exposure rush) use a monotone
/// remap of normalized time so early frames advance fewer simulation steps than late ones.
#[must_use]
pub fn checkpoint_indices_multi_act(total_steps: usize, num_frames: usize) -> Vec<usize> {
    if total_steps == 0 || num_frames == 0 {
        return Vec::new();
    }
    let last = total_steps.saturating_sub(1);
    let nf = num_frames.max(1);
    let mut out = Vec::with_capacity(nf);
    for fi in 0..nf {
        let t = (fi as f64 + 1.0) / nf as f64;
        // Piecewise easing: slow start (t^0.55), middle linear-ish, fast finish (t^1.35 blended).
        let u = if t < 0.35 {
            (t / 0.35).powf(0.55) * 0.25
        } else if t < 0.65 {
            let local = (t - 0.35) / 0.30;
            0.25 + local.powf(0.95) * 0.35
        } else {
            let local = (t - 0.65) / 0.35;
            0.60 + local.powf(1.35) * 0.40
        };
        let step = ((last as f64) * u).round() as usize;
        out.push(step.min(last));
    }
    // Ensure strictly non-decreasing (duplicate frames collapse on merge upstream).
    let mut prev = 0;
    for s in &mut out {
        if *s < prev {
            *s = prev;
        }
        prev = *s;
    }
    out
}

/// Uniform checkpoints (legacy): every `frame_interval` steps plus final frame.
#[must_use]
pub fn checkpoint_indices_uniform(total_steps: usize, frame_interval: usize) -> Vec<usize> {
    if total_steps == 0 {
        return Vec::new();
    }
    let frame_interval = frame_interval.max(1);
    let mut checkpoints = Vec::new();
    let mut checkpoint = frame_interval;
    while checkpoint < total_steps {
        if checkpoint > 0 {
            checkpoints.push(checkpoint);
        }
        checkpoint += frame_interval;
    }
    let final_step = total_steps - 1;
    if checkpoints.last().copied() != Some(final_step) {
        checkpoints.push(final_step);
    }
    checkpoints
}

/// Resolve checkpoint list for pass-2 given total steps and desired frame count.
#[must_use]
pub fn resolve_video_checkpoints(total_steps: usize, use_director: bool) -> Vec<usize> {
    if use_director {
        checkpoint_indices_multi_act(total_steps, DEFAULT_TARGET_FRAMES as usize)
    } else {
        let fi = (total_steps / DEFAULT_TARGET_FRAMES as usize).max(1);
        checkpoint_indices_uniform(total_steps, fi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_act_is_non_decreasing_and_in_range() {
        let out = checkpoint_indices_multi_act(10_000, 1_800);
        assert_eq!(out.len(), 1_800);
        assert!(out.iter().all(|&s| s < 10_000));
        for win in out.windows(2) {
            assert!(win[0] <= win[1], "checkpoints must be non-decreasing");
        }
    }

    #[test]
    fn test_multi_act_starts_small_ends_at_last_step() {
        let total = 10_000;
        let out = checkpoint_indices_multi_act(total, 600);
        let first_quarter = out[out.len() / 4];
        let last = *out.last().expect("non-empty");
        assert!(first_quarter < total / 2, "Act I should advance slowly; got {first_quarter}");
        assert_eq!(last, total - 1);
    }

    #[test]
    fn test_uniform_checkpoints_last_is_total_minus_one() {
        let out = checkpoint_indices_uniform(100, 10);
        assert_eq!(*out.last().expect("non-empty"), 99);
    }

    #[test]
    fn test_empty_inputs_produce_empty_outputs() {
        assert!(checkpoint_indices_multi_act(0, 1).is_empty());
        assert!(checkpoint_indices_uniform(0, 1).is_empty());
    }

    #[test]
    fn test_resolve_video_checkpoints_dispatches() {
        let director = resolve_video_checkpoints(1_000, true);
        let uniform = resolve_video_checkpoints(1_000, false);
        assert!(!director.is_empty());
        assert!(!uniform.is_empty());
        assert_ne!(director, uniform);
    }
}
