//! Global pipeline toggles set from the CLI before rendering (v2 features).
//!
//! These are intentionally simple atomics so the hot spectral path does not carry
//! extra configuration references through every closure.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

/// Plummer softening length squared (0 = disabled). Bits as `f64`.
static SIM_SOFTENING_EPS2: AtomicU64 = AtomicU64::new(0);

/// Number of shutter samples per simulation step (1 = off).
static SHUTTER_SAMPLES: AtomicU8 = AtomicU8::new(1);

/// Use perspective + thin-lens style `CoC` when true.
static PERSPECTIVE_CAMERA: AtomicBool = AtomicBool::new(true);

/// Enable comet wake overlay in spectral accumulation.
static COMET_WAKE: AtomicBool = AtomicBool::new(true);

/// Enable volumetric-style nebula seeded by trajectory (replaces pure 2-D noise when true).
static VOLUMETRIC_NEBULA: AtomicBool = AtomicBool::new(true);

/// `0` = legacy radial dispersion, `1` = achromatic polynomial CA in `convert_spd_buffer_to_rgba`.
static CA_MODEL: AtomicU8 = AtomicU8::new(1);

/// Artistic speed-of-light scalar used for `Doppler` wavelength shifts (larger = subtler).
static DOPPLER_C_ART: AtomicU64 = AtomicU64::new(400.0f64.to_bits());

/// Scene-linear exposure multiplier before `AgX` tonemapping (metering). Default `1.0`.
static SCENE_EXPOSURE_SCALE: AtomicU64 = AtomicU64::new(1.0f64.to_bits());

/// Enable lens flare / starburst post pass (CLI turns on for full v2 runs).
static LENS_FLARE: AtomicBool = AtomicBool::new(false);

/// Enable body sphere splats at the end of each accumulation batch.
static BODY_SPHERES: AtomicBool = AtomicBool::new(true);

/// Use multi-act director for video checkpoint spacing.
static MULTI_ACT_DIRECTOR: AtomicBool = AtomicBool::new(true);

/// Configure Plummer softening (`epsilon` in world units; 0 disables).
pub fn set_sim_softening_epsilon(eps: f64) {
    let bits = if eps > 0.0 { (eps * eps).to_bits() } else { 0 };
    SIM_SOFTENING_EPS2.store(bits, Ordering::Relaxed);
}

/// Current Plummer ε² (0 if disabled).
#[must_use]
pub fn sim_softening_eps2() -> f64 {
    f64::from_bits(SIM_SOFTENING_EPS2.load(Ordering::Relaxed))
}

/// Shutter samples per step (clamped 1..=32).
pub fn set_shutter_samples(n: u8) {
    let n = n.clamp(1, 32);
    SHUTTER_SAMPLES.store(n, Ordering::Relaxed);
}

/// Current shutter oversample count per simulation step (at least 1).
#[must_use]
pub fn shutter_samples() -> u8 {
    SHUTTER_SAMPLES.load(Ordering::Relaxed).max(1)
}

/// Enable or disable perspective projection + thin-lens `CoC`.
pub fn set_perspective_camera(on: bool) {
    PERSPECTIVE_CAMERA.store(on, Ordering::Relaxed);
}

/// Whether perspective camera projection is active.
#[must_use]
pub fn perspective_camera_enabled() -> bool {
    PERSPECTIVE_CAMERA.load(Ordering::Relaxed)
}

/// Enable or disable the comet-head / wake spectral overlay.
pub fn set_comet_wake(on: bool) {
    COMET_WAKE.store(on, Ordering::Relaxed);
}

/// Whether comet wake accumulation is enabled.
#[must_use]
pub fn comet_wake_enabled() -> bool {
    COMET_WAKE.load(Ordering::Relaxed)
}

/// Enable trajectory-seeded volumetric nebula instead of pure 2D noise.
pub fn set_volumetric_nebula(on: bool) {
    VOLUMETRIC_NEBULA.store(on, Ordering::Relaxed);
}

/// Whether volumetric nebula generation is enabled.
#[must_use]
pub fn volumetric_nebula_enabled() -> bool {
    VOLUMETRIC_NEBULA.load(Ordering::Relaxed)
}

/// `0` legacy CA sampling, `1` physical-style CA.
pub fn set_ca_model_physical(on: bool) {
    CA_MODEL.store(u8::from(on), Ordering::Relaxed);
}

/// Whether the physical chromatic aberration model is selected (vs legacy radial).
#[must_use]
pub fn ca_model_physical() -> bool {
    CA_MODEL.load(Ordering::Relaxed) == 1
}

/// Artistic c for Doppler; use `f64::INFINITY` to disable.
pub fn set_doppler_c_art(c: f64) {
    DOPPLER_C_ART.store(c.to_bits(), Ordering::Relaxed);
}

/// Artistic speed-of-light scale used for line-of-sight Doppler shifts.
#[must_use]
pub fn doppler_c_art() -> f64 {
    f64::from_bits(DOPPLER_C_ART.load(Ordering::Relaxed))
}

/// Manual scene-linear exposure gain applied with auto metering.
pub fn set_scene_exposure_scale(s: f64) {
    SCENE_EXPOSURE_SCALE.store(s.clamp(0.05, 20.0).to_bits(), Ordering::Relaxed);
}

/// Current manual scene exposure multiplier.
#[must_use]
pub fn scene_exposure_scale() -> f64 {
    f64::from_bits(SCENE_EXPOSURE_SCALE.load(Ordering::Relaxed))
}

/// Enable or disable lens flare / diffraction streak post processing.
pub fn set_lens_flare(on: bool) {
    LENS_FLARE.store(on, Ordering::Relaxed);
}

/// Whether lens flare post processing is enabled.
#[must_use]
pub fn lens_flare_enabled() -> bool {
    LENS_FLARE.load(Ordering::Relaxed)
}

/// Enable or disable Gaussian body sphere splats in the spectral buffer.
pub fn set_body_spheres(on: bool) {
    BODY_SPHERES.store(on, Ordering::Relaxed);
}

/// Whether body sphere drawing is enabled.
#[must_use]
pub fn body_spheres_enabled() -> bool {
    BODY_SPHERES.load(Ordering::Relaxed)
}

/// Enable non-uniform video checkpoints (multi-act director pacing).
pub fn set_multi_act_director(on: bool) {
    MULTI_ACT_DIRECTOR.store(on, Ordering::Relaxed);
}

/// Whether multi-act director checkpoint spacing is used for video.
#[must_use]
pub fn multi_act_director_enabled() -> bool {
    MULTI_ACT_DIRECTOR.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize test access to global atomics so they don't tramp on each other.
    static FLAG_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_sim_softening_round_trips() {
        let _lock = FLAG_LOCK.lock().expect("lock");
        set_sim_softening_epsilon(0.0);
        assert_eq!(sim_softening_eps2(), 0.0);
        set_sim_softening_epsilon(0.25);
        assert!((sim_softening_eps2() - 0.0625).abs() < 1e-12);
        set_sim_softening_epsilon(0.0);
    }

    #[test]
    fn test_shutter_samples_clamped() {
        let _lock = FLAG_LOCK.lock().expect("lock");
        set_shutter_samples(0);
        assert_eq!(shutter_samples(), 1);
        set_shutter_samples(200);
        assert_eq!(shutter_samples(), 32);
        set_shutter_samples(8);
        assert_eq!(shutter_samples(), 8);
    }

    #[test]
    fn test_boolean_flags_round_trip() {
        let _lock = FLAG_LOCK.lock().expect("lock");
        set_perspective_camera(false);
        assert!(!perspective_camera_enabled());
        set_perspective_camera(true);
        assert!(perspective_camera_enabled());

        set_comet_wake(false);
        assert!(!comet_wake_enabled());
        set_comet_wake(true);
        assert!(comet_wake_enabled());

        set_volumetric_nebula(false);
        assert!(!volumetric_nebula_enabled());
        set_volumetric_nebula(true);
        assert!(volumetric_nebula_enabled());

        set_body_spheres(false);
        assert!(!body_spheres_enabled());
        set_body_spheres(true);
        assert!(body_spheres_enabled());

        set_multi_act_director(false);
        assert!(!multi_act_director_enabled());
        set_multi_act_director(true);
        assert!(multi_act_director_enabled());

        set_ca_model_physical(false);
        assert!(!ca_model_physical());
        set_ca_model_physical(true);
        assert!(ca_model_physical());

        set_lens_flare(true);
        assert!(lens_flare_enabled());
        set_lens_flare(false);
        assert!(!lens_flare_enabled());
    }

    #[test]
    fn test_scalar_scene_exposure_clamped() {
        let _lock = FLAG_LOCK.lock().expect("lock");
        set_scene_exposure_scale(100.0);
        assert!(scene_exposure_scale() <= 20.0);
        set_scene_exposure_scale(0.00001);
        assert!(scene_exposure_scale() >= 0.05);
        set_scene_exposure_scale(1.0);
        assert!((scene_exposure_scale() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_doppler_c_art_round_trip() {
        let _lock = FLAG_LOCK.lock().expect("lock");
        set_doppler_c_art(2048.0);
        assert!((doppler_c_art() - 2048.0).abs() < 1e-9);
        set_doppler_c_art(400.0);
    }
}
