#![no_main]
use libfuzzer_sys::fuzz_target;
use three_body_problem::spectrum::NUM_BINS;
use three_body_problem::spectrum_simd;

fuzz_target!(|data: &[u8]| {
    if data.len() < NUM_BINS * 8 {
        return;
    }
    let mut spd = [0.0f64; NUM_BINS];
    for (i, chunk) in data[..NUM_BINS * 8].chunks_exact(8).enumerate() {
        spd[i] = f64::from_le_bytes(chunk.try_into().unwrap());
    }
    let _ = spectrum_simd::spd_to_rgba_simd(&spd);
});
