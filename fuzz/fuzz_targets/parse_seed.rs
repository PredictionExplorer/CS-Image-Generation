#![no_main]
use libfuzzer_sys::fuzz_target;
use three_body_problem::app;

fuzz_target!(|data: &str| {
    let _ = app::parse_seed(data);
});
