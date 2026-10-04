//! Helpers shared by the unit tests of several modules.

use std::process::Command;

/// Whether every one of `tools` (`ffmpeg`, `ffprobe`) runs here. When one does not, the
/// calling test `test` should return early: locally it is skipped with a message; under CI
/// (the `CI` environment variable is set) this panics instead, so that a runner without the
/// tools fails rather than passes without testing.
pub(crate) fn media_tools_available(test: &str, tools: &[&str]) -> bool {
    let missing: Vec<&str> = tools
        .iter()
        .copied()
        .filter(|tool| {
            !Command::new(tool).arg("-version").output().is_ok_and(|output| output.status.success())
        })
        .collect();
    if missing.is_empty() {
        return true;
    }
    assert!(
        std::env::var_os("CI").is_none(),
        "{test} needs {missing:?} on PATH, and CI is set: install FFmpeg in this job"
    );
    eprintln!("skipping {test}: {missing:?} not on PATH");
    false
}
