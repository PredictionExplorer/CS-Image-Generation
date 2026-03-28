//! Performance instrumentation for pipeline stage profiling.
//!
//! Provides [`StageTimer`] to measure wall-clock time and memory usage for each
//! pipeline stage, and [`PerformanceProfile`] to aggregate and serialize the
//! results into `generation_log.json`.

use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::info;

/// Resident set size in megabytes, sampled via `getrusage(RUSAGE_SELF)`.
///
/// Returns 0.0 on platforms where `getrusage` is unavailable or fails.
fn current_rss_mb() -> f64 {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        // SAFETY: zeroed rusage is valid, and getrusage with RUSAGE_SELF is safe.
        unsafe {
            let mut usage: libc::rusage = std::mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
                // macOS reports ru_maxrss in bytes; Linux reports in kilobytes.
                #[cfg(target_os = "macos")]
                {
                    usage.ru_maxrss as f64 / (1024.0 * 1024.0)
                }
                #[cfg(target_os = "linux")]
                {
                    usage.ru_maxrss as f64 / 1024.0
                }
            } else {
                0.0
            }
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0.0
    }
}

/// Number of logical CPUs available to the process.
fn logical_cpu_count() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Captures wall-clock time, memory, and thread metrics for a single pipeline stage.
///
/// Usage:
/// ```ignore
/// let timer = StageTimer::start("Borda Search");
/// run_borda_selection(...);
/// let timing = timer.finish_with_throughput("100000 candidates, 1000000 steps");
/// ```
pub struct StageTimer {
    name: String,
    start: Instant,
    rss_mb_start: f64,
    rayon_threads: usize,
}

impl StageTimer {
    /// Begin timing a named stage.
    pub fn start(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
            rss_mb_start: current_rss_mb(),
            rayon_threads: rayon::current_num_threads(),
        }
    }

    /// Finish timing and return the completed [`StageTiming`] record.
    pub fn finish(self) -> StageTiming {
        self.finish_with_throughput(None::<&str>)
    }

    /// Finish timing with an optional throughput description (e.g. "100000 sims").
    pub fn finish_with_throughput(self, throughput: Option<impl Into<String>>) -> StageTiming {
        let wall_clock_secs = self.start.elapsed().as_secs_f64();
        let rss_mb_end = current_rss_mb();
        let throughput = throughput.map(|s| s.into());

        info!(
            stage = %self.name,
            wall_clock_secs = format!("{wall_clock_secs:.3}"),
            peak_rss_mb = format!("{rss_mb_end:.0}"),
            threads = self.rayon_threads,
            throughput = throughput.as_deref().unwrap_or(""),
            "Stage complete",
        );

        StageTiming {
            stage_name: self.name,
            wall_clock_secs,
            peak_rss_mb_start: self.rss_mb_start,
            peak_rss_mb_end: rss_mb_end,
            rayon_threads: self.rayon_threads,
            throughput,
            sub_stages: None,
        }
    }
}

/// Timing record for a single pipeline stage, serialized into `generation_log.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTiming {
    pub stage_name: String,
    pub wall_clock_secs: f64,
    pub peak_rss_mb_start: f64,
    pub peak_rss_mb_end: f64,
    pub rayon_threads: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub throughput: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub_stages: Option<Vec<StageTiming>>,
}

/// Aggregated performance profile for the entire generation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub total_wall_clock_secs: f64,
    pub num_logical_cpus: usize,
    pub rayon_threads: usize,
    pub stages: Vec<StageTiming>,
}

impl PerformanceProfile {
    /// Create a new empty profile, recording the current system context.
    pub fn new() -> Self {
        Self {
            total_wall_clock_secs: 0.0,
            num_logical_cpus: logical_cpu_count(),
            rayon_threads: rayon::current_num_threads(),
            stages: Vec::new(),
        }
    }

    /// Add a completed stage timing.
    pub fn push(&mut self, timing: StageTiming) {
        self.stages.push(timing);
    }

    /// Set the total wall-clock time (measured externally from the full pipeline).
    pub fn set_total(&mut self, total_secs: f64) {
        self.total_wall_clock_secs = total_secs;
    }

    /// Log a human-readable summary table via `tracing::info!`.
    pub fn log_summary(&self) {
        let mut lines = String::with_capacity(2048);
        lines.push('\n');
        lines.push_str("┌──────────────────────────┬───────────┬────────────┬─────────┬──────────────────────────────────────────┐\n");
        lines.push_str("│ Stage                    │  Time (s) │  RSS (MB)  │ Threads │ Throughput                               │\n");
        lines.push_str("├──────────────────────────┼───────────┼────────────┼─────────┼──────────────────────────────────────────┤\n");

        for s in &self.stages {
            let throughput_str = s.throughput.as_deref().unwrap_or("");
            lines.push_str(&format!(
                "│ {:<24} │ {:>9.3} │ {:>10.0} │ {:>7} │ {:<40} │\n",
                s.stage_name, s.wall_clock_secs, s.peak_rss_mb_end, s.rayon_threads, throughput_str,
            ));

            if let Some(subs) = &s.sub_stages {
                for sub in subs {
                    lines.push_str(&format!(
                        "│   {:<22} │ {:>9.3} │ {:>10.0} │ {:>7} │ {:<40} │\n",
                        sub.stage_name,
                        sub.wall_clock_secs,
                        sub.peak_rss_mb_end,
                        sub.rayon_threads,
                        sub.throughput.as_deref().unwrap_or(""),
                    ));
                }
            }
        }

        lines.push_str("├──────────────────────────┼───────────┼────────────┼─────────┼──────────────────────────────────────────┤\n");
        lines.push_str(&format!(
            "│ {:<24} │ {:>9.3} │ {:>10} │ {:>7} │ {:<40} │\n",
            "TOTAL",
            self.total_wall_clock_secs,
            "",
            self.rayon_threads,
            format!("{} logical CPUs", self.num_logical_cpus),
        ));
        lines.push_str("└──────────────────────────┴───────────┴────────────┴─────────┴──────────────────────────────────────────┘");

        info!("Performance summary:{lines}");
    }
}

impl Default for PerformanceProfile {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_timer_measures_elapsed_time() {
        let timer = StageTimer::start("test_stage");
        std::thread::sleep(std::time::Duration::from_millis(15));
        let timing = timer.finish();
        assert!(
            timing.wall_clock_secs >= 0.010,
            "expected >= 10ms, got {:.4}s",
            timing.wall_clock_secs
        );
        assert_eq!(timing.stage_name, "test_stage");
    }

    #[test]
    fn test_stage_timer_with_throughput() {
        let timer = StageTimer::start("throughput_stage");
        let timing = timer.finish_with_throughput(Some("500 items"));
        assert_eq!(timing.throughput.as_deref(), Some("500 items"));
    }

    #[test]
    fn test_stage_timer_without_throughput() {
        let timer = StageTimer::start("no_throughput");
        let timing = timer.finish();
        assert!(timing.throughput.is_none());
    }

    #[test]
    fn test_rss_is_non_negative() {
        let rss = current_rss_mb();
        assert!(rss >= 0.0, "RSS should be non-negative, got {rss}");
    }

    #[test]
    fn test_timing_fields_populated() {
        let timer = StageTimer::start("fields_test");
        let timing = timer.finish_with_throughput(Some("42 units"));
        assert!(!timing.stage_name.is_empty());
        assert!(timing.wall_clock_secs >= 0.0);
        assert!(timing.peak_rss_mb_start >= 0.0);
        assert!(timing.peak_rss_mb_end >= 0.0);
        assert!(timing.rayon_threads > 0);
        assert!(timing.sub_stages.is_none());
    }

    #[test]
    fn test_performance_profile_serialization_roundtrip() {
        let mut profile = PerformanceProfile::new();
        profile.set_total(99.5);
        profile.push(StageTiming {
            stage_name: "Phase1".to_string(),
            wall_clock_secs: 42.0,
            peak_rss_mb_start: 100.0,
            peak_rss_mb_end: 200.0,
            rayon_threads: 8,
            throughput: Some("1000 items".to_string()),
            sub_stages: None,
        });
        profile.push(StageTiming {
            stage_name: "Phase2".to_string(),
            wall_clock_secs: 57.5,
            peak_rss_mb_start: 200.0,
            peak_rss_mb_end: 350.0,
            rayon_threads: 8,
            throughput: None,
            sub_stages: Some(vec![StageTiming {
                stage_name: "sub_task_a".to_string(),
                wall_clock_secs: 30.0,
                peak_rss_mb_start: 200.0,
                peak_rss_mb_end: 280.0,
                rayon_threads: 8,
                throughput: None,
                sub_stages: None,
            }]),
        });

        let json = serde_json::to_string_pretty(&profile).expect("serialize");
        let deserialized: PerformanceProfile =
            serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.total_wall_clock_secs, 99.5);
        assert_eq!(deserialized.stages.len(), 2);
        assert_eq!(deserialized.stages[0].stage_name, "Phase1");
        assert_eq!(deserialized.stages[0].throughput.as_deref(), Some("1000 items"));
        assert!(deserialized.stages[1].sub_stages.is_some());
        assert_eq!(deserialized.stages[1].sub_stages.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_performance_profile_empty_serialization() {
        let profile = PerformanceProfile::new();
        let json = serde_json::to_string(&profile).expect("serialize");
        assert!(json.contains("\"stages\":[]"));
        let deserialized: PerformanceProfile =
            serde_json::from_str(&json).expect("deserialize");
        assert!(deserialized.stages.is_empty());
    }

    #[test]
    fn test_logical_cpu_count_positive() {
        assert!(logical_cpu_count() > 0);
    }

    #[test]
    fn test_log_summary_does_not_panic() {
        let mut profile = PerformanceProfile::new();
        profile.set_total(10.0);
        profile.push(StageTiming {
            stage_name: "TestStage".to_string(),
            wall_clock_secs: 5.0,
            peak_rss_mb_start: 50.0,
            peak_rss_mb_end: 100.0,
            rayon_threads: 4,
            throughput: Some("100 items".to_string()),
            sub_stages: None,
        });
        profile.log_summary();
    }
}
