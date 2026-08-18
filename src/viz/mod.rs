//! Visualization subsystem: 69 planned modes that transform a finished run
//! (trajectory, palette, SPD) into additional artworks, each behind a CLI
//! flag. Architecture and per-mode specs live in `docs/VIZ_MASTER_PLAN.md`.
//!
//! Wave 0 (this module tree) provides the framework: catalog, selection,
//! context with lazy derived data, artifact sink, and the stage runner.
//! Wave 1 ships the first eight modes.

pub mod catalog;
pub mod common;
pub mod context;
pub mod modes;
pub mod sink;

use crate::error::Result;
use catalog::ModeEntry;
use context::VizContext;
use sink::ArtifactSink;
use tracing::{info, warn};

/// Execution phase of a mode, per the planner in the master plan (I.6).
///
/// SPD-phase modes run inside the render block while the accumulated
/// spectral buffer is still alive; trajectory-phase modes run at the end of
/// the pipeline after the core package is complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VizPhase {
    /// Requires the accumulated per-pixel SPD buffer.
    Spd,
    /// Requires only trajectory data (and optionally the frame tap).
    Trajectory,
}

/// A single visualization mode, executed after the main render.
pub trait VizMode {
    /// Catalog entry for this mode (id, flag, metadata).
    fn entry(&self) -> &'static ModeEntry;

    /// Which stage phase this mode must run in.
    fn phase(&self) -> VizPhase {
        VizPhase::Trajectory
    }

    /// Whether this mode consumes frames tapped from the main video render.
    fn needs_frame_tap(&self) -> bool {
        false
    }

    /// Render all artifacts into the sink. Must be seed-deterministic.
    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()>;
}

/// Validated selection of modes to execute.
pub struct VizSelection {
    entries: Vec<&'static ModeEntry>,
}

impl VizSelection {
    /// Resolve raw `--viz` values (flags, categories, or `all`; comma lists
    /// allowed) against the catalog.
    ///
    /// Unknown names and explicitly requested unimplemented flags are
    /// errors; `all` and category names silently select only implemented
    /// modes (planned ones are logged).
    pub fn resolve(specs: &[String]) -> Result<Self> {
        let mut selected: Vec<&'static ModeEntry> = Vec::new();
        let mut push = |entry: &'static ModeEntry| {
            if !selected.iter().any(|existing| existing.flag == entry.flag) {
                selected.push(entry);
            }
        };

        for spec in specs.iter().flat_map(|value| value.split(',')) {
            let token = spec.trim();
            if token.is_empty() {
                continue;
            }
            if token.eq_ignore_ascii_case("all") {
                for entry in catalog::CATALOG.iter().filter(|entry| entry.implemented) {
                    push(entry);
                }
                continue;
            }
            if catalog::categories().contains(&token) {
                let mut any = false;
                for entry in catalog::CATALOG
                    .iter()
                    .filter(|entry| entry.category == token && entry.implemented)
                {
                    push(entry);
                    any = true;
                }
                if !any {
                    warn!("viz category '{token}' has no implemented modes yet; skipping");
                }
                continue;
            }
            match catalog::find(token) {
                Some(entry) if entry.implemented => push(entry),
                Some(entry) => {
                    return Err(std::io::Error::other(format!(
                        "viz mode '{}' ({}) is planned but not implemented yet; \
                         see docs/VIZ_MASTER_PLAN.md",
                        entry.flag, entry.id
                    ))
                    .into());
                }
                None => {
                    return Err(std::io::Error::other(format!(
                        "unknown viz mode '{token}'; run with --viz-list to see the catalog"
                    ))
                    .into());
                }
            }
        }

        // Deterministic execution order: catalog (ledger) order.
        selected.sort_by_key(|entry| entry.id);
        Ok(Self { entries: selected })
    }

    /// Whether no modes were selected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether any selected mode consumes the main-render frame tap.
    #[must_use]
    pub fn needs_frame_tap(&self) -> bool {
        self.entries
            .iter()
            .filter_map(|entry| modes::build(entry.flag))
            .any(|mode| mode.needs_frame_tap())
    }

    /// Whether any selected mode runs in the given phase.
    #[must_use]
    pub fn has_phase(&self, phase: VizPhase) -> bool {
        self.entries
            .iter()
            .filter_map(|entry| modes::build(entry.flag))
            .any(|mode| mode.phase() == phase)
    }

    /// Flags of selected modes in the given phase (for skip warnings).
    #[must_use]
    pub fn phase_flags(&self, phase: VizPhase) -> Vec<&'static str> {
        self.entries
            .iter()
            .filter(|entry| modes::build(entry.flag).is_some_and(|mode| mode.phase() == phase))
            .map(|entry| entry.flag)
            .collect()
    }

    /// Selected catalog entries in execution order.
    #[must_use]
    pub fn entries(&self) -> &[&'static ModeEntry] {
        &self.entries
    }
}

/// Accumulates artifact records and failures across the stage's phases.
///
/// Individual mode failures are logged and collected; [`Self::finish`]
/// returns an error listing them only after all phases have run, so one
/// failure never blocks other modes and never endangers the core package
/// (which is fully written before the trajectory phase runs).
#[derive(Default)]
pub struct VizStageState {
    records: Vec<sink::ArtifactRecord>,
    failures: Vec<String>,
}

impl VizStageState {
    /// Create an empty stage state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Execute every selected mode belonging to `phase`.
    pub fn run_phase(&mut self, ctx: &VizContext<'_>, selection: &VizSelection, phase: VizPhase) {
        for entry in selection.entries() {
            let Some(mode) = modes::build(entry.flag) else {
                self.failures.push(format!("{} (no implementation registered)", entry.flag));
                continue;
            };
            if mode.phase() != phase {
                continue;
            }
            info!("VIZ {} `{}`: {}...", entry.id, entry.flag, entry.title);
            let started = std::time::Instant::now();
            match ArtifactSink::new(ctx.seed_dir, entry.flag) {
                Ok(mut mode_sink) => match mode.run(ctx, &mut mode_sink) {
                    Ok(()) => {
                        let records = mode_sink.into_records();
                        info!(
                            "   => viz `{}` done: {} artifact(s) in {:.1}s",
                            entry.flag,
                            records.len(),
                            started.elapsed().as_secs_f64()
                        );
                        self.records.extend(records);
                    }
                    Err(error) => {
                        warn!("viz mode `{}` failed: {error}", entry.flag);
                        self.failures.push(format!("{} ({error})", entry.flag));
                    }
                },
                Err(error) => {
                    warn!("viz mode `{}` could not create its sink: {error}", entry.flag);
                    self.failures.push(format!("{} ({error})", entry.flag));
                }
            }
        }
    }

    /// Write the aggregated viz manifest and report the stage outcome.
    pub fn finish(self, seed_dir: &str) -> Result<()> {
        sink::write_viz_manifest(seed_dir, &self.records)?;
        info!(
            "VIZ stage complete: {} artifact(s), {} failure(s)",
            self.records.len(),
            self.failures.len()
        );
        if self.failures.is_empty() {
            Ok(())
        } else {
            Err(std::io::Error::other(format!("viz modes failed: {}", self.failures.join(", ")))
                .into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_all_selects_only_implemented() {
        let selection = VizSelection::resolve(&["all".to_string()]).expect("'all' must resolve");
        assert_eq!(selection.entries().len(), 15);
        assert!(selection.entries().iter().all(|entry| entry.implemented));
    }

    #[test]
    fn spd_phase_detection() {
        let selection = VizSelection::resolve(&["thin-film,braid".to_string()]).expect("resolve");
        assert!(selection.has_phase(VizPhase::Spd));
        assert!(selection.has_phase(VizPhase::Trajectory));
        assert_eq!(selection.phase_flags(VizPhase::Spd), vec!["thin-film"]);
        let trajectory_only = VizSelection::resolve(&["braid".to_string()]).expect("resolve");
        assert!(!trajectory_only.has_phase(VizPhase::Spd));
    }

    #[test]
    fn resolve_rejects_unknown_and_unimplemented() {
        assert!(VizSelection::resolve(&["no-such-mode".to_string()]).is_err());
        assert!(VizSelection::resolve(&["basin-map".to_string()]).is_err());
    }

    #[test]
    fn resolve_comma_lists_and_dedupes() {
        let selection =
            VizSelection::resolve(&["braid,syzygy-wheel".to_string(), "braid".to_string()])
                .expect("valid flags must resolve");
        assert_eq!(selection.entries().len(), 2);
    }

    #[test]
    fn frame_tap_detection() {
        let with_tap = VizSelection::resolve(&["slit-scan".to_string()]).expect("resolve");
        assert!(with_tap.needs_frame_tap());
        let without = VizSelection::resolve(&["braid".to_string()]).expect("resolve");
        assert!(!without.needs_frame_tap());
    }

    #[test]
    fn every_implemented_catalog_entry_has_a_registered_mode() {
        for entry in catalog::CATALOG.iter().filter(|entry| entry.implemented) {
            let mode = modes::build(entry.flag);
            assert!(mode.is_some(), "{} has no implementation", entry.flag);
            assert_eq!(
                mode.expect("just checked").entry().flag,
                entry.flag,
                "mode reports a different catalog entry"
            );
        }
    }
}
