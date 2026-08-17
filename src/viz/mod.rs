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

/// A single visualization mode, executed after the main render.
pub trait VizMode {
    /// Catalog entry for this mode (id, flag, metadata).
    fn entry(&self) -> &'static ModeEntry;

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

    /// Selected catalog entries in execution order.
    #[must_use]
    pub fn entries(&self) -> &[&'static ModeEntry] {
        &self.entries
    }
}

/// Execute every selected mode, write the viz manifest, and report.
///
/// Individual mode failures are logged and collected; the stage returns an
/// error listing them only after all modes have been attempted, so one
/// failure never blocks the rest.
pub fn run_viz_stage(ctx: &VizContext<'_>, selection: &VizSelection) -> Result<()> {
    let mut all_records = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for entry in selection.entries() {
        let Some(mode) = modes::build(entry.flag) else {
            failures.push(format!("{} (no implementation registered)", entry.flag));
            continue;
        };
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
                    all_records.extend(records);
                }
                Err(error) => {
                    warn!("viz mode `{}` failed: {error}", entry.flag);
                    failures.push(format!("{} ({error})", entry.flag));
                }
            },
            Err(error) => {
                warn!("viz mode `{}` could not create its sink: {error}", entry.flag);
                failures.push(format!("{} ({error})", entry.flag));
            }
        }
    }

    sink::write_viz_manifest(ctx.seed_dir, &all_records)?;
    info!("VIZ stage complete: {} artifact(s), {} failure(s)", all_records.len(), failures.len());

    if failures.is_empty() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!("viz modes failed: {}", failures.join(", "))).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_all_selects_only_implemented() {
        let selection = VizSelection::resolve(&["all".to_string()]).expect("'all' must resolve");
        assert_eq!(selection.entries().len(), 8);
        assert!(selection.entries().iter().all(|entry| entry.implemented));
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
