//! Static catalog of every visualization mode defined in
//! `docs/VIZ_MASTER_PLAN.md`.
//!
//! The catalog is the single source of truth for mode ids, CLI flag names,
//! categories, cost classes, and implementation status. `--viz-list` prints
//! it; flag validation resolves against it; the ledger consistency test
//! asserts it matches the master plan document.

/// One visualization mode as registered in the master plan ledger.
#[derive(Clone, Copy, Debug)]
pub struct ModeEntry {
    /// Stable ledger id (`V01`..`V69`).
    pub id: &'static str,
    /// CLI flag name accepted by `--viz` (kebab-case, frozen).
    pub flag: &'static str,
    /// Short human title.
    pub title: &'static str,
    /// Ledger category.
    pub category: &'static str,
    /// Cost class from the master plan (A cheapest .. D budgeted).
    pub cost: char,
    /// Whether the mode is implemented in this build.
    pub implemented: bool,
}

/// Compact constructor keeping the 69-entry table readable.
const fn entry(
    id: &'static str,
    flag: &'static str,
    title: &'static str,
    category: &'static str,
    cost: char,
    implemented: bool,
) -> ModeEntry {
    ModeEntry { id, flag, title, category, cost, implemented }
}

/// Every mode from the master plan ledger, in ledger order.
pub const CATALOG: &[ModeEntry] = &[
    entry("V01", "alien-vision", "The Same Light, Other Eyes", "spectral", 'A', true),
    entry("V02", "hyperspectral-flythrough", "Flying Through the Spectrum", "spectral", 'C', true),
    entry("V03", "spectral-centroid", "What Color the Light Really Is", "spectral", 'A', true),
    entry("V04", "prism-portrait", "The Artwork Through Glass", "spectral", 'A', true),
    entry("V05", "spectrum-card", "Stellar Classification Card", "spectral", 'A', true),
    entry("V06", "thin-film", "Oil-Slick Twin", "spectral", 'A', true),
    entry("V07", "braid", "The Orbit Is a Braid", "physics", 'A', true),
    entry("V08", "shape-sphere", "The Planet of Shapes", "physics", 'B', true),
    entry("V09", "gw-chirp", "The Sound of Spacetime", "physics", 'A', true),
    entry("V10", "sonification", "The Orbit's Score", "physics", 'B', false),
    entry("V11", "recurrence", "Fingerprint of Chaos", "physics", 'B', false),
    entry("V12", "field-lines", "The Gravitational Engraving", "physics", 'B', true),
    entry("V13", "syzygy-wheel", "The Rhythm Clock", "physics", 'A', true),
    entry("V14", "triangle-centers", "The Constellation of Centers", "physics", 'B', true),
    entry("V15", "medial-recursion", "Vortex of Triangles", "physics", 'B', true),
    entry("V16", "chord-progression", "The Harmony of Distances", "physics", 'B', false),
    entry("V17", "epicycles", "The Impossible Machine", "physics", 'B', false),
    entry("V18", "chrono-grid", "Motion Study Sheet", "time", 'B', true),
    entry("V19", "slit-scan", "The Whole Film in One Image", "time", 'A', true),
    entry("V20", "strobe", "Phantom Triangles", "time", 'B', true),
    entry("V21", "comet", "Forever Redrawing", "time", 'C', true),
    entry("V22", "editorial-retime", "Drama-Adaptive Time", "time", 'C', true),
    entry("V23", "epilogue", "How This Artwork Dies", "time", 'C', true),
    entry("V24", "multiverse", "The Garden of Forking Orbits", "time", 'C', true),
    entry("V25", "three-shadows", "The Cave Wall Triptych", "frames", 'B', true),
    entry("V26", "corotating", "The Same Dance from the Dance Floor", "frames", 'C', true),
    entry("V27", "ride-along", "What Body Three Sees", "frames", 'C', true),
    entry("V28", "bullet-time", "The Held Breath", "frames", 'C', true),
    entry("V29", "retarded-time", "Where Their Light Says They Are", "frames", 'B', true),
    entry("V30", "lensing", "Gravity Bends the Gallery", "frames", 'B', true),
    entry("V31", "dust-nebula", "Gravity's Weather", "matter", 'C', true),
    entry("V32", "light-echoes", "Three Boats on a Dark Pond", "matter", 'C', true),
    entry("V33", "physarum", "The Organism Rediscovers the Orbit", "matter", 'C', true),
    entry("V34", "frost", "Winter Claims the Window", "matter", 'C', true),
    entry("V35", "lightning", "The Storm Record", "matter", 'C', true),
    entry("V36", "marbling", "Suminagashi Stirred by Gravity", "matter", 'C', true),
    entry("V37", "roche", "Lobes That Touch", "matter", 'C', true),
    entry("V38", "galaxy-collision", "The Antennae, Choreographed", "matter", 'C', true),
    entry("V39", "reconnection", "Field Lines That Snap", "matter", 'C', true),
    entry("V40", "aurora", "Curtains Over the Void", "matter", 'C', true),
    entry("V41", "winding-glass", "Topological Stained Glass", "topology", 'B', true),
    entry("V42", "basin-map", "Where Your Artwork Lives in Chaos", "topology", 'D', true),
    entry("V43", "worldtube", "The Spacetime Sculpture", "topology", 'C', true),
    entry("V44", "neon", "Signage from the End of the Universe", "scene3d", 'C', true),
    entry("V45", "chandelier", "The Room Lit by the Orbit", "scene3d", 'D', true),
    entry("V46", "turntable", "Museum Turntable", "scene3d", 'A', true),
    entry("V47", "trailer", "Sixty Seconds, Auto-Edited", "cinema", 'C', false),
    entry("V48", "mission-control", "The 1969 Broadcast", "cinema", 'C', false),
    entry("V49", "broadcast", "The Five-Act Short Film", "cinema", 'D', false),
    entry("V50", "sculpture-export", "The Printable Object", "exports", 'B', true),
    entry("V51", "plotter-svg", "Ink and Thread", "exports", 'A', true),
    entry("V52", "depth-pack", "The Third Dimension, Packaged", "exports", 'B', true),
    entry("V53", "webgl-viewer", "Hold Your Orbit", "exports", 'B', false),
    entry("V54", "oscilloscope", "Sound That Draws", "exports", 'B', true),
    entry("V55", "hologram", "A Recording of the Wavefront", "exports", 'D', false),
    entry("V56", "tilt", "The Poster That Plays", "exports", 'B', false),
    entry("V57", "instrument", "Play Your Orbit", "exports", 'C', false),
    entry("V58", "ephemeris-poster", "The Almanac Page", "posters", 'B', false),
    entry("V59", "blueprint", "Two Archival Restylings", "posters", 'B', false),
    entry("V60", "dwell-nebula", "The Ergodic Ghost", "posters", 'B', true),
    entry("V61", "topo-contours", "The Terrain of Light", "posters", 'B', true),
    entry("V62", "terra", "Terra Trium Corporum", "posters", 'C', true),
    entry("V63", "celestial-atlas", "The Collection as a Sky", "posters", 'C', false),
    entry("V64", "powers-of-fate", "The Dive", "combos", 'D', false),
    entry("V65", "witness", "First Person, Honest Optics", "combos", 'D', false),
    entry("V66", "rose-window", "Gravity Builds a Cathedral", "combos", 'D', false),
    entry("V67", "vanitas", "The Life and Death of an Artwork", "combos", 'D', false),
    entry("V68", "reliquary", "The Monument to Almost", "combos", 'D', false),
    entry("V69", "pond", "The Surface of a Dark Pond", "combos", 'D', false),
];

/// Look up a catalog entry by its CLI flag name.
#[must_use]
pub fn find(flag: &str) -> Option<&'static ModeEntry> {
    CATALOG.iter().find(|candidate| candidate.flag == flag)
}

/// All catalog categories in ledger order, deduplicated.
#[must_use]
pub fn categories() -> Vec<&'static str> {
    let mut seen = Vec::new();
    for candidate in CATALOG {
        if !seen.contains(&candidate.category) {
            seen.push(candidate.category);
        }
    }
    seen
}

/// Render the catalog as a human-readable table for `--viz-list`.
#[must_use]
pub fn render_list() -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    out.push_str("ID   FLAG                      CATEGORY  COST  STATUS       TITLE\n");
    for candidate in CATALOG {
        let status = if candidate.implemented { "implemented" } else { "planned" };
        let _ = writeln!(
            out,
            "{:<4} {:<25} {:<9} {:<5} {:<12} {}",
            candidate.id,
            candidate.flag,
            candidate.category,
            candidate.cost,
            status,
            candidate.title
        );
    }
    let implemented = CATALOG.iter().filter(|candidate| candidate.implemented).count();
    let _ = writeln!(
        out,
        "\n{} modes total, {} implemented. Select with --viz <flags|category|all>.",
        CATALOG.len(),
        implemented
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_69_unique_entries() {
        assert_eq!(CATALOG.len(), 69);
        let mut flags: Vec<&str> = CATALOG.iter().map(|candidate| candidate.flag).collect();
        flags.sort_unstable();
        flags.dedup();
        assert_eq!(flags.len(), 69, "duplicate flag names in catalog");
        let mut ids: Vec<&str> = CATALOG.iter().map(|candidate| candidate.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 69, "duplicate ids in catalog");
    }

    #[test]
    fn catalog_ids_are_sequential() {
        for (index, candidate) in CATALOG.iter().enumerate() {
            let expected = format!("V{:02}", index + 1);
            assert_eq!(candidate.id, expected, "catalog order must match ledger order");
        }
    }

    #[test]
    fn ledger_document_matches_catalog() {
        let doc = include_str!("../../docs/VIZ_MASTER_PLAN.md");
        for candidate in CATALOG {
            let ledger_row = format!("| {} | `{}` |", candidate.id, candidate.flag);
            assert!(
                doc.contains(&ledger_row),
                "master plan ledger is missing or disagrees on {} `{}`",
                candidate.id,
                candidate.flag
            );
        }
    }

    #[test]
    fn implemented_waves_are_marked() {
        for flag in [
            // Wave 1
            "braid",
            "gw-chirp",
            "syzygy-wheel",
            "slit-scan",
            "winding-glass",
            "plotter-svg",
            "oscilloscope",
            "turntable",
            // Wave 2 (SPD family)
            "alien-vision",
            "spectral-centroid",
            "prism-portrait",
            "spectrum-card",
            "thin-film",
            "dwell-nebula",
            "topo-contours",
            // Wave 3 (re-accumulation family)
            "triangle-centers",
            "medial-recursion",
            "chrono-grid",
            "strobe",
            "comet",
            "three-shadows",
            "ride-along",
            "retarded-time",
            "depth-pack",
            // Wave 4 (fields & frames)
            "field-lines",
            "corotating",
            "lensing",
            "roche",
            "reconnection",
            // Wave 5 (particles, agents, media)
            "dust-nebula",
            "galaxy-collision",
            "physarum",
            "frost",
            "lightning",
            "marbling",
            "light-echoes",
            // Wave 6 (3D scene family)
            "worldtube",
            "hyperspectral-flythrough",
            "shape-sphere",
            "aurora",
            "neon",
            "chandelier",
            "sculpture-export",
            "bullet-time",
            // Wave 7 (ensembles & cartography)
            "epilogue",
            "multiverse",
            "editorial-retime",
            "basin-map",
            "terra",
        ] {
            let found = find(flag).expect("implemented flag must exist");
            assert!(found.implemented, "{flag} must be marked implemented");
        }
        assert_eq!(CATALOG.iter().filter(|candidate| candidate.implemented).count(), 49);
    }
}
