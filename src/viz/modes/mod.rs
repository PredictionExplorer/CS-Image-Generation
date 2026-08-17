//! Implemented visualization modes (Wave 1 of `docs/VIZ_MASTER_PLAN.md`).

pub mod braid;
pub mod gw_chirp;
pub mod oscilloscope;
pub mod plotter_svg;
pub mod slit_scan;
pub mod syzygy_wheel;
pub mod turntable;
pub mod winding_glass;

use crate::viz::VizMode;

/// Instantiate the implementation for a catalog flag, if one exists.
#[must_use]
pub fn build(flag: &str) -> Option<Box<dyn VizMode>> {
    match flag {
        "braid" => Some(Box::new(braid::Braid)),
        "gw-chirp" => Some(Box::new(gw_chirp::GwChirp)),
        "syzygy-wheel" => Some(Box::new(syzygy_wheel::SyzygyWheel)),
        "slit-scan" => Some(Box::new(slit_scan::SlitScan)),
        "winding-glass" => Some(Box::new(winding_glass::WindingGlass)),
        "plotter-svg" => Some(Box::new(plotter_svg::PlotterSvg)),
        "oscilloscope" => Some(Box::new(oscilloscope::Oscilloscope)),
        "turntable" => Some(Box::new(turntable::Turntable)),
        _ => None,
    }
}
