//! Implemented visualization modes (Waves 1-2 of `docs/VIZ_MASTER_PLAN.md`).

pub mod alien_vision;
pub mod braid;
pub mod dwell_nebula;
pub mod gw_chirp;
pub mod oscilloscope;
pub mod plotter_svg;
pub mod prism_portrait;
pub mod slit_scan;
pub mod spectral_centroid;
pub mod spectrum_card;
pub mod syzygy_wheel;
pub mod thin_film;
pub mod topo_contours;
pub mod turntable;
pub mod winding_glass;

use crate::viz::VizMode;

/// Instantiate the implementation for a catalog flag, if one exists.
#[must_use]
pub fn build(flag: &str) -> Option<Box<dyn VizMode>> {
    match flag {
        "alien-vision" => Some(Box::new(alien_vision::AlienVision)),
        "spectral-centroid" => Some(Box::new(spectral_centroid::SpectralCentroid)),
        "prism-portrait" => Some(Box::new(prism_portrait::PrismPortrait)),
        "spectrum-card" => Some(Box::new(spectrum_card::SpectrumCard)),
        "thin-film" => Some(Box::new(thin_film::ThinFilm)),
        "braid" => Some(Box::new(braid::Braid)),
        "gw-chirp" => Some(Box::new(gw_chirp::GwChirp)),
        "syzygy-wheel" => Some(Box::new(syzygy_wheel::SyzygyWheel)),
        "slit-scan" => Some(Box::new(slit_scan::SlitScan)),
        "winding-glass" => Some(Box::new(winding_glass::WindingGlass)),
        "plotter-svg" => Some(Box::new(plotter_svg::PlotterSvg)),
        "oscilloscope" => Some(Box::new(oscilloscope::Oscilloscope)),
        "turntable" => Some(Box::new(turntable::Turntable)),
        "dwell-nebula" => Some(Box::new(dwell_nebula::DwellNebula)),
        "topo-contours" => Some(Box::new(topo_contours::TopoContours)),
        _ => None,
    }
}
