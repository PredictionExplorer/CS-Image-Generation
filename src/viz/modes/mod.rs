//! Implemented visualization modes (Waves 1-4 of `docs/VIZ_MASTER_PLAN.md`).

pub mod alien_vision;
pub mod braid;
pub mod chrono_grid;
pub mod comet;
pub mod corotating;
pub mod depth_pack;
pub mod dwell_nebula;
pub mod field_lines;
pub mod gw_chirp;
pub mod lensing;
pub mod medial_recursion;
pub mod oscilloscope;
pub mod plotter_svg;
pub mod prism_portrait;
pub mod reconnection;
pub mod retarded_time;
pub mod ride_along;
pub mod roche;
pub mod slit_scan;
pub mod spectral_centroid;
pub mod spectrum_card;
pub mod strobe;
pub mod syzygy_wheel;
pub mod thin_film;
pub mod three_shadows;
pub mod topo_contours;
pub mod triangle_centers;
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
        "triangle-centers" => Some(Box::new(triangle_centers::TriangleCenters)),
        "medial-recursion" => Some(Box::new(medial_recursion::MedialRecursion)),
        "chrono-grid" => Some(Box::new(chrono_grid::ChronoGrid)),
        "slit-scan" => Some(Box::new(slit_scan::SlitScan)),
        "strobe" => Some(Box::new(strobe::Strobe)),
        "comet" => Some(Box::new(comet::Comet)),
        "three-shadows" => Some(Box::new(three_shadows::ThreeShadows)),
        "ride-along" => Some(Box::new(ride_along::RideAlong)),
        "retarded-time" => Some(Box::new(retarded_time::RetardedTime)),
        "winding-glass" => Some(Box::new(winding_glass::WindingGlass)),
        "plotter-svg" => Some(Box::new(plotter_svg::PlotterSvg)),
        "oscilloscope" => Some(Box::new(oscilloscope::Oscilloscope)),
        "turntable" => Some(Box::new(turntable::Turntable)),
        "dwell-nebula" => Some(Box::new(dwell_nebula::DwellNebula)),
        "topo-contours" => Some(Box::new(topo_contours::TopoContours)),
        "depth-pack" => Some(Box::new(depth_pack::DepthPack)),
        "field-lines" => Some(Box::new(field_lines::FieldLines)),
        "corotating" => Some(Box::new(corotating::Corotating)),
        "lensing" => Some(Box::new(lensing::Lensing)),
        "roche" => Some(Box::new(roche::Roche)),
        "reconnection" => Some(Box::new(reconnection::Reconnection)),
        _ => None,
    }
}
