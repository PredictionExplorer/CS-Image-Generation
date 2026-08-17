//! V46 `turntable` -- Museum Turntable (adapter).
//!
//! Thin adapter absorbing the existing orbit renderer into the viz
//! framework: `--viz turntable` renders the 360-degree sweep with
//! viz-standard output paths and quality mapping. The standalone
//! `--orbit-video` flag remains untouched.

use crate::app;
use crate::error::Result;
use crate::render::SpectralScene;
use crate::render::orbit::OrbitVideoConfig;
use crate::viz::VizMode;
use crate::viz::catalog::{self, ModeEntry};
use crate::viz::context::{VizContext, VizQuality};
use crate::viz::sink::ArtifactSink;

/// The turntable adapter mode.
pub struct Turntable;

/// Round a dimension down to an even value (yuv420p requirement).
fn even_dim(value: u32) -> u32 {
    (value & !1).max(16)
}

impl VizMode for Turntable {
    fn entry(&self) -> &'static ModeEntry {
        catalog::find("turntable").expect("turntable is in the catalog")
    }

    fn run(&self, ctx: &VizContext<'_>, sink: &mut ArtifactSink) -> Result<()> {
        let (width, height, seconds, fps, stride) = match ctx.quality {
            VizQuality::Final => (even_dim(ctx.width), even_dim(ctx.height), 24.0, 30, 1),
            VizQuality::Draft => (even_dim(ctx.width / 2), even_dim(ctx.height / 2), 12.0, 30, 4),
        };
        let config =
            OrbitVideoConfig { width, height, fps, seconds, tilt_deg: 18.0, step_stride: stride };
        let web = sink.path("turntable.mp4");
        let hq = sink.path("turntable_hq.mp4");
        app::render_orbit_video(
            SpectralScene::new(ctx.positions, ctx.colors, ctx.body_alphas),
            ctx.levels,
            ctx.settings,
            &config,
            app::VideoOutputPaths { web: &web, high_quality: &hq },
            ctx.fast_encode,
        )?;
        sink.record("turntable.mp4", "video");
        sink.record("turntable_hq.mp4", "video");
        Ok(())
    }
}
