//! The ember edition: the selected orbit drawn in sumi ink by the fluid it stirs.
//!
//! The three bodies of the selected orbit move across the sheet exactly as the main edition draws
//! them (its projection space, viewing rotation, drift and frame) and are dragged, as small
//! Brinkman-penalised bodies, through a doubly periodic two-dimensional Navier–Stokes fluid
//! (Re = 300). Each body is an ellipse of constant area that the tidal field of
//! the other two stretches (a disc when it is alone, up to 3 : 1 at the orbit's closest moments),
//! and its material follows the irrotational flow that carries that outline. Whenever a parcel of
//! water brushes past a body where the body's boundary layer is spinning (|ω| above a gate), it
//! picks up ink. Ink is carried and diluted by the flow; it stays black for a moment, then fades
//! to a pale grey wash on a clock set in film time, so every orbit's film breathes at the same
//! pace. The bodies themselves are solid: no ink lies inside them.
//! The result is shaded spectrally (36 bands, Kubelka–Munk with a Saunderson surface) as
//! pine-soot sumi on a mottled kozo sheet under a warm gallery light.
//!
//! # Pipeline
//!
//! 1. [`view`] and `orbit` — the main edition's view re-applied to the recorded orbit, the
//!    orbit-to-fluid time map (the median body speed is the fluid's reference speed) and the
//!    bodies' tidal shapes.
//! 2. `fluid` — pseudo-spectral vorticity solver (Lawson IF-RK4, 2/3 dealiasing, hyperviscosity,
//!    sponge) built on the deterministic `fft`.
//! 3. `trace` and `ink` — every frame, each ink node is traced back along exact characteristics
//!    to the previous frame, recording gated soak-zone contacts; older ink is carried by clamped
//!    Catmull-Rom semi-Lagrangian interpolation.
//! 4. `look`, `optics` and `paper` — tone law, spectral shading of the paper and encoding.
//! 5. [`pipeline`] — orchestration: one frame per main-video checkpoint (the last frame is the
//!    still), and the slow film's frames between them.
//! 6. [`certificate`] — the per-package determinism certificate `metadata/ember.json`, written and
//!    read back.
//!
//! The design (equations, conventions, determinism rules, defaults) is documented for
//! maintainers in `docs/ember-design.md`, the product view in `docs/ember-edition.md`.
//!
//! # Determinism
//!
//! The frame streams and the still are a pure function of the orbit, the view, the output size,
//! the frame schedule, the paper seed and [`EmberConfig`]: bit-identical on every IEEE-754 CPU architecture. The module uses only
//! exactly rounded arithmetic, the pure-Rust [`libm`](https://docs.rs/libm) crate for
//! transcendental functions (through `math`), its own FFT, fixed-order reductions, and parallelism
//! only over independent outputs. See `metadata/ember.json` for the per-package certificate.

pub mod certificate;
pub mod config;
pub mod error;
pub(crate) mod fft;
pub(crate) mod fluid;
pub(crate) mod ink;
pub(crate) mod look;
pub(crate) mod math;
pub(crate) mod optics;
pub(crate) mod orbit;
pub(crate) mod paper;
pub mod pipeline;
pub(crate) mod trace;
pub mod view;

pub use certificate::{CertificateError, EmberCertificate};
pub use config::EmberConfig;
pub use error::{EmberError, EmberResult};
#[doc(hidden)]
pub use fft::bench as fft_bench;
pub use pipeline::{
    EmberFrame, EmberMode, EmberPlan, EmberRequest, EmberStats, EmberSummary, EmberTimings,
    plan_ember, render_ember,
};
pub use view::{View, ViewDrift, ViewFrame, ViewProjection};
