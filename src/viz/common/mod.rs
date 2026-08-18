//! Shared infrastructure for visualization modes: derived kinematics,
//! event detection, the spectral drawing canvas, audio helpers, and vector
//! exporters. See `docs/VIZ_MASTER_PLAN.md` Part II.

pub mod accum;
pub mod agents;
pub mod audio;
pub mod compositor;
pub mod contours;
pub mod display;
pub mod events;
pub mod fields;
pub mod fluid;
pub mod kinematics;
pub mod particles;
pub mod raster;
pub mod resim;
pub mod spd;
pub mod style;
pub mod text;
pub mod tube_render;
pub mod vector_export;
pub mod wave;
