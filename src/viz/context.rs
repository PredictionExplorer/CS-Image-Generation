//! Shared context handed to every visualization mode: borrowed run data plus
//! lazily computed derived series (kinematics, events) and the frame-tap
//! data collected during the main video render.

use crate::render::{ChannelLevels, OklabColor, SpectralRenderSettings};
use crate::sim::{Body, Sha3RandomByteStream};
use crate::spectrum::NUM_BINS;
use crate::viz::common::events::Events;
use crate::viz::common::kinematics::Kinematics;
use nalgebra::Vector3;
use std::sync::OnceLock;

/// Output quality for viz artifacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VizQuality {
    /// Development aid: halved canvases, quartered frame counts.
    Draft,
    /// Full masterpiece-bar output (default).
    Final,
}

impl VizQuality {
    /// Scale a pixel dimension for this quality (even-rounded).
    #[must_use]
    pub fn scale_dim(self, value: u32) -> u32 {
        match self {
            Self::Draft => ((value / 2) & !1).max(16),
            Self::Final => value,
        }
    }

    /// Scale a frame or sample count for this quality.
    #[must_use]
    pub fn scale_count(self, value: usize) -> usize {
        match self {
            Self::Draft => (value / 4).max(1),
            Self::Final => value,
        }
    }
}

/// Frames observed from the main video render for tap-consuming modes.
pub struct FrameTapData {
    /// Number of frames observed.
    pub frames: usize,
    /// Source frame width in pixels.
    pub width: u32,
    /// Source frame height in pixels.
    pub height: u32,
    /// Sampled column x (trajectory density centroid).
    pub column_x: u32,
    /// Per-frame column samples, `frames * height * 3` u16 values.
    pub linear: Vec<u16>,
    /// Number of samples along the ring.
    pub ring_samples: usize,
    /// Per-frame ring samples, `frames * ring_samples * 3` u16 values.
    pub ring: Vec<u16>,
}

/// Collector wired into the main render's frame sink.
pub struct FrameTapCollector {
    data: FrameTapData,
    center: (f32, f32),
    radius: f32,
}

/// Read one u16 sample from a native-endian `rgb48` byte stream.
#[inline]
fn frame_u16(bytes: &[u8], index: usize) -> u16 {
    u16::from_ne_bytes([bytes[2 * index], bytes[2 * index + 1]])
}

impl FrameTapCollector {
    /// Number of samples taken along the slit ring.
    pub const RING_SAMPLES: usize = 2048;

    /// Create a collector for frames of the given size.
    ///
    /// `centroid` is the trajectory density centroid in pixel coordinates;
    /// the linear slit is the column through it, the ring is centred on it.
    #[must_use]
    pub fn new(width: u32, height: u32, centroid: (f32, f32)) -> Self {
        let column_x = (centroid.0.round() as u32).min(width.saturating_sub(1));
        let radius = 0.38 * f64::from(width.min(height)) as f32;
        Self {
            data: FrameTapData {
                frames: 0,
                width,
                height,
                column_x,
                linear: Vec::new(),
                ring_samples: Self::RING_SAMPLES,
                ring: Vec::new(),
            },
            center: centroid,
            radius,
        }
    }

    /// Observe one `rgb48` (native-endian) frame from the encoder stream.
    pub fn observe(&mut self, frame_bytes: &[u8]) {
        let width = self.data.width as usize;
        let height = self.data.height as usize;
        debug_assert_eq!(frame_bytes.len(), width * height * 6);

        // Linear slit: copy the centroid column.
        let column = self.data.column_x as usize;
        for row in 0..height {
            let pixel = row * width + column;
            for channel in 0..3 {
                self.data.linear.push(frame_u16(frame_bytes, pixel * 3 + channel));
            }
        }

        // Radial slit: bilinear ring samples.
        let sample_bilinear = |x: f32, y: f32, channel: usize| -> u16 {
            let xc = x.clamp(0.0, (width - 1) as f32);
            let yc = y.clamp(0.0, (height - 1) as f32);
            let x0 = xc.floor() as usize;
            let y0 = yc.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            let fx = f64::from(xc) - x0 as f64;
            let fy = f64::from(yc) - y0 as f64;
            let at = |px: usize, py: usize| -> f64 {
                f64::from(frame_u16(frame_bytes, (py * width + px) * 3 + channel))
            };
            let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
            let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
            (top * (1.0 - fy) + bottom * fy).round() as u16
        };
        for sample in 0..self.data.ring_samples {
            let theta = sample as f64 / self.data.ring_samples as f64 * std::f64::consts::TAU;
            let x = self.center.0 + self.radius * theta.cos() as f32;
            let y = self.center.1 + self.radius * theta.sin() as f32;
            for channel in 0..3 {
                self.data.ring.push(sample_bilinear(x, y, channel));
            }
        }

        self.data.frames += 1;
    }

    /// Finish collection, returning the tap data for the viz stage.
    #[must_use]
    pub fn finish(self) -> FrameTapData {
        self.data
    }
}

/// Reduced-resolution frames captured from the main render at selected
/// frame indices (V47 macro push-ins, V56 lenticular time flips).
pub struct FrameArchiveData {
    /// Archive frame width (quarter of the render width).
    pub width: u32,
    /// Archive frame height.
    pub height: u32,
    /// Total main-video frames observed.
    pub source_frames: usize,
    /// Captured `(frame index, rgb8)` pairs in frame order.
    pub frames: Vec<(usize, Vec<u8>)>,
}

impl FrameArchiveData {
    /// The captured frame nearest to `index`, if any were captured.
    #[must_use]
    pub fn nearest(&self, index: usize) -> Option<&(usize, Vec<u8>)> {
        self.frames.iter().min_by_key(|(frame, _)| frame.abs_diff(index))
    }
}

/// Collector capturing quarter-resolution copies of selected frames.
pub struct FrameArchiveCollector {
    targets: Vec<usize>,
    data: FrameArchiveData,
    full_width: usize,
    full_height: usize,
    counter: usize,
}

impl FrameArchiveCollector {
    /// Create a collector for the given sorted, deduplicated target frames.
    #[must_use]
    pub fn new(full_width: u32, full_height: u32, mut targets: Vec<usize>) -> Self {
        targets.sort_unstable();
        targets.dedup();
        let width = (full_width / 4).max(16);
        let height = (full_height / 4).max(16);
        Self {
            targets,
            data: FrameArchiveData { width, height, source_frames: 0, frames: Vec::new() },
            full_width: full_width as usize,
            full_height: full_height as usize,
            counter: 0,
        }
    }

    /// Observe one `rgb48` (native-endian) frame from the encoder stream.
    pub fn observe(&mut self, frame_bytes: &[u8]) {
        let index = self.counter;
        self.counter += 1;
        self.data.source_frames = self.counter;
        if self.targets.binary_search(&index).is_err() {
            return;
        }
        // 4x4 box downsample from rgb48 to rgb8.
        let out_w = self.data.width as usize;
        let out_h = self.data.height as usize;
        let mut rgb8 = vec![0u8; out_w * out_h * 3];
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut sums = [0u32; 3];
                let mut count = 0u32;
                for sy in 0..4 {
                    let y = oy * 4 + sy;
                    if y >= self.full_height {
                        continue;
                    }
                    for sx in 0..4 {
                        let x = ox * 4 + sx;
                        if x >= self.full_width {
                            continue;
                        }
                        let pixel = (y * self.full_width + x) * 3;
                        for (channel, sum) in sums.iter_mut().enumerate() {
                            *sum += u32::from(frame_u16(frame_bytes, pixel + channel)) >> 8;
                        }
                        count += 1;
                    }
                }
                let base = (oy * out_w + ox) * 3;
                for channel in 0..3 {
                    rgb8[base + channel] = (sums[channel] / count.max(1)) as u8;
                }
            }
        }
        self.data.frames.push((index, rgb8));
    }

    /// Finish collection, returning the archive for the viz stage.
    #[must_use]
    pub fn finish(self) -> FrameArchiveData {
        self.data
    }
}

/// Trajectory density centroid in pixel space (strided, deterministic).
#[must_use]
pub fn trajectory_centroid_px(
    positions: &[Vec<Vector3<f64>>],
    ctx: &crate::render::context::RenderContext,
) -> (f32, f32) {
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut count = 0.0_f64;
    for body in positions {
        for point in body.iter().step_by(211) {
            let (px, py) = ctx.to_pixel(point.x, point.y);
            sum_x += f64::from(px);
            sum_y += f64::from(py);
            count += 1.0;
        }
    }
    if count == 0.0 {
        return (0.0, 0.0);
    }
    ((sum_x / count) as f32, (sum_y / count) as f32)
}

/// Borrowed run data plus lazy derived state, handed to every mode.
pub struct VizContext<'a> {
    /// Projected per-body trajectories (`[body][step]`).
    pub positions: &'a [Vec<Vector3<f64>>],
    /// Per-body `OkLab` color sequences.
    pub colors: &'a [Vec<OklabColor>],
    /// Per-body base opacities.
    pub body_alphas: &'a [f64],
    /// Initial body states of the winning candidate (masses).
    pub bodies: &'a [Body],
    /// Frozen tonemap levels from pass 1.
    pub levels: &'a ChannelLevels,
    /// Spectral render settings of the main render.
    pub settings: SpectralRenderSettings<'a>,
    /// Main output width in pixels.
    pub width: u32,
    /// Main output height in pixels.
    pub height: u32,
    /// Seed as lowercase hex (no `0x` prefix).
    pub seed_hex: &'a str,
    /// Seed package directory (`output/<name>`).
    pub seed_dir: &'a str,
    /// Requested artifact quality.
    pub quality: VizQuality,
    /// Whether fast video encoding was requested.
    pub fast_encode: bool,
    /// Directory of prior seed packages for multi-seed modes (V63); the
    /// current seed is charted alone when absent.
    pub seeds_dir: Option<&'a str>,
    /// Frame tap data if a tap-consuming mode was requested (video runs only).
    pub frame_tap: Option<FrameTapData>,
    /// Reduced-resolution frame archive (event windows + drama samples),
    /// present only when an archive-consuming mode was requested.
    pub frame_archive: Option<FrameArchiveData>,
    /// Accumulated per-pixel SPD buffer (present only during the SPD phase).
    pub accum_spd: Option<&'a [[f64; NUM_BINS]]>,
    /// Energy field retained from the main render for the trajectory phase
    /// (compact substitute for the dropped SPD buffer).
    retained_energy: Option<Vec<f32>>,
    base_rng: &'a Sha3RandomByteStream,
    kinematics: OnceLock<Kinematics>,
    events: OnceLock<Events>,
    energy_field: OnceLock<Vec<f32>>,
}

impl<'a> VizContext<'a> {
    /// Bundle borrowed run data into a context.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        positions: &'a [Vec<Vector3<f64>>],
        colors: &'a [Vec<OklabColor>],
        body_alphas: &'a [f64],
        bodies: &'a [Body],
        levels: &'a ChannelLevels,
        settings: SpectralRenderSettings<'a>,
        width: u32,
        height: u32,
        seed_hex: &'a str,
        seed_dir: &'a str,
        quality: VizQuality,
        fast_encode: bool,
        seeds_dir: Option<&'a str>,
        frame_tap: Option<FrameTapData>,
        frame_archive: Option<FrameArchiveData>,
        accum_spd: Option<&'a [[f64; NUM_BINS]]>,
        retained_energy: Option<Vec<f32>>,
        base_rng: &'a Sha3RandomByteStream,
    ) -> Self {
        Self {
            positions,
            colors,
            body_alphas,
            bodies,
            levels,
            settings,
            width,
            height,
            seed_hex,
            seed_dir,
            quality,
            fast_encode,
            seeds_dir,
            frame_tap,
            frame_archive,
            accum_spd,
            retained_energy,
            base_rng,
            kinematics: OnceLock::new(),
            events: OnceLock::new(),
            energy_field: OnceLock::new(),
        }
    }

    /// Derived kinematic series (computed once on first use).
    pub fn kinematics(&self) -> &Kinematics {
        self.kinematics.get_or_init(|| Kinematics::compute(self.positions, self.bodies))
    }

    /// Detected orbit events (computed once on first use).
    pub fn events(&self) -> &Events {
        self.events.get_or_init(|| Events::detect(self.positions, self.kinematics()))
    }

    /// Per-pixel total SPD energy: the retained main-render field if present
    /// (trajectory phase), else computed from the live SPD (SPD phase);
    /// `None` when neither source exists (`--image-only`).
    pub fn energy_field(&self) -> Option<&[f32]> {
        if let Some(retained) = &self.retained_energy {
            return Some(retained.as_slice());
        }
        let spd = self.accum_spd?;
        Some(
            self.energy_field.get_or_init(|| crate::viz::common::spd::energy_field(spd)).as_slice(),
        )
    }

    /// Fork a deterministic RNG for a mode (`domain = viz/<flag>/v1`).
    #[must_use]
    pub fn fork_rng(&self, mode_flag: &str) -> Sha3RandomByteStream {
        let domain = format!("viz/{mode_flag}/v1");
        self.base_rng.fork(domain.as_bytes())
    }

    /// Mean `OkLab` color of one body's sequence.
    #[must_use]
    pub fn mean_color(&self, body: usize) -> OklabColor {
        let sequence = &self.colors[body];
        if sequence.is_empty() {
            return (0.75, 0.0, 0.0);
        }
        let mut sum = (0.0, 0.0, 0.0);
        for &(l, a, b) in sequence.iter().step_by(101) {
            sum.0 += l;
            sum.1 += a;
            sum.2 += b;
        }
        let count = sequence.iter().step_by(101).count().max(1) as f64;
        (sum.0 / count, sum.1 / count, sum.2 / count)
    }

    /// Number of simulation steps.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.positions.first().map_or(0, Vec::len)
    }
}
