//! GPU-accelerated spectral line splatting via wgpu compute shaders.
//!
//! Uploads world positions and segment color/alpha data once, then for each
//! frame computes velocity HDR multipliers from CPU-projected positions
//! (matching the CPU path exactly) and dispatches projection + splatting.

use super::color::OklabColor;
use super::context::BoundingBox;
use super::drawing::oklab_hue_to_wavelength;
use crate::render::camera::Camera3D;
use crate::render::constants::{VELOCITY_HDR_BOOST_FACTOR, VELOCITY_HDR_BOOST_THRESHOLD};
use crate::spectral_constants;
use crate::spectrum::NUM_BINS;
use bytemuck::{self, Pod, Zeroable};
use nalgebra::Vector3;
use tracing::info;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;
const ENERGY_PRESCALE: f32 = 1e6;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniforms {
    eye: [f32; 3],
    _pad0: f32,
    fwd: [f32; 3],
    _pad1: f32,
    right: [f32; 3],
    _pad2: f32,
    true_up: [f32; 3],
    half_fov_tan: f32,
    focus_dist: f32,
    z_floor: f32,
    depth_scale: f32,
    num_steps: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SplatUniforms {
    width: u32,
    height: u32,
    hdr_scale: f32,
    num_segments: u32,
    bbox_min_x: f32,
    bbox_min_y: f32,
    bbox_width: f32,
    bbox_height: f32,
    energy_prescale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    project_pipeline: wgpu::ComputePipeline,
    splat_pipeline: wgpu::ComputePipeline,
    world_buf: Option<wgpu::Buffer>,
    projected_buf: Option<wgpu::Buffer>,
    seg_buf: Option<wgpu::Buffer>,
    spd_buf: Option<wgpu::Buffer>,
    readback_buf: Option<wgpu::Buffer>,
    splat_uniform_buf: Option<wgpu::Buffer>,
    total_steps: usize,
    num_segments: usize,
    pixel_count: usize,
    spd_byte_size: u64,
    dt: f64,
}

impl GpuContext {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("Failed to find a GPU adapter");

        info!(
            name = adapter.get_info().name,
            backend = ?adapter.get_info().backend,
            "GPU adapter selected"
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("spectral-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 1 << 30,
                    max_buffer_size: 1 << 30,
                    max_compute_invocations_per_workgroup: 256,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create GPU device");

        let project_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("project.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/project.wgsl").into()),
        });

        let splat_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/splat.wgsl").into()),
        });

        let project_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("project-pipeline"),
                layout: None,
                module: &project_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let splat_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("splat-pipeline"),
                layout: None,
                module: &splat_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        info!("GPU compute pipelines compiled");

        Self {
            device,
            queue,
            project_pipeline,
            splat_pipeline,
            world_buf: None,
            projected_buf: None,
            seg_buf: None,
            spd_buf: None,
            readback_buf: None,
            splat_uniform_buf: None,
            total_steps: 0,
            num_segments: 0,
            pixel_count: 0,
            spd_byte_size: 0,
            dt: 0.0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        original_positions: &[Vec<Vector3<f64>>],
        colors: &[Vec<OklabColor>],
        body_alphas: &[f64],
        total_steps: usize,
        hdr_scale: f64,
        width: u32,
        height: u32,
        bounds: &BoundingBox,
        dt: f64,
    ) {
        let num_bodies = original_positions.len();
        self.total_steps = total_steps;
        self.num_segments = total_steps * 3;
        self.pixel_count = (width as usize) * (height as usize);
        self.spd_byte_size = (self.pixel_count * NUM_BINS * 4) as u64;
        self.dt = dt;

        let world_data = pack_positions(original_positions, total_steps);
        self.world_buf = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("world-positions"),
            contents: bytemuck::cast_slice(&world_data),
            usage: wgpu::BufferUsages::STORAGE,
        }));

        let projected_size = (total_steps * num_bodies * 16) as u64;
        self.projected_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("projected-positions"),
            size: projected_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        // Static segment data: color bins + alphas (no HDR mults -- those are per-frame)
        let seg_data = pack_segment_colors(colors, body_alphas, total_steps);
        self.seg_buf = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("segment-data"),
            contents: bytemuck::cast_slice(&seg_data),
            usage: wgpu::BufferUsages::STORAGE,
        }));

        self.spd_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spd-buffer"),
            size: self.spd_byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.readback_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spd-readback"),
            size: self.spd_byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        let splat_uniforms = SplatUniforms {
            width,
            height,
            hdr_scale: hdr_scale as f32,
            num_segments: self.num_segments as u32,
            bbox_min_x: bounds.min_x as f32,
            bbox_min_y: bounds.min_y as f32,
            bbox_width: bounds.width as f32,
            bbox_height: bounds.height as f32,
            energy_prescale: ENERGY_PRESCALE,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        self.splat_uniform_buf =
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("splat-uniforms"),
                contents: bytemuck::bytes_of(&splat_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            }));

        let spd_mb = self.spd_byte_size as f64 / (1024.0 * 1024.0);
        info!(
            steps = total_steps,
            segments = self.num_segments,
            spd_mb = format!("{spd_mb:.1}"),
            "GPU buffers uploaded"
        );
    }

    /// Render a single frame. Computes velocity HDR multipliers from
    /// CPU-projected positions to exactly match the CPU rendering path.
    pub fn render_frame(
        &self,
        camera: &Camera3D,
        original_positions: &[Vec<Vector3<f64>>],
        camera_step: usize,
        step_end: usize,
    ) -> Vec<[f32; NUM_BINS]> {
        let world_buf = self.world_buf.as_ref().expect("call prepare() first");
        let projected_buf = self.projected_buf.as_ref().unwrap();
        let seg_buf = self.seg_buf.as_ref().unwrap();
        let spd_buf = self.spd_buf.as_ref().unwrap();
        let readback_buf = self.readback_buf.as_ref().unwrap();
        let splat_uniform_buf = self.splat_uniform_buf.as_ref().unwrap();

        // Compute velocity HDR multipliers from projected positions (matches CPU exactly)
        let frame_positions = camera.project_all_positions_at_step(original_positions, camera_step);
        let hdr_mults = compute_hdr_multipliers(&frame_positions, self.dt, step_end);

        // Pack: seg_data (static, already on GPU) is in seg_buf at indices 0..num_segments.
        // HDR mults go into a separate per-frame buffer at indices num_segments..2*num_segments.
        let mut hdr_data: Vec<[f32; 4]> = Vec::with_capacity(self.num_segments);
        for &m in &hdr_mults {
            hdr_data.push([m, 0.0, 0.0, 0.0]);
        }
        // Pad remainder with 1.0 if step_end < total_steps
        for _ in hdr_mults.len()..self.num_segments {
            hdr_data.push([1.0, 0.0, 0.0, 0.0]);
        }

        let hdr_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hdr-mults"),
            contents: bytemuck::cast_slice(&hdr_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let cam_data = camera.camera_uniforms_at_step(camera_step);
        let cam_uniforms = CameraUniforms {
            eye: cam_data.eye,
            _pad0: 0.0,
            fwd: cam_data.fwd,
            _pad1: 0.0,
            right: cam_data.right,
            _pad2: 0.0,
            true_up: cam_data.true_up,
            half_fov_tan: cam_data.half_fov_tan,
            focus_dist: cam_data.focus_dist,
            z_floor: cam_data.z_floor,
            depth_scale: cam_data.depth_scale,
            num_steps: step_end as u32,
        };
        let cam_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera-uniforms"),
            contents: bytemuck::bytes_of(&cam_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let num_bodies = 3u32;
        let proj_groups = (step_end as u32 * num_bodies).div_ceil(WORKGROUP_SIZE);
        let splat_groups = (step_end as u32 * 3).div_ceil(WORKGROUP_SIZE);

        let proj_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("project-bind"),
            layout: &self.project_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: world_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: projected_buf.as_entire_binding() },
            ],
        });

        let splat_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat-bind"),
            layout: &self.splat_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: splat_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: projected_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: seg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: spd_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: hdr_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(spd_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.project_pipeline);
            pass.set_bind_group(0, &proj_bind, &[]);
            pass.dispatch_workgroups(proj_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.splat_pipeline);
            pass.set_bind_group(0, &splat_bind, &[]);
            pass.dispatch_workgroups(splat_groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(spd_buf, 0, readback_buf, 0, self.spd_byte_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().expect("GPU readback failed");

        let data = slice.get_mapped_range();
        let f32_data: &[f32] = bytemuck::cast_slice(&data);

        let inv_prescale = 1.0f32 / ENERGY_PRESCALE as f32;
        let mut result = vec![[0.0f32; NUM_BINS]; self.pixel_count];
        for (px_idx, pixel) in result.iter_mut().enumerate() {
            for bin in 0..NUM_BINS {
                pixel[bin] = f32_data[px_idx * NUM_BINS + bin] * inv_prescale;
            }
        }

        drop(data);
        readback_buf.unmap();

        result
    }
}

/// Compute velocity HDR multipliers matching `VelocityHdrCalculator` exactly.
fn compute_hdr_multipliers(
    projected_positions: &[Vec<Vector3<f64>>],
    dt: f64,
    step_end: usize,
) -> Vec<f32> {
    let edges: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
    let mut mults = Vec::with_capacity(step_end * 3);

    for step in 0..step_end {
        for &(b0, b1) in &edges {
            if step + 1 < projected_positions[0].len() {
                let v0 = single_velocity_mult(projected_positions, step, b0, dt);
                let v1 = single_velocity_mult(projected_positions, step, b1, dt);
                mults.push(((v0 + v1) * 0.5) as f32);
            } else {
                mults.push(1.0);
            }
        }
    }
    mults
}

#[inline]
fn single_velocity_mult(positions: &[Vec<Vector3<f64>>], step: usize, body: usize, dt: f64) -> f64 {
    let p0 = positions[body][step];
    let p1 = positions[body][step + 1];
    let velocity = (p1 - p0).norm() / dt;
    let normalized = (velocity / VELOCITY_HDR_BOOST_THRESHOLD).min(1.0);
    1.0 + normalized * (VELOCITY_HDR_BOOST_FACTOR - 1.0)
}

fn pack_positions(positions: &[Vec<Vector3<f64>>], step_end: usize) -> Vec<[f32; 4]> {
    let num_bodies = positions.len();
    let mut data = Vec::with_capacity(step_end * num_bodies);
    for step in 0..step_end {
        for body in 0..num_bodies {
            let p = positions[body][step];
            data.push([p[0] as f32, p[1] as f32, p[2] as f32, 0.0]);
        }
    }
    data
}

/// Pack static per-segment data: spectral bin positions + body alphas.
/// HDR multipliers are computed per-frame and uploaded separately.
fn pack_segment_colors(
    colors: &[Vec<OklabColor>],
    body_alphas: &[f64],
    step_end: usize,
) -> Vec<[f32; 4]> {
    let mut seg_data = Vec::with_capacity(step_end * 3);
    let edges: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];

    for step in 0..step_end {
        for &(b0, b1) in &edges {
            let (_, a0, b0_c) = colors[b0][step];
            let (_, a1, b1_c) = colors[b1][step];
            let wl0 = oklab_hue_to_wavelength(a0, b0_c);
            let wl1 = oklab_hue_to_wavelength(a1, b1_c);
            let bin0 = spectral_constants::wavelength_to_bin(wl0) as f32;
            let bin1 = spectral_constants::wavelength_to_bin(wl1) as f32;
            seg_data.push([bin0, bin1, body_alphas[b0] as f32, body_alphas[b1] as f32]);
        }
    }
    seg_data
}

/// Raw camera pose data for GPU upload.
pub struct CameraRawUniforms {
    pub eye: [f32; 3],
    pub fwd: [f32; 3],
    pub right: [f32; 3],
    pub true_up: [f32; 3],
    pub half_fov_tan: f32,
    pub focus_dist: f32,
    pub z_floor: f32,
    pub depth_scale: f32,
}
