//! GPU-accelerated spectral line splatting via wgpu compute shaders.
//!
//! Provides the same visual output as the CPU path but uses the GPU for
//! the projection and SDF splatting kernels.  The SPD buffer lives on the
//! GPU as fixed-point u32 atomics and is read back as f64 for the existing
//! post-processing pipeline.

use super::color::OklabColor;
use super::context::BoundingBox;
use super::drawing::oklab_hue_to_wavelength;
use crate::render::camera::Camera3D;
use crate::spectral_constants;
use crate::spectrum::NUM_BINS;
use bytemuck::{self, Pod, Zeroable};
use nalgebra::Vector3;
use tracing::info;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

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
}

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    project_pipeline: wgpu::ComputePipeline,
    splat_pipeline: wgpu::ComputePipeline,
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

        let project_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("project.wgsl"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/project.wgsl").into(),
                ),
            });

        let splat_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("splat.wgsl"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/splat.wgsl").into(),
                ),
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

        Self { device, queue, project_pipeline, splat_pipeline }
    }

    /// Render a single frame: project all positions through the camera at
    /// `camera_step`, clear the SPD buffer, splat all segments 0..step_end,
    /// and read back the result as f64 for the downstream CPU pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_frame(
        &self,
        original_positions: &[Vec<Vector3<f64>>],
        colors: &[Vec<OklabColor>],
        body_alphas: &[f64],
        camera: &Camera3D,
        camera_step: usize,
        step_end: usize,
        hdr_scale: f64,
        width: u32,
        height: u32,
        bounds: &BoundingBox,
        velocity_positions: &[Vec<Vector3<f64>>],
        dt: f64,
    ) -> Vec<[f64; NUM_BINS]> {
        let num_bodies = original_positions.len();
        let pixel_count = (width as usize) * (height as usize);

        let world_data = self.pack_positions(original_positions, step_end);
        let cam_uniforms = self.build_camera_uniforms(camera, camera_step, step_end);

        let world_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("world-positions"),
            contents: bytemuck::cast_slice(&world_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let projected_size = (step_end * num_bodies * 16) as u64;
        let projected_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("projected-positions"),
            size: projected_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let cam_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera-uniforms"),
            contents: bytemuck::bytes_of(&cam_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // --- Projection dispatch ---
        let proj_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("project-bind"),
            layout: &self.project_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: world_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: projected_buf.as_entire_binding() },
            ],
        });

        let total_invocations = (step_end * num_bodies) as u32;
        let proj_groups = total_invocations.div_ceil(WORKGROUP_SIZE);

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.project_pipeline);
            pass.set_bind_group(0, &proj_bind_group, &[]);
            pass.dispatch_workgroups(proj_groups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // --- Prepare segment data ---
        let num_segments = step_end * 3;
        let (seg_data, hdr_mults) = self.pack_segment_data(
            colors, body_alphas, velocity_positions, dt, step_end, hdr_scale,
        );

        let mut seg_combined: Vec<[f32; 4]> = Vec::with_capacity(num_segments * 2);
        seg_combined.extend_from_slice(&seg_data);
        for &m in &hdr_mults {
            seg_combined.push([m, 0.0, 0.0, 0.0]);
        }

        let seg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("segment-data"),
            contents: bytemuck::cast_slice(&seg_combined),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // SPD buffer as u32 fixed-point atomics
        let spd_size = (pixel_count * NUM_BINS * 4) as u64;
        let spd_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spd-buffer"),
            size: spd_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        {
            let mut view = spd_buf.slice(..).get_mapped_range_mut();
            view.fill(0);
        }
        spd_buf.unmap();

        let splat_uniforms = SplatUniforms {
            width,
            height,
            hdr_scale: hdr_scale as f32,
            num_segments: num_segments as u32,
            bbox_min_x: bounds.min_x as f32,
            bbox_min_y: bounds.min_y as f32,
            bbox_width: bounds.width as f32,
            bbox_height: bounds.height as f32,
        };

        let splat_uniform_buf =
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("splat-uniforms"),
                contents: bytemuck::bytes_of(&splat_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let splat_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat-bind"),
            layout: &self.splat_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: splat_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: projected_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: seg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: spd_buf.as_entire_binding() },
            ],
        });

        let splat_groups = (num_segments as u32).div_ceil(WORKGROUP_SIZE);

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.splat_pipeline);
            pass.set_bind_group(0, &splat_bind_group, &[]);
            pass.dispatch_workgroups(splat_groups, 1, 1);
        }

        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spd-readback"),
            size: spd_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&spd_buf, 0, &readback_buf, 0, spd_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back
        let slice = readback_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().expect("GPU readback failed");

        let data = slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);

        let mut result = vec![[0.0f64; NUM_BINS]; pixel_count];
        for (px_idx, pixel) in result.iter_mut().enumerate() {
            for bin in 0..NUM_BINS {
                let fixed = u32_data[px_idx * NUM_BINS + bin];
                let value = (fixed as i32) as f64 / 65536.0;
                pixel[bin] = value;
            }
        }

        drop(data);
        readback_buf.unmap();

        result
    }

    fn pack_positions(
        &self,
        positions: &[Vec<Vector3<f64>>],
        step_end: usize,
    ) -> Vec<[f32; 4]> {
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

    fn build_camera_uniforms(
        &self,
        camera: &Camera3D,
        camera_step: usize,
        step_end: usize,
    ) -> CameraUniforms {
        let proj = camera.project_point(&Vector3::zeros(), camera_step);
        let _ = proj;

        let test_origin = Vector3::new(0.0, 0.0, 0.0);
        let test_x = Vector3::new(1.0, 0.0, 0.0);
        let test_y = Vector3::new(0.0, 1.0, 0.0);
        let test_z = Vector3::new(0.0, 0.0, 1.0);

        let p0 = camera.project_point(&test_origin, camera_step);
        let px = camera.project_point(&test_x, camera_step);
        let py = camera.project_point(&test_y, camera_step);
        let pz = camera.project_point(&test_z, camera_step);

        let _ = (p0, px, py, pz);

        let projected_all = camera.project_all_positions_at_step(
            &[vec![test_origin], vec![test_origin], vec![test_origin]],
            camera_step,
        );
        let _ = projected_all;

        let cam_data = camera.camera_uniforms_at_step(camera_step);

        CameraUniforms {
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
        }
    }

    fn pack_segment_data(
        &self,
        colors: &[Vec<OklabColor>],
        body_alphas: &[f64],
        velocity_positions: &[Vec<Vector3<f64>>],
        dt: f64,
        step_end: usize,
        _hdr_scale: f64,
    ) -> (Vec<[f32; 4]>, Vec<f32>) {
        let num_segments = step_end * 3;
        let mut seg_data = Vec::with_capacity(num_segments);
        let mut hdr_mults = Vec::with_capacity(num_segments);

        for step in 0..step_end {
            let edges: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
            for &(b0, b1) in &edges {
                let (_, a0, b0_c) = colors[b0][step];
                let (_, a1, b1_c) = colors[b1][step];
                let wl0 = oklab_hue_to_wavelength(a0, b0_c);
                let wl1 = oklab_hue_to_wavelength(a1, b1_c);
                let bin0 = spectral_constants::wavelength_to_bin(wl0) as f32;
                let bin1 = spectral_constants::wavelength_to_bin(wl1) as f32;
                let alpha0 = body_alphas[b0] as f32;
                let alpha1 = body_alphas[b1] as f32;
                seg_data.push([bin0, bin1, alpha0, alpha1]);

                let hdr_mult = if step > 0 {
                    let d0 = (velocity_positions[b0][step] - velocity_positions[b0][step - 1]).norm();
                    let d1 = (velocity_positions[b1][step] - velocity_positions[b1][step - 1]).norm();
                    let avg_speed = (d0 + d1) * 0.5 / dt;
                    (1.0 / (1.0 + avg_speed * 0.01)).clamp(0.1, 2.0)
                } else {
                    1.0
                };
                hdr_mults.push(hdr_mult as f32);
            }
        }
        (seg_data, hdr_mults)
    }
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
