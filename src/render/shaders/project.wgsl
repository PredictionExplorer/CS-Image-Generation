struct CameraUniforms {
    eye: vec3<f32>,
    _pad0: f32,
    fwd: vec3<f32>,
    _pad1: f32,
    right: vec3<f32>,
    _pad2: f32,
    true_up: vec3<f32>,
    half_fov_tan: f32,
    focus_dist: f32,
    z_floor: f32,
    depth_scale: f32,
    num_steps: u32,
}

@group(0) @binding(0) var<uniform> cam: CameraUniforms;
@group(0) @binding(1) var<storage, read> world_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> projected: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = cam.num_steps * 3u;
    if idx >= total {
        return;
    }

    let wp = world_pos[idx].xyz;
    let d = wp - cam.eye;
    let z_cam = max(dot(d, cam.fwd), cam.z_floor);
    let x_cam = dot(d, cam.right);
    let y_cam = dot(d, cam.true_up);

    let proj_x = x_cam / (z_cam * cam.half_fov_tan);
    let proj_y = y_cam / (z_cam * cam.half_fov_tan);
    let depth = (z_cam - cam.focus_dist) * cam.depth_scale;

    projected[idx] = vec4<f32>(proj_x, proj_y, depth, 0.0);
}
