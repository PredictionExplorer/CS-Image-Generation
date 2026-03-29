// SDF line splatting -- one invocation per line segment (3 per timestep).
// Matches the CPU draw_line_segment_aa_spectral_rows algorithm.

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

// Per-segment data packed: (bin0_f, bin1_f, alpha0, alpha1, hdr_mult)
@group(0) @binding(0) var<uniform> params: SplatUniforms;
@group(0) @binding(1) var<storage, read> projected: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> seg_data: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> spd: array<atomic<u32>>;

fn float_to_fixed(v: f32) -> u32 {
    return bitcast<u32>(i32(v * 65536.0));
}

fn fixed_to_float(v: u32) -> f32 {
    return f32(bitcast<i32>(v)) / 65536.0;
}

fn atomic_add_fixed(idx: u32, value: f32) {
    let fixed_val = bitcast<u32>(i32(value * 65536.0));
    if fixed_val == 0u {
        return;
    }
    atomicAdd(&spd[idx], fixed_val);
}

fn world_to_pixel(proj_x: f32, proj_y: f32) -> vec2<f32> {
    let nx = (proj_x - params.bbox_min_x) / params.bbox_width;
    let ny = (proj_y - params.bbox_min_y) / params.bbox_height;
    return vec2<f32>(nx * f32(params.width), ny * f32(params.height));
}

const NUM_BINS: u32 = 64u;
const LAMBDA_START: f32 = 380.0;
const BIN_WIDTH: f32 = 5.0; // (700-380)/64 = 5.0

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seg_idx = gid.x;
    if seg_idx >= params.num_segments {
        return;
    }

    let step = seg_idx / 3u;
    let edge = seg_idx % 3u;
    let body0 = edge;
    let body1 = (edge + 1u) % 3u;

    let idx0 = step * 3u + body0;
    let idx1 = step * 3u + body1;
    let p0 = projected[idx0];
    let p1 = projected[idx1];

    let pix0 = world_to_pixel(p0.x, p0.y);
    let pix1 = world_to_pixel(p1.x, p1.y);

    let x0 = pix0.x; let y0 = pix0.y; let z0 = p0.z;
    let x1 = pix1.x; let y1 = pix1.y; let z1 = p1.z;
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dz = z1 - z0;

    let len_sq = dx * dx + dy * dy;
    let len_3d = sqrt(dx * dx + dy * dy + dz * dz);

    let base_thickness = 1.2;
    let thickness = clamp(base_thickness / (0.1 + len_3d * 0.5), 0.2, 4.0);
    let avg_z = (z0 + z1) * 0.5;
    let coc = abs(avg_z * 0.05);
    let effective_thickness = thickness + coc;

    let pad = i32(ceil(effective_thickness * 2.5));
    let min_px = max(i32(min(x0, x1)) - pad, 0);
    let max_px = min(i32(max(x0, x1)) + pad, i32(params.width) - 1);
    let min_py = max(i32(min(y0, y1)) - pad, 0);
    let max_py = min(i32(max(y0, y1)) + pad, i32(params.height) - 1);

    if min_px > max_px || min_py > max_py {
        return;
    }

    let sd = seg_data[seg_idx];
    let bin0_f = sd.x;
    let bin1_f = sd.y;
    let alpha0 = sd.z;
    let alpha1 = sd.w;
    let hdr_mult = seg_data[params.num_segments + seg_idx].x;

    let energy_conservation = thickness / effective_thickness;
    let depth_fade = clamp(exp(-abs(avg_z) * 0.002), 0.05, 1.0);
    let base_energy_mult = params.hdr_scale * hdr_mult * depth_fade * energy_conservation;

    let inv_thick_sq = 1.0 / (effective_thickness * effective_thickness);

    for (var py = min_py; py <= max_py; py++) {
        for (var px = min_px; px <= max_px; px++) {
            let pax = f32(px) - x0;
            let pay = f32(py) - y0;

            var h: f32;
            if len_sq > 1e-6 {
                h = clamp((pax * dx + pay * dy) / len_sq, 0.0, 1.0);
            } else {
                h = 0.5;
            }

            let proj_x_d = pax - dx * h;
            let proj_y_d = pay - dy * h;
            let dist_sq = proj_x_d * proj_x_d + proj_y_d * proj_y_d;

            let energy = exp(-dist_sq * inv_thick_sq);
            if energy < 0.01 {
                continue;
            }

            let alpha = alpha0 * (1.0 - h) + alpha1 * h;
            let final_energy = energy * alpha * base_energy_mult;

            let bin_f = bin0_f * (1.0 - h) + bin1_f * h;
            let bin_left = min(u32(floor(bin_f)), NUM_BINS - 1u);
            let bin_right = min(bin_left + 1u, NUM_BINS - 1u);
            let w_right = bin_f - floor(bin_f);

            let pixel_idx = u32(py) * params.width + u32(px);
            let base = pixel_idx * NUM_BINS;

            if bin_right == bin_left {
                atomic_add_fixed(base + bin_left, final_energy);
            } else {
                atomic_add_fixed(base + bin_left, final_energy * (1.0 - w_right));
                atomic_add_fixed(base + bin_right, final_energy * w_right);
            }
        }
    }
}
