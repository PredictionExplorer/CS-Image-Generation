#version 430 core
// Actual single-valued surface intersection with analytical direct lighting.
// The BRDF and finite-layer RGB pigment optics are approximations, not a path trace.
in vec2 screen_position;
out vec4 frag_color;
uniform sampler2D u_paint, u_geometry, u_finish;
uniform vec3 u_pigment_0, u_pigment_1, u_pigment_2, u_scattering, u_substrate;
uniform vec2 u_visible_size, u_full_size, u_grid_size, u_output_size;
uniform vec3 u_camera_right, u_camera_up, u_camera_view, u_key_direction;
uniform float u_height_scale, u_max_height, u_layer_scale;
uniform float u_ambient, u_key_strength, u_fill_strength, u_anisotropy;
uniform float u_roughness_scale, u_roughness_bias;
uniform float u_grain_height, u_grain_scale, u_shadow_strength, u_occlusion_strength, u_exposure;
uniform int u_family, u_tone_map;
const float PI = 3.141592653589793;

vec2 field_uv(vec2 p) { return p / u_full_size + 0.5; }
bool inside(vec2 uv) { return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0))); }
float surface_height(vec2 p) {
    vec2 uv = field_uv(p);
    return inside(uv) ? textureLod(u_geometry, uv, 0.0).r * u_height_scale : 0.0;
}

vec3 hit_surface(vec3 origin, vec3 view) {
    // A frontal camera hits the exact pixel column; tilted views march through
    // the tight physical height interval. This includes self-occlusion/parallax.
    if (dot(view.xy, view.xy) < 1e-12) return vec3(origin.xy, surface_height(origin.xy));
    vec3 start = origin + view * ((u_max_height - origin.z) / view.z);
    vec3 finish = origin - view * (origin.z / view.z);
    vec3 upper = start;
    vec3 lower = finish;
    for (int i = 1; i <= 128; ++i) {
        vec3 point = mix(start, finish, float(i) / 128.0);
        if (point.z <= surface_height(point.xy)) { lower = point; break; }
        upper = point;
    }
    for (int i = 0; i < 8; ++i) {
        vec3 point = (upper + lower) * 0.5;
        if (point.z > surface_height(point.xy)) upper = point;
        else lower = point;
    }
    return (upper + lower) * 0.5;
}

float hash21(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}
float support_noise(vec2 p) {
    vec2 cell = floor(p), f = fract(p);
    vec2 w = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(cell), hash21(cell + vec2(1, 0)), w.x),
               mix(hash21(cell + vec2(0, 1)), hash21(cell + 1.0), w.x), w.y) * 2.0 - 1.0;
}
vec3 normal_at(vec3 point, float coverage) {
    vec2 e = u_full_size / u_grid_size;
    vec2 slope = vec2(
        (surface_height(point.xy + vec2(e.x, 0)) - surface_height(point.xy - vec2(e.x, 0))) / (2.0 * e.x),
        (surface_height(point.xy + vec2(0, e.y)) - surface_height(point.xy - vec2(0, e.y))) / (2.0 * e.y));
    float footprint = max(u_visible_size.x / u_output_size.x, u_visible_size.y / u_output_size.y) / u_camera_view.z;
    float attenuation = 1.0 - smoothstep(u_grain_scale * 0.2, u_grain_scale * 0.65, footprint);
    float amplitude = u_grain_height / u_grain_scale * attenuation * mix(1.0, 0.3, coverage);
    vec2 grain_point = point.xy / u_grain_scale;
    slope += amplitude * vec2(support_noise(grain_point + vec2(0.1, 0)) - support_noise(grain_point - vec2(0.1, 0)),
                               support_noise(grain_point + vec2(0, 0.1)) - support_noise(grain_point - vec2(0, 0.1))) / 0.2;
    return normalize(vec3(-slope, 1.0));
}

vec3 absorption_ratio(vec3 r) { return (1.0 - r) * (1.0 - r) / (2.0 * max(r, vec3(1e-6))); }
float one_minus_exp_negative(float x) {
    if (x < 0.01) return x * (1.0 + x * (-0.5 + x * (1.0/6.0 + x * (-1.0/24.0 + x/120.0))));
    return 1.0 - exp(-x);
}
float finite_layer(float ratio, float thickness, float substrate) {
    if (thickness <= 0.0) return substrate;
    float a = 1.0 + ratio, b = sqrt(ratio * (ratio + 2.0));
    float r, t;
    if (b <= 1e-6) { r = thickness / (1.0 + thickness); t = 1.0 / (1.0 + thickness); }
    else {
        float exponent = b * thickness;
        float difference = one_minus_exp_negative(2.0 * exponent);
        float denominator = b * (2.0 - difference) + a * difference;
        r = difference / denominator;
        t = 2.0 * b * exp(-exponent) / denominator;
    }
    return r + t * t * substrate / max(1.0 - r * substrate, 1e-12);
}
vec3 paint_color(vec2 uv) {
    vec3 strength = textureLod(u_paint, uv, 0.0).rgb * u_scattering;
    float scattering = strength.x + strength.y + strength.z;
    vec3 absorption = strength.x * absorption_ratio(u_pigment_0)
                    + strength.y * absorption_ratio(u_pigment_1)
                    + strength.z * absorption_ratio(u_pigment_2);
    vec3 ratio = absorption / max(scattering, 1e-20);
    float thickness = scattering * u_layer_scale;
    return vec3(finite_layer(ratio.r, thickness, u_substrate.r),
                finite_layer(ratio.g, thickness, u_substrate.g),
                finite_layer(ratio.b, thickness, u_substrate.b));
}

float shadow(vec3 point, vec3 light) {
    float pixel = max(u_full_size.x / u_grid_size.x, u_full_size.y / u_grid_size.y);
    float travel = min((u_max_height - point.z + pixel) / light.z, 0.035);
    float visibility = 1.0;
    // Increasing step lengths concentrate samples around small paint ridges.
    // The height clearance estimates a broad emitter's penumbra deterministically.
    for (int i = 1; i <= 24; ++i) {
        float q = float(i) / 24.0;
        float distance = pixel + travel * q * q;
        vec3 ray = point + light * distance;
        float clearance = ray.z + pixel * 0.15 - surface_height(ray.xy);
        visibility = min(visibility, clamp(clearance / max(distance * 0.14, pixel * 0.1), 0.0, 1.0));
    }
    return mix(1.0, visibility, u_shadow_strength);
}

float occlusion(vec3 point) {
    float sum = 0.0;
    float pixel = max(u_full_size.x / u_grid_size.x, u_full_size.y / u_grid_size.y);
    for (int i = 0; i < 8; ++i) {
        float a = float(i) * PI / 4.0;
        vec2 d = vec2(cos(a), sin(a));
        float distance = max(pixel * 3.0, 0.0006);
        float near_h = (surface_height(point.xy + d * distance) - point.z) / distance;
        float far_h = (surface_height(point.xy + d * distance * 4.0) - point.z) / (distance * 4.0);
        sum += clamp(max(near_h, far_h), 0.0, 1.0);
    }
    return 1.0 - u_occlusion_strength * sum / 8.0;
}

float lambda_ggx(vec3 w, vec3 n, vec3 t, vec3 b, vec2 alpha) {
    float wz = max(dot(w, n), 1e-5);
    float x = dot(w, t) * alpha.x, y = dot(w, b) * alpha.y;
    return (sqrt(1.0 + (x*x + y*y)/(wz*wz)) - 1.0) * 0.5;
}
vec3 light_brdf(vec3 n, vec3 t, vec3 b, vec3 v, vec3 l, vec3 color, float roughness, float coherence, float paint_presence) {
    float nl = max(dot(n, l), 0.0), nv = max(dot(n, v), 1e-5);
    if (nl <= 0.0) return vec3(0);
    vec3 h = normalize(v + l);
    float ratio = sqrt(1.0 - 0.9 * u_anisotropy * coherence);
    float alpha = max(roughness * roughness, 0.0064);
    vec2 a = vec2(alpha / ratio, alpha * ratio);
    float ht = dot(h, t)/a.x, hb = dot(h, b)/a.y, hn = max(dot(h, n), 0.0);
    float denominator = ht*ht + hb*hb + hn*hn;
    float distribution = 1.0 / max(PI * a.x*a.y * denominator*denominator, 1e-9);
    float geometry = 1.0 / (1.0 + lambda_ggx(v,n,t,b,a) + lambda_ggx(l,n,t,b,a));
    // Nocturne's bare porous carbon ground has a low effective interface
    // reflectance; a binder-rich painted passage has a stronger dielectric lobe.
    float f0 = u_family == 0 ? 0.025 : u_family == 1 ? 0.04 : mix(0.006, 0.052, paint_presence);
    float fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    float specular = distribution * geometry * fresnel / max(4.0 * nv * nl, 1e-5);
    // Lambertian paint reflectance plus energy-reduced dielectric interface.
    return ((1.0 - fresnel) * color / PI + vec3(specular)) * nl * PI;
}

void main() {
    vec3 origin = u_camera_right * screen_position.x * u_visible_size.x * 0.5
                + u_camera_up * screen_position.y * u_visible_size.y * 0.5;
    vec3 point = hit_surface(origin, u_camera_view);
    vec2 uv = field_uv(point.xy);
    if (!inside(uv)) { frag_color = vec4(u_substrate, 1); return; }
    vec4 material = textureLod(u_geometry, uv, 0.0);
    vec2 axial = textureLod(u_finish, uv, 0.0).xy;
    float coherence = clamp(length(axial), 0.0, 1.0);
    float angle = coherence > 1e-6 ? atan(axial.y, axial.x) * 0.5 : 0.0;
    vec3 n = normal_at(point, material.a);
    vec3 raw_t = vec3(cos(angle), sin(angle), 0.0);
    vec3 t = normalize(raw_t - n * dot(raw_t, n));
    vec3 b = cross(n, t);
    float roughness = clamp((material.b * u_roughness_scale + u_roughness_bias)
                            * mix(1.0, 0.78, material.g), 0.09, 0.98);
    if (u_family == 0) roughness = max(roughness, mix(0.55, 0.4, material.g));
    float paint_presence = 1.0 - exp(-12.0 * material.a);
    if (u_family == 2) {
        // A direction field may have a unit fallback even on unpainted ground.
        // Carbon's authored directional sheen belongs to deposited material;
        // uncovered support stays matte instead of reflecting a false brush.
        roughness = mix(0.93, roughness, paint_presence);
        coherence *= paint_presence;
    }
    vec3 color = paint_color(uv);
    // Wet paint is slightly darker; the independent roughness change carries
    // most of the visible drying response without a synthetic animated overlay.
    color *= mix(1.0, 0.91, material.g);
    float ao = occlusion(point);
    vec3 radiance = color * u_ambient * ao * (0.65 + 0.35 * max(n.z, 0.0));
    vec3 key_t = normalize(cross(vec3(0, 0, 1), u_key_direction + vec3(1e-6, 0, 0)));
    vec3 key_b = cross(u_key_direction, key_t);
    float visibility = shadow(point, u_key_direction);
    // Fixed world-space quadrature does not crawl or change with output cadence.
    for (int i = 0; i < 5; ++i) {
        vec2 offset = i == 0 ? vec2(0) : i == 1 ? vec2(-0.16,0) : i == 2 ? vec2(0.16,0)
                    : i == 3 ? vec2(0,-0.1) : vec2(0,0.1);
        vec3 light = normalize(u_key_direction + key_t * offset.x + key_b * offset.y);
        radiance += light_brdf(n,t,b,u_camera_view,light,color,roughness,coherence,paint_presence) * u_key_strength * visibility / 5.0;
    }
    vec3 fill = normalize(vec3(-u_key_direction.xy, 0.8));
    radiance += light_brdf(n,t,b,u_camera_view,fill,color,roughness,coherence,paint_presence) * u_fill_strength * ao;
    radiance = max(radiance * u_exposure, vec3(0));
    if (u_tone_map == 1) {
        float luminance = dot(radiance, vec3(0.2126, 0.7152, 0.0722));
        radiance *= (1.0 + luminance / 16.0) / (1.0 + luminance);
        radiance = clamp(radiance, 0.0, 1.0);
    }
    frag_color = vec4(radiance, 1);
}
