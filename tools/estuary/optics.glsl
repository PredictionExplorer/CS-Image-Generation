#version 330 core

// Finite-layer Kubelka--Munk approximation in linear sRGB. This is an authored
// RGB material model, not a spectral fit to specific manufactured pigments.
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_paint;
uniform vec3 u_pigment_0;
uniform vec3 u_pigment_1;
uniform vec3 u_pigment_2;
uniform vec3 u_scattering;
uniform vec3 u_substrate;
uniform float u_layer_scale;
uniform float u_grain;
uniform vec2 u_grain_frequency;

vec3 absorption_ratio(vec3 reflectance) {
    vec3 safe_color = max(reflectance, vec3(1e-6));
    return (1.0 - safe_color) * (1.0 - safe_color) / (2.0 * safe_color);
}

float one_minus_exp_negative(float x) {
    // Avoid float32 cancellation for optically thin paint.
    if (x < 0.01) {
        return x * (1.0 + x * (-0.5 + x * (1.0 / 6.0 + x * (-1.0 / 24.0 + x / 120.0))));
    }
    return 1.0 - exp(-x);
}

float finite_layer(float ratio, float thickness, float substrate) {
    if (thickness <= 0.0) {
        return substrate;
    }
    float a = 1.0 + ratio;
    float b = sqrt(ratio * (ratio + 2.0));
    float layer_r;
    float layer_t;
    if (b <= 1e-6) {
        // Exact pure-scattering limit; avoids the white-pigment 0/0 case.
        layer_r = thickness / (1.0 + thickness);
        layer_t = 1.0 / (1.0 + thickness);
    } else {
        float exponent = b * thickness;
        float difference = one_minus_exp_negative(2.0 * exponent);
        float denominator = b * (2.0 - difference) + a * difference;
        layer_r = difference / denominator;
        layer_t = 2.0 * b * exp(-exponent) / denominator;
    }
    return layer_r + layer_t * layer_t * substrate / max(1.0 - layer_r * substrate, 1e-12);
}

float grain_hash(vec2 point) {
    // Stable spatial hash; no frame index, output resolution or time input.
    vec3 p = fract(vec3(point.xyx) * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float substrate_grain(vec2 point) {
    vec2 cell = floor(point);
    vec2 local = fract(point);
    vec2 weight = local * local * (3.0 - 2.0 * local);
    return mix(
        mix(grain_hash(cell), grain_hash(cell + vec2(1.0, 0.0)), weight.x),
        mix(grain_hash(cell + vec2(0.0, 1.0)), grain_hash(cell + vec2(1.0)), weight.x),
        weight.y
    ) * 2.0 - 1.0;
}

void main() {
    vec3 density = clamp(texture(u_paint, v_uv).rgb, vec3(0.0), vec3(1e6));
    vec3 strength = density * u_scattering;
    float scattering = strength.x + strength.y + strength.z;
    vec3 absorption = strength.x * absorption_ratio(u_pigment_0)
                    + strength.y * absorption_ratio(u_pigment_1)
                    + strength.z * absorption_ratio(u_pigment_2);
    vec3 ratio = absorption / (scattering > 0.0 ? scattering : 1.0);
    float thickness = scattering * u_layer_scale;
    vec3 reflectance = vec3(
        finite_layer(ratio.r, thickness, u_substrate.r),
        finite_layer(ratio.g, thickness, u_substrate.g),
        finite_layer(ratio.b, thickness, u_substrate.b)
    );
    // Minute roughness of a matte, stationary support. It remains fixed as the
    // paint passes over it; it must not be interpreted as advected pigment.
    float grain = substrate_grain(v_uv * u_grain_frequency);
    reflectance *= 1.0 + u_grain * grain;
    frag_color = vec4(clamp(reflectance, vec3(0.0), vec3(1.0)), 1.0);
}
