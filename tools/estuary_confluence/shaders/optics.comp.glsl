#version 430 core
// PIGMENT_COUNT is inserted after this version line by the archive-bound adapter.
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f, binding=0) writeonly uniform image2D u_color_output;
uniform sampler2DArray u_phases;
uniform sampler2D u_mixing;
uniform vec3 u_ratios[PIGMENT_COUNT];
uniform float u_scattering[PIGMENT_COUNT];
uniform vec3 u_substrate;
uniform float u_layer_scale, u_mix_control, u_mass_reference;
uniform int u_layered, u_crisp, u_glazed;
uniform float u_glaze_min_mass_ratio, u_glaze_max_mass_ratio;
const int GROUPS = (PIGMENT_COUNT + 3) / 4;

float one_minus_exp_negative(float x) {
    if (x < 0.01) return x * (1.0 + x * (-0.5 + x * (1.0/6.0 + x * (-1.0/24.0 + x/120.0))));
    return 1.0 - exp(-x);
}
void finite_rt(float ratio, float thickness, out float r, out float t) {
    if (thickness <= 0.0) { r=0.0; t=1.0; return; }
    float a=1.0+ratio, b=sqrt(ratio*(ratio+2.0));
    if (b <= 1e-6) { r=thickness/(1.0+thickness); t=1.0/(1.0+thickness); return; }
    float x=b*thickness;
    float d=one_minus_exp_negative(2.0*x);
    float denominator=b*(2.0-d)+a*d;
    r=d/denominator;
    t=2.0*b*exp(-x)/denominator;
}
void rgb_rt(vec3 ratios, float thickness, out vec3 r, out vec3 t) {
    for (int j=0; j<3; ++j) finite_rt(ratios[j], thickness, r[j], t[j]);
}
void phase_density(ivec2 xy, int layer, out float density[PIGMENT_COUNT]) {
    for (int group=0; group<GROUPS; ++group) {
        vec4 channels=texelFetch(u_phases, ivec3(xy,layer*GROUPS+group),0);
        for (int j=0; j<4; ++j) {
            int i=group*4+j;
            if (i<PIGMENT_COUNT) density[i]=channels[j];
        }
    }
}
void layer_rt(float density[PIGMENT_COUNT], float mixedness, out vec3 r, out vec3 t) {
    float mass=0.0, scattering=0.0;
    vec3 absorption=vec3(0);
    for (int i=0; i<PIGMENT_COUNT; ++i) {
        mass+=density[i];
        float amount=density[i]*u_scattering[i];
        scattering+=amount;
        absorption+=amount*u_ratios[i];
    }
    rgb_rt(absorption/max(scattering,1e-30),scattering*u_layer_scale,r,t);
    if (mixedness>=1.0 || mass<=0.0) return;
    vec3 area_r=vec3(0),area_t=vec3(0);
    for (int i=0; i<PIGMENT_COUNT; ++i) {
        vec3 pr,pt;
        rgb_rt(u_ratios[i],mass*u_scattering[i]*u_layer_scale,pr,pt);
        float fraction=density[i]/mass;
        area_r+=fraction*pr;
        area_t+=fraction*pt;
    }
    r=mix(area_r,r,mixedness);
    t=mix(area_t,t,mixedness);
}
vec3 add_layer(vec3 bottom,vec3 r,vec3 t) {
    return r+t*t*bottom/max(vec3(1)-r*bottom,vec3(1e-12));
}
void main() {
    ivec2 xy=ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(xy,imageSize(u_color_output)))) return;
    float mixedness=mix(1.0,clamp(texelFetch(u_mixing,xy,0).r,0.0,1.0),u_mix_control);
    float density[PIGMENT_COUNT];
    // Crisp is an optical interpretation, never a change to the physical state.
    // Keep the real mass in alpha so the contour cannot depend on palette RGB.
    float total_mass=0.0;
    if (u_crisp==1) {
        for (int layer=0;layer<3;++layer) {
            phase_density(xy,layer,density);
            for (int i=0;i<PIGMENT_COUNT;++i) total_mass+=density[i];
        }
        if (total_mass<=1e-20) {
            imageStore(u_color_output,xy,vec4(0));
            return;
        }
    }
    float density_scale=u_crisp==1 ? u_mass_reference/total_mass : 1.0;
    if (u_glazed==1) density_scale=clamp(total_mass,
        u_mass_reference*u_glaze_min_mass_ratio,
        u_mass_reference*u_glaze_max_mass_ratio)/total_mass;
    vec3 color=u_substrate;
    if (u_layered==1) {
        for (int layer=0;layer<3;++layer) {
            phase_density(xy,layer,density);
            if (u_crisp==1) {
                for (int i=0;i<PIGMENT_COUNT;++i) density[i]*=density_scale;
            }
            vec3 r,t;
            layer_rt(density,mixedness,r,t);
            color=add_layer(color,r,t);
        }
    } else {
        for (int i=0;i<PIGMENT_COUNT;++i) density[i]=0.0;
        for (int layer=0;layer<3;++layer) {
            float phase[PIGMENT_COUNT];
            phase_density(xy,layer,phase);
            for (int i=0;i<PIGMENT_COUNT;++i) density[i]+=phase[i];
        }
        if (u_crisp==1) {
            for (int i=0;i<PIGMENT_COUNT;++i) density[i]*=density_scale;
        }
        vec3 r,t;
        layer_rt(density,mixedness,r,t);
        color=add_layer(color,r,t);
    }
    color=clamp(color,0.0,1.0);
    // Mass-premultiplied color avoids a substrate-colored interpolation fringe
    // next to an empty texel. The surface unpremultiplies before illumination.
    imageStore(u_color_output,xy,u_crisp==1 ? vec4(color*total_mass,total_mass) : vec4(color,1.0));
}
