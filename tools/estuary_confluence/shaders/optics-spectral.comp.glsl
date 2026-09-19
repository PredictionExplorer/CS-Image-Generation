#version 430 core
// Synthetic 38-band spectra are reconstructed once on CPU, never per pixel.
// Pigment mass and phase fractions remain those supplied by the simulation.
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D u_color_output;
uniform sampler2DArray u_phases;
uniform sampler2D u_mixing;
uniform float u_spectral_ks[PIGMENT_COUNT*38];
uniform float u_scattering[PIGMENT_COUNT];
uniform float u_substrate_spectrum[38];
uniform vec3 u_rgb_weights[38];
uniform vec3 u_substrate;
uniform float u_layer_scale,u_mix_control,u_mass_reference;
uniform int u_layered,u_crisp,u_glazed;
uniform float u_glaze_min_mass_ratio,u_glaze_max_mass_ratio;
const int GROUPS=(PIGMENT_COUNT+3)/4;

float one_minus_exp_negative(float x) {
    if(x<.01)return x*(1.+x*(-.5+x*(1./6.+x*(-1./24.+x/120.))));
    return 1.-exp(-x);
}
void finite_rt(float ratio,float thickness,out float r,out float t) {
    if(thickness<=0.){r=0.;t=1.;return;}
    float a=1.+ratio,b=sqrt(ratio*(ratio+2.));
    if(b<=1e-6){r=thickness/(1.+thickness);t=1./(1.+thickness);return;}
    float x=b*thickness,d=one_minus_exp_negative(2.*x);
    float denominator=b*(2.-d)+a*d;
    r=d/denominator;t=2.*b*exp(-x)/denominator;
}
vec3 gamut_map(vec3 rgb) {
    float neutral=clamp(dot(rgb,vec3(.2126,.7152,.0722)),0.,1.);
    vec3 delta=rgb-neutral;
    float amount=1.;
    for(int j=0;j<3;++j) {
        float bound=delta[j]<0. ? neutral/max(-delta[j],1e-30)
            : (1.-neutral)/max(delta[j],1e-30);
        amount=min(amount,bound);
    }
    return clamp(vec3(neutral)+amount*delta,0.,1.);
}
void main() {
    ivec2 xy=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(xy,imageSize(u_color_output))))return;
    float density[3*PIGMENT_COUNT];
    float total_mass=0.;
    for(int layer=0;layer<3;++layer) {
        for(int group=0;group<GROUPS;++group) {
            vec4 channels=texelFetch(u_phases,ivec3(xy,layer*GROUPS+group),0);
            for(int j=0;j<4;++j) {
                int i=group*4+j;
                if(i<PIGMENT_COUNT){density[layer*PIGMENT_COUNT+i]=channels[j];total_mass+=channels[j];}
            }
        }
    }
    if(total_mass<=1e-20) {
        imageStore(u_color_output,xy,u_crisp==1 ? vec4(0) : vec4(u_substrate,1));
        return;
    }
    int layer_count=u_layered==1 ? 3 : 1;
    if(u_layered==0) {
        for(int i=0;i<PIGMENT_COUNT;++i)density[i]+=density[PIGMENT_COUNT+i]+density[2*PIGMENT_COUNT+i];
    }
    float factor=u_crisp==1 ? u_mass_reference/total_mass : 1.;
    if(u_glazed==1)factor=clamp(total_mass,u_mass_reference*u_glaze_min_mass_ratio,
        u_mass_reference*u_glaze_max_mass_ratio)/total_mass;
    float phase_mass[3],phase_scatter[3];
    int phase_pigments[3];
    for(int layer=0;layer<layer_count;++layer) {
        phase_mass[layer]=0.;phase_scatter[layer]=0.;phase_pigments[layer]=0;
        for(int i=0;i<PIGMENT_COUNT;++i) {
            float amount=density[layer*PIGMENT_COUNT+i]*factor;
            density[layer*PIGMENT_COUNT+i]=amount;
            phase_mass[layer]+=amount;phase_scatter[layer]+=amount*u_scattering[i];
            if(amount>0.)phase_pigments[layer]+=1;
        }
    }
    float mixedness=mix(1.,clamp(texelFetch(u_mixing,xy,0).r,0.,1.),u_mix_control);
    vec3 rgb=vec3(0);
    for(int band=0;band<38;++band) {
        float color=u_substrate_spectrum[band];
        for(int layer=0;layer<layer_count;++layer) {
            if(phase_mass[layer]<=0.)continue;
            float absorption=0.;
            for(int i=0;i<PIGMENT_COUNT;++i) {
                absorption+=density[layer*PIGMENT_COUNT+i]*u_scattering[i]*u_spectral_ks[i*38+band];
            }
            float r,t;
            finite_rt(absorption/max(phase_scatter[layer],1e-30),phase_scatter[layer]*u_layer_scale,r,t);
            float intimate_color=r+t*t*color/max(1.-r*color,1e-12);
            // Compose each pure column with the lower reflector BEFORE area
            // averaging. Averaging R/T first would invent cross-column paths.
            if(mixedness<1. && phase_pigments[layer]>1) {
                float area_color=0.;
                for(int i=0;i<PIGMENT_COUNT;++i) {
                    float amount=density[layer*PIGMENT_COUNT+i];
                    if(amount<=0.)continue;
                    float pr,pt;
                    finite_rt(u_spectral_ks[i*38+band],phase_mass[layer]*u_scattering[i]*u_layer_scale,pr,pt);
                    float column=pr+pt*pt*color/max(1.-pr*color,1e-12);
                    area_color+=(amount/phase_mass[layer])*column;
                }
                color=mix(area_color,intimate_color,mixedness);
            } else color=intimate_color;
        }
        rgb+=clamp(color,0.,1.)*u_rgb_weights[band];
    }
    rgb=gamut_map(rgb);
    imageStore(u_color_output,xy,u_crisp==1 ? vec4(rgb*total_mass,total_mass) : vec4(rgb,1));
}
