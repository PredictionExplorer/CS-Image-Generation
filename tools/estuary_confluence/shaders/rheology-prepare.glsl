#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D guide_output;
layout(rgba32f,binding=1) writeonly uniform image2D delta_output;
uniform sampler2D u_upper_state,u_lower_state,u_upper,u_lower,u_upper_other,u_lower_other;
uniform ivec2 u_size,u_coarse_size;
uniform bool u_has_other;
uniform float u_aspect,u_domain,u_minimum,u_resistance_strength;
#ifdef RHEOLOGY_OCCUPANCY_REFERENCE
uniform float u_occupancy_mass_reference;
#endif
#ifdef RHEOLOGY_TRAITS
uniform sampler2D u_upper_trait,u_lower_trait,u_upper_contact,u_lower_contact;
uniform float u_trait_amplitude;
#endif
#include "rheology-streamfunction.glsl"
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_coarse_size)))return;
    // Disjoint integer cells cover every native pigment texel once.
    ivec2 lo=p*u_size/u_coarse_size,hi=(p+1)*u_size/u_coarse_size;
    float mass=0.,moment=0.;
#ifdef RHEOLOGY_TRAITS
    float affinity_moment=0.;
#endif
    for(int y=lo.y;y<hi.y;y++)for(int x=lo.x;x<hi.x;x++) {
        ivec2 q=ivec2(x,y);
        float a=dot(max(texelFetch(u_upper,q,0),vec4(0.)),vec4(1.));
        float b=dot(max(texelFetch(u_lower,q,0),vec4(0.)),vec4(1.));
        if(u_has_other) {
            a+=dot(max(texelFetch(u_upper_other,q,0),vec4(0.)),vec4(1.));
            b+=dot(max(texelFetch(u_lower_other,q,0),vec4(0.)),vec4(1.));
        }
        moment+=a*texelFetch(u_upper_state,q,0).x+b*texelFetch(u_lower_state,q,0).x;
#ifdef RHEOLOGY_TRAITS
        // These are transported material properties and recorded contact dose,
        // not freshly sampled texture noise. Both layers use actual paint mass.
        affinity_moment+=a*clamp(texelFetch(u_upper_trait,q,0).x,-1.,1.)
                           *clamp(texelFetch(u_upper_contact,q,0).x,0.,1.)
                        +b*clamp(texelFetch(u_lower_trait,q,0).x,-1.,1.)
                           *clamp(texelFetch(u_lower_contact,q,0).x,0.,1.);
#endif
        mass+=a+b;
    }
    float cells=float((hi.x-lo.x)*(hi.y-lo.y));
    float average_mass=mass/cells;
    float structure=mass>u_minimum?clamp(moment/mass,0.,1.):0.;
#ifdef RHEOLOGY_OCCUPANCY_REFERENCE
    // A paint-amount scale changes resistance only, retaining the independent
    // small concentration cutoff used to carry and rebuild material history.
    float resistance=u_resistance_strength*structure*structure*average_mass/(average_mass+u_occupancy_mass_reference);
#else
    float resistance=u_resistance_strength*structure*structure*average_mass/(average_mass+u_minimum);
#endif
#ifdef RHEOLOGY_TRAITS
    float affinity=mass>u_minimum?clamp(affinity_moment/mass,-1.,1.):0.;
    resistance*=1.+u_trait_amplitude*affinity;
#endif
    vec2 world=((vec2(p)+.5)/vec2(u_coarse_size)*2.-1.)*vec2(u_aspect*u_domain,u_domain);
    imageStore(guide_output,p,vec4(source_potential(world),resistance,0.,0.));
    imageStore(delta_output,p,vec4(0.));
}
