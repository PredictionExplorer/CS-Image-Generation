#version 430
// Authored contact chemistry and objective in-plane fabric, not particle physics.
// Reads pigment; never creates pigment, geometry, or renderer-space noise.
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D upper_output;
layout(rgba32f,binding=1) writeonly uniform image2D lower_output;
uniform sampler2D u_upper_origin,u_lower_origin,u_upper_state,u_lower_state;
uniform sampler2D u_upper,u_lower,u_upper_other,u_lower_other,u_carrier,u_velocity;
uniform ivec2 u_size;
uniform uvec4 u_seed;
uniform bool u_has_other;
uniform float u_domain,u_dt,u_lower_scale,u_minimum_concentration;
uniform float u_contact_rate,u_origin_distance,u_composition_threshold;
uniform float u_fabric_rate,u_fabric_relaxation,u_aggregation_rate,u_breakup_rate;
uniform float u_nucleation_scale,u_nucleation_contrast;
#ifdef MATERIAL_VARIATION
layout(rgba32f,binding=2) writeonly uniform image2D upper_trait_output;
layout(rgba32f,binding=3) writeonly uniform image2D lower_trait_output;
uniform sampler2D u_upper_traits,u_lower_traits;
uniform float u_trait_amplitude;
#endif

float relaxed(float x) {
    // Stable 1-exp(-x), including tiny canonical substeps in float32.
    if(x<.01)return x*(1.+x*(-.5+x*(1./6.+x*(-1./24.+x/120.))));
    return 1.-exp(-x);
}
float lattice(ivec2 p) {
    uint h=(uint(p.x)^u_seed.x)*0x9e3779b9u;
    h^=(uint(p.y)^u_seed.y)*0x85ebca6bu;
    h^=u_seed.z;
    h=(h^(h>>16u))*0x7feb352du;
    h=(h^(h>>15u))*0x846ca68bu;
    h^=(h>>16u)^u_seed.w;
    return float(h>>8u)/16777216.;
}
float nucleation(vec2 origin) {
    vec2 point=origin/u_nucleation_scale;
    ivec2 p=ivec2(floor(point));
    vec2 f=fract(point);f=f*f*(3.-2.*f);
    float n=mix(mix(lattice(p),lattice(p+ivec2(1,0)),f.x),
                mix(lattice(p+ivec2(0,1)),lattice(p+ivec2(1,1)),f.x),f.y);
    float islands=smoothstep(.3,.75,n);islands=islands*islands*islands;
    return mix(1.,islands,u_nucleation_contrast);
}
vec2 velocity(ivec2 p) {
    return texelFetch(u_velocity,clamp(p,ivec2(0),u_size-1),0).xy;
}
vec2 rotate(vec2 q,float angle) {
    float c=cos(angle),s=sin(angle);
    return vec2(c*q.x-s*q.y,s*q.x+c*q.y);
}
vec4 react(vec4 old,vec2 origin,float contact,float wet,vec2 strain,float spin
#ifdef MATERIAL_VARIATION
           ,vec2 traits
#endif
) {
    float dose=clamp(old.x+(1.-old.x)*relaxed(u_contact_rate*contact*wet*u_dt),0.,1.);
    float raw_magnitude=length(strain);
    vec2 axis=raw_magnitude>1e-20?strain/raw_magnitude:vec2(0.);
    float magnitude=min(40.,raw_magnitude);
    float align=u_fabric_rate*contact*wet*magnitude;
#ifdef MATERIAL_VARIATION
    vec2 variation=u_trait_amplitude*contact*wet*traits;
    align*=1.+variation.y;
#endif
    float relaxation=align+u_fabric_relaxation*wet;
    float blend=relaxed(relaxation*u_dt);
    float target_weight=relaxation>0.?align/relaxation:0.;
    // The axial representation rotates twice as far as physical material.
    // Each half of this symmetric split is spin*dt in axial coordinates.
    vec2 q=rotate(old.yz,spin*u_dt)*(1.-blend)+axis*(dose*target_weight*blend);
    q=rotate(q,spin*u_dt);
    q*=min(1.,dose/max(length(q),1e-20));
    float formation=u_aggregation_rate*contact*wet*nucleation(origin);
    float breakup=u_breakup_rate*wet*magnitude;
#ifdef MATERIAL_VARIATION
    formation*=1.+variation.x;
    breakup*=1.-variation.x;
#endif
    float rate=formation+breakup;
    float equilibrium=rate>0.?formation/rate:0.;
    float aggregate=clamp(old.w+(equilibrium-old.w)*relaxed(rate*u_dt),0.,1.);
    return vec4(dose,q,aggregate);
}
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    vec4 upper=max(texelFetch(u_upper,p,0),vec4(0.));
    vec4 lower=max(texelFetch(u_lower,p,0),vec4(0.));
    vec4 upper_other=vec4(0.),lower_other=vec4(0.);
    if(u_has_other) {
        upper_other=max(texelFetch(u_upper_other,p,0),vec4(0.));
        lower_other=max(texelFetch(u_lower_other,p,0),vec4(0.));
    }
    float a=dot(upper+upper_other,vec4(1.));
    float b=dot(lower+lower_other,vec4(1.));
    vec2 oa=texelFetch(u_upper_origin,p,0).xy;
    vec2 ob=texelFetch(u_lower_origin,p,0).xy;
    float contact=0.;
    if(a>u_minimum_concentration && b>u_minimum_concentration) {
        float difference=.5*(dot(abs(upper/a-lower/b),vec4(1.))
                            +dot(abs(upper_other/a-lower_other/b),vec4(1.)));
        float composition=smoothstep(u_composition_threshold,1.,difference);
        float origin_gate=smoothstep(u_origin_distance*.25,u_origin_distance,length(oa-ob));
        contact=max(origin_gate,composition)*2.*min(a,b)/(a+b);
    }
    float wet=clamp(texelFetch(u_carrier,p,0).x,0.,1.);
    float dx=2.*u_domain/float(u_size.y);
    vec2 vx=(velocity(p+ivec2(1,0))-velocity(p-ivec2(1,0)))/(2.*dx);
    vec2 vy=(velocity(p+ivec2(0,1))-velocity(p-ivec2(0,1)))/(2.*dx);
    vec2 strain=.5*vec2(vx.x-vy.y,vx.y+vy.x);
    float spin=.5*(vx.y-vy.x);
    vec4 sa=vec4(0.),sb=vec4(0.);
#ifdef MATERIAL_VARIATION
    vec2 ta=a>u_minimum_concentration?texelFetch(u_upper_traits,p,0).xy:vec2(0.);
    vec2 tb=b>u_minimum_concentration?texelFetch(u_lower_traits,p,0).xy:vec2(0.);
    if(a>u_minimum_concentration)sa=react(texelFetch(u_upper_state,p,0),oa,contact,wet,strain,spin,ta);
    if(b>u_minimum_concentration)sb=react(texelFetch(u_lower_state,p,0),ob,contact,wet,
                                       strain*u_lower_scale,spin*u_lower_scale,tb);
    imageStore(upper_trait_output,p,vec4(ta,0.,0.));
    imageStore(lower_trait_output,p,vec4(tb,0.,0.));
#else
    if(a>u_minimum_concentration)sa=react(texelFetch(u_upper_state,p,0),oa,contact,wet,strain,spin);
    if(b>u_minimum_concentration)sb=react(texelFetch(u_lower_state,p,0),ob,contact,wet,
                                       strain*u_lower_scale,spin*u_lower_scale);
#endif
    imageStore(upper_output,p,sa);
    imageStore(lower_output,p,sb);
}
