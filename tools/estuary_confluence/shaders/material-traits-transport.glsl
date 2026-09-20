#version 430
// Two initialized material properties move with actual old pigment mass.
// Phase 0: forward gather; 1: reverse gather weighted by forward support;
// 2: bounded MacCormack correction. Temporary z is a guide, never pigment.
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D trait_output;
uniform sampler2D u_field,u_forward,u_backward,u_paint,u_paint_other,u_velocity;
uniform ivec2 u_size;
uniform int u_phase;
uniform bool u_has_other,u_keep_support;
uniform float u_aspect,u_domain,u_dt,u_minimum_concentration;

vec2 velocity(vec2 uv) {
    vec2 point=clamp(uv*vec2(u_size)-.5,vec2(0.),vec2(u_size-1));
    ivec2 p=ivec2(floor(point)),q=min(p+ivec2(1),u_size-1);
    vec2 f=fract(point);
    return mix(mix(texelFetch(u_velocity,p,0).xy,
                   texelFetch(u_velocity,ivec2(q.x,p.y),0).xy,f.x),
               mix(texelFetch(u_velocity,ivec2(p.x,q.y),0).xy,
                   texelFetch(u_velocity,q,0).xy,f.x),f.y);
}
vec2 trace(vec2 uv) {
    vec2 scale=vec2(.5/u_aspect,.5)/u_domain;
    return uv-u_dt*velocity(uv-.5*u_dt*velocity(uv)*scale)*scale;
}
float paint_mass(ivec2 p) {
    float result=dot(max(texelFetch(u_paint,p,0),vec4(0.)),vec4(1.));
    if(u_has_other)result+=dot(max(texelFetch(u_paint_other,p,0),vec4(0.)),vec4(1.));
    return result;
}
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    if(u_phase==0 && u_dt==0.) {
        vec2 value=paint_mass(p)>u_minimum_concentration?texelFetch(u_field,p,0).xy:vec2(0.);
        imageStore(trait_output,p,vec4(value,0.,0.));return;
    }
    vec2 point=clamp(trace((vec2(p)+.5)/vec2(u_size))*vec2(u_size)-.5,
                     vec2(0.),vec2(u_size-1));
    ivec2 base=ivec2(floor(point));
    vec2 f=fract(point),sum=vec2(0.),low=vec2(3.402823e38),high=-low;
    float support=0.;
    for(int y=0;y<2;y++)for(int x=0;x<2;x++) {
        ivec2 q=min(base+ivec2(x,y),u_size-1);
        float amount=u_phase==1?max(texelFetch(u_field,q,0).z,0.):paint_mass(q);
        float weight=(x==0?1.-f.x:f.x)*(y==0?1.-f.y:f.y);
        if(amount>u_minimum_concentration && weight>0.) {
            vec2 value=texelFetch(u_field,q,0).xy;
            low=min(low,value);high=max(high,value);
            weight*=amount;sum+=value*weight;support+=weight;
        }
    }
    vec2 value=support>u_minimum_concentration?sum/support:vec2(0.);
    if(u_phase==2) {
        vec4 forward=texelFetch(u_forward,p,0),backward=texelFetch(u_backward,p,0);
        value=forward.xy;
        bool occupied=forward.z>u_minimum_concentration;
        if(occupied && paint_mass(p)>u_minimum_concentration
                    && backward.z>u_minimum_concentration && support>u_minimum_concentration)
            value=clamp(value+.5*(texelFetch(u_field,p,0).xy-backward.xy),low,high);
        if(!occupied)value=vec2(0.);
    }
    imageStore(trait_output,p,vec4(clamp(value,vec2(-1.),vec2(1.)),
        u_keep_support && support>u_minimum_concentration?support:0.,0.));
}
