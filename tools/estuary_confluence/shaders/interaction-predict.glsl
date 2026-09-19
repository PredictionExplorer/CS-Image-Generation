#version 430
// Forward/reverse attribute predictor. Temporary origin.z carries the positive
// interpolated mass guide; this is never written into actual pigment density.
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D origin_output;
layout(rgba32f,binding=1) writeonly uniform image2D state_output;
uniform sampler2D u_origin,u_state,u_paint,u_paint_other,u_velocity;
uniform ivec2 u_size;
uniform bool u_has_other,u_use_mass_guide;
uniform float u_aspect,u_domain,u_dt,u_minimum_concentration;

vec2 velocity(vec2 uv) {
    // Hardware texture filtering may quantize interpolation weights. Explicit
    // float32 interpolation keeps the forward/reverse error estimate accurate.
    vec2 point=clamp(uv*vec2(u_size)-.5,vec2(0.),vec2(u_size-1));
    ivec2 p=ivec2(floor(point)),q=min(p+ivec2(1),u_size-1);
    vec2 f=fract(point);
    return mix(mix(texelFetch(u_velocity,p,0).xy,
                   texelFetch(u_velocity,ivec2(q.x,p.y),0).xy,f.x),
               mix(texelFetch(u_velocity,ivec2(p.x,q.y),0).xy,
                   texelFetch(u_velocity,q,0).xy,f.x),f.y);
}
vec2 trace(vec2 uv) {
    vec2 scale=vec2(0.5/u_aspect,0.5)/u_domain;
    vec2 middle=uv-.5*u_dt*velocity(uv)*scale;
    return uv-u_dt*velocity(middle)*scale;
}
float mass(ivec2 p) {
    if(u_use_mass_guide)return max(texelFetch(u_origin,p,0).z,0.);
    float amount=dot(max(texelFetch(u_paint,p,0),vec4(0.)),vec4(1.));
    if(u_has_other)amount+=dot(max(texelFetch(u_paint_other,p,0),vec4(0.)),vec4(1.));
    return amount;
}
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    vec2 point=clamp(trace((vec2(p)+.5)/vec2(u_size))*vec2(u_size)-.5,
                     vec2(0.),vec2(u_size-1));
    ivec2 base=ivec2(floor(point));
    vec2 f=fract(point),origin=vec2(0.);
    vec4 state=vec4(0.);
    float support=0.;
    for(int y=0;y<2;y++)for(int x=0;x<2;x++) {
        ivec2 q=min(base+ivec2(x,y),u_size-1);
        float amount=mass(q);
        float weight=(x==0?1.-f.x:f.x)*(y==0?1.-f.y:f.y);
        weight*=amount>u_minimum_concentration?amount:0.;
        origin+=texelFetch(u_origin,q,0).xy*weight;
        state+=texelFetch(u_state,q,0)*weight;
        support+=weight;
    }
    if(support>u_minimum_concentration) {
        origin/=support;state/=support;
    } else { origin=vec2(0.);state=vec4(0.);support=0.; }
    imageStore(origin_output,p,vec4(origin,support,0.));
    imageStore(state_output,p,state);
}
