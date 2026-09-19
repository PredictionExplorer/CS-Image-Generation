#version 430
// Positive, actual-pigment-mass weighted auxiliary history transport.
// Neither unoccupied neighbors nor vacuum interpolation may contribute history.
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D origin_output;
layout(rgba32f,binding=1) writeonly uniform image2D state_output;
uniform sampler2D u_origin,u_state,u_paint,u_paint_other,u_velocity;
uniform ivec2 u_size;
uniform bool u_has_other;
uniform float u_aspect,u_domain,u_dt,u_minimum_concentration;

vec2 trace(vec2 uv) {
    vec2 scale=vec2(0.5/u_aspect,0.5)/u_domain;
    vec2 half_uv=uv-0.5*u_dt*texture(u_velocity,uv).xy*scale;
    return uv-u_dt*texture(u_velocity,half_uv).xy*scale;
}

void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    vec2 traced=clamp(trace((vec2(p)+.5)/vec2(u_size))*vec2(u_size)-.5,
                      vec2(0.),vec2(u_size-1));
    ivec2 base=ivec2(floor(traced));
    vec2 f=fract(traced);
    vec4 origin=vec4(0.),state=vec4(0.);
    float amount=0.;
    for(int y=0;y<2;y++)for(int x=0;x<2;x++) {
        ivec2 q=min(base+ivec2(x,y),u_size-1);
        float mass=dot(max(texelFetch(u_paint,q,0),vec4(0.)),vec4(1.));
        if(u_has_other)mass+=dot(max(texelFetch(u_paint_other,q,0),vec4(0.)),vec4(1.));
        float weight=(x==0?1.-f.x:f.x)*(y==0?1.-f.y:f.y);
        weight*=mass>u_minimum_concentration?mass:0.;
        origin+=texelFetch(u_origin,q,0)*weight;
        state+=texelFetch(u_state,q,0)*weight;
        amount+=weight;
    }
    if(amount>u_minimum_concentration) {
        origin/=amount;
        state/=amount;
        state.x=clamp(state.x,0.,1.);
        state.w=clamp(state.w,0.,1.);
        state.yz*=min(1.,state.x/max(length(state.yz),1e-20));
    } else { origin=vec4(0.);state=vec4(0.); }
    imageStore(origin_output,p,origin);
    imageStore(state_output,p,state);
}
