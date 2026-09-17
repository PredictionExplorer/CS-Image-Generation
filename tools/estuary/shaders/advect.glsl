#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D destination;
uniform sampler2D u_input;
uniform sampler2D u_velocity;
uniform ivec2 u_size;
uniform float u_aspect;
uniform float u_domain;
uniform float u_dt;
vec2 trace(vec2 uv) {
    vec2 scale=vec2(0.5/u_aspect,0.5)/u_domain;
    vec2 half_uv=uv-0.5*u_dt*texture(u_velocity,uv).xy*scale;
    return uv-u_dt*texture(u_velocity,half_uv).xy*scale;
}
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size))) return;
    vec2 uv=(vec2(p)+0.5)/vec2(u_size);
    imageStore(destination,p,texture(u_input,trace(uv)));
}
