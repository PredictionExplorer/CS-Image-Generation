#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D output_potential;
uniform sampler2D u_delta;
uniform ivec2 u_size;
uniform float u_aspect,u_domain,u_flow_domain;
vec4 weights(float f) {
    float a=1.-f;
    return vec4(a*a*a,3.*f*f*f-6.*f*f+4.,-3.*f*f*f+3.*f*f+3.*f+1.,f*f*f)/6.;
}
float cubic(vec2 uv) {
    vec2 size=vec2(textureSize(u_delta,0)),p=uv*size-.5;
    vec2 base=floor(p),f=fract(p);
    vec4 x=weights(f.x),y=weights(f.y);
    vec2 gx=vec2(x.x+x.y,x.z+x.w),gy=vec2(y.x+y.y,y.z+y.w);
    vec2 px=vec2(base.x-1.+x.y/gx.x,base.x+1.+x.w/gx.y);
    vec2 py=vec2(base.y-1.+y.y/gy.x,base.y+1.+y.w/gy.y);
    return gx.x*gy.x*texture(u_delta,(vec2(px.x,py.x)+.5)/size).x
          +gx.y*gy.x*texture(u_delta,(vec2(px.y,py.x)+.5)/size).x
          +gx.x*gy.y*texture(u_delta,(vec2(px.x,py.y)+.5)/size).x
          +gx.y*gy.y*texture(u_delta,(vec2(px.y,py.y)+.5)/size).x;
}
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    vec2 uv=(vec2(p)+.5)/vec2(u_size),world=(uv*2.-1.)*vec2(u_aspect*u_domain,u_domain);
    // One central-curl cell plus a half-cell float-boundary margin prevents
    // correction in the stationary guard. Never mask velocity itself.
    vec2 extent=vec2(u_aspect*u_flow_domain,u_flow_domain)-vec2(3.*u_domain/float(u_size.y));
    vec2 t=max(vec2(0.),1.-world*world/(extent*extent));
    float potential=cubic(uv)*t.x*t.x*t.y*t.y;
    // Fixed outer ring makes the correction exactly no-through at domain edges.
    if(any(equal(p,ivec2(0))) || any(equal(p,u_size-1)))potential=0.;
    imageStore(output_potential,p,vec4(potential,0.,0.,0.));
}
