#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D output_state;
uniform sampler2D u_state,u_paint,u_paint_other,u_velocity;
uniform ivec2 u_size;
uniform bool u_has_other;
uniform float u_aspect,u_domain,u_dt,u_minimum;
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    vec2 uv=(vec2(p)+.5)/vec2(u_size),scale=vec2(.5/u_aspect,.5)/u_domain;
    vec2 half_uv=uv-.5*u_dt*texture(u_velocity,uv).xy*scale;
    vec2 trace=clamp((uv-u_dt*texture(u_velocity,half_uv).xy*scale)*vec2(u_size)-.5,vec2(0.),vec2(u_size-1));
    ivec2 base=ivec2(floor(trace));vec2 f=fract(trace);
    float state=0.,amount=0.;
    for(int y=0;y<2;y++)for(int x=0;x<2;x++) {
        ivec2 q=min(base+ivec2(x,y),u_size-1);
        float mass=dot(max(texelFetch(u_paint,q,0),vec4(0.)),vec4(1.));
        if(u_has_other)mass+=dot(max(texelFetch(u_paint_other,q,0),vec4(0.)),vec4(1.));
        float weight=(x==0?1.-f.x:f.x)*(y==0?1.-f.y:f.y)*(mass>u_minimum?mass:0.);
        state+=texelFetch(u_state,q,0).x*weight;amount+=weight;
    }
    imageStore(output_state,p,vec4(amount>u_minimum?clamp(state/amount,0.,1.):0.,0.,0.,0.));
}
