#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D destination;
uniform sampler2D u_original,u_forward,u_backward,u_velocity;
uniform ivec2 u_size;
uniform float u_aspect,u_domain,u_dt,u_brush;
uniform vec4 u_segments[3],u_doses[3];
uniform bool u_signed;
vec2 trace(vec2 uv){
    vec2 scale=vec2(.5/u_aspect,.5)/u_domain;
    vec2 half_uv=uv-.5*u_dt*texture(u_velocity,uv).xy*scale;
    return uv-u_dt*texture(u_velocity,half_uv).xy*scale;
}
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    vec2 uv=(vec2(p)+.5)/vec2(u_size);
    ivec2 donor=ivec2(floor(trace(uv)*vec2(u_size)-.5));
    vec4 lo=vec4(1e30),hi=vec4(-1e30);
    for(int y=0;y<2;y++)for(int x=0;x<2;x++){
        vec4 c=texelFetch(u_original,clamp(donor+ivec2(x,y),ivec2(0),u_size-1),0);
        lo=min(lo,c);hi=max(hi,c);
    }
    vec4 c=texelFetch(u_forward,p,0)+.5*(texelFetch(u_original,p,0)-texelFetch(u_backward,p,0));
    c=clamp(c,lo,hi);
    vec2 world=(uv*2.-1.)*vec2(u_aspect,1.)*u_domain;
    for(int i=0;i<3;i++){
        vec2 a=u_segments[i].xy,b=u_segments[i].zw,s=b-a;
        float t=clamp(dot(world-a,s)/max(dot(s,s),1e-20),0.,1.);
        vec2 delta=world-mix(a,b,t);
        c+=u_doses[i]*(1.-smoothstep(.36,1.,dot(delta,delta)/(u_brush*u_brush)));
    }
    imageStore(destination,p,u_signed?c:max(c,vec4(0.)));
}
