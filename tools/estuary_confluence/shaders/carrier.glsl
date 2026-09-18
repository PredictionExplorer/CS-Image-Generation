#version 430
// Carrier fields are transported with pigment: water, intimate mixing, axial direction.
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D destination;
uniform sampler2D u_input,u_tooth,u_velocity;
uniform ivec2 u_size;
uniform float u_aspect,u_domain,u_dt,u_time,u_drying,u_wetting,u_fade,u_brush,u_mixing_rate;
uniform vec4 u_segments[3],u_events[3]; // xy center, z radius, w integrated water dose
uniform vec3 u_travel;
vec2 velocity(ivec2 p){return texelFetch(u_velocity,clamp(p,ivec2(0),u_size-1),0).xy;}
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    vec4 c=texelFetch(u_input,p,0),tooth=texelFetch(u_tooth,p,0);
    vec2 world=((vec2(p)+.5)/vec2(u_size)*2.-1.)*vec2(u_aspect,1.)*u_domain;
    float fresh=0.;
    for(int i=0;i<3;i++){
        vec2 a=u_segments[i].xy,b=u_segments[i].zw,s=b-a;
        float t=clamp(dot(world-a,s)/max(dot(s,s),1e-20),0.,1.);
        vec2 delta=world-mix(a,b,t);
        float brush=1.-smoothstep(.36,1.,dot(delta,delta)/(u_brush*u_brush*4.));
        fresh+=u_wetting*u_fade*u_travel[i]/max(u_brush,1e-6)*brush;
        vec2 e=world-u_events[i].xy;
        fresh+=u_events[i].w*(1.-smoothstep(.04,1.,dot(e,e)/max(u_events[i].z*u_events[i].z,1e-8)));
    }
    float drying=u_drying*(.35+.65*smoothstep(.15,.95,u_time))*tooth.y;
    float wet=clamp(c.x*exp(-drying*u_dt)+fresh,0.,1.);
    float dx=2.*u_domain/float(u_size.y);
    vec2 vx=(velocity(p+ivec2(1,0))-velocity(p-ivec2(1,0)))/(2.*dx);
    vec2 vy=(velocity(p+ivec2(0,1))-velocity(p-ivec2(0,1)))/(2.*dx);
    float strain=min(12.,length(vec2(vx.x-vy.y,vx.y+vy.x)));
    // Finite-rate intimate mixing follows strain while wet. Dry paint retains
    // its previous state. Fresh solvent alone cannot unmix existing pigment.
    float rate=u_mixing_rate*wet*(.08+strain*.12);
    float intimate=1.-(1.-clamp(c.y,0.,1.))*exp(-rate*u_dt);
    vec2 v=velocity(p),axial=vec2(v.x*v.x-v.y*v.y,2.*v.x*v.y)/max(dot(v,v),1e-10);
    float follow=(1.-exp(-min(20.,length(v))*wet*u_dt));
    vec2 direction=mix(c.zw,axial,follow);
    imageStore(destination,p,vec4(wet,intimate,direction));
}
