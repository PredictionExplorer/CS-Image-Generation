#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D mobile_out;
layout(rgba32f,binding=1) writeonly uniform image2D sediment_out;
layout(rgba32f,binding=2) writeonly uniform image2D direction_out;
uniform sampler2D u_mobile,u_sediment,u_tooth,u_velocity,u_direction;
uniform ivec2 u_size;
uniform float u_aspect,u_domain,u_dt,u_time,u_drying,u_wetting,u_granulation,u_bank_strength,u_brush_radius,u_fade;
uniform vec3 u_settling,u_release,u_travel;
uniform vec4 u_segments[3];

float water(ivec2 p){return texelFetch(u_mobile,clamp(p,ivec2(0),u_size-1),0).a;}
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    vec2 uv=(vec2(p)+0.5)/vec2(u_size);
    vec4 m=texelFetch(u_mobile,p,0),d=texelFetch(u_sediment,p,0);
    vec4 tooth=texelFetch(u_tooth,p,0);
    vec2 velocity=texelFetch(u_velocity,p,0).xy;
    vec2 world=(uv*2.-1.)*vec2(u_aspect,1.)*u_domain;
    float fresh=0.;
    for(int i=0;i<3;i++){
        vec2 a=u_segments[i].xy,b=u_segments[i].zw,s=b-a;
        float t=clamp(dot(world-a,s)/max(dot(s,s),1e-20),0.,1.);
        float q=dot(world-mix(a,b,t),world-mix(a,b,t))/(u_brush_radius*u_brush_radius*4.);
        fresh+=u_travel[i]/max(u_brush_radius,1e-6)*(1.-smoothstep(.36,1.,q));
    }
    float drying=u_drying*(.35+.65*smoothstep(.15,.95,u_time))*tooth.y;
    float wet=clamp(m.a*exp(-drying*u_dt)+fresh*u_wetting*u_fade,0.,1.);
    vec2 gradient=vec2(water(p+ivec2(1,0))-water(p-ivec2(1,0)),water(p+ivec2(0,1))-water(p-ivec2(0,1)));
    // Scale the finite difference into physical canvas coordinates. A one-pixel
    // wet-front detector would otherwise change the process with resolution.
    float bank=clamp(length(gradient)*float(u_size.y)/(4.*u_domain)*.04,0.,1.);
    float dry=1.-wet;
    float grain=1.+u_granulation*(tooth.x*2.-1.);
    vec3 settle=u_settling*(.008+dry*dry*dry*2.6)
        *vec3(1.,mix(1.,grain,.35),grain*grain)
        *(1.+u_bank_strength*bank*vec3(.3,1.,1.8));
    float shear=length(velocity)/(1.+length(velocity));
    vec3 release=u_release*wet*wet*(.15+.85*shear);
    vec3 rate=settle+release;
    vec3 factor=vec3(0.);
    for(int i=0;i<3;i++){
        float x=rate[i]*u_dt;
        // expm1 series prevents cancellation at the small canonical timestep.
        float decay=x<.001?x*(1.-.5*x+x*x/6.):1.-exp(-x);
        factor[i]=rate[i]>0.?decay/rate[i]:u_dt;
    }
    vec3 transfer=(settle*m.rgb-release*d.rgb)*factor;
    vec3 next_m=max(m.rgb-transfer,vec3(0.)),next_d=max(d.rgb+transfer,vec3(0.));
    vec2 axial=vec2(velocity.x*velocity.x-velocity.y*velocity.y,2.*velocity.x*velocity.y);
    axial/=max(dot(velocity,velocity),1e-10);
    vec2 direction=texelFetch(u_direction,p,0).xy;
    float received=dot(max(transfer,vec3(0.)),vec3(1.));
    float old_mass=dot(d.rgb,vec3(1.));
    direction=mix(direction,axial,clamp(received/(old_mass+received+1e-8),0.,1.));
    imageStore(mobile_out,p,vec4(next_m,wet));
    imageStore(sediment_out,p,vec4(next_d,0.));
    imageStore(direction_out,p,vec4(direction,0.,0.));
}
