#version 430
// Exactly conservative local exchange; advection itself is interpolated.
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D mobile_out;
layout(rgba32f,binding=1) writeonly uniform image2D deposit_out;
layout(rgba32f,binding=2) writeonly uniform image2D underpaint_out;
uniform sampler2D u_mobile,u_deposit,u_underpaint,u_carrier,u_tooth,u_velocity;
uniform sampler2D u_mobile_other,u_deposit_other;
uniform bool u_has_other;
uniform ivec2 u_size;
uniform float u_domain,u_dt,u_shoreline_strength,u_granulation,u_underpaint_release,u_burial_rate;
uniform vec4 u_settling,u_release,u_grain;
float water(ivec2 p){return texelFetch(u_carrier,clamp(p,ivec2(0),u_size-1),0).x;}
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    vec4 m=texelFetch(u_mobile,p,0),d=texelFetch(u_deposit,p,0),u=texelFetch(u_underpaint,p,0);
    vec4 tooth=texelFetch(u_tooth,p,0);
    float wet=water(p),dry=1.-wet;
    float speed=length(texelFetch(u_velocity,p,0).xy),shear=speed/(1.+speed);
    vec2 gradient=vec2(water(p+ivec2(1,0))-water(p-ivec2(1,0)),water(p+ivec2(0,1))-water(p-ivec2(0,1)));
    float bank=clamp(length(gradient)*float(u_size.y)/(4.*u_domain)*.04,0.,1.);
    vec4 grain=1.+u_granulation*u_grain*(tooth.x*2.-1.);
    vec4 settle=u_settling*(.008+dry*dry*dry*2.6)*grain*grain
        *(1.+u_shoreline_strength*bank*(.3+u_grain*1.5));
    vec4 release=u_release*wet*wet*(.15+.85*shear);
    vec4 rate=settle+release,factor;
    for(int i=0;i<4;i++){
        float x=rate[i]*u_dt;
        float decay=x<.001?x*(1.-.5*x+x*x/6.):1.-exp(-x);
        factor[i]=rate[i]>0.?decay/rate[i]:u_dt;
    }
    vec4 transfer=(settle*m-release*d)*factor;
    m-=transfer;d+=transfer;
    // Thin exposed areas release old pigment into the mobile phase. Release
    // diminishes under a thick overlayer; it never creates or deletes pigment.
    float cover=dot(m+d,vec4(1.));
    if(u_has_other)cover+=dot(texelFetch(u_mobile_other,p,0)+texelFetch(u_deposit_other,p,0),vec4(1.));
    vec4 lift=u*(1.-exp(-u_release*u_underpaint_release*wet*wet*exp(-cover*8.)*u_dt));
    // Earlier settled paint matures into the retained lower layer. The
    // chronological strata arise from material history, not duplicate images.
    vec4 buried=d*(1.-exp(-u_burial_rate*dry*dry*u_dt));
    imageStore(mobile_out,p,max(m+lift,vec4(0.)));
    imageStore(deposit_out,p,max(d-buried,vec4(0.)));
    imageStore(underpaint_out,p,max(u-lift+buried,vec4(0.)));
}
