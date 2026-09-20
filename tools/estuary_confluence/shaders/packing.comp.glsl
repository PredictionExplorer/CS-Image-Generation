#version 430 core
// Versioned finite packing equilibrium. Pair writes are disjoint in each pass;
// every transfer has an equal opposite transfer, with no floating atomics.
layout(local_size_x=16,local_size_y=16) in;
uniform sampler2D u_geometry,u_paint;
uniform sampler2DArray u_phases,u_interaction;
#ifdef DIRECTIONAL_RELIEF
layout(rgba32f,binding=0) coherent uniform image2D potential;
uniform float u_directional_strength,u_directional_anisotropy;
#else
layout(rg32f,binding=0) coherent uniform image2D potential;
#endif
layout(r32f,binding=1) coherent uniform image2D film;
uniform int u_mode,u_axis,u_stride,u_parity,u_glazed;
uniform float u_strength,u_relaxation,u_height_scale,u_mass_threshold,u_mass_reference;
uniform float u_glaze_relief_strength;
const float FILM_FRACTION=0.12;
const int GROUPS=(PIGMENT_COUNT+3)/4;
layout(std430,binding=1) buffer PackingSummary{uint maximum_relative;};
shared uint group_maximum[256];

void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy),size=imageSize(film);
    if(u_mode==2){
        uint encoded=0u;
        if(all(lessThan(p,size))){
            float base=imageLoad(potential,p).r;
            float delta=imageLoad(film,p).r-base;
            float relative=base>0.0?FILM_FRACTION*delta/base:0.0;
            imageStore(film,p,vec4(relative,0,0,0));
            encoded=floatBitsToUint(max(relative,0.0));
        }
        uint index=gl_LocalInvocationIndex;
        group_maximum[index]=encoded;
        barrier();
        for(uint stride=128u;stride>0u;stride>>=1u){
            if(index<stride)group_maximum[index]=max(group_maximum[index],group_maximum[index+stride]);
            barrier();
        }
        if(index==0u)atomicMax(maximum_relative,group_maximum[0]);
        return;
    }
    if(any(greaterThanEqual(p,size)))return;
    if(u_mode==0){
        float mass=texelFetch(u_paint,p,0).a;
        float h=texelFetch(u_geometry,p,0).r*u_height_scale;
        if(u_glazed==1){
            float bank=smoothstep(0.6,2.2,mass/u_mass_reference);
            h*=mix(1.0,bank,u_glaze_relief_strength);
        }
        float base=mass>=u_mass_threshold?h*FILM_FRACTION:0.0;
        float upper=0.0,lower=0.0;
        for(int group=0;group<GROUPS;++group){
            vec4 a=texelFetch(u_phases,ivec3(p,2*GROUPS+group),0);
            vec4 b=texelFetch(u_phases,ivec3(p,group),0);
            for(int j=0;j<4;++j)if(4*group+j<PIGMENT_COUNT){upper+=a[j];lower+=b[j];}
        }
        float gu=texelFetch(u_interaction,ivec3(p,0),0).w;
        float gl=texelFetch(u_interaction,ivec3(p,1),0).w;
        float aggregate=gu==gl?gu:(gu*upper+gl*lower)/max(upper+lower,1e-20);
#ifdef DIRECTIONAL_RELIEF
        vec4 hu=texelFetch(u_interaction,ivec3(p,0),0);
        vec4 hl=texelFetch(u_interaction,ivec3(p,1),0);
        vec4 history=(hu*upper+hl*lower)/max(upper+lower,1e-20);
        float contact=clamp(history.x,0.0,1.0);
        vec2 fabric=contact>0.0?history.yz/sqrt(max(contact,1e-12)):vec2(0.0);
        imageStore(potential,p,vec4(base,aggregate,fabric));
#else
        imageStore(potential,p,vec4(base,aggregate,0,0));
#endif
        imageStore(film,p,vec4(base,0,0,0));
        return;
    }
    // CPU axes follow array order (axis0=y,axis1=x).
#ifdef DIRECTIONAL_RELIEF
    ivec2 unit_step=u_axis==0?ivec2(0,1):u_axis==1?ivec2(1,0)
                   :u_axis==2?ivec2(1,1):ivec2(1,-1);
    ivec2 offset=unit_step*u_stride;
    int coordinate=u_axis==0?p.y:p.x;
#else
    ivec2 offset=u_axis==1?ivec2(u_stride,0):ivec2(0,u_stride);
    int coordinate=u_axis==1?p.x:p.y;
#endif
    if((coordinate/u_stride)%2!=u_parity)return;
    ivec2 q=p+offset;
#ifdef DIRECTIONAL_RELIEF
    if(any(lessThan(q,ivec2(0))))return;
#endif
    if(any(greaterThanEqual(q,size)))return;
    vec2 a=imageLoad(potential,p).rg,b=imageLoad(potential,q).rg;
    if(a.x<=0.0||b.x<=0.0)return;
#ifndef DIRECTIONAL_RELIEF
    ivec2 unit_step=u_axis==1?ivec2(1,0):ivec2(0,1);
#endif
    // Coarse solver edges never cross even a one-cell gap in the actual paint.
    for(int i=1;i<u_stride;++i)if(imageLoad(potential,p+unit_step*i).r<=0.0)return;
#ifdef DIRECTIONAL_RELIEF
    if(u_axis>=2){
        // Supercover paths include both cells at each diagonal corner. No
        // affinity or film crosses a one-cell slit or a corner-only connection.
        for(int i=0;i<u_stride;++i){
            ivec2 r=p+unit_step*i;
            if(imageLoad(potential,r+ivec2(unit_step.x,0)).r<=0.0
              ||imageLoad(potential,r+ivec2(0,unit_step.y)).r<=0.0)return;
        }
    }
    if(u_mode==3){
        vec2 qa=imageLoad(potential,p).zw,qb=imageLoad(potential,q).zw;
        vec2 mean=(qa+qb)*0.5;
        float confidence=min(length(mean),1.0);
        vec2 d=vec2(unit_step);
        vec2 axial=vec2(d.x*d.x-d.y*d.y,2.0*d.x*d.y)/dot(d,d);
        float alignment=clamp(dot(mean,axial)/max(confidence,1e-12),-1.0,1.0);
        float along=0.5+0.5*alignment;along=along*along;along=along*along;
        float weight=0.5*u_relaxation*u_directional_strength*confidence
                    *mix(1.0,along,u_directional_anisotropy);
        precise float transfer=weight*(a.y-b.y);
        imageStore(potential,p,vec4(a.x,a.y-transfer,qa));
        imageStore(potential,q,vec4(b.x,b.y+transfer,qb));
        return;
    }
#endif
    float fa=imageLoad(film,p).r,fb=imageLoad(film,q).r;
    if(a.y==b.y&&fa==a.x&&fb==b.x)return;
    precise float difference=fa*b.x-fb*a.x;
    difference-=u_strength*a.x*b.x*(a.y-b.y);
    float transfer=u_relaxation*difference/(a.x+b.x);
    float low=max(fa-(1.0+u_strength)*a.x,(1.0-u_strength)*b.x-fb);
    float high=min(fa-(1.0-u_strength)*a.x,(1.0+u_strength)*b.x-fb);
    transfer=clamp(transfer,low,high);
    imageStore(film,p,vec4(fa-transfer,0,0,0));
    imageStore(film,q,vec4(fb+transfer,0,0,0));
}
