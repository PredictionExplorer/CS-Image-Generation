#version 430
// Persistent paint amount/wetness and surface height/axial direction/roughness.
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D next_paint;
layout(rgba32f,binding=1) writeonly uniform image2D next_surface;
uniform sampler2D u_paint, u_surface;
uniform ivec2 u_size;
uniform float u_aspect,u_domain,u_width,u_drag,u_lift,u_deposit,u_dry,u_height,u_bristles,u_dt;
uniform int u_nocturne;
uniform vec4 u_segments[3];
uniform vec4 u_tools[3]; // contact pressure, loading fraction, original-knot arc travel
uniform vec2 u_directions[3];

uint scramble(uint x) {
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}
float hash(ivec2 p) {
    return float(scramble(uint(p.x)^scramble(uint(p.y)+0x9e3779b9u))&0xffffffu)/16777215.0;
}
float noise(vec2 p) {
    ivec2 cell=ivec2(floor(p));vec2 f=fract(p);f=f*f*(3.0-2.0*f);
    return mix(mix(hash(cell),hash(cell+ivec2(1,0)),f.x),
               mix(hash(cell+ivec2(0,1)),hash(cell+ivec2(1,1)),f.x),f.y);
}
float grain(vec2 p) {
    // Stationary aperiodic substrate tooth, never regenerated per frame.
    return 0.72*noise(p*180.0)+0.28*noise(p*43.0+vec2(17.0,-9.0));
}
void main() {
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(pixel,u_size)))return;
    vec2 uv=(vec2(pixel)+0.5)/vec2(u_size);
    vec2 world=(uv*2.0-1.0)*vec2(u_aspect,1.0)*u_domain;
    vec4 paint=texelFetch(u_paint,pixel,0);
    vec4 surface=texelFetch(u_surface,pixel,0);
    paint.a*=exp(-u_dry*u_dt);
    float tooth=grain(world);
    for(int i=0;i<3;i++) {
        float pressure=u_tools[i].x;
        if(pressure<0.00001)continue;
        vec2 a=u_segments[i].xy,b=u_segments[i].zw,delta=b-a;
        float t=clamp(dot(world-a,delta)/max(dot(delta,delta),1e-15),0.0,1.0);
        vec2 tangent=u_directions[i],normal=vec2(-tangent.y,tangent.x);
        vec2 offset=world-mix(a,b,t);
        float red_width=u_nocturne==1?0.68:0.28;
        float half_width=u_width*(0.35+0.65*sqrt(pressure))*(i==1?1.4:(i==2?red_width:1.0));
        float across=dot(offset,normal)/half_width;
        float along=dot(offset,tangent)/(half_width*0.55);
        float distance=pow(pow(abs(across),6.0)+pow(abs(along),6.0),1.0/6.0);
        if(distance>1.10)continue;
        float edge=1.0-smoothstep(0.92,1.00,distance+(tooth-0.5)*0.035);
        float travel=u_tools[i].z;
        float dose=travel/max(half_width,0.02);
        float load=u_tools[i].y;
        float fibre=dot(offset,normal);
        float bristle=0.7*noise(vec2(fibre*260.0,float(i)*17.0))
                     +0.3*noise(vec2(fibre*73.0,13.0+float(i)*7.0));
        float grooves=mix(1.0,0.24+0.76*bristle,u_bristles);
        // Preserve Nocturne's established surface response exactly.
        float broken=smoothstep(0.25,0.70,tooth+load*0.12+pressure*0.05);
        float contact=edge*pressure*mix(1.0,grooves*broken,u_nocturne==1?0.22:1.0);
        if(u_nocturne==0) {
            // Loaded paint bridges the support. Substrate breakup emerges as
            // the finite loading fraction falls, rather than perforating every
            // fresh stroke and accumulating a pelt-like pattern through drag.
            float dryness=pow(1.0-clamp(load,0.0,1.0),2.0);
            float groove_strength=u_bristles*(0.15+0.85*dryness);
            grooves=mix(1.0,0.24+0.76*bristle,groove_strength);
            broken=mix(1.0,broken,u_bristles*dryness);
            contact=edge*pressure*grooves*broken;
        }
        // The trace already contains the step displacement. Multiplying its
        // blend by travel would incorrectly make drag vanish as dt gets small.
        float shear=clamp(contact*(0.25+0.75*paint.a),0.0,1.0);
        vec2 donor=uv-delta*vec2(0.5/u_aspect,0.5)/u_domain*u_drag*contact;
        vec4 dragged_paint=texture(u_paint,donor);
        vec4 dragged_surface=texture(u_surface,donor);
        paint=mix(paint,dragged_paint,shear);
        surface=mix(surface,dragged_surface,shear);
        float lift=1.0-exp(-dose*contact*u_lift*(i==2?2.5:0.12));
        vec3 deposit=vec3(0.0);
        deposit[i]=u_deposit*dose*contact*load*(i==0?0.32:(i==1?0.76:0.038));
        if(u_nocturne==1) {
            // Carbon material is redistributed; the three channels describe
            // dark carbon, pale mineral traces and a restrained oxide residue.
            deposit*=vec3(0.72,0.08,0.18);
        }
        paint.rgb=paint.rgb*(1.0-lift)+deposit;
        paint.a=clamp(paint.a+0.45*dot(deposit,vec3(1.0))+contact*dose*0.06,0.0,1.0);
        float mark=1.0-exp(-dose*contact*(u_nocturne==1?4.0:2.0));
        vec2 axial=vec2(tangent.x*tangent.x-tangent.y*tangent.y,2.0*tangent.x*tangent.y);
        surface.yz=mix(surface.yz,axial,mark);
        float desired_roughness=(u_nocturne==1?0.29:0.48)+0.22*(1.0-load)+0.10*tooth;
        surface.w=mix(surface.w,desired_roughness,mark);
        // Physical relief follows deposited amount; bristle tooth only occurs
        // where the tool actually made a mark and persists in this state.
        float amount=dot(paint.rgb,vec3(0.30,1.0,0.45));
        float ridge=u_height*amount*(1.0+u_bristles*0.10*(bristle-0.5));
        surface.x=mix(surface.x,max(ridge,0.0),max(mark,lift));
    }
    surface.x=max(surface.x,0.0);
    surface.w=clamp(surface.w+u_dt*u_dry*0.012,0.08,0.95);
    float axial_length=length(surface.yz);
    if(axial_length>1.0)surface.yz/=axial_length;
    imageStore(next_paint,pixel,max(paint,vec4(0.0)));
    imageStore(next_surface,pixel,surface);
}
