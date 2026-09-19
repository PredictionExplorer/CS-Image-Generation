#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rg32f, binding=0) writeonly uniform image2D velocity;
layout(std430, binding=1) buffer Maximums { float maximums[]; };
uniform ivec2 u_size;
uniform float u_aspect;
uniform float u_domain;
uniform float u_flow_domain;
uniform vec4 u_tools[3];
uniform vec3 u_pairs[3];
uniform float u_radius;
uniform float u_strength;
uniform float u_pair_gain;
uniform float u_pair_strain;
uniform vec3 u_strains[3]; // pair axis.xy and bounded signed extension rate
uniform vec2 u_carrier;
shared float speeds[256];

void main() {
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy);
    vec2 p=(vec2(pixel)+0.5)/vec2(u_size)*2.0-1.0;
    p*=vec2(u_aspect*u_domain,u_domain);
    float psi=u_carrier.x*p.y-u_carrier.y*p.x;
    vec2 v=u_carrier;
    float s2=u_radius*u_radius;
    for(int i=0;i<3;i++) {
        vec2 d=p-u_tools[i].xy;
        vec2 u=u_tools[i].zw;
        float g=exp(-0.5*dot(d,d)/s2);
        float q=u.x*d.y-u.y*d.x;
        psi+=u_strength*q*g;
        v+=u_strength*g*(u+vec2(-d.y,d.x)*q/s2);
        vec2 z=p-u_pairs[i].xy;
        float ps2=2.89*s2;
        float pg=exp(-0.5*dot(z,z)/ps2);
        float spin=u_pair_gain*u_pairs[i].z;
        psi+=spin*ps2*pg;
        v+=spin*pg*vec2(-z.y,z.x);
        if(u_pair_strain>0.0) {
            vec2 axis=u_strains[i].xy;
            vec2 normal=vec2(-axis.y,axis.x);
            float x=dot(z,axis),y=dot(z,normal);
            float amplitude=u_pair_strain*u_strains[i].z*pg;
            // A Gaussian quadrupole pulls along the real pair axis, and folds
            // across it. Differentiate the scalar rather than clamping velocity.
            psi+=amplitude*x*y;
            vec2 gradient=amplitude*(y*axis+x*normal-x*y/ps2*z);
            v+=vec2(gradient.y,-gradient.x);
        }
    }
    // The state texture retains its guard domain. The stream function has a
    // separate, compact support so pigment remains on the visible painting.
    // Both the function and its first derivative vanish at this boundary.
    float x=max(0.0,1.0-p.x*p.x/(u_aspect*u_aspect*u_flow_domain*u_flow_domain));
    float y=max(0.0,1.0-p.y*p.y/(u_flow_domain*u_flow_domain));
    float boundary=x*x*y*y;
    vec2 curl_boundary=vec2(-4.0*p.y*x*x*y,4.0*p.x*x*y*y/(u_aspect*u_aspect))/(u_flow_domain*u_flow_domain);
    v=v*boundary+psi*curl_boundary;
    bool inside=all(lessThan(pixel,u_size));
    if(inside) imageStore(velocity,pixel,vec4(v,0,0));
    uint k=gl_LocalInvocationIndex;
    speeds[k]=inside?length(v):0.0;
    barrier();
    for(uint stride=128u;stride>0u;stride/=2u) {
        if(k<stride) speeds[k]=max(speeds[k],speeds[k+stride]);
        barrier();
    }
    if(k==0u) maximums[gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x]=speeds[0];
}
