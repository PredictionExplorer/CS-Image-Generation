#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rg32f,binding=0) writeonly uniform image2D output_velocity;
layout(std430,binding=1) buffer Maximums {float maximums[];};
uniform sampler2D u_potential,u_source_velocity;
uniform ivec2 u_size;
uniform float u_domain;
shared float speeds[256];
float value(ivec2 p){return texelFetch(u_potential,clamp(p,ivec2(0),u_size-1),0).x;}
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    bool inside=all(lessThan(p,u_size));
    vec2 velocity=vec2(0.);
    if(inside) {
        float denominator=4.*u_domain/float(u_size.y);
        vec2 curl=vec2(value(p+ivec2(0,1))-value(p-ivec2(0,1)),value(p-ivec2(1,0))-value(p+ivec2(1,0)))/denominator;
        velocity=texelFetch(u_source_velocity,p,0).xy-curl;
        imageStore(output_velocity,p,vec4(velocity,0.,0.));
    }
    uint k=gl_LocalInvocationIndex;
    speeds[k]=inside?length(velocity):0.;
    barrier();
    for(uint stride=128u;stride>0u;stride/=2u){if(k<stride)speeds[k]=max(speeds[k],speeds[k+stride]);barrier();}
    if(k==0u)maximums[gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x]=speeds[0];
}
