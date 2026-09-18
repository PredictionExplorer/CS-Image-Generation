#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(std430,binding=0) buffer PartialMass { vec4 partial_mass[]; };
uniform sampler2D u_input;
uniform ivec2 u_size;
shared vec4 masses[256];
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    uint k=gl_LocalInvocationIndex;
    masses[k]=all(lessThan(p,u_size))?texelFetch(u_input,p,0):vec4(0.);
    barrier();
    for(uint stride=128u;stride>0u;stride/=2u){
        if(k<stride)masses[k]+=masses[k+stride];
        barrier();
    }
    if(k==0u)partial_mass[gl_WorkGroupID.y*gl_NumWorkGroups.x+gl_WorkGroupID.x]=masses[0];
}
