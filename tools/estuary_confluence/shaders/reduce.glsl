#version 430
// Conservative box integration for display snapshots only. No state writes.
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D destination;
uniform sampler2D u_input;
uniform ivec2 u_output_size;
uniform int u_factor;
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_output_size)))return;
    vec4 sum=vec4(0.);
    for(int y=0;y<u_factor;y++)for(int x=0;x<u_factor;x++)sum+=texelFetch(u_input,p*u_factor+ivec2(x,y),0);
    imageStore(destination,p,sum/float(u_factor*u_factor));
}
