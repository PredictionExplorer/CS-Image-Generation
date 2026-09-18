#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D destination;
uniform sampler2D u_input;
uniform ivec2 u_size;
uniform vec4 u_factors;
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    // Positive uniform factors retain zero support and cannot create chalk.
    imageStore(destination,p,texelFetch(u_input,p,0)*u_factors);
}
