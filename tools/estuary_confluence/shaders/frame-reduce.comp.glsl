#version 430 core
// Exact equal-area 2x2 filter in linear display RGB, before PNG's transfer curve.
layout(local_size_x=16,local_size_y=16) in;
uniform sampler2D u_input;
layout(rgba32f,binding=0) writeonly uniform image2D reduced_output;
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,imageSize(reduced_output))))return;
    ivec2 q=p*2;
    // Match the top-down CPU image's row order; arithmetic stays float32.
    precise vec3 value=texelFetch(u_input,q+ivec2(0,1),0).rgb;
    value+=texelFetch(u_input,q+ivec2(1,1),0).rgb;
    value+=texelFetch(u_input,q,0).rgb;
    value+=texelFetch(u_input,q+ivec2(1,0),0).rgb;
    imageStore(reduced_output,p,vec4(value*.25,1));
}
