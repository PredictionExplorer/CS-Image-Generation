#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D output_delta;
uniform sampler2D u_guide,u_delta;
uniform ivec2 u_size;
uniform vec2 u_coefficients;
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    float result=0.;
    if(all(greaterThan(p,ivec2(0))) && all(lessThan(p,u_size-1))) {
        vec2 guide=texelFetch(u_guide,p,0).xy;
        float neighbors=u_coefficients.x*(texelFetch(u_delta,p+ivec2(1,0),0).x+texelFetch(u_delta,p-ivec2(1,0),0).x)
                       +u_coefficients.y*(texelFetch(u_delta,p+ivec2(0,1),0).x+texelFetch(u_delta,p-ivec2(0,1),0).x);
        result=(guide.x*guide.y+neighbors)/(1.+guide.y+2.*(u_coefficients.x+u_coefficients.y));
    }
    imageStore(output_delta,p,vec4(result,0.,0.,0.));
}
