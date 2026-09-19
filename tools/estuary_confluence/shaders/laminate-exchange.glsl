#version 430
// Exact positive local color-fraction relaxation, at fixed layer amounts.
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D upper_output;
layout(rgba32f,binding=1) writeonly uniform image2D lower_output;
uniform sampler2D u_upper,u_lower,u_upper_other,u_lower_other,u_carrier;
uniform ivec2 u_size;
uniform bool u_has_other;
uniform float u_exposure,u_minimum;
float relaxed(float x){
    // Stable 1-exp(-x), preserving small finite exchange at many source steps.
    if(x<.01)return x*(1.+x*(-.5+x*(1./6.+x*(-1./24.+x/120.))));
    return 1.-exp(-x);
}
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    vec4 upper=texelFetch(u_upper,p,0),lower=texelFetch(u_lower,p,0);
    float a=dot(upper,vec4(1.)),b=dot(lower,vec4(1.));
    if(u_has_other){
        a+=dot(texelFetch(u_upper_other,p,0),vec4(1.));
        b+=dot(texelFetch(u_lower_other,p,0),vec4(1.));
    }
    float to_upper=0.,to_lower=0.;
    if(a>u_minimum && b>u_minimum){
        float wet=clamp(texelFetch(u_carrier,p,0).x,0.,1.);
        float q=relaxed(u_exposure*wet);
        to_upper=q*a/(a+b);to_lower=q*b/(a+b);
    }
    imageStore(upper_output,p,upper*(1.-to_lower)+lower*to_upper);
    imageStore(lower_output,p,lower*(1.-to_upper)+upper*to_lower);
}
