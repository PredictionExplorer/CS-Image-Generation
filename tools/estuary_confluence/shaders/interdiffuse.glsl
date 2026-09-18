#version 430
// Conservative pigment-fraction exchange inside a fixed local paint amount.
// One symmetric face flux acts oppositely on the two neighboring cells.
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D destination;
uniform sampler2D u_input,u_other,u_carrier;
uniform ivec2 u_size;
uniform bool u_has_other;
uniform float u_lambda,u_minimum;

float total(ivec2 p,vec4 local){
    float amount=dot(local,vec4(1.));
    if(u_has_other)amount+=dot(texelFetch(u_other,p,0),vec4(1.));
    return amount;
}

void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    vec4 center=texelFetch(u_input,p,0);
    float amount=total(p,center);
    float wet=clamp(texelFetch(u_carrier,p,0).x,0.,1.);
    vec4 change=vec4(0.);
    const ivec2 neighbors[4]=ivec2[4](ivec2(-1,0),ivec2(1,0),ivec2(0,-1),ivec2(0,1));
    if(amount>u_minimum && wet>0.){
        for(int k=0;k<4;k++){
            ivec2 q=p+neighbors[k];
            if(any(lessThan(q,ivec2(0))) || any(greaterThanEqual(q,u_size)))continue;
            vec4 other=texelFetch(u_input,q,0);
            float other_amount=total(q,other);
            float other_wet=clamp(texelFetch(u_carrier,q,0).x,0.,1.);
            if(other_amount>u_minimum && other_wet>0.){
                float face=u_lambda*min(wet,other_wet)*min(amount,other_amount);
                change+=face*(other/other_amount-center/amount);
            }
        }
    }
    // lambda<=.24 guarantees a positive convex update, with a margin below
    // the four-neighbor stability limit. No clipping or normalization is used.
    imageStore(destination,p,center+change);
}
