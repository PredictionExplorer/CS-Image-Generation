#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D upper_output;
layout(rgba32f,binding=1) writeonly uniform image2D lower_output;
uniform sampler2D u_upper_state,u_lower_state,u_upper,u_lower,u_upper_other,u_lower_other,u_carrier,u_velocity;
uniform ivec2 u_size;
uniform bool u_has_other;
uniform float u_domain,u_minimum,u_dt,u_lower_scale,u_rebuild_rate,u_dry_rebuild_rate,u_breakdown_rate,u_shear_scale;
vec2 velocity(ivec2 p){return texelFetch(u_velocity,clamp(p,ivec2(0),u_size-1),0).xy;}
float evolve(float state,float wet,float shear,float amount) {
    if(amount<=u_minimum)return 0.;
    float rebuild=u_rebuild_rate+u_dry_rebuild_rate*(1.-wet);
    float breakdown=u_breakdown_rate*shear/(u_shear_scale+shear),rate=rebuild+breakdown;
    float equilibrium=rate>0.?rebuild/rate:state;
    // Exponential is bounded for any accepted dt; no explicit-Euler overshoot.
    return clamp(state+(equilibrium-state)*(1.-exp(-rate*u_dt)),0.,1.);
}
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    float a=dot(max(texelFetch(u_upper,p,0),vec4(0.)),vec4(1.)),b=dot(max(texelFetch(u_lower,p,0),vec4(0.)),vec4(1.));
    if(u_has_other){a+=dot(max(texelFetch(u_upper_other,p,0),vec4(0.)),vec4(1.));b+=dot(max(texelFetch(u_lower_other,p,0),vec4(0.)),vec4(1.));}
    float denominator=4.*u_domain/float(u_size.y);
    vec2 dx=(velocity(p+ivec2(1,0))-velocity(p-ivec2(1,0)))/denominator;
    vec2 dy=(velocity(p+ivec2(0,1))-velocity(p-ivec2(0,1)))/denominator;
    float shear=sqrt(2.*dx.x*dx.x+2.*dy.y*dy.y+(dx.y+dy.x)*(dx.y+dy.x));
    float wet=clamp(texelFetch(u_carrier,p,0).x,0.,1.);
    imageStore(upper_output,p,vec4(evolve(texelFetch(u_upper_state,p,0).x,wet,shear,a),0.,0.,0.));
    imageStore(lower_output,p,vec4(evolve(texelFetch(u_lower_state,p,0).x,wet,shear*u_lower_scale,b),0.,0.,0.));
}
