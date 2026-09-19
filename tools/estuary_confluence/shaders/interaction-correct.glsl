#version 430
// Bounded MacCormack history correction. Only occupied old donors supply bounds.
// No correction is applied through missing old-target/backward support. Origins
// leave this pass as xy/0/0: the temporary mass guide is not retained as history.
layout(local_size_x=16,local_size_y=16) in;
layout(rgba32f,binding=0) writeonly uniform image2D origin_output;
layout(rgba32f,binding=1) writeonly uniform image2D state_output;
uniform sampler2D u_origin,u_state,u_forward_origin,u_forward_state;
uniform sampler2D u_backward_origin,u_backward_state,u_paint,u_paint_other,u_velocity;
uniform ivec2 u_size;
uniform bool u_has_other;
uniform float u_aspect,u_domain,u_dt,u_minimum_concentration;

vec2 velocity(vec2 uv) {
    // Match the predictor's unquantized midpoint velocity interpolation.
    vec2 point=clamp(uv*vec2(u_size)-.5,vec2(0.),vec2(u_size-1));
    ivec2 p=ivec2(floor(point)),q=min(p+ivec2(1),u_size-1);
    vec2 f=fract(point);
    return mix(mix(texelFetch(u_velocity,p,0).xy,
                   texelFetch(u_velocity,ivec2(q.x,p.y),0).xy,f.x),
               mix(texelFetch(u_velocity,ivec2(p.x,q.y),0).xy,
                   texelFetch(u_velocity,q,0).xy,f.x),f.y);
}
vec2 trace(vec2 uv) {
    vec2 scale=vec2(0.5/u_aspect,0.5)/u_domain;
    vec2 middle=uv-.5*u_dt*velocity(uv)*scale;
    return uv-u_dt*velocity(middle)*scale;
}
float mass(ivec2 p) {
    float amount=dot(max(texelFetch(u_paint,p,0),vec4(0.)),vec4(1.));
    if(u_has_other)amount+=dot(max(texelFetch(u_paint_other,p,0),vec4(0.)),vec4(1.));
    return amount;
}
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(any(greaterThanEqual(p,u_size)))return;
    vec4 forward_origin=texelFetch(u_forward_origin,p,0);
    vec4 backward_origin=texelFetch(u_backward_origin,p,0);
    vec2 origin=forward_origin.xy;
    vec4 state=texelFetch(u_forward_state,p,0);
    bool occupied=forward_origin.z>u_minimum_concentration;
    if(occupied && mass(p)>u_minimum_concentration
                && backward_origin.z>u_minimum_concentration) {
        vec2 point=clamp(trace((vec2(p)+.5)/vec2(u_size))*vec2(u_size)-.5,
                         vec2(0.),vec2(u_size-1));
        ivec2 base=ivec2(floor(point));
        vec2 f=fract(point),origin_low=vec2(3.402823e38),origin_high=-origin_low;
        vec4 state_low=vec4(3.402823e38),state_high=-state_low;
        bool donor_found=false;
        for(int y=0;y<2;y++)for(int x=0;x<2;x++) {
            ivec2 q=min(base+ivec2(x,y),u_size-1);
            float weight=(x==0?1.-f.x:f.x)*(y==0?1.-f.y:f.y);
            if(weight>0. && mass(q)>u_minimum_concentration) {
                vec2 donor_origin=texelFetch(u_origin,q,0).xy;
                vec4 donor_state=texelFetch(u_state,q,0);
                origin_low=min(origin_low,donor_origin);origin_high=max(origin_high,donor_origin);
                state_low=min(state_low,donor_state);state_high=max(state_high,donor_state);
                donor_found=true;
            }
        }
        if(donor_found) {
            origin=clamp(origin+.5*(texelFetch(u_origin,p,0).xy-backward_origin.xy),
                         origin_low,origin_high);
            state=clamp(state+.5*(texelFetch(u_state,p,0)-texelFetch(u_backward_state,p,0)),
                        state_low,state_high);
        }
    }
    if(occupied) {
        state.x=clamp(state.x,0.,1.);state.w=clamp(state.w,0.,1.);
        state.yz*=min(1.,state.x/max(length(state.yz),1e-20));
    } else { origin=vec2(0.);state=vec4(0.); }
    imageStore(origin_output,p,vec4(origin,0.,0.));
    imageStore(state_output,p,state);
}
