#version 430 core
// Read-only material capture. Every destination belongs to Surface, never Engine.
layout(local_size_x=16,local_size_y=16) in;
const int GROUPS=(PIGMENT_COUNT+3)/4;
uniform sampler2D u_phase[3*GROUPS]; // underpaint, deposit, mobile
uniform sampler2D u_carrier,u_tooth;
uniform float u_specific_volumes[PIGMENT_COUNT];
uniform float u_height_scale_mm,u_substrate_height_m;
uniform int u_chalk_index,u_material_model;
#ifdef INTERACTION_CAPTURE
uniform sampler2D u_origin_upper,u_origin_lower,u_interaction_upper,u_interaction_lower;
layout(rgba32f,binding=4) writeonly uniform image2DArray interaction_output;
#endif
layout(rgba32f,binding=0) writeonly uniform image2DArray phase_output;
layout(rgba32f,binding=1) writeonly uniform image2D geometry_output;
layout(rg32f,binding=2) writeonly uniform image2D direction_output;
layout(r32f,binding=3) writeonly uniform image2D mixing_output;
layout(std430,binding=0) buffer CaptureSummary { uint maximum_height; uint invalid_fields; };
shared uint group_heights[256];
shared uint group_errors[256];
bool finite(float value){return !isnan(value)&&!isinf(value);}
float coverage(float x){
    if(x<.01)return x*(1.+x*(-.5+x*(1./6.+x*(-1./24.+x/120.))));
    return 1.-exp(-x);
}
#ifdef INTERACTION_CAPTURE
bool valid_origin(vec4 value){
    return !any(isnan(value))&&!any(isinf(value))
        &&all(lessThanEqual(abs(value.xy),vec2(1e6)))&&all(equal(value.zw,vec2(0)));
}
bool valid_interaction(vec4 value){
    return !any(isnan(value))&&!any(isinf(value))
        &&value.x>=0.&&value.x<=1.&&value.w>=0.&&value.w<=1.
        &&length(value.yz)<=value.x+1e-5;
}
#endif
void main(){
    ivec2 p=ivec2(gl_GlobalInvocationID.xy),size=imageSize(geometry_output);
    uint local=gl_LocalInvocationIndex;
    uint invalid=0u,encoded_height=0u;
    if(all(lessThan(p,size))){
#ifdef INTERACTION_CAPTURE
        vec4 origin_upper=texelFetch(u_origin_upper,p,0),origin_lower=texelFetch(u_origin_lower,p,0);
        vec4 history_upper=texelFetch(u_interaction_upper,p,0),history_lower=texelFetch(u_interaction_lower,p,0);
        if(!valid_origin(origin_upper)||!valid_origin(origin_lower)
          ||!valid_interaction(history_upper)||!valid_interaction(history_lower))invalid=1u;
        imageStore(interaction_output,ivec3(p,0),history_upper);
        imageStore(interaction_output,ivec3(p,1),history_lower);
#endif
        float density[3*PIGMENT_COUNT];
        for(int phase=0;phase<3;++phase){
            for(int group=0;group<GROUPS;++group){
                vec4 value=texelFetch(u_phase[phase*GROUPS+group],p,0);
                for(int component=0;component<4;++component){
                    int channel=group*4+component;
                    if(channel<PIGMENT_COUNT){
                        float amount=value[component];
                        if(!finite(amount)||amount<0.||amount>1e6)invalid=1u;
                        density[phase*PIGMENT_COUNT+channel]=amount;
                    }else value[component]=0.;
                }
                imageStore(phase_output,ivec3(p,phase*GROUPS+group),value);
            }
        }
        precise float total=0.,dry_mass=0.,chalk=0.,mass_height=0.;
        for(int i=0;i<PIGMENT_COUNT;++i){
            float under=density[i],deposit=density[PIGMENT_COUNT+i],mobile=density[2*PIGMENT_COUNT+i];
            precise float pigment=(mobile+deposit)+under;
            if(!finite(pigment)||pigment>1e6)invalid=1u;
            total+=pigment;
            dry_mass+=u_material_model==1?deposit:deposit+under;
            if(i==u_chalk_index)chalk=pigment;
            precise float weighted=u_material_model==1?(mobile+under)*.22:(deposit+under)+mobile*.22;
            mass_height+=weighted*u_specific_volumes[i];
        }
        vec4 carrier=texelFetch(u_carrier,p,0),tooth=texelFetch(u_tooth,p,0);
        if(any(isnan(carrier))||any(isinf(carrier))||any(isnan(tooth))||any(isinf(tooth)))invalid=1u;
        float wet=clamp(carrier.x,0.,1.),mixed=clamp(carrier.y,0.,1.);
        precise float squared=carrier.z*carrier.z+carrier.w*carrier.w;
        float norm=sqrt(squared);
        vec2 direction=norm<1e-9?vec2(1,0):carrier.zw/max(norm,1e-9);
        float dry_share=total>1e-9?dry_mass/total:0.;
        float white=total>1e-9?chalk/total:0.;
        precise float height=mass_height*u_height_scale_mm;
        height*=.001;
        height+=u_substrate_height_m*tooth.a;
        precise float roughness=.3+.26*dry_share;
        roughness+=.11*white;
        roughness+=.06*tooth.x*dry_share;
        roughness-=.12*wet;
        roughness=clamp(roughness,.12,.88);
        float amount=coverage(total*3.);
        if(!finite(height)||height<0.||height>.05||!finite(amount))invalid=1u;
        encoded_height=finite(height)&&height>=0.?floatBitsToUint(height):0u;
        imageStore(geometry_output,p,vec4(height,wet,roughness,amount));
        imageStore(direction_output,p,vec4(direction,0,0));
        imageStore(mixing_output,p,vec4(mixed,0,0,0));
    }
    group_heights[local]=encoded_height;group_errors[local]=invalid;
    barrier();
    for(uint stride=128u;stride>0u;stride>>=1u){
        if(local<stride){group_heights[local]=max(group_heights[local],group_heights[local+stride]);group_errors[local]|=group_errors[local+stride];}
        barrier();
    }
    if(local==0u){atomicMax(maximum_height,group_heights[0]);atomicOr(invalid_fields,group_errors[0]);}
}
