// White-albedo specialization derived from Portsmouth, Kutz and Hill (2024),
// EON eqs.10,15,19: https://arxiv.org/abs/2410.18026 . Independently implemented.
// Returns pi*BRDF; multiplying pigment reflectance preserves its hemisphere
// integral. This is an angular surface-scattering model, not spectral mixing.
const float EON_PI=3.141592653589793;
const float EON_C1=0.5-2.0/(3.0*EON_PI);
const float EON_C2=2.0/3.0-28.0/(15.0*EON_PI);

// Authored percolation-inspired optical response, not particle connectivity.
// Material g stays untouched; a unit exponent is exactly the original response.
float aggregate_response(float g,float contrast){
    if(contrast==1.0)return g;
    float ratio=min(g,1.0-g)/max(g,1.0-g);
    float power=pow(ratio,contrast);
    return g<=0.5?power/(1.0+power):1.0/(1.0+power);
}

float eon_albedo_deficit(float cosine) {
    float mu=clamp(cosine,0.0,1.0);
    float sine=sqrt(max(0.0,(1.0-mu)*(1.0+mu)));
    // Algebraically equivalent to FON's exact directional albedo, with the
    // removable grazing singularity and cancellation eliminated analytically.
    // Some hardware acos implementations lose ~7e-5 radians near grazing.
    // atan(sin,cos) evaluates the same angle with a stable endpoint and avoids
    // that error being amplified by the small directional-albedo deficit.
    float g=sine*(atan(sine,mu)-sine*mu)
        +(2.0/3.0)*(sine*mu*(1.0+sine+sine*sine)/(1.0+sine)-sine);
    return clamp(EON_C1-g/EON_PI,0.0,EON_C1);
}

float white_eon(float incoming_cosine,float outgoing_cosine,float tangent_dot,float roughness) {
    if(roughness==0.0)return 1.0;
    float denominator=max(incoming_cosine,outgoing_cosine);
    // Exactly grazing/grazing has zero projected measure. Positive angle
    // cosines retain the analytical lobe without an arbitrary brightness cap.
    if(denominator<=0.0)return 0.0;
    float angular=tangent_dot>0.0?tangent_dot/denominator:tangent_dot;
    float multiple=eon_albedo_deficit(incoming_cosine)*eon_albedo_deficit(outgoing_cosine)
        /(EON_C1-EON_C2);
    return (1.0+roughness*(angular+multiple))/(1.0+EON_C1*roughness);
}
