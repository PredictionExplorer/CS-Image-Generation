// Analytic prescribed source potential; amplitude/conditioning matches flow.glsl.
uniform float u_flow_domain,u_radius,u_strength,u_pair_gain,u_pair_strain;
uniform vec4 u_tools[3];
uniform vec3 u_pairs[3],u_strains[3];
uniform vec2 u_carrier;
float source_potential(vec2 p) {
    float psi=u_carrier.x*p.y-u_carrier.y*p.x;
    float s2=u_radius*u_radius;
    for(int i=0;i<3;i++) {
        vec2 d=p-u_tools[i].xy,u=u_tools[i].zw;
        psi+=u_strength*(u.x*d.y-u.y*d.x)*exp(-.5*dot(d,d)/s2);
        vec2 z=p-u_pairs[i].xy;
        float ps2=2.89*s2,pg=exp(-.5*dot(z,z)/ps2);
        psi+=u_pair_gain*u_pairs[i].z*ps2*pg;
        if(u_pair_strain>0.) {
            vec2 axis=u_strains[i].xy,normal=vec2(-axis.y,axis.x);
            psi+=u_pair_strain*u_strains[i].z*pg*dot(z,axis)*dot(z,normal);
        }
    }
    vec2 t=max(vec2(0.),1.-p*p/(vec2(u_aspect*u_aspect,1.)*u_flow_domain*u_flow_domain));
    return psi*t.x*t.x*t.y*t.y;
}
