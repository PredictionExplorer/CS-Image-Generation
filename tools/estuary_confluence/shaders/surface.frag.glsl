#version 430 core
// Actual single-valued surface intersection with analytical direct lighting.
// The BRDF and finite-layer RGB pigment optics are approximations, not a path trace.
in vec2 screen_position;
out vec4 frag_color;
uniform sampler2D u_paint, u_geometry, u_finish;
uniform sampler2DArray u_phases;
uniform float u_scattering[PIGMENT_COUNT];
uniform vec3 u_substrate, u_ground;
uniform vec2 u_visible_size, u_full_size, u_grid_size, u_output_size;
uniform vec3 u_camera_right, u_camera_up, u_camera_view, u_key_direction;
uniform float u_height_scale, u_max_height;
uniform float u_ambient, u_key_strength, u_fill_strength, u_anisotropy;
uniform float u_roughness_scale, u_roughness_bias;
uniform float u_grain_height, u_grain_scale, u_shadow_strength, u_occlusion_strength, u_exposure;
uniform int u_tone_map, u_crisp, u_glazed;
uniform float u_mass_threshold, u_mass_reference, u_glaze_relief_strength;
#ifdef PACKING_SURFACE
uniform sampler2D u_packing;
#endif
#ifdef INTERACTION_SURFACE
uniform sampler2DArray u_interaction; // Surface-owned upper/lower frozen histories
uniform float u_silk_strength, u_grain_strength;
#ifdef GRAIN_CONTRAST
uniform float u_grain_contrast;
#endif
// INTERACTION_SCATTERING
#endif
const float PI = 3.141592653589793;
const int GROUPS = (PIGMENT_COUNT + 3) / 4;

vec2 field_uv(vec2 p) { return p / u_full_size + 0.5; }
bool inside(vec2 uv) { return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0))); }
#ifdef PACKING_SURFACE
float packed_height(float legacy_height,vec2 uv){
    return legacy_height*(1.0+textureLod(u_packing,uv,0.0).r);
}
#else
float packed_height(float legacy_height,vec2 uv){return legacy_height;}
#endif
float surface_height(vec2 p) {
    vec2 uv = field_uv(p);
    if (!inside(uv)) return 0.0;
    if (u_crisp==1 && textureLod(u_paint,uv,0.0).a<u_mass_threshold) return 0.0;
    if (u_glazed==1) {
        float ratio=textureLod(u_paint,uv,0.0).a/u_mass_reference;
        // Only physically denser paint banks keep the full authored relief.
        // This is a bounded appearance transform; the archived height is intact.
        float bank=smoothstep(0.6,2.2,ratio);
        float scale=mix(1.0,bank,u_glaze_relief_strength);
        return packed_height(textureLod(u_geometry,uv,0.0).r*u_height_scale*scale,uv);
    }
    return packed_height(textureLod(u_geometry, uv, 0.0).r * u_height_scale,uv);
}

void upper_material(vec2 uv, float total_mass, out float share, out float porosity) {
    float upper_mass=0.0,scatter=0.0;
    for (int group=0;group<GROUPS;++group) {
        vec4 amount=textureLod(u_phases,vec3(uv,2*GROUPS+group),0.0);
        for (int j=0;j<4;++j) {
            int i=4*group+j;
            if (i<PIGMENT_COUNT) {
                upper_mass+=amount[j];
                scatter+=amount[j]*u_scattering[i];
            }
        }
    }
    share=clamp(upper_mass/max(total_mass,1e-20),0.0,1.0);
    float strength=scatter/max(upper_mass,1e-20);
    porosity=strength/(1.0+strength);
}

#ifdef INTERACTION_SURFACE
vec4 material_interaction(vec2 uv) {
    float upper=0.0, lower=0.0;
    for (int group=0;group<GROUPS;++group) {
        vec4 a=textureLod(u_phases,vec3(uv,2*GROUPS+group),0.0);
        vec4 b=textureLod(u_phases,vec3(uv,group),0.0);
        for (int j=0;j<4;++j) if (4*group+j<PIGMENT_COUNT) {
            upper+=a[j]; lower+=b[j];
        }
    }
    // Material fractions control which history is visible. Empty support never
    // acquires an interaction response, including at interpolated boundaries.
    float total=upper+lower;
    if(total<=1e-20)return vec4(0);
    return (textureLod(u_interaction,vec3(uv,0),0.0)*upper
           +textureLod(u_interaction,vec3(uv,1),0.0)*lower)/total;
}
#endif

vec3 hit_surface(vec3 origin, vec3 view) {
    // A frontal camera hits the exact pixel column; tilted views march through
    // the tight physical height interval. This includes self-occlusion/parallax.
    if (dot(view.xy, view.xy) < 1e-12) return vec3(origin.xy, surface_height(origin.xy));
    vec3 start = origin + view * ((u_max_height - origin.z) / view.z);
    vec3 finish = origin - view * (origin.z / view.z);
    vec3 upper = start;
    vec3 lower = finish;
    for (int i = 1; i <= 128; ++i) {
        vec3 point = mix(start, finish, float(i) / 128.0);
        if (point.z <= surface_height(point.xy)) { lower = point; break; }
        upper = point;
    }
    for (int i = 0; i < 8; ++i) {
        vec3 point = (upper + lower) * 0.5;
        if (point.z > surface_height(point.xy)) upper = point;
        else lower = point;
    }
    return (upper + lower) * 0.5;
}

float hash21(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}
float support_noise(vec2 p) {
    vec2 cell = floor(p), f = fract(p);
    vec2 w = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(cell), hash21(cell + vec2(1, 0)), w.x),
               mix(hash21(cell + vec2(0, 1)), hash21(cell + 1.0), w.x), w.y) * 2.0 - 1.0;
}
vec3 normal_at(vec3 point, float coverage) {
    vec2 e = u_full_size / u_grid_size;
    vec2 slope = vec2(
        (surface_height(point.xy + vec2(e.x, 0)) - surface_height(point.xy - vec2(e.x, 0))) / (2.0 * e.x),
        (surface_height(point.xy + vec2(0, e.y)) - surface_height(point.xy - vec2(0, e.y))) / (2.0 * e.y));
    float footprint = max(u_visible_size.x / u_output_size.x, u_visible_size.y / u_output_size.y) / u_camera_view.z;
    float attenuation = 1.0 - smoothstep(u_grain_scale * 0.2, u_grain_scale * 0.65, footprint);
    float amplitude = u_grain_height / u_grain_scale * attenuation * mix(1.0, 0.3, coverage);
    vec2 grain_point = point.xy / u_grain_scale;
    slope += amplitude * vec2(support_noise(grain_point + vec2(0.1, 0)) - support_noise(grain_point - vec2(0.1, 0)),
                               support_noise(grain_point + vec2(0, 0.1)) - support_noise(grain_point - vec2(0, 0.1))) / 0.2;
    return normalize(vec3(-slope, 1.0));
}

// Phase optics are evaluated once at material resolution, then retained on GPU.
vec3 paint_color(vec2 uv) {
    vec4 paint=textureLod(u_paint,uv,0.0);
    return u_crisp==1 ? paint.rgb/max(paint.a,1e-20) : paint.rgb;
}

float shadow(vec3 point, vec3 light) {
    float pixel = max(u_full_size.x / u_grid_size.x, u_full_size.y / u_grid_size.y);
    float travel = min((u_max_height - point.z + pixel) / light.z, 0.035);
    float visibility = 1.0;
    // Increasing step lengths concentrate samples around small paint ridges.
    // The height clearance estimates a broad emitter's penumbra deterministically.
    for (int i = 1; i <= 24; ++i) {
        float q = float(i) / 24.0;
        float distance = pixel + travel * q * q;
        vec3 ray = point + light * distance;
        float clearance = ray.z + pixel * 0.15 - surface_height(ray.xy);
        visibility = min(visibility, clamp(clearance / max(distance * 0.14, pixel * 0.1), 0.0, 1.0));
    }
    return mix(1.0, visibility, u_shadow_strength);
}

float occlusion(vec3 point) {
    float sum = 0.0;
    float pixel = max(u_full_size.x / u_grid_size.x, u_full_size.y / u_grid_size.y);
    for (int i = 0; i < 8; ++i) {
        float a = float(i) * PI / 4.0;
        vec2 d = vec2(cos(a), sin(a));
        float distance = max(pixel * 3.0, 0.0006);
        float near_h = (surface_height(point.xy + d * distance) - point.z) / distance;
        float far_h = (surface_height(point.xy + d * distance * 4.0) - point.z) / (distance * 4.0);
        sum += clamp(max(near_h, far_h), 0.0, 1.0);
    }
    return 1.0 - u_occlusion_strength * sum / 8.0;
}

float lambda_ggx(vec3 w, vec3 n, vec3 t, vec3 b, vec2 alpha) {
    float wz = max(dot(w, n), 1e-5);
    float x = dot(w, t) * alpha.x, y = dot(w, b) * alpha.y;
    return (sqrt(1.0 + (x*x + y*y)/(wz*wz)) - 1.0) * 0.5;
}
vec3 light_brdf(vec3 n, vec3 t, vec3 b, vec3 v, vec3 l, vec3 color, float roughness, float coherence, float paint_presence
#ifdef INTERACTION_SURFACE
    ,float diffuse_roughness
#endif
) {
    float nl = max(dot(n, l), 0.0), nv = max(dot(n, v), 1e-5);
    if (nl <= 0.0) return vec3(0);
    vec3 h = normalize(v + l);
#ifdef INTERACTION_SURFACE
    float ratio = sqrt(1.0 - 0.9 * coherence);
#else
    float ratio = sqrt(1.0 - 0.9 * u_anisotropy * coherence);
#endif
    float alpha = max(roughness * roughness, 0.0064);
    vec2 a = vec2(alpha / ratio, alpha * ratio);
    float ht = dot(h, t)/a.x, hb = dot(h, b)/a.y, hn = max(dot(h, n), 0.0);
    float denominator = ht*ht + hb*hb + hn*hn;
    float distribution = 1.0 / max(PI * a.x*a.y * denominator*denominator, 1e-9);
    float geometry = 1.0 / (1.0 + lambda_ggx(v,n,t,b,a) + lambda_ggx(l,n,t,b,a));
    // Porous fresco has a modest effective dielectric interface reflection.
    float f0 = 0.025;
    if (u_glazed==1) f0=mix(0.012,0.028,paint_presence);
    float fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    float specular = distribution * geometry * fresnel / max(4.0 * nv * nl, 1e-5);
#ifdef INTERACTION_SURFACE
    if(diffuse_roughness>0.0) {
        // Only angular redistribution changes. White EON retains the existing
        // pigment reflectance's directional-hemispherical diffuse integral.
        float true_nv=max(dot(n,v),0.0);
        float tangent_dot=dot(l-n*nl,v-n*true_nv);
        float angular=white_eon(nl,true_nv,tangent_dot,diffuse_roughness);
        return ((1.0-fresnel)*color/PI*angular+vec3(specular))*nl*PI;
    }
#endif
    // Lambertian paint reflectance plus energy-reduced dielectric interface.
    return ((1.0 - fresnel) * color / PI + vec3(specular)) * nl * PI;
}

void main() {
    vec3 origin = u_camera_right * screen_position.x * u_visible_size.x * 0.5
                + u_camera_up * screen_position.y * u_visible_size.y * 0.5;
    vec3 point = hit_surface(origin, u_camera_view);
    vec2 uv = field_uv(point.xy);
    if (!inside(uv)) { frag_color = vec4(u_crisp==1 ? u_ground : u_substrate, 1); return; }
    float crisp_coverage=1.0;
    if (u_crisp==1) {
        float mass=textureLod(u_paint,uv,0.0).a;
        // Derivatives convert the fixed material-space contour into one-pixel
        // area coverage. They do not create a broad wash or change the cutoff.
        float pixel_span=fwidth(mass);
        crisp_coverage=pixel_span>1e-12
            ? clamp((mass-u_mass_threshold)/pixel_span+0.5,0.0,1.0)
            : step(u_mass_threshold,mass);
        if (crisp_coverage<=0.0) { frag_color=vec4(u_ground,1); return; }
    }
    vec4 material = textureLod(u_geometry, uv, 0.0);
    vec2 axial = textureLod(u_finish, uv, 0.0).xy;
    float coherence = clamp(length(axial), 0.0, 1.0);
    float angle = coherence > 1e-6 ? atan(axial.y, axial.x) * 0.5 : 0.0;
#ifdef INTERACTION_SURFACE
    vec4 history=material_interaction(uv);
    float fabric=clamp(length(history.yz),0.0,1.0);
    float silk=u_silk_strength*fabric;
#ifdef GRAIN_CONTRAST
    float aggregate=u_grain_strength*aggregate_response(clamp(history.w,0.0,1.0),u_grain_contrast);
#else
    float aggregate=u_grain_strength*clamp(history.w,0.0,1.0);
#endif
    coherence*=u_anisotropy;
    if(silk>1e-6) {
        // No velocity, hue or procedural noise: the nematic tensor stores a
        // material-space axis. Camera motion merely reveals that frozen axis.
        angle=0.5*atan(history.z,history.y);
        coherence=mix(coherence,0.9,silk);
    }
#endif
    vec3 n = normal_at(point, material.a);
    vec3 raw_t = vec3(cos(angle), sin(angle), 0.0);
    vec3 t = normalize(raw_t - n * dot(raw_t, n));
    vec3 b = cross(n, t);
    float roughness = clamp((material.b * u_roughness_scale + u_roughness_bias)
                            * mix(1.0, 0.78, material.g), 0.09, 0.98);
    roughness = max(roughness, mix(0.55, 0.4, material.g));
    float paint_presence = 1.0 - exp(-12.0 * material.a);
    vec3 color = paint_color(uv);
    // Wet paint is slightly darker; the independent roughness change carries
    // most of the visible drying response without a synthetic animated overlay.
    if (u_glazed==1) {
        float mass=textureLod(u_paint,uv,0.0).a;
        float upper_share,porosity;
        upper_material(uv,mass,upper_share,porosity);
        // A wet upper layer can carry a satin interface; exposed lower paint
        // and stronger-scattering pigments remain more matte. No screen noise.
        float coating=material.g*upper_share*smoothstep(0.25,1.0,mass/u_mass_reference);
        float dry_roughness=clamp(0.6+0.25*material.b+0.09*porosity,0.55,0.92);
        float wet_roughness=0.25+0.15*porosity;
        roughness=clamp(mix(dry_roughness,wet_roughness,coating)*u_roughness_scale
                        +u_roughness_bias,0.18,0.96);
        paint_presence=coating;
        color*=mix(1.0,0.96,material.g*upper_share);
    } else color *= mix(1.0, 0.91, material.g);
#ifdef INTERACTION_SURFACE
    // These are bounded surface-scattering interpretations of recorded fabric
    // and aggregation. No pigment coefficient, mass, height or normal changes.
    // Grain disrupts the smooth satin interface; its variation is exclusively
    // the transported aggregate fraction, never a screen-space texture.
    roughness=mix(roughness,max(0.2,roughness*0.55),silk);
    roughness=mix(roughness,max(roughness,0.88),aggregate);
#endif
    float ao = occlusion(point);
    vec3 radiance = color * u_ambient * ao * (0.65 + 0.35 * max(n.z, 0.0));
    vec3 key_t = normalize(cross(vec3(0, 0, 1), u_key_direction + vec3(1e-6, 0, 0)));
    vec3 key_b = cross(u_key_direction, key_t);
    float visibility = shadow(point, u_key_direction);
    // Fixed world-space quadrature does not crawl or change with output cadence.
    for (int i = 0; i < 5; ++i) {
        vec2 offset = i == 0 ? vec2(0) : i == 1 ? vec2(-0.16,0) : i == 2 ? vec2(0.16,0)
                    : i == 3 ? vec2(0,-0.1) : vec2(0,0.1);
        vec3 light = normalize(u_key_direction + key_t * offset.x + key_b * offset.y);
        radiance += light_brdf(n,t,b,u_camera_view,light,color,roughness,coherence,paint_presence
#ifdef INTERACTION_SURFACE
            ,aggregate
#endif
        ) * u_key_strength * visibility / 5.0;
    }
    vec3 fill = normalize(vec3(-u_key_direction.xy, 0.8));
    radiance += light_brdf(n,t,b,u_camera_view,fill,color,roughness,coherence,paint_presence
#ifdef INTERACTION_SURFACE
        ,aggregate
#endif
    ) * u_fill_strength * ao;
    radiance = max(radiance * u_exposure, vec3(0));
    if (u_tone_map == 1) {
        float luminance = dot(radiance, vec3(0.2126, 0.7152, 0.0722));
        radiance *= (1.0 + luminance / 16.0) / (1.0 + luminance);
        radiance = clamp(radiance, 0.0, 1.0);
    }
    radiance=clamp(radiance,0.0,1.0);
    if (u_crisp==1) radiance=mix(u_ground,radiance,crisp_coverage);
    frag_color = vec4(radiance, 1);
}
