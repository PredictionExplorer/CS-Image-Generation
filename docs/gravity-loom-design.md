# Gravity Loom — implemented construction

Gravity Loom is implemented in `src/atelier/loom.rs`. `LoomConfig` controls its
geometry and dyes; `scene(&OrbitSeries, time, &config)` produces strands for the
shared CPU renderer. This document describes v08; curated film recipes can
override the defaults below.

## The object and its meaning

Three woven relationship panels, AB, BC, and CA, form a rounded shell with a broad
belly, narrow feathered ends, and staggered windows. A permanent S-curve and
longitudinal twist establish the silhouette. Recorded motion deforms the rails
and bows as it travels through a fixed history window.

The shell’s X coordinate represents chronology. Its shape is an artistic mapping
of original positions and actual three-dimensional pair distances. It does not
depict literal body trajectories or simulate a freely moving cloth.

## 1. Genuine history and a fixed source plane

For current source fraction `t`, history length `H`, and material coordinate
`u ∈ [0,1]`, sample `tau = t-H*(1-u)`. The oldest history is at `u=0`; the current
source position is at `u=1`. The complete window must exist in `OrbitSeries`.
Missing prehistory returns an error. A verified prelude at least `H` long supports
the fully formed first frame; there is no wrapping or extrapolation.

Let `p_i(tau)` be a body’s position after the source’s single fixed world
normalization. Two fixed orthonormal `source_axes`, `a` and `b`, give

```
y = dot(p_i,a); z = dot(p_i,b)
q_i = (0,y,z) / hypot(source_radius, hypot(y,z))
```

This smooth mapping guarantees `|q_i| < 1`. The default axes select original X/Y:
`a=(1,0,0)`, `b=(0,1,0)`. Validation requires unit lengths and orthogonality within
`1e-6`. A seed-specific plane may be fitted once from the complete source and
stored explicitly in the recipe; it never follows the moving window.

Sampling 20,001 evenly spaced rows of seed `0xb7f327f9f722` found that YZ retained
about 50% of temporal position variance, XY retained 87%, and a fixed principal-
component plane retained 92%. This explains the change from the original YZ
projection.

Pair closeness always uses the unprojected positions:

```
d_ij = |p_i-p_j|
c_ij = 1 / (1 + (d_ij/pair_distance_scale)^2)
```

All three pairs share the same distance scale. The implementation uses `hypot`
for numerical range. It does not substitute a nearest-companion measurement for
a particular pair’s distance.

## 2. Static silhouette and moving rails

With fixed unit axes `X`, `Y`, and `Z`, the centerline and conch radius are

```
C(u) = L*(u-0.5)*X + A_y*sin(2*pi*u)*Y + A_z*sin(pi*u)^2*Z
S5(u) = 6*u^5 - 15*u^4 + 10*u^3
R(u) = radius_tail + (radius_mouth-radius_tail)*S5(u)
     + radius_bulge*u^3*(1-u)^2/0.03456
```

The alternative spindle uses
`R(u)=radius_min+(radius_max-radius_min)*sin(pi*u)^2`. Both profiles remain
strictly positive. Three rest rail angles are separated by 120°:

```
theta_i(u) = pi/2 + 2*pi*i/3 + rail_rotation + twist*(u-0.5)
e_i(u) = (0,cos(theta_i),sin(theta_i))
Q_i(u,t) = C(u) + R(u)*(e_i(u) + rho*q_i(tau))
```

`rho` is `source_influence`, restricted to `[0,0.40]`. Source displacement stays
below `rho*R(u)`, including at the narrow ends. Twist is a fixed function of
chronology, never a whole-object spin over time.

## 3. Rounded polar panels and their bounds

The default `cross_section: "polar"` interpolates between adjacent rails around
the shell. Let `r_i` and `phi_i` be a rail’s actual transverse radius and angle
about `C`. Angles stay anchored to their rest directions and are unwrapped
continuously. For each ordered pair A→B, B→C, C→A, take the positive angular span
`Delta_phi`; the final pair wraps through one full turn.

For across-panel coordinate `v ∈ [0,1]`,

```
k = compression*c_ij
F = v + k*sin(2*pi*v)/(2*pi)
B = R(u)*(bow_base + bow_response*c_ij)
r = r_i + (r_j-r_i)*S5(F) + B*sin(pi*F)^2
phi = phi_i + Delta_phi*F
S(u,v,t) = C(u) + r*(0,cos(phi),sin(phi))
```

Radial interpolation and bow have zero across-panel radial slope at the rails.
Adjacent panels therefore share smooth surface normals. The earlier chord-and-
bow construction remains selectable as `cross_section: "bowed"` for explicit
comparison recipes.

The polar construction guarantees

```
F_v >= 1-compression > 0
r >= R(u)*(1-rho) > 0
Delta_phi >= 2*pi/3 - 2*asin(rho) > 0
S_u.x = L; S_v.x = 0; S_t.x = 0

|S_u cross S_v| >= L*R(u)*(1-rho)*(2*pi/3-2*asin(rho))*(1-compression)
```

The base shell thus retains an ordered radial cross-section and positive area.
Surface derivatives are analytical around measured rail values; source-dependent
derivatives use nearby genuine samples, with one-sided differences at source
endpoints. The outward normal is `normalize(S_v cross S_u)`.

## 4. Permanent windows and real weaving

Each panel removes a fixed rotated ellipse in material coordinates:

| Panel | Center `(u,v)` | Half-axes | Tilt |
| --- | --- | --- | --- |
| AB | `(0.43,0.48)` | `(0.39,0.37)` | −6° |
| BC | `(0.59,0.53)` | `(0.35,0.40)` | +8° |
| CA | `(0.50,0.48)` | `(0.42,0.36)` | −5° |

The windows remove roughly 44–48% of parameter area. Their topology is fixed;
source motion changes their world-space shape. Constant-U and constant-V thread
curves are split at analytical quadratic intersections with each ellipse.
Separate surviving pieces are never connected across a hole.

Warp group `m` lies near `v_m=(m+0.5)/N_w`; weft row `n` lies near
`u_n=(n+0.5)/N_f`. Their offsets along the actual surface normal are

```
h(u) = crossing_lift*R(u)/1.1
warp_lift(u,m) =  h(u)*cos(pi*(N_f*u - 0.5 + m))
weft_lift(n,v) = -h(u_n)*cos(pi*(N_w*v - 0.5 + n))
```

Group-center crossings have opposite heights, alternating by `(-1)^(n+m)`.
Fine fibers have fixed offsets within each group; crossing height is validated
against fiber radius, variation, and group width. These are separate 3D curves,
with no supporting opaque surface.

Each window has a continuous braid; each shared rail cable is emitted once.
Component-end tapers reduce yarn radii and crossing lift where threads meet a
rim. Lattice indices, braid phases, and dye variations remain fixed through time.
A runtime check rejects a lifted warp polyline that folds backward along
chronology. This construction does not include a global fiber-contact solver.

## 5. Feathered terminal finish

The oldest and newest ends have independent smooth feather lengths. With `S5`
clamped to `[0,1]`, their shared envelope is

```
T(u) = S5(u/terminal_feather_old)
     * S5((1-u)/terminal_feather_new)
```

A zero length replaces its factor with one. The envelope scales all yarn radii
and crossing lift, plus boundary-fiber radii and braid offsets. Terminal rings
therefore vanish when feathering is enabled; interior threads remain unchanged.
No source sampling, aperture topology, or lattice indices change. Defaults
feather the oldest 5% and newest 8% of the shell.

## Defaults and artistic controls

| Control | Default |
| --- | --- |
| Construction / profile | polar / conch |
| History / required first-frame prelude | 0.22 / at least 0.22 |
| Axis length / centerline Y,Z amplitudes | 7.2 / 0.42, 0.32 |
| Conch tail, mouth, extra belly radius | 0.06, 0.15, 1.25 |
| Spindle minimum / maximum radius | 0.06 / 1.35 |
| Fixed longitudinal twist | 180° |
| Source plane / influence / soft radius | XY / 0.22 / 2.0 |
| Pair distance scale | 1.4 |
| Base bow / pair response / compression | 0.055 / 0.16 / 0.40 |
| Warp groups / weft rows | 128 / 256 |
| Fibers per warp / weft group | 2 / 2 |
| Warp / weft / aperture sampling intervals | 3072 / 512 / 1536 |
| Group width relative to cell pitch | 0.34 |
| Fiber radius / crossing lift at radius 1.1 | 0.00035 / 0.0015 |
| Terminal feather, old / new | 0.05 / 0.08 |

Default dyes are champagne, bronze, and pearl, with metallic contributions
0.68, 0.75, and 0.15, and roughness 0.32. Blue accents affect about 3.5% of yarn
groups; selected other groups receive stronger gilding. Geometry supplies the
fine weave, so material `fiber_strength` is zero. Film recipes can adjust dyes
and lighting independently of the recorded motion.

For more readable motion, first choose a fixed source plane that captures the
orbit. Influence 0.4 and pair response 0.35–0.4 remain within the construction’s
bounds. A soft radius around 1.0 is a useful starting point: reducing it increases
displacement but eventually saturates radial changes, so smaller values do not
uniformly increase motion. The stationary centerline and radius profile preserve
the broad silhouette.

`panels: [true,false,false]` isolates AB for a material proof. Full-film curation
uses a fixed camera, checkpoints across the complete source, and motion review
for readability and shimmer. Tests cover projection validity, pair-measurement
independence, polar joins and derivatives, area bounds, analytical cuts,
alternating crossings, repeatability, temporal continuity, missing prehistory,
and terminal feathering.
