# Gravity Loom — construction notes

Status: design only. Implement after the first Tidal Calligraphy film is finished.

## The object

One suspended woven conch: a slender old end opens into a generous belly and an
asymmetric mouth. Its centerline makes a shallow, permanent S-curve across the
frame. Three pairwise textile panels surround a dark central chamber, with large
staggered openings through which the far-side weave is only partly visible.
Fine threads form actual alternating warp/weft crossings; quiet braided rims
finish the openings. The three families are AB, BC, and CA.

Chronology has a permanent left-to-right direction. The panels breathe and their
threads catch traveling highlights as the source changes, while the overall
object keeps its orientation and generous empty spaces.

This is a parametric artistic interpretation of recorded motion. The source
positions and physical pair distances are unchanged inputs. The time axis,
stationary conch profile, bounded rail neighborhoods, bowed panels, and weave
are designed mappings; this is not a simulation of celestial bodies pulling a
physical textile.

## 1. Permanent chronology and organic outer silhouette

Let `T = (1,0,0)` be the time axis, and let `P` project onto the YZ plane. Let
`u ∈ [0,1]` run from oldest to newest history, and `t` be the current source
fraction. Sample genuine source history at

```
tau(u,t) = t - H * (1-u)
```

Use a verified prelude at least `H` long for a fully formed opening. Do not wrap
time or fabricate pre-zero samples.

The centerline bends in YZ while retaining an exact, monotone X coordinate:

```
C(u)  = L*(u-0.5)*T + A_y*sin(2*pi*u)*Y + A_z*sin(pi*u)^2*Z
C'(u) = L*T + 2*pi*A_y*cos(2*pi*u)*Y + pi*A_z*sin(2*pi*u)*Z
```

Start with `L = 7.2`, `A_y = 0.42`, and `A_z = 0.32`. The maximum transverse slope
is bounded by about 0.40, giving a gentle bend. `C` never changes with frame time.
Keep cross-sections in the fixed YZ plane; do not rotate them to follow the
centerline tangent. This preserves the simple chronology invariant below.

Compare only two finished silhouette treatments in the first proof:

**Preferred teardrop/conch.** A narrow tail, fuller belly, and open mouth:

```
S5(u)  = 6*u^5 - 15*u^4 + 10*u^3
S5'(u) = 30*u^2*(1-u)^2
K      = (3/5)^3*(2/5)^2 = 0.03456
E(u)   = u^3*(1-u)^2 / K
E'(u)  = u^2*(1-u)*(3-5*u) / K

R(u)  = R_tail + (R_mouth-R_tail)*S5(u) + R_bulge*E(u)
R'(u) = (R_mouth-R_tail)*S5'(u) + R_bulge*E'(u)
```

Use `R_tail = 0.32`, `R_mouth = 0.78`, and `R_bulge = 0.50`. The radius stays at
least 0.32, has zero derivative at both ends, and grows to roughly 1.1–1.2 in the
belly. Its upper bound is `R_mouth + R_bulge = 1.28`.

**Spindle alternative.** A simpler open-ended, symmetrical envelope:

```
R(u)  = R_min + (R_max-R_min)*sin(pi*u)^2
R'(u) = pi*(R_max-R_min)*sin(2*pi*u)
```

Use `R_min = 0.42` and `R_max = 1.18`. Both profiles have genuinely open ends;
neither collapses to a point. Start without twisting the cross-sections. Only if
the silhouette proof needs it, consider a small, fixed twist along `u`; never
drive whole-object orientation from frame time.

## 2. Source-driven rails with local separation bounds

Place unit rail directions `r_i` in the YZ plane at 90°, 210°, and 330°. Define
one shared, fixed source normalization:

```
q_i(tau) = P*p_i(tau)/M, with |q_i| <= 1
Q_i(u,t) = C(u) + R(u)*(r_i + rho*q_i(tau))
```

Choose `M` from the complete visible source and required prelude, with a bound
on the actual position interpolant. Knot extrema alone can miss interpolation
overshoot. A conservative maximum projected norm of the cubic Hermite curves'
Bezier control points supplies a convex-hull bound. Use the same `M` for all
bodies and all frames, with a positive fallback for a stationary source.

Start with `rho = 0.22`; require `0 <= rho <= 0.40`. Scaling source displacement
by the local radius keeps the narrow tail as well separated as the broad belly.
No per-frame normalization, force clamping, or change to recorded positions is
involved. The rail derivative is

```
Q_i,u = C' + R'*(r_i + rho*q_i) + R*rho*H*q_i,tau
```

Use actual position derivatives, or consistent geometric finite differences.
`BodySample.speed` is a normalized artistic measurement, not this derivative.

For every pair,

```
|Q_j-Q_i| >= R(u)*(sqrt(3)-2*rho)
```

At the teardrop defaults this is at least about 0.413 world units. The radius is
positive and the three displacement neighborhoods retain their cyclic order.
Thus the rail frame remains defined even at a close physical encounter.

The omitted spatial coordinate is not presented as literal position. Actual
three-dimensional pair separation is retained in the panel-shaping signal below.

## 3. Bowed panels and explicit derivative invariants

Use the oriented pairs A→B, B→C, and C→A. At each history row, define

```
D_ij = Q_j - Q_i
N_ij = normalize(D_ij cross T)
```

The fixed time axis and separated rail neighborhoods keep this frame defined.
Orient it outward once using the rest triangle. No Frenet frame, arbitrary
normal sign switches, or per-frame object rotation is needed.

Let `c_ij(tau)` be a gently smoothed closeness signal derived from the actual
three-dimensional distance `|p_i-p_j|`, with one fixed normalization shared by
the three pair families. It ranges from zero (far) to one (near).
Compute these pair distances directly; `BodySample.proximity` describes the
nearest companion and must not be mistaken for a specific AB/BC/CA measurement.

For material coordinate `v ∈ [0,1]` across a panel, use

```
k = 0.40*c_ij
F(u,v,t) = v + k*sin(2*pi*v)/(2*pi)
b = R(u)*(0.11 + 0.23*c_ij)

S_ij(u,v,t) = Q_i + F*D_ij + b*sin(pi*F)^2*N_ij
```

Scale the bow with the local radius too. Smooth `c_ij` over the complete source,
not separately over each moving window, and keep its interpolation in `[0,1]`.

For implementation and normal calculation, let `e = D/|D|`, `g = sin(pi*F)^2`,
and `g_F = pi*sin(2*pi*F)`. Then

```
e_u = (D_u - e*(e dot D_u))/|D|
N_u = e_u cross T
F_u = k_u*sin(2*pi*v)/(2*pi)
F_v = 1 + k*cos(2*pi*v)
b_u = R'*(0.11 + 0.23*c_ij) + R*0.23*c_ij,u

S_u = Q_i,u + F_u*D + F*D_u + (b_u*g + b*g_F*F_u)*N + b*g*N_u
S_v = F_v*(D + b*g_F*N)
```

`F_v >= 0.60`, `S_u.x = L`, `S_v.x = 0`, and `S_t.x = 0` exactly. Because
`D` and `N` are perpendicular, the base panel has the lower Jacobian bound

```
|S_u cross S_v| >= L*0.60*R_min*(sqrt(3)-2*rho) > 0
```

Here `R_min` is the lower bound of the chosen radius profile. This is about
1.79 at the teardrop defaults. It guarantees a regular panel and ordered
chronology; a base panel cannot fold back along the time axis. It does not
certify clearance between all woven fibers or adjacent panels, which still
needs a geometry check and the visual proof.

An approaching pair narrows the middle of its woven opening and deepens its
bow. Separation lets the opening expand. The changing signal travels through
the fixed history axis. Keep these changes broad and smooth enough to read as
one gesture rather than high-frequency vibration.

## 4. Staggered permanent apertures and connected edges

Give each panel a different fixed, gently tilted material-space ellipse:

```
x = u-u_0; y = v-v_0
xi  = x*cos(theta) + y*sin(theta)
eta = -x*sin(theta) + y*cos(theta)
hole: (xi/a)^2 + (eta/b)^2 < 1
```

| Panel | Center `(u_0,v_0)` | Half-axes `(a,b)` | Tilt |
| --- | --- | --- | --- |
| AB | `(0.43,0.48)` | `(0.39,0.37)` | −6° |
| BC | `(0.59,0.53)` | `(0.35,0.40)` | +8° |
| CA | `(0.50,0.48)` | `(0.42,0.36)` | −5° |

These remove about 44–48% of each panel's parameter area. Their staggering
creates unequal overlapping views through the chamber. The S-curve and radius
envelope make their world-space outlines organic. Keep centers, axes, and tilts
fixed for the whole film; no threshold animation or changing topology.

Validate containment with the exact axis-aligned ellipse extents:

```
h_u = sqrt((a*cos(theta))^2 + (b*sin(theta))^2)
h_v = sqrt((a*sin(theta))^2 + (b*cos(theta))^2)
```

Require `u_0 +/- h_u` and `v_0 +/- h_v` inside `[0.035,0.965]`. The proposed
openings satisfy this margin.

Split thread curves analytically where they meet each ellipse; a constant-U or
constant-V fiber still has a quadratic intersection after ellipse rotation.
Never join the two surviving pieces across the hole. End them at a fine
continuous braided rim.
Ease both crossing lift and thread radius to zero over the final boundary region
so ends meet the rim cleanly. Apply the same treatment at the outer rails.

Emit each of the three shared rail cables once, rather than duplicating bright
edges for both adjoining panels.

## 5. Real over/under weaving, with a breathing yarn scale

Keep weave indices fixed in material coordinates, not tied to the current frame
number or the changing first sample of a history window.

For longitudinal warp group `m` at `v_m = (m+0.5)/N_w`, and crosswise weft row `n`
at `u_n = (n+0.5)/N_f`, use opposite offsets along the actual panel normal. The
half-cell offsets keep these groups distinct from the shared boundary cables.
Choose the outward surface normal `nu = normalize(S_v cross S_u)`; it includes
the centerline and panel slopes, whereas the earlier `N_ij` is only the
cross-sectional bow direction.

```
h(u) = h_ref*R(u)/R_ref
warp lift(u,m) =  h(u)*cos(pi*(N_f*u - 0.5 + m))
weft lift(n,v) = -h(u_n)*cos(pi*(N_w*v - 0.5 + n))
```

At a crossing `(u_n,v_m)`, one centerline is at `+h(u_n)*(-1)^(n+m)` and the other at
the opposite height. This produces actual alternate over/under crossings in 3D.
It is not a texture painted onto coincident curves.

Scale fiber radius by the same `R(u)/R_ref`, and size group offsets from the
local lattice pitch. This prevents the narrow tail from becoming a dense plug
of constant-thickness wire. At cut boundaries, taper radius and lift together.

Each yarn group contains a few fine parallel fibers. Use group-scale lift
phases, small offsets within the group, and restrained fixed radius/tint
variation. Choose lift to clear the complete bundle thickness. Fixed crossing
parity and fixed apertures keep the animation temporally coherent. The normal
offset curves need their own clearance checks: require positive longitudinal
derivatives on warp fibers and lift small relative to local pitch and bend
radius. The regularity bound for the base panel alone does not prove this.

Warp in the AB panel uses A's dye and weft uses B's; similarly for BC and CA.
Their overlap expresses the pair without labels on the artwork.

## 6. Resolvable detail and smooth fiber highlights

Start with three fibers per yarn group in both directions. Add five only if the
intended 4K camera resolves their spacing in the broad sections. At the narrow
tail, fine fibers may merge into an analytically filtered yarn; increasing the
count there adds cost without a visible benefit. Keep counts fixed through time.

Use the initial sampling counts below, then check projected chord error against
0.2–0.25 pixel and check tangent change under raking light. Smooth geometric
curves with segment-constant shading can still show zipper-like highlights.

The current strand renderer shades with a constant tangent per segment. If the
weave close-up exposes this, compute a tangent at every curve point and
interpolate it along each rendered segment:

```
T(s) = normalize((1-s)*T_0 + s*T_1)
```

Prefer analytical derivatives of the lifted curves, or arc-length-aware central
differences. Compute tangents separately for each clipped component, never
across an aperture gap. This can live in renderer preparation without changing
the public `Strand` data structure. Preserve analytical coverage and the merge
of neighboring segment footprints; change shading interpolation only. Resolve
this in the first material proof if needed, before a full Loom film.

## Initial high-detail recipe

| Parameter | Starting value |
| --- | --- |
| History fraction H | 0.22 |
| Verified prelude | at least 0.22 |
| Time-axis length L | 7.2 world units |
| Static centerline offsets A_y / A_z | 0.42 / 0.32 |
| Preferred radius tail / mouth / bulge | 0.32 / 0.78 / 0.50 |
| Reference radius R_ref for fiber sizing | 1.1 |
| Maximum source displacement | 0.22 × local R(u) |
| Panel bow | local R(u) × (0.11–0.34) |
| Closeness compression gain | 0.40 |
| Apertures | three fixed staggered ellipses from the table above |
| Warp groups per panel | 56 |
| Weft rows per panel | 144 |
| Fibers per warp/weft group | 3 / 3; compare 5 only when visibly resolved |
| Fiber radius at R_ref | about 0.00045–0.00055 |
| Group width | about 24% of its lattice pitch |
| Crossing lift h_ref | about 0.0035 |
| Longitudinal warp intervals | 1536 |
| Weft intervals across a complete row | 192 |
| Surface film between fibers | off initially; optional very faint web later |

These are high-detail starting values for actual 4K stills: roughly one million
strand segments before aperture clipping with three fibers per group. Keep
subpixel fibers analytically filtered, and verify that additional geometry is
visible in the intended framing.

## Palette, light, and camera

- A: champagne; B: soft bronze; C: warm pearl. Example linear dye colors are
  `(0.82,0.68,0.45)`, `(0.50,0.28,0.13)`, and `(0.78,0.73,0.65)`. Use a muted blue
  accent in at most roughly 5% of fine yarn groups, keeping the overall object
  warm. Their paired yarn colors remain distinct without becoming three colored
  hero bands.
- Start around roughness 0.32–0.38, anisotropy 0.80, and sheen 0.45, with no
  emission. Aim for soft dyed silk highlights; hard brass glints could make the
  object look like a metal strainer. Dim companion fibers provide depth.
  Cylindrical fibers do not have an independently visible back face. Keep the
  apertures truly dark and legible against a nearly black background.
- One long raking strip light, a cool rear strip, and low fill. The modeled
  crossings provide depth occlusion and changing local tangent highlights;
  avoid promising cast shadows that the current raster compositor does not add.
- Start with an oblique fixed camera around `(3.4,2.0,10)`, target zero, world-Y
  up, orthographic height around 5.0–5.2. Fit the complete deformed geometry,
  including the prelude, at all six source checkpoints. Retain both open ends
  and the large windows so the conch silhouette remains readable. Keep this
  camera fixed through the film.

## Risks and decisions

1. **Industrial tube:** a straight centerline, uniform radius, aligned windows,
   and sharp metallic highlights can reinforce one another. The static S-curve,
   broad radius envelope, staggered apertures, and warm soft sheen must establish
   the object at full-frame scale before thread detail can succeed.
2. **Opaque tube:** too much material or overlap. Keep the lens apertures large,
   the central chamber open, and the base film absent or extremely light.
3. **Conveyor-belt flicker:** resetting weave parity or clipping topology by
   frame. Keep the lattice and holes fixed in UV while source history deforms it.
4. **Neon cage:** too-bright outline cables. Shared rails and aperture rims should
   finish the object quietly rather than dominate it.
5. **A static diagram:** insufficient source influence. Increase panel bow and
   proximity response first; retain the fixed chronology axis and safe rail
   neighborhoods.
6. **Unresolved glitter:** excessive fine geometry or segmented tangent shading.
   Judge a raking-light 4K crop, interpolate tangents when needed, and keep
   genuinely visible yarn hierarchy rather than adding invisible strands.

## First proof after Calligraphy is complete

Build the connected three-panel object at finished detail, with a close-up of
one AB region showing unambiguous alternating crossings. The complete object is
necessary to judge the silhouette and the views through staggered windows.

Compare only the S-teardrop and S-spindle profiles, using the same camera,
material, and yarn counts, at two source times with different pair separation.
Choose the silhouette first, then verify fiber softness and real over/under
depth in the close-up. Favor the teardrop if its open mouth and uneven belly
read as a graceful woven conch; retain the spindle if it makes cleaner openings.

Review fixed-camera frames at source fractions `0`, `0.25`, `0.50`, `0.75`,
`0.927`, and `1`, including genuine prelude geometry at the opening. Check
rail separation, panel regularity, lifted-fiber clearance, aperture continuity,
and clean temporally stable highlights. Only after that visual proof should
the second full film begin.

This document records the design only. No Loom implementation or render has been started.
