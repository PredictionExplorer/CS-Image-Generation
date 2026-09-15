# Light Cast by Gravity

Status: design only. No implementation has started.

## Intended image

Three moving pools of refracted light produce long ivory crescents, delicate
cusps, and occasional interlocking luminous knots. Broad quiet areas give those
marks room. Close approaches merge and rearrange the caustics; separation opens
them again. The image contains light on a receiving surface, with no woven grid,
curtain, filament, or visible supporting geometry.

The preferred first treatment is warm ivory ground: bright caustics and soft
depleted regions resemble light through sculpted glass onto fine paper. A dark
field is a second display treatment, useful for comparing the same optical
construction without changing its motion.

## 1. Three source-driven phase lenses

Use original positions at the current source fraction `t ∈ [0,1]`. There is no
history window and no prelude requirement. A fixed source plane, stored in the
recipe, maps the two strongest motion components to lens centers:

```
c_i(t) = (3.1*tanh(dot(p_i,a)/1.5), 1.7*tanh(dot(p_i,b)/1.5))
```

For b7, start with the same fixed temporal principal-component plane used for
Loom. Both coordinates move the optical forms by substantial distances. Keep
ellipse orientations fixed; do not add a spinning camera or a free animation
phase. The third source component and smooth proximity may modulate optical
power gently, around ±10%, while center movement remains the dominant gesture.

Model each form as a smooth, thin glass boss on a phase plate. For local rotated
ellipse coordinates `xi,eta`, let

```
s = xi^2/a_i^2 + eta^2/b_i^2
h_i = H_i*(1-s)^4*(1 + coma_i*xi/a_i), for s < 1
h_i = 0, otherwise
```

The small fixed coma term breaks perfect symmetry into graceful open crescents.
Require `|coma_i| <= 0.15`; thickness stays positive. Height and its first three
derivatives meet zero at the aperture boundary. Overlapping forms add thickness,
equivalent to closely stacked thin phase elements.

This is a paraxial thin-element optical model, not full three-dimensional glass
transport. Its assumptions and optical settings belong in the recipe.

## 2. The ray map creates the caustics

Illuminate the phase plate with a uniform parallel beam. At source-plane point
`x`, a wavelength channel with refractive index `n_lambda` reaches the receiver at

```
OPD_lambda(x) = (n_lambda - 1)*sum_i h_i(x)
R_lambda(x) = x + D*gradient(OPD_lambda(x))
J_lambda = I + D*Hessian(OPD_lambda(x))
```

Gradients and Hessians are analytical. Caustic folds occur where the ray map
compresses strongly; `det(J)=0` identifies a fold. Ellipticity and weak coma
unfold circular symmetry into cusps and asymmetric arcs. The cusps are produced
by focusing, rather than drawn curves with arbitrary glow.

Start with dimensionless power
`K=8*D*(n_green-1)*H_i/(a_i*b_i)` around 1.5–1.7. This gives useful focusing
while keeping the thickness and ray slopes modest. Check slopes during the
first proof; if they become large, reduce power or increase the optical distance
while adjusting thickness, rather than silently applying a direction clamp.

## 3. Deterministic, energy-aware CPU accumulation

Use a fixed fine source-plane grid and split each cell into two triangles. Map
its vertices through `R_lambda`. Each source triangle carries positive flux
`Phi = incident_irradiance * source_area`. Deposit this flux over its mapped
receiver triangle using pixel-overlap coverage, preserving total flux. Reversed
orientation is valid: use unsigned area and accumulate every overlapping branch.
Do not shade with an unbounded `1/abs(det(J))` at individual samples.

An identity ray map produces the uniform background analytically. For cells
affected by any lens, subtract their unrefracted contribution and add their
refracted contribution. Cover the union once, including where lenses overlap;
tracing each lens as a separate complete beam would count light multiple times.
Cells entirely outside all lens supports require no optical work.

Give the receiver a finite, normalized optical footprint of roughly 0.8–1.2
pixels at 4K. This represents a small source-angle/sensor blur and regularizes
geometric singularities. For a nearly collapsed mapped triangle, deposit its
flux over a normalized line or point footprint instead of dividing by a tiny
area. A long degenerate triangle must retain its line extent. The transition
between triangle and degenerate footprints must be continuous.

Track received and escaped flux separately. Do not renormalize photons that
fall beyond the receiver crop back into the image. Retain double precision for
signed redistribution and accumulation; only tiny numerical negative residuals
may be rounded away after checking the energy error.

Process independent output tiles in parallel, with contributions accumulated in
a fixed source-triangle order inside each tile. Use no random sample changes,
unordered floating-point atomics, or per-frame exposure normalization. The
fixed mesh and continuous finite footprints keep focusing events stable in time.

## 4. Detail and dispersion

The first proof uses a dense fixed mesh, not a sparse photon preview. Evaluate
ray-map midpoint error against approximately 0.2 receiver pixels, especially
near folds. Compare a doubled-resolution native crop before choosing the final
mesh. If adaptive refinement is later necessary, use a shared fixed refinement
pattern for the sequence; avoid changing leaf topology at a frame threshold.

Start with three wavelength channels, approximately red/green/blue, and weak
dispersion. They share the same geometry and grid. Keep most caustics nearly
white, with only a narrow spectral separation at selected edges. The first
model is a three-band color approximation; a denser spectral quadrature is an
optional refinement if it visibly improves the proof.

Accumulate four shutter samples in linear light before the existing fixed tone
curve. A finite footprint and conservative coverage address both photon noise
and the flicker that point-like caustics would produce at 60 fps. No interference
or diffraction fringes are claimed by this geometric model.

## Starting high-detail recipe

| Control | Initial value |
| --- | --- |
| Lens centers | fixed source plane; motion scales 3.1 / 1.7 |
| Source soft scales | 1.5 / 1.5 |
| Lens semi-axes | `(1.00,0.72)`, `(0.90,0.65)`, `(1.10,0.80)` |
| Fixed lens orientations | −20°, +35°, +78° |
| Dimensionless powers K | 1.55 / 1.70 / 1.45 |
| Fixed coma | +0.08 / −0.10 / +0.06 |
| Receiver distance D | 8 world units |
| Refractive indices R/G/B | 1.497 / 1.500 / 1.504 |
| Source grid / domain | 2304×1536 / 9.6×6.4 world units |
| Initial receiver field | 10.667×6.0 world units, fixed |
| Output | 3840×2160, 60 fps, four shutter samples |
| Finite optical footprint | about 0.9 pixels at 4K |

Freeze the receiver field after checking the complete mapped envelope, including
overlapping lenses and dispersion. Track any escaped flux. Preserve substantial
negative space rather than fitting a changing tight crop around the caustics.

## Display and implementation boundary

For ivory ground, show full direct irradiance with a warm diffuse receiver and
a small independent ambient contribution. Focusing creates bright marks and
redistribution creates darker regions. Keep one exposure for the entire film.

For dark field, display positive focusing gain above the uniform reference,
`max(E/E0-1,0)`, using a fixed contrast curve. Label this as an artistic contrast
view, since a uniformly illuminated physical receiver would retain its baseline.

Implement a dedicated Rust optical accumulator returning the existing linear
frame type, then reuse tone mapping, PNG output, receipts, and video encoding.
The alpha-compositing mesh renderer should not substitute for flux accumulation.
No Blender, GPU, or new numerical dependency is required by this construction.

## First proof and catchable failures

Produce finished 4K stills at source fractions 0, 0.25, 0.50, 0.75, 0.927, and 1
using the same lens model and receiver field. Compare ivory and dark treatments
without changing lens parameters. Then inspect a native 60 fps close-approach
excerpt and a doubled-resolution crop around a fine cusp.

Reject stationary decorative rings, excessive rainbow separation, uniformly
busy lace, clipped bright plateaus, and temporal sparkle. Check that empty areas
remain deliberate and the three source-driven movements are obvious. Numerical
gates are an identity-map baseline, conserved redistributed flux, a stable
degenerate-triangle limit, repeatable frames, and convergence near a cusp. Only
after those checks and the visual proof should the fourth full film begin.
