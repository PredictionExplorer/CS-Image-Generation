# Light Cast by Gravity

Implemented in `src/atelier/light.rs` and `src/atelier/light/accumulate.rs`.
The first full-detail optical proofs are being prepared. The public
`render_linear` entry point returns linear RGB plus validated per-sample optical
diagnostics; visual curation precedes the fourth complete film.

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

The small fixed coma term adds asymmetry. Require `|coma_i| <= 0.15`; thickness
stays positive. An optional fixed quadratic bend curves an elongated boss into a
crescent-shaped optical form:

```
u = xi/a_i
v = eta/b_i - bend_i*u^2
s = u^2 + v^2
h_i = H_i*(1-s)^4*(1 + coma_i*u), for s < 1
```

Require `|bend_i| <= 0.5`; start visual experiments around ±0.25–0.35. Bend is
fixed per lens and introduces no additional motion or clock. Zero bend retains
the original arithmetic path and is omitted from serialized lens settings, so
existing straight-lens recipes and their hashes remain unchanged. All curved
gradients and Hessians include the full coordinate-warp chain rule.

Height and its first three derivatives meet zero at either aperture boundary.
Overlapping forms add thickness, equivalent to closely stacked thin phase
elements. Curved candidate cells use a conservative world bounding box: the
ordinary rotated ellipse bounds expand by the projected transverse displacement
`b_i*abs(bend_i)`. Unaffected candidates cancel through the exact identity path.
The ray-slope bound includes the additional derivative contribution
`H_i*(27/32)*abs(bend_i)*(1+abs(coma_i))/a_i`; increasing receiver distance lowers
this bound while preserving the configured dimensionless optical power.

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
receiver triangle using exact clipped-cell moments in a bilinear receiving
basis at the configured screen supersampling, preserving total flux. Reversed
orientation is valid: use unsigned area and accumulate every overlapping branch.
Do not shade with an unbounded `1/abs(det(J))` at individual samples.

An identity ray map produces the uniform background analytically. For cells
affected by any lens, subtract their unrefracted contribution and add their
refracted contribution. Cover the union once, including where lenses overlap;
tracing each lens as a separate complete beam would count light multiple times.
Cells entirely outside all lens supports require no optical work.

The receiver has a finite normalized footprint, defaulting to approximately
0.9 final pixels at AA3. Three separable normalized box passes provide a compact
Gaussian-like source-angle/sensor blur. Nearly collapsed triangles blend into
their nonuniform projected line distribution or a point footprint, preserving
both flux and line extent. Padding precedes filtering so light just outside the
crop can still enter through this footprint.

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

Twelve wavelength bands group all 64 existing `spectrum::BIN_XYZ_LUT` entries
into contiguous ranges. Each band sums its original XYZ weights and uses an
XYZ-weighted representative wavelength. Their weights retain the repository's
D65 normalization, without another division by band count. Irradiance is
accumulated into XYZ and converted only after all twelve bands are present;
the legacy SIMD alpha/tone path is not used.

Dispersion follows `n(lambda)=n_ref+dispersion*((lambda_ref/lambda)^2-1)`, with
`n_ref=1.5`, `lambda_ref=550 nm`, and `dispersion=0.008`. Keep most caustics nearly
white, with narrow spectral separation at selected edges. Complete XYZ colors
outside nonnegative linear sRGB are gently desaturated at constant luminance.

The initial Light preset accumulates sixteen half-frame shutter samples in
linear light before the existing fixed tone curve. A finite footprint and
conservative coverage address both photon noise
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
| Optional fixed bend | 0 / 0 / 0; experimental range approximately ±0.25–0.35 |
| Receiver distance D | 8 world units |
| Spectral integration | 12 contiguous groups from the existing 64-bin XYZ table |
| Reference index / wavelength / dispersion | 1.500 / 550 nm / 0.008 |
| Source grid / domain | 2304×1536 / 9.6×6.4 world units |
| Initial receiver field | 10.667×6.0 world units, fixed |
| Output | 3840×2160, 60 fps, AA3, sixteen shutter samples |
| Finite optical footprint | about 0.9 pixels at 4K |

Freeze the receiver field after checking the complete mapped envelope, including
overlapping lenses and dispersion. Track any escaped flux. Preserve substantial
negative space rather than fitting a changing tight crop around the caustics.

## Display and implementation boundary

For ivory ground, show full direct irradiance with a warm diffuse receiver and
a small independent ambient contribution. Focusing creates bright marks and
redistribution creates darker regions. Keep one exposure for the entire film.

For dark field, display positive focusing gain above the uniform reference.
The default remains `max(E/E0-1,0)*gain_scale`. An optional `gain_exponent` in
`[1,3]` applies `max(delta_E/E0,0)^gain_exponent*gain_scale` separately to each
wavelength band before XYZ accumulation. Exponent one retains the exact original
arithmetic and is omitted from serialized settings, preserving old recipe
hashes. Exponent two with a lower fixed gain can suppress broad weak pools and
emphasize the narrow caustic arcs; it changes neither ray transport nor physical
flux budgets. Ivory ignores this exponent. Reject non-finite display signals
or colors instead of passing them through gamut compression.

Label dark field as an artistic contrast view, since a uniformly illuminated
physical receiver would retain its baseline. Keep its exponent and gain fixed
throughout a sequence; this is not an exposure adjustment that follows a frame.

A dedicated Rust optical accumulator returns the existing linear frame format,
then reuses tone mapping, PNG output, receipts, and video encoding. A frontal
camera addresses the receiver at world z=0; its target XY, orthographic height,
and in-plane roll are honored. Unsupported tilt is rejected. The phase plate
sits virtually one optical distance behind the receiver.

Every shutter sample records source time, lens states, support-union counts,
twelve spectral budgets, ray-slope bounds, sampled interpolation error, and
received/escaped reference and refracted flux. Signed crop redistribution is
distinct from nonnegative received and escaped flux. Public diagnostic validation
rejects missing bands, non-finite values, and inconsistent budgets. Per-band
flux uses a common reference irradiance times world area; the aggregate is
Y-weighted luminance flux, not calibrated radiant watts. No new dependency,
Blender, or GPU is required.

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
