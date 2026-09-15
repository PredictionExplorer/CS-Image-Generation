# Orbital Engraving — first detailed design

The first implementation is in `src/atelier/engraving.rs` and its `filter`
module. Numerical checks pass; visual selection is still in progress. Use the
unchanged `0xb7f327f9f722` orbit and the existing 1802-frame, 4K/60 clock.

## Visual direction

Three open, asymmetric rosettes engraved in platinum, smoked blue and a little
pale copper on a nearly black aubergine plate. Hundreds of fine, nested cuts
form large crescents; paired cuts gather into broad moving interference lobes.
The image should resemble an extraordinary intaglio instrument viewed square
on: flat, exact, restrained, with deep empty openings and exquisite local detail.

Use hierarchy: three dominant silhouettes, two to five broad gatherings in each,
then the individual grooves. Keep their mouths open and differently oriented.
Avoid a complete circular medallion, a page filled with ornament, fabric shading,
perspective tubes, stars, framing borders, and luminous fog. The interference
here is the designed coincidence of two engraved patterns, not a claim about
gravitational waves or a second optical caustics simulation.

## Frozen source map

Call `OrbitSeries::sample(t)` at the same recorded fractions as the other studies.
Use the fixed three-axis basis already archived in `03-aurora-b7.json`:

```text
a = ( 0.9775173637530767,  0.07416319035834007, -0.19738192611560845)
b = (-0.13152225722147193, 0.946134321388188,   -0.29585763763452494)
c = ( 0.16480806842082663, 0.31516609444416055,  0.9346168377983626)
q_i = tanh((dot(p_i,a), dot(p_i,b), dot(p_i,c)) / 1.5)
center_i = (2.7 q_ix, 1.35 q_iy)
```

All three centers therefore follow the actual source through one fixed mapping.
Keep ellipse orientations, camera and carrier counts fixed. Companion separation
drives broad registration changes below. There is no independent animation
phase, accumulated arbitrary spin, moving principal-axis fit or arc-length
carrier. No history is needed; the first engraving exists at the first source
sample, and the last uses the genuine last sample.

## Open, lobed engraving coordinates

For each body, rotate `x-center_i` by its fixed negative orientation and divide
by its two semi-axes, giving `(u,v)`. Define `r=hypot(u,v)` and

```text
rho = r * (1 + 0.065 cos(3 theta) + 0.025 sin(5 theta))
```

Only evaluate the angular terms in the annular support `rho > 0.28`. Compute
their sines/cosines from `(u/r,v/r)` using polynomial recurrences, so there is
no `atan2` seam. The radial multiplier is at least 0.91: nested contours stay
ordered, with a gentle three-lobed outline and finer five-lobed inflection.

The support is a smooth annulus: fade in over `rho=0.28..0.33`, remain full,
then fade out over `0.96..1.02`. Remove one 64-degree mouth using a cosine-based
angular gate, with 12-degree smooth shoulders on each side. Use quintic
smoothstep for these envelopes. The center opening and mouth join into one
large connected region of negative space; there is no painted disc underneath.
The engraving ends taper as the envelope vanishes.

Two almost matching contour families live on each rosette. For the next body
`j=(i+1)%3`, define phases in **cycles**, not radians:

```text
Phi_i = N_i rho
delta_i = 0.70 q_iz + 1.05 (q_jy - q_iy)
steer_i = 0.55 (q_jx - q_ix)
Psi_i = B_i rho
      + D_i (u*u-v*v)/(1+r*r)
      + 0.75 (2*u*v)/(1+r*r)
      + steer_i*u + delta_i
phi_plus  = Phi_i + Psi_i/2
phi_minus = Phi_i - Psi_i/2
```

`Phi` supplies fine nested grooves. The much slower `Psi` supplies broad,
asymmetric registration lobes. Relative body motion moves those lobes while
the entire engraving follows its own body's trajectory. Inspect both effects
in an approach and release; fine phase motion must not obscure center motion.

## Grooves and selective reflected color

Use the positive finite harmonic profile

```text
C(phi) = cos(pi*phi)^(2m), with m=8
P_i = 0.38 C(phi_plus) + 0.30 C(phi_minus)
    + 1.20 C(phi_plus) C(phi_minus)
```

The single-family terms preserve the individual cuts. The product gives a
stronger reflected accent where the two families register; its low-frequency
terms create the broad gatherings. This is an explicit graphic material rule.
Do not draw a separate blurred approximation of the desired interference.

Multiply `P_i` by the support and a broad fixed directional factor in `0.35..1`
based on the smooth contour normal and one fixed upper-left lighting direction.
Keep this factor soft enough to leave complete grooves legible. Add the three
nonnegative colored contributions to a fixed dark ground in linear RGB, then
use the existing fixed tone curve once. Additive body layers avoid introducing
unplanned cross-body pattern products. Their crossings retain both families.

Suggested linear-RGB reflection tints: A `(0.72,0.82,0.90)` pewter, B
`(0.35,0.57,0.69)` smoked blue, C `(0.72,0.50,0.39)` pale copper. Relative
strengths `1.0 / 0.80 / 0.48` make copper an accent. Start with common gain
`0.85`, exposure `0.0`, ground `(0.0030,0.0018,0.0048)`, and **zero bloom**.
Aim for silver and graphite, with a few pale intersections and ample headroom.

## Filter the combined phases, preserving the intended interference

Point-sampled grooves, even with regular supersampling, will eventually shimmer.
Filtering each carrier separately and then multiplying loses the broad beat.
Instead, expand the complete material rule before filtering. The profile has
the exact finite Fourier coefficients

```text
C(phi) = sum[n=-m..m] a_n exp(i*2*pi*n*phi)
a_n = binomial(2m,m-n) / 4^m
```

For a product term `(n,k)`, its combined phase is
`chi=(n+k)Phi+(n-k)Psi/2`, with coefficient `a_n*a_k`. The terms `k=-n` have
phase `n*Psi`: they survive when the much finer carrier is unresolved. Include
all coefficients in the first implementation; 17 single terms and 289 product
terms are bounded, deterministic CPU work. Pair conjugates and use harmonic
recurrences to avoid a trigonometric call for every term.

For a locally affine combined phase over a rectangular footprint, multiply its
complex exponential by

```text
sinc(pi * dchi/dx * pixel_width)
* sinc(pi * dchi/dy * pixel_height)
* sinc(pi * dchi/dt * exposure_cell_width)
```

Here `sinc(z)=sin(z)/z`, derivatives use final-pixel coordinates and recorded
source fraction, and the footprint is centered at the evaluation point. This
is exact box integration for an affine phase. It preserves the correct DC
term as grooves become subpixel. Filter the *combined* `(n,k)` phase in both
space and time, including the cancellation in its derivatives.

Start with 2×2 spatial subfootprints and 16 equal cells across the existing
half-frame shutter. Each cell receives this analytic integration, then cells
are averaged in linear light. Sixteen temporal point samples alone are not
the proposed filter. Extend only Engraving's linear-render path to accept the
exposure-cell interval; retain the shared midpoint times, clock and receipts.
At endpoints split cells at the existing `[0,1]` clamp and integrate the held
endpoint part at constant source position. Never sample invented future motion.

Spatial phase gradients are analytic. Temporal slopes can be obtained from
the same continuous source interpolant, or checked symmetric samples inside a
cell; use one-sided samples at endpoints. Compare actual phases at cell corners
and temporal quarter-points against the affine prediction. Subdivide where the
highest retained phase deviates by more than `0.03` radians or the envelope
changes by more than `0.01`. Freeze a conservative refinement schedule from
full-orbit probes if per-frame threshold switching becomes visible. This
criterion controls phase curvature, not carrier frequency: analytic integration
already handles rapid locally linear motion. Confirm convergence; do not mask
errors with exposure changes, arbitrary phase clamps, or indiscriminate blur.

## First finished-detail recipe and proof

| Control | A | B | C |
| --- | --- | --- | --- |
| Semi-axes | 2.12 / 1.17 | 1.83 / 1.45 | 1.60 / 0.93 |
| Fixed orientation | −24° | +48° | +101° |
| Mouth angle in local coordinates | −38° | +104° | +234° |
| Carrier count N | 64 | 74 | 54 |
| Broad radial beat B | 2.7 | 3.1 | 2.3 |
| Broad saddle D | +1.8 | −1.3 | +1.5 |

Use a frontal orthographic camera, height `6.8`, target `(0,0,0)`, fixed up.
Start at 3840×2160; the nominal narrow-axis groove periods are about 5–6 pixels
before the mild contour deformation. The exact gradient determines the actual
local period and filtering. The bounded source motion and lobed supports fit
this field conservatively, but inspect the complete sampled envelope before
freezing composition. Do not normalize the field or exposure per frame.

Implement a dedicated `atelier::engraving::render_linear` returning `Vec<V3>`
plus diagnostics, following the Light path rather than constructing thousands
of tubes. Reuse PNGs, fixed postprocessing, frame receipts and film packaging.
Add an optional Engraving config block; retain old recipe hashes and old pixels.
Tiles are independent, with fixed body/harmonic accumulation order. No new
dependency, GPU, random temporal samples or history reconstruction is needed.

### Artistic acceptance

- Review native 4K frames at 0, 0.25, 0.50, 0.75, the close approach near 0.92746,
  and 1, then real 60 fps excerpts from calm, approach, release and ending.
- At thumbnail scale, read three sweeping, open engraved forms and broad
  gatherings. At native size, resolve crisp hairline cuts with rounded endings.
- The three centers travel substantially with the source. Broad lobes visibly
  shift with companion separation. A frozen source produces a frozen image.
- Preserve a large quiet background and a connected opening in each rosette.
  Reject a filled rosette center, equally bright concentric rings, busy mesh,
  luminous haze, harsh rainbow coloring, and dominant copper/gold.
- Compare a doubled-resolution crop downsampled in linear light with native
  filtering. Broad bands must agree in position and contrast; no crawling
  fringes may appear solely because the camera scale changes slightly.
- Check a locally affine phase against a dense numerical box integral, exact
  DC limits, `Psi=0` registration, and intentional difference-frequency survival
  when both carriers are unresolved. Test temporal integration against a much
  finer shutter reference in the fastest passage. Repeated-worker renders must
  agree. These numerical checks support the visual decision; they do not make it.

Archive the chosen recipe, phase-gradient ranges, minimum local groove period,
largest temporal phase excursion, refinement/error summaries and comparison
crops. Begin the complete fifth film only after the still and motion proof
passes, then retain both verified 4K/60 movies and canonical RGB16 frames.

## First implementation notes

- `RenderConfig.aa=2` controls the initial 2×2 analytic spatial subcells. There
  is no second independent AA control. Generic defaults retain Cartesian source
  axes; the b7 recipe must explicitly include the fixed basis above.
- The direct linear renderer receives the exact raw exposure-cell interval,
  splits it at 0 and 1, and integrates held and live pieces in proportion to
  their original durations. Diagnostic validation independently checks those
  durations and the unchanged requested midpoint.
- Each live interval prepares one dyadic table from `OrbitSeries`, with four
  intervals even at the finest allowed temporal depth. Cell checks include
  spatial corners/edge centers, temporal quarter points and mixed space/time
  corners. Maximum depths default to five spatial and five temporal levels;
  exhausting refinement while above tolerance is an error, not a blurred or
  silently undersampled fallback.
- Diagnostics distinguish inspected from accepted phase residuals, report
  accepted envelope variation, phase gradients, temporal harmonic excursions,
  endpoint holds and refinement counts. The phase residuals remain sampled
  numerical evidence. Culling uses `OrbitSeries::position_bounds`, whose cubic
  Bezier control hulls bound continuous motion; projecting the padded boxes and
  applying the monotone tanh map preserves that protection between samples.
- The filter evaluates all original coefficients through conjugate pairing,
  returns raw values for range validation, and retains the broad zero-carrier
  terms. Its only allowed negative correction is roundoff within `1e-12`.


## Curved opening refinement

The first v16 full-detail proofs felt too much like broken records. v17 adds a
fixed, radius-dependent angular warp `theta_warped = theta + radial_twist*r` to
both the lobe field and mouth. It changes shape without introducing a time
phase. Per-rosette lobe amplitudes are configurable. The original zero twist
and `[.065,.025]` lobes remain omitted from serialized recipes, preserving the
archived v16 recipe hash and default arithmetic.

The radial derivative is bounded below by
`1-|a|-|b| - |twist|*outer_radius*(3|a|+5|b|)/(1-|a|-|b|)`.
Validation requires this to exceed .10. Proposed twists `.8/-.65/1.0` and lobes
`[.10,.035]` give lower bounds `.417/.501/.305`. Thus radial cuts stay ordered,
and the warped mouth stays one connected opening on every nested contour.
Analytic gradients retain the original polar basis and include the warp's
radial chain-rule term. Support and mouth culling conservatively include the
full angular range. Regression cases cover gradients, seams, radial ordering,
legacy serialization, and a previously unsafe mouth-culling configuration.

The initial v17 comparison uses exposure .35 stops and copper strength .65.
The `flow` treatment retains carriers `64/74/54`; `fine` uses `90/100/76`.
Neither is selected until full-resolution and motion proofs are inspected.
