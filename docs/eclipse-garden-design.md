# Eclipse Garden — first detailed design

Design only. Implement after the fifth study is selected; use the same frozen
`0xb7f327f9f722` orbit, 1802 frames, 3840×2160 and 60 fps. All generation and
rendering remain deterministic Rust CPU work with existing dependencies.

## Artistic direction

An almost black garden of three large, asymmetric petals. Their silhouettes
cut into displaced fields of warm pearl light, revealing thick-to-thin crescents,
rose-colored tips and a few copper seams. During an approach, the dark petals
join, swallow interior light and leave a newly shaped outer crescent. During
separation, a thin luminous opening slowly becomes a petal of light.

Keep darkness dominant. Two slender bent leaves and one broader calyx make a
recognizable composition before any fine detail is visible. The secondary
scale is the changing luminous openings; the smallest scale is a delicate,
curved corona rooted in selected bright arcs. Avoid three circular discs with
uniform halos, complete bright outlines, a busy floral illustration, stars,
free rotations, twinkling particles and smoke filling the empty background.

This is an artistic construction of opaque silhouettes and luminous fields,
not a gravitational-optics or astronomical eclipse simulation.

## Source map and time

Use `OrbitSeries::sample(t)` and the fixed orthonormal basis archived in
`03-aurora-b7.json` and written out in `engraving-design.md`. For those axes
`a,b,c`, use

```text
q_i = tanh((dot(p_i,a), dot(p_i,b), dot(p_i,c)) / 1.5)
center_i = (2.55 q_ix, 1.35 q_iy)
long_axis_i(t) = base_long_axis_i * (1 + 0.06 q_iz)
```

Centers visibly follow the recorded bodies. The third component supplies a
small continuous opening/closing of the petals; translation remains the main
movement. Ellipse orientation and the lighting direction are fixed. Source
positions already carry the one common `OrbitSeries` normalization; do not fit
or normalize them again per body or frame. No prehistory, time warping, moving
camera or accumulated animation phase is required.

## Smooth, asymmetric dark petals

For body i, rotate a point relative to its center by the fixed negative angle,
then divide by its short and long semi-axes to obtain `(u,v)`. Define

```text
w(v) = 1 + eta*tanh(v)
z = (u - kappa*(1-v*v)) / w(v)
F_i = z*z + v*v - 1
```

`F_i<0` is the opaque interior. The positive width `w` gives a fuller shoulder
on one end; the shear bends the leaf. With `|eta|<=0.22`, `w>=0.78` everywhere.
Each section with `|v|<1` has two ordered boundary points; the shape remains a
single regular oval with rounded tips, without folded surfaces or self-crossing.
The explicit contour is useful for corona anchors:

```text
v = sin(theta)
u = (1 + eta*tanh(v))*cos(theta) + kappa*cos(theta)^2
```

Use analytical world-space gradients of F. A near-boundary distance coordinate
is `d_i=F_i/sqrt(dot(gradient(F_i),gradient(F_i))+epsilon^2)`, with
`epsilon=1e-6/world_unit`. It is not an exact global signed distance. Deep inside
or outside, masks are already constant; only the narrow boundary neighborhood
uses its distance scale. Check that neighborhood against contour distances in
the proof and refine the distance calculation if its error affects the image.

Join the three dark forms with the symmetric smooth minimum

```text
d0 = min(d_A,d_B,d_C)
D = d0 - h*log(sum_i exp(-(d_i-d0)/h))
T = quintic_smoothstep(-edge_width, +edge_width, D)
```

Use `h=0.025` world units and `edge_width=0.003`. `T` is light transmission:
zero inside the union, one outside. The largest smooth-union expansion relative
to the minimum coordinate is `h*log(3)`, about 0.0275 world units. This small,
declared soft joining region produces a graceful neck at a merger. Log-sum-exp
is stable and symmetric; there is no nearest-body switch, depth sorting or
body-order seam. Do not normalize its gradient near a merger saddle.

## Pearl crescents produced by occlusion

Behind each dark petal, place a luminous copy of its contour, shifted toward a
single upper-left light direction. Use the same source center and deformation.
Let `d'_i(x)=d_i(x-offset_i)` and

```text
E_i = gain_i * exp(-0.5*(d'_i/sigma_i)^2) * W_i
```

`W_i` is a broad directional reflection envelope based on the copy's smooth
outward contour normal and the fixed light direction, ranging from 0.06 to 1.
Use `0.06+0.94*max(dot(normal,light_direction),0)^2`; normals are needed only
near the emitter contour, where their gradient is regular. Fade the Gaussian
smoothly to zero between three and four sigma to preserve truly quiet space.

The offset copy and common dark union do the sculpting. The illuminated side
exposes a broad crescent; the other side is covered. Overlaps erase interior
arcs rather than adding bright outlines through an opaque body. Light from a
neighbor can survive in an opening, producing a pearl wedge or narrow copper
seam without a separately animated flourish.

Vary color slowly across each crescent: warm white at the root, soft shell pink
through its middle, muted copper only at the dim outer tail. Example linear-RGB
tints are pearl `(1.00,0.88,0.71)`, rose `(0.76,0.36,0.32)` and copper
`(0.55,0.22,0.10)`. Interpolate in linear light using the positive outward
distance along the visible crescent. Most luminous area should remain pearl.
Start with body gains `1.25 / 0.90 / 0.65`; retain one exposure for the film.

## Fine corona from the first proof

Build 1536 anchored curves per emitter. Distribute their initial anchors evenly
in contour arc length using a fixed 8192-interval table, and preserve each
anchor's theta as the source deforms the petal. For contour position `r_j`,
outward normal `n_j`, tangent `s_j`, and distance `a=0..L_j`, use

```text
curve_j(a) = r_j + offset_i + a*n_j + bend_i*(a*a/L_j)*s_j
bend_i = base_bend_i + 0.08 q_iz
```

Use 64 intervals per curve initially, refining if projected midpoint error
exceeds 0.10 final pixels. Typical lengths are 0.07–0.24 world units and radii
0.00055–0.0011 world units. A small fixed set of 36 irregularly spaced anchors
per body extends to 0.38–0.48 world units. These longer hairs sit primarily in
the brighter shoulder and taper to zero at their ends.

Length, radius and strength vary through a fixed smooth field along the anchor
indices: seeded control values every eight anchors with C2 interpolation,
multiplied by a few broad fixed angular gatherings. The seed and selected long
anchors belong in the recipe. This variation never changes with frame number;
the geometry carries its identity through motion. Prefer quiet groups of fine
hairs to independently noisy strands. Use the same directional envelope as the
crescent, a smooth root fade, and a long end taper. Their total contribution
starts at about 12% of the broad crescent's integrated luminance.

Evaluate the final linear radiance at each spatial and temporal sample as

```text
L = T * (background + sum_i E_i + corona)
  + (1-T) * dark_petal_color
```

The common transmission masks both broad light and every fine hair. Apply it
before spatial or temporal averaging; multiplying separately averaged masks
and light would leak thin halos through moving occluders. There is no surface
highlight painted on the dark interior. Start with background
`(0.0005,0.00035,0.00065)` and dark petals `(0.00010,0.00007,0.00012)`.

## Stable filtering and first full-detail parameters

Use analytic finite-width coverage for isolated curve segments and the
silhouette's local affine distance, with deterministic 3×3 spatial strata for
the complete composite. Where a curve footprint crosses a dark edge, clip its
integral against the locally affine mask or jointly subdivide the product of
transmission and light until its mean channel error is below `1e-4`; do not
multiply two independent coverage averages. Also subdivide where the implicit
midpoint/edge residual exceeds 0.05 pixels. Merger saddles require direct
implicit evaluation rather than the affine approximation. Gaussian line
footprints have a final-pixel sigma of at least 0.25 pixels, in addition to the
intended physical radius. This is a documented reconstruction footprint, not
a broad glow effect.

Begin with **128 fixed midpoint samples over the existing half-frame shutter**.
Full-source source/geometry probes must show that adjacent exposure samples move
every visible contour and corona point by at most 0.15 final pixels. If not,
increase the fixed sample count for the entire film. The previous Light center
bound was about 31 pixels per frame, but new shapes and corona tips need their
own bound. Sample the original continuous source at every exposure time and
retain the existing endpoint clamp; never invent post-recording motion.

| Control | A: bent leaf | B: broad calyx | C: slender leaf |
| --- | --- | --- | --- |
| Short / long semi-axis | 0.82 / 1.70 | 1.14 / 1.30 | 0.66 / 1.62 |
| Fixed angle | −32° | +52° | +117° |
| Shear kappa | +0.20 | −0.16 | +0.24 |
| Shoulder eta | +0.18 | −0.20 | +0.12 |
| Light offset in world XY | (−0.18,+0.23) | (−0.16,+0.20) | (−0.14,+0.19) |
| Luminous contour sigma | 0.085 | 0.075 | 0.065 |
| Fixed corona bend | +0.18 | −0.15 | +0.12 |

Start with a frontal orthographic camera, target `(0,0,0)`, fixed up, height
`7.4`, exposure `0.0`, and bloom `0.01` or zero. Check the entire deformed
contour plus corona envelope before freezing the crop. Keep the image airy;
do not fill the frame by refitting during an approach.

A dedicated `atelier::eclipse::render_linear` can combine implicit masks,
tile-binned corona curves and emitted radiance, then reuse tone mapping, RGB16
PNGs, receipts and film packaging. Prepare each exposure's body states and
curves once. Accumulate tiles in fixed body/curve order; introduce no floating
atomics or per-frame random samples. Preserve previous study hashes and pixels.

## Acceptance before the sixth full film

- Inspect native 4K at source 0, 0.25, 0.5, 0.75, 0.92746 and 1, plus actual
  60 fps calm, approach, merger, separation and ending excerpts. The dominant
  image must be shaped darkness and luminous openings; corona is a discovery
  at full size, not the composition's main subject.
- Require distinctly tapered asymmetric forms, selective pearl crescents and
  deliberate open background. Reject circular halo symbols, white plateaus,
  pink neon, continuously glowing outlines and a forest of equal-length hairs.
- At mergers, buried light disappears continuously and new openings widen
  continuously. Reject seams, sudden bright joints, nearest-body color changes,
  dark-edge chatter or light leaking through the opaque interiors.
- Freeze source time and verify an identical image; inspect persistent hair
  identities in motion. No fixed-source sparkle or arbitrary phase may remain.
- Compare native results with a doubled-resolution linear-light crop and a
  doubled shutter count at the fastest passage. Curves, dark boundaries and
  thin openings should converge together. Check symmetric union behavior,
  `0<=T<=1`, regular individual contours, finite merger saddles, complete
  occlusion and identical output across worker counts.

Archive the selected recipe, full-source bounds, spatial/temporal convergence
crops, maximum per-sample displacement and union/filter diagnostics. Only after
the visual proof passes should the complete film start; deliver both verified
4K/60 movies, canonical frames and the same provenance as the other studies.
