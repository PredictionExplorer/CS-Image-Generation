# Aurora Veils

## Intended image

Three enormous curtains of light sweep across a blue-black void. Jade lower
edges open into ice-cyan rays and a faint violet upper haze. Some regions are
almost absent; others gather into bright, finely striated folds. Dark channels
between the curtains are as deliberate as the light. Bodies approach, their
curtains gather and overlap, then separate into broad open sweeps.

This is an atmospheric emission study. Its dominant marks run vertically,
across broad sheets. Calligraphy follows the paths with narrow bands; Loom
builds an enclosed woven form. Aurora has no cross-weave, closed shell, cloth
solver, tumbling object, or studio-lit metallic surface.

## Source mapping: large movement first

Freeze the principal axes of the complete recorded source once. The selected
first two axes capture about 92% of this seed's temporal positional variance
(91.61% measured over 20,001 sampled rows after centering each body). Both must
drive the visible composition. Never select an arbitrary fixed YZ projection
or recompute the axes per frame. Normalize with explicit fixed source scales
of 1.5 world units and a bounded tanh response, recorded in the recipe.

Let `qxi(s), qyi(s), qzi(s)` be body i's normalized principal components,
`t` the visible source fraction, `u` recent chronology, and `v` curtain height.
Use actual history `s=t-h(1-u)` including verified physical prehistory.

A first bounded construction is:

```text
x = 1.8 qxi(t) + Li (u - 0.5)
y = 1.8 qyi(s) + Hi(u,s) v
z = zi + Di(s) + Ai(s) sin(pi v) Fi(s)
```

Here unsheared `x` is strictly monotone in u and `Hi` is strictly positive.
After constructing this graph, apply a smooth world-height shear:

```text
lean_i(t) = 0.7 * (0.65 qxi(t) + 0.35 qyi(t))
X = x + lean_i(t) * tanh((y - 0.8) / 1.3)
Y = y
Z = z
```

This gives each body a different source-driven curve through the luminous
height. Its inverse subtracts the same function of Y, so sheets stay open and
regular, without self-intersection. The XY area determinant is unchanged.
Chronology stays monotone in the unsheared coordinate; rendered world-X need
not be monotone. Adding a bend directly in local `v` would not provide this
guarantee when the source-driven lower edge is steep.

The current first source component moves the
entire curtain by an amount comparable to its height. The second component
moves its lower edge by the same large amount across history. These are the
principal gestures, not small modulation on a fixed shape.

Starting values: widths `Li=[5.2,4.8,5.6]`, history `h=0.26`, and height
`Hi=0.25+(1.7+0.6 proximity_i(s)+0.2 speed_i(s))*sin(pi u)^2`. Check a longer history if the
actual source only gives small ripples. Use speed and proximity for smooth
height and ray-density changes, keeping geometry bounded. Fold phase comes
from absolute source arc length, with roughly two or three broad folds per
history window; no unrelated periodic time animation. The third source
component contributes depth. Lanes `zi=[-1.2,0,1.2]`, `|Di|<=0.18`, and
`|Ai Fi|<=0.28` preserve inter-sheet separation while allowing projected overlap.

Measured visible-source lateral spans are A 1.32, B 3.13 and C 2.81 world units;
lower-edge vertical spans are A 1.95, B 3.01 and C 2.38. The rigorous bounds,
including arbitrary genuine prehistory, are unsheared X ±4.6, Y [-1.8,4.55]
and Z ±1.66. The height shear adds at most 0.7 horizontally, giving final
X ±5.3. The fixed 16:9 camera retains lateral margin.

The newest end of the history corresponds to the current body but feathers
optically into darkness; it is not a visible marker or solid seam. Source
endpoint times remain 0 and 1; prehistory only forms the opening curtains.
This is a designed mapping of gravitational motion, not a simulation of
atmospheric aurora.

## Light, detail and composition

- Use dominant jade/celadon, secondary ice cyan, and restrained violet high haze.
  Keep the brightest lower seams nearly white only at selected gatherings.
  Avoid three equally saturated rainbow curtains.
- Render thin, low-absorption emissive sheets with fine curved rays. Use
  broad smooth density variations to create alternating luminous fans and dark
  channels. Feather sides and upper ends until they disappear; no rectangle
  outlines, solid hems, punched holes, or uniformly bright mesh.
- Start with 1024 by 192 surface intervals per curtain, smooth derivative normals,
  and 3072 rays with 160 intervals each. Broad brightness gatherings follow
  source arc length. Fine radius and upper-length variation uses fixed ray
  identity, with deterministic aperiodic value noise blended at half, one and
  1.9 times its characteristic scale. The compatible `ray_detail_period` setting
  is a scale rather than a repeating period: 13 gives roughly 6.5–25-ray detail;
  the default remains 17. Quintic interpolation has continuous first and second
  derivatives, and separate seeds vary radius and upper length. This keeps
  the finest detail from flashing through source-phase cycles during a close
  encounter. Detail should survive native 4K but
  support the large gesture when viewed small.
- Use a fixed shared world-Y color profile for sheet and ray emission, giving
  the same height the same hue. It spans the full designed vertical envelope;
  the large moving lower edge must not be reduced to accommodate the palette.
- Begin near frontal: camera `(2,1.4,12)`, target `(0,1.3,0)`, up `(0,1,0)`,
  orthographic height `7.0` at 16:9. Fit/check the actual full-motion envelope.
  Slight depth reveals overlapping veils; no camera orbit or changing zoom.
  Aim for substantial dark space through the center and above the forms.
- Reduce reflected studio light strongly. The material emits its own light;
  bloom is a restrained optical finish, not the substance of the curtain.

## Renderer fit and one required optical extension

The current `Scene` supports smooth triangle sheets and analytic fine `Strand`s;
its deterministic depth-sorted optical stack already supports overlapping
transparent layers. Surface UVs are available, while strands currently use
arc length rather than authored height UVs. A shared world-height profile can
therefore keep both in agreement without a general strand-coordinate redesign.

Add an opt-in continuous Aurora profile that scales emitted radiance and optical
depth, with explicit smooth end/side envelopes. **Current emission is added
independently of optical depth: reducing absorption alone does not fade light.**
Account for this in ray ends and sheet boundaries. Existing unprofiled materials
must preserve their output. Favor smooth low-dimensional fields over hundreds
of abrupt material zones; retain deterministic temporal and spatial filtering.

## Acceptance gates

1. Inspect full-detail 4K proofs at opening, 10%, 25%, 50%, 75%, closest approach
   and ending. Require obvious changes in placement, lower-edge shape and negative
   space. If it resembles a stationary curtain with shimmering texture, change
   the source mapping while retaining the full-detail configuration.
2. Inspect native 4K with full rays: broad luminous volumes, visible transparency,
   fine interior structure, no hard luminous cutoffs or uniform neon wash.
3. Watch actual approach and release passages at 60 fps. Require flowing
   source-driven motion, no brightness popping, aliasing or disappearing layers.
   Only then freeze a recipe for the complete 1802-frame film.

## First 4K proof revision

The v09 images had broad flowing silhouettes but read as a glowing equalizer:
straight parallel rays, a nearly uniform bright lower band, and a hard luminous
hem. The height shear above addresses the ray geometry. It retains the full
1024×192 sheets and 3072×160 rays. The accompanying material trials soften the
lower fade to 0.12–0.18 of height, reduce sheet emission to 0.015–0.03, and use
ray radii around 0.00025–0.00032. The world-height emission curve should avoid
its first broad green plateau. These changes are to be judged together in real
4K proofs before selecting a recipe.

Rays still follow the analytic sheet without an artificial depth shift. A
0.001-world-unit front offset could reduce centerline/triangle order changes,
but does not guarantee ordering across a sloped ray's antialiased footprint.
The quieter, thinner sheet is the first proof condition; add a documented bias
only if actual motion reveals a remaining defect.

The v10 dense proof improved the atmosphere with 6144 finer rays per body and
quieter sheets. Remaining comb-like repetition came from the former sinusoidal
radius/length pattern. The final texture revision replaces those fine sine
waves with the fixed-index value noise above; chronological ray spacing, broad source
gatherings, source clocks, geometry density and palette are otherwise unchanged.
