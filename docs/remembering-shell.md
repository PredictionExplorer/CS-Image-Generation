# The Shell That Remembers

A growing, open sheet with thickness, driven by features of a recorded three-body
triangle. This is a **derived shape encoding**: the source recording is unchanged,
but the shell's vertices are authored construction geometry, not physical body
positions or copies of their paths. Curated geometry and studio revisions belong
in [the recipe directory](../tools/remembering_shell/recipes/); this guide does not
select a final recipe.

## Architecture and API

- `src/remembering_shell.rs` provides `Recipe::validate()` and
  `build(&OrbitSeries, &Recipe) -> SilkResult<BuiltShell>`.
- `BuiltShell` contains the closed material-boundary mesh, one supplied shading
  normal per vertex, and source-mapping/sampling diagnostics.
- `src/bin/remembering_shell.rs` resolves recipes, verifies the frozen `.orbit`
  source, checks native f32 geometry and archives the mesh and its identity.
- `tools/remaining_form/render.py` supplies the pinned Blender/Cycles CPU scene
  adapter. `tools/remaining_form/film.py` supplies the compatible frame controller.
- `tools/remembering_shell/audit_mesh.py` performs a separate bounded geometric
  intersection audit of the exported PLY.

An open arc gives the construction a mouth and two free edges. Outer and inner
walls, rounded long lips and end caps form a **closed material boundary**; the opening
is an actual gap in the form, not missing mesh polygons. There is no enclosing
ellipsoid whose silhouette must survive the process. Whether this produces a
convincing shell, scroll or folded form still requires inspection from several views.

## What the source controls

`OrbitSeries` retains the original 3D trajectories after its one shared translation
and uniform scale. Features are sampled at `feature_samples + 1` fixed source
times over `[0,1]`. Pair identities remain ordered.

For squared edge lengths `A=d01²`, `B=d12²`, `C=d20²`, the two shape components are:

```text
x = (2A - B - C) / (A + B + C)
y = sqrt(3) * (B - C) / (A + B + C)
```

The complete feature preparation also computes:

- **Size:** bounded log perimeter relative to the whole-recording median perimeter.
- **Closeness:** a smooth combination of all three pair distances; its reference
  distance is the median perimeter divided by twelve.
- **Turning:** `sum(edge × edge_velocity) / sum(edge²)`, projected onto the normal
  of the largest-area sampled triangle. Its magnitude is normalized by the
  recording's 90th-percentile absolute spin, bounded with `tanh`, smoothed and
  integrated. This is a rotation proxy, not physical angular momentum.
- **Travel:** normalized summed `BodySample.arc_length`, which drives rib spacing.

Size, closeness, shape and spin use a Gaussian with `smoothing` as its source-time
standard deviation. The support extends to three standard deviations, truncated
and renormalized at recording ends. Cumulative travel retains its original samples.
Componentwise monotone cubic Hermite interpolation (PCHIP) limits overshoot between
the frozen knots; it is C1, not a claim of globally C2 feature interpolation.

Preparation always uses the complete recording. This allows neighboring future
samples to influence a smoothed feature. It is a fixed design process, not a
causal biological growth simulation. Features and their normalization are never
recomputed from the currently visible prefix.

## Growth clock and permanent history

Let `u` be the full construction coordinate and `f=seed_fraction`. The authored
starter occupies `0 <= u <= f` and exists at source time zero. There is no invented
physical prehistory. For the rest of the surface:

```text
s = clamp((u - f) / (1 - f), 0, 1)
source_time = 6s³ - 8s⁴ + 3s⁵
```

This fixed monotone clock has zero first and second derivatives at the seed join.
The builder inverts it by deterministic bisection to honor `source_fraction`.
Construction growth and source time therefore are not related by a simple linear
stretch after the seed.

Completed canonical rows use the same `u=i/longitudinal_segments` in every frame.
A moving terminal row and its cap finish each prefix. Earlier canonical wall
positions and normals remain unchanged; the terminal row/cap are reconstructed
as growth advances. Output fps never determines feature integration or geometric
sampling. Neither old geometry nor texture phase is normalized by the current end.

## Authored shape and source modulation

The axis, starter, thickness and open-arc grammar are explicit art controls.
The axis has positive azimuthal motion (`turns >= 0.05`) and nonnegative `rise`,
avoiding the singular projected-radial frame permitted by a purely radial axis.
Source turning rotates the growing **profile**, rather than changing the axis tangent.

The regularized base growth laws are:

```text
width = 0.045 + (rim_radius - 0.045) * (0.25u + 0.75u²)
axis radius = 0.1 + axis_radius * (0.65u + 0.35u²)
half thickness h = thickness/2 * (0.55 + 0.45u²(3 - 2u))
```

The axis sweeps the configured `turns`; height follows `1.55*width + rise*u`.
Source size and encounter severity modulate width exponentially. Ordered shape
components modulate a second angular harmonic. `triad_relief` (default `0.18`)
adds threefold folding, strengthened by `clamp(1-x²-y²,0,1)` after interpolation.
Profile rotation, authored twist, integrated source turning and `lip_asymmetry`
control orientation and unequal extension of the free edges.

`aperture_degrees` is the **occupied arc span**: its missing sector is
`360 - aperture_degrees`. The conservative recipe constraint
`lobe_response * (0.65*sqrt(2) + triad_relief) <= 0.9` keeps the radial factor
positive. Positive radius and a nonsingular axis frame do not certify absence
of local offset foldover or global self-intersection.

### Rib placement

Rib phase mixes construction chronology and normalized source travel:
`cycles = ribs * (0.45u + 0.55*travel)`. Relief is zero throughout the authored
starter and activates smoothly through the fixed source clock. In addition to
the nominal sampling check, the prepared source must advance by no more than
`0.1` rib cycles in any canonical longitudinal interval when relief is active.

For transverse coordinate `v` in `[-1,1]`, let `W=(1-v²)²` and let `r(u)` be the
nonnegative rib height. With smooth base position `S` and base normal `n`:

```text
outer wall = S + n * (h + rW)
inner wall = S - n * (h + 0.5rW)
```

The base surface and its frame contain no ribs. The local normal gap is
`2h + 1.5rW > 0`; the rounded lip charts are unchanged because `W` and its first
derivative vanish at both free edges. This avoids applying wall thickness along
the normals of a high-curvature corrugated midsurface. Different parts of the
surface can still intersect and require the separate audit. Shading normals are
evaluated from the actual wall's parametric finite differences and cap frames.

## Build, photograph and animate

```sh
cargo build --release --bin remembering_shell
/path/to/remembering_shell config --output /path/to/shell.json
/path/to/remembering_shell resolve --config /path/to/shell.json --output /path/to/resolved.json
/path/to/remembering_shell --threads 4 build \
  --orbit /path/to/source.orbit --config /path/to/resolved.json \
  --output /path/to/shell-mesh --time 0.6
```

The geometry recipe is a flat shell `Recipe`, with no Remaining Form `field` block
or voxel-resolution setting. Resolution comes from its longitudinal, transverse,
lip and feature sampling controls. The output contains `mesh.ply`, `recipe.json`
and `build.json`. Add `--resume` to the same request for verified reuse.

The archive identity includes source bytes, builder executable and resolved recipe.
The source is checked before and after loading. Resume verifies the embedded
identity, archived recipe and PLY hash. Native f32 preflight rejects nonfinite
positions, collapsed faces and orientation reversals; it repairs nothing. Supplied
parametric normals are written with the canonical f64 geometry.

Use the Blender 4.5.14 setup and still command documented in
[the shared rendering guide](remaining-form.md#photograph-the-same-object).
Keep one uniform physical scale and a declared rigid pose. The reusable film
controller accepts the shell builder through its common `resolve`/`build` interface:

```sh
python3 tools/remaining_form/film.py \
  --builder /path/to/remembering_shell \
  --blender /path/to/blender-4.5.14-linux-x64/blender \
  --render-script /path/to/repo/tools/remaining_form/render.py \
  --ffmpeg /path/to/ffmpeg --orbit /path/to/source.orbit \
  --geometry-recipe /path/to/shell.json --studio-recipe /path/to/fixed-studio.json \
  --output /path/to/shell-film --frames 144 --fps 24 --workers 1
```

Freeze numeric `ground.height_units` before animation. Frames use explicit source
fractions; the optional final camera examination reuses the completed mesh. The
controller's historical phase label `excavation` denotes growth when this builder
is selected. It does not interpolate meshes or provide temporal supersampling.
Renderer, color-management, image and movie receipts retain their own identities;
cross-platform pixel equality is not promised.

## Geometry audits and tests

The build checks closed oriented manifold topology, finite positions, unit normals,
positive signed volume and native precision. Its `self_intersections_checked`
flag remains false. Audit the actual exported mesh separately:

```sh
python3 tools/remembering_shell/audit_mesh.py \
  --mesh /path/to/shell-mesh/mesh.ply --output /path/to/audit-f64.json
python3 tools/remembering_shell/audit_mesh.py \
  --mesh /path/to/shell-mesh/mesh.ply --output /path/to/audit-f32.json --native-f32
```

The auditor streams BVH candidate pairs, uses adaptive exact orientation and
rational plane intersections, and permits only contacts confined to shared
topological vertices/edges. Its triangle limit is two million. Candidate-pair work defaults to ten million
and can be explicitly bounded up to fifty million. Reaching a limit or interruption is **incomplete**,
not a pass. An audit applies only to the recorded mesh/hash and precision; a
lower-resolution proof does not certify a denser output or every growth prefix.

```sh
cargo test --lib remembering_shell::
cargo test --bin remembering_shell
cargo clippy --lib --bins --tests -- -D warnings
python3 -m unittest discover -s tools/remembering_shell -p 'test_*.py'
```

Tests cover clock inversion, seed continuity, bounded endpoint derivatives,
PCHIP ranges, monotone/local rib phase, retained growth rows, positive wall gap,
closed boundaries, archive verification and legal/illegal triangle contacts.
Inspect silhouette, mouth depth and free-edge shape before adding fine ribs.
Compare views and source recordings: numerical validity alone does not establish
a distinctive form or a useful visual account of the motion.

## Selected study

The versioned recipes in `tools/remembering_shell/recipes/` select the petalled
spiral with uncoated celadon inside and ivory outside. `shell.json` is used for
both stills and motion, including the same 64 ribs and mesh sampling.

For smooth visible growth, pass `--source-times` with the supplied
`source-times.json`, `--frames 120`, `--turntable-frames 48`,
`--turntable-degrees 25`, `--start-hold 0.5`, and `--end-hold 1.5`. The explicit
source times apply the fixed source clock to a uniform construction progression.
They retain the complete recorded interval and all earlier geometry. At 24 fps
the selected film is nine seconds. The generic controller accepts one through
four parallel frame jobs, each limited to four threads.

The final 1,278,752-triangle mesh was audited using an explicit 20-million-pair
work cap. All 15,066,274 candidate pairs completed without intersections or
ambiguous contacts. The audit defaults to 10 million pairs; a caller can choose
a bounded cap up to 50 million. The full-resolution production mesh has the
same content hash as the audited mesh. This is a geometric intersection check,
not a physical fabrication or strength certification.
