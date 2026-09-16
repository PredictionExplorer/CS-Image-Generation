# The Remaining Form

Experimental sculpture geometry and offline photography from a frozen three-body
recording. Geometry and photographic recipes are separate. This guide describes
the implemented mechanisms and archive contracts; it does not select a final art recipe.
Use the versioned [recipe files](../tools/remaining_form/recipes/README.md) and artifact receipts
for the selected geometry, pose, material and lighting settings.

## Architecture

| Component | Responsibility |
| --- | --- |
| `src/atelier/source.rs` | Immutable `OrbitSeries`: original 3D paths, one shared translation/scale, transported tool frames |
| `src/remaining/field.rs` | Fixed stock, declared reveal cuts, cumulative excavation and grid diagnostics |
| `src/remaining/field/geometry.rs` | Conservative bounds and nearest-triangle acceleration for the stock |
| `src/remaining/mesh.rs` | Deterministic isosurface extraction, bounded endpoint conditioning, topology checks and binary PLY |
| `src/bin/remaining_form.rs` | Recipe resolution, source verification, mesh archives and verified reuse |
| `tools/remaining_form/render.py` | Blender/Cycles CPU scene construction, physical material settings and image archives |
| `tools/remaining_form/film.py` | Independent cumulative frames, final-object camera examination, receipt verification and encoding |

## Field semantics and units

`OrbitSeries` maps the recording's largest original coordinate span to four
world units. It preserves relative 3D geometry through a single translation and
uniform scale. No display projection, drift, per-frame fit or prehistory enters
this study. Renderer rotation changes the pose of the complete finished mesh.

The fixed stock is either a rounded union of `stock_samples` instantaneous source
triangles or the explicitly configured control ellipsoid. The triangle envelope
is a coarse geometric approximation: inspect `centers_outside_stock` and refine
`stock_samples` when necessary. The triangle stock uses the complete recording;
both stock choices remain fixed while excavation advances.

Each body carries the same ellipsoid. `cutter_axes` gives its **support semi-axes**
along the transported tangent, normal and binormal, in normalized world units.
These are not the final opening radii. The compact kernel is
`K(q) = (1-q)^4 (1+4q)` for ellipsoidal radius `q < 1`, and zero otherwise.

The dose is `C(x,T) = dose_strength * integral_0^T sum_i K_i(x,t) dt`.
Time is calibrated to the fixed recording interval `[0,1]`. `time_samples` counts
canonical integration intervals; output fps does not change them. Kernel fields
are interpolated positively between canonical samples and integrated exactly
over complete and partial intervals. The resulting weights are nonnegative and
nondecreasing. Returning during additional source time adds dose; compressing
identical visits into the same total duration preserves their occupation integral.

Without rim rounding, material is removed at `C >= dose_threshold`.
The fixed art controls are:

- `reveal`: retain `normal.dot(x) <= offset`, using a unit normal.
- `aperture`: subtract the ellipsoid described by `center` and `axes`.
- `rim_rounding`: apply a smooth maximum at the blank/dose intersection:
  `max(a,b) + max(k-abs(a-b),0)^2/(4k)`. Zero uses the exact maximum.

The plane and aperture establish viewing openings; they are explicitly authored
geometry rather than orbital events. Rim rounding removes additional material
near the intersection. Its width is in **implicit-field units**, not an exact
physical bevel radius. Keeping it below the smallest cutter semi-axis preserves
zero-dose blank membership; larger values can also round away initial material.
None of these mechanisms fills previously excavated material or repairs a mesh
independently each frame.

The scalar grid is negative inside surviving material. It is not a signed-distance
field and must not be used directly for sphere-tracing step lengths. Resolution
counts the maximum nodes on an axis; spacing is isotropic, storage is x-fastest,
and the bounds include a physical margin plus two protective cells. Spatial bricks
hold canonically ordered cutter candidates. Parallel slabs own their output and
preserve accumulation order.

## Extraction and diagnostic limits

Each grid cube uses six consistently aligned tetrahedra. Intersections share
global grid-edge identities. Exact zeros share node identities. For native f32
renderer compatibility, roots sufficiently close to a grid endpoint also share
that endpoint. The allowed displacement is
`min(4 * f32::EPSILON * max(1, grid-coordinate magnitude), 0.001 * spacing)`.
Canonical coordinates remain f64.

`extraction` records the tolerance, unique snapped edge count and maximum actual
displacement. Affine tetrahedral field gradients determine face orientation.
Only collapsed canonical-zero triangulation is omitted. This bounded numerical
conditioning changes neither source trajectories nor sampled field values;
there is no component selection, general vertex welding or per-frame material
healing. Introduced pinches fail the audit. The f32 preflight casts coordinates
to f32, promotes them back to f64, then checks area: this distinguishes actual
collapsed geometry from cancellation in f32 cross products. The renderer repeats
these checks on its imported coordinates. Zero-area faces remain errors.

Mesh diagnostics cover oriented manifold edges and vertex links, duplicate and
degenerate triangles, referenced vertices, connected boundary surfaces, signed
volumes and Euler characteristic. **General triangle/triangle self-intersections
are not checked**; `self_intersections_checked` remains false.

### Surface normals

The recipe's `surface_normals` defaults to `field`: shading follows the continuous
implicit field rather than the distribution of triangles around each vertex.
Compact-kernel, ellipsoid, plane, aperture, and smooth-intersection derivatives
are analytic. Triangle-envelope stock uses a declared central difference at
`0.02 * grid_spacing`. An undefined derivative receives a counted full-field
finite-difference fallback; an unresolved zero/nonfinite gradient fails.

The receipt records the normal source, gradient range, difference step and
fallback count. `surface_normals: "mesh"` retains area-weighted triangle normals
for diagnostic comparisons. Neither choice changes mesh positions, faces,
sampled field values, or material history. Supplied PLY normals must be finite
unit vectors with one entry per vertex.

### Interpreting diagnostics

- `max_dose` is a sampled maximum near the underlying stock, including regions
  excluded by fixed reveals. It is not an analytic maximum or a cavity-size prediction.
- Stock/remaining node volumes are coarse grid estimates. Stock counts follow
  the fixed reveals; removal counts include excavation and declared rim rounding.
- A separate inward cavity wall contributes a negative surface-component volume.
  Multiple surface components therefore do not necessarily mean detached solids.
- `solid_grid_components` uses face adjacency. Thin diagonal connections and
  tiny islands need spatial refinement before being interpreted as real parts.
- `max_source_step` and `minimum_cutter_radius_cells` help choose temporal and
  spatial resolution. Neither alone certifies a thin wall or chamber connection.

## Build a mesh

Run these commands from the repository, using the selected frozen builder binary:

```sh
cargo build --release --bin remaining_form
/path/to/remaining_form config --output /path/to/geometry.json
/path/to/remaining_form resolve --config /path/to/geometry.json --output /path/to/resolved.json
/path/to/remaining_form --threads 4 build \
  --orbit /path/to/source.orbit \
  --config /path/to/resolved.json \
  --output /path/to/sculpture \
  --time 0.65 --resolution 256
```

`--time` is the cumulative endpoint: the example includes all exposure from zero
through 0.65. `--time` and `--resolution` overrides appear in the archived resolved
recipe. The output contains `mesh.ply`, `recipe.json` and `build.json`.

Add `--resume` to the same invocation for reuse. The build identity includes the
orbit hash, builder executable hash and resolved recipe. The source is hashed
before and after reading; reuse checks the archived recipe and mesh content.
An orphan mesh without a complete receipt is preserved and requires a new output
directory. A per-directory advisory lock prevents concurrent writers.

## Photograph the same object

The adapter requires **Blender 4.5.14**, background mode and Cycles CPU. The isolated
Linux installation uses `blender-4.5.14-linux-x64/`. Its recorded distribution
archive, `blender-4.5.14-linux-x64.tar.xz`, has SHA-256:

`9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3`

Verify that archive before extracting it into a dedicated directory. The adapter
records the actual executable hash, build identity and bundled color-management
file hashes; it rejects an `OCIO` environment override. Version checking alone is
not a substitute for retaining the recorded binary and archive provenance.

```sh
/path/to/blender-4.5.14-linux-x64/blender \
  --factory-startup --background --threads 4 --python-exit-code 1 \
  --python /path/to/repo/tools/remaining_form/render.py -- \
  --mesh /path/to/sculpture/mesh.ply \
  --recipe /path/to/studio.json \
  --output /path/to/view-front --view front
```

`studio-clay.json` and `studio-porcelain.json` are starting configurations. The
studio specifies cameras, lights, one rigid model rotation, metres per source
unit and material scattering distances in millimetres. Blender imports native
f32 vertex coordinates; the archived f64 PLY remains the geometry authority.
No displacement, remeshing or nonuniform model scaling is applied.

Each view archives the resolved recipe, request, `.blend` scene, display RGB16
`render.png`, scene-linear RGB32 `render.exr` and a completed `receipt.json`.
`--resume` requires identical inputs/runtime/settings and verifies every archived
artifact. Fixed seeds and samples support repeatability; cross-platform pixel
identity is not promised.

## Motion and final-object examination

The Python archive helpers use POSIX advisory locks (Linux/macOS).
Freeze a numeric `ground.height_units` in the studio before making a film. A null
height is an automatic still-proof fit and would move as the sculpture changes.

```sh
python3 tools/remaining_form/film.py \
  --builder /path/to/remaining_form \
  --blender /path/to/blender-4.5.14-linux-x64/blender \
  --render-script /path/to/repo/tools/remaining_form/render.py \
  --ffmpeg /path/to/ffmpeg \
  --orbit /path/to/source.orbit \
  --geometry-recipe /path/to/geometry.json \
  --studio-recipe /path/to/fixed-studio.json \
  --output /path/to/film \
  --frames 144 --fps 24 --workers 1 \
  --turntable-frames 48 --turntable-degrees 35
```

The excavation samples the complete `[0,1]` interval with a fixed camera. The
optional examination reuses the final mesh and moves the camera. Opening and
closing holds repeat verified frames. Frames are point samples: this workflow
does not interpolate changing mesh topology or perform temporal supersampling.
It allows one or two concurrent frame jobs, each using four child threads.

Film archives retain input hashes and per-frame mesh/render evidence. Encoding
produces H.264 `yuv420p`; the complete movie is decoded and its frame count checked
before its completion receipt is published. `--resume` verifies the existing
request and delegates artifact checks to the original build/render programs.

## Checks

```sh
cargo test --lib remaining::
cargo test --bin remaining_form
cargo clippy --lib --bins --tests -- -D warnings
python3 -m unittest discover -s tools/remaining_form -p 'test_*.py'
```

The Rust tests exercise cumulative monotonicity, repeat visits, temporal
convergence, actual depth, acceleration parity, worker/order independence,
fixed reveals, rim rounding, manifold examples and native-precision conditioning.
Python contract tests cover recipes, scene archives and film orchestration without
requiring a render. Refine source sampling and spatial resolution, inspect multiple
views, and check moving topology separately from these numerical contracts.
