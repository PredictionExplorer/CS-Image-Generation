# Estuary: Into depth

This study photographs the **actual pigment fields from a completed Estuary
painting** as relief, separated layers, or a combination of both. The color,
concentration and recorded source history remain inspectable throughout.

The depth is an artistic mapping. The underlying Estuary simulation is
two-dimensional pigment transport; neither the height field nor the stacked
sheets claim to be a three-dimensional fluid simulation or measured paint.

## The three families

| Recipe `family` | Geometry | Source states |
| --- | --- | --- |
| `relief` | One closed pigment-height surface with a finite-thickness base | Completed painting |
| `layered` | Separated closed, flat sheets | Verified earlier states and the completed painting |
| `hybrid` | Separated closed pigment-height surfaces | Verified earlier states and the completed painting |

Height comes from `sum(concentration * specific_volume)`, with authored relative
volumes for the three channels. Display RGB does not determine height. The
renderer applies `relief_mm` to that relative field. `canvas_width_m`,
`base_thickness_mm` and `layer_gap_mm` define explicit dimensions; meshes use
metres. Height samples sit at cell centers, so the outermost geometry is inset
by half a mesh cell from the declared domain boundary. Each subsequent layer
clears the preceding layer's highest point.

`relief_smoothing_mm` optionally spreads the geometric height with a Gaussian of
that physical standard deviation. Positive normalized weights and symmetric
boundary extension preserve the height integral. Color, UVs, archived pigment
concentrations and the original simulation remain untouched. The selected
studies use 0.35 mm to suppress mesh-scale scalloping under raking light.

The intended palette assigns channel 0 to blue, channel 1 to ivory, and channel 2
to vermilion. These independent concentration fractions control roughness,
transmission, subsurface response and optional metallic accents. Linear pigment
reflectance supplies the material color. The lowest sheet is made
nontransmitting. Optional micrometre-scale bump changes shading normals only.
These are Principled material approximations, not measured spectral pigments.

The optional `transmission_model: "clear-glaze"` mixes an opaque pigment BSDF
with a separately tinted glass BSDF, using complementary weights from the
pigment fractions. `glass_tint` is linear RGB and `glass_roughness` controls its
surface response. The default `"pigment"` mode retains the original Principled
transmission. This distinction is an authored material choice, not a claim of
measured optical properties.

## Stages and contracts

1. **Authenticate the completed painting.** `prepare.py` checks the source
   request and receipt, the complete source endpoint, the resolved recipe, the
   original orbit, and the exact archived pigment and linear-image files.
2. **Authenticate earlier history when needed.** `history.py` replays the
   original immutable Estuary runtime on the original GPU and dependency runtime.
   Requested times must land on canonical simulation steps. The ledger becomes
   complete only when the replayed final `final-state.npy` has exactly the same
   SHA-256 as the original. An earlier state is never invented from the final
   image, faded into existence, or accepted from an unrelated seed.
3. **Prepare a bundle.** `prepare.py` produces `bundle.npz` and `manifest.json`.
   Concentrations are resampled using exact pixel-area averages over the full
   guard domain. Texture and mesh resolutions are independent. Arrays are
   bottom-to-top, with a bottom-left UV origin; the central painting and larger
   domain bounds are recorded. The manifest binds the original source, render
   receipts, preparation settings, code and any verified history ledger.
4. **Photograph the geometry.** `render.py` constructs closed solids, embeds the
   linear textures, builds the material graph, and uses pinned Blender/Cycles
   with OptiX. CPU fallback is disabled. The resolved recipe, camera poses,
   editable scene, display image, linear image and actual GPU identity are
   archived.
5. **Compare and publish.** `experiment.py` freezes copies of the renderer,
   sibling `materials.py`, bundle and every recipe. It runs a finite queue,
   verifies every completed artifact, optionally encodes motion, and updates the
   gallery only with completed cases.

`layered` and `hybrid` require verified earlier history. A bundle with just the
completed state is accepted for `relief` only.

## Environment

Run preparation, replay and the experiment controller from the repository root.
The Python dependency lock is
[`../estuary/requirements.txt`](../estuary/requirements.txt). Install it with
`python -m pip install --require-hashes -r tools/estuary/requirements.txt` in a
virtual environment. Replay additionally requires **the original Python,
NumPy, glcontext, GPU/driver and Estuary code identities**; installing a different
compatible version does not satisfy that contract.

Rendering uses the official **Blender 4.5.14** build and its bundled Python/NumPy.
Motion encoding uses explicit FFmpeg and ffprobe executables. The experiment
controller and gallery are otherwise standard-library Python.

For this workspace, a usable local preparation/test interpreter is:

```text
/Users/tarasbobrovytsky/Dev/CS-Estuary/.venv-estuary/bin/python
```

The rendering server workspace is `/home/user/estuary-depth` on
`user@100.76.88.48`. Its source, history, bundles and logs are separate from the
original paintings in `/home/user/estuary/production/fc5cf08`. Keep experiment
versions in new directories and leave the original archives immutable.

## Commands

The following examples use explicit paths to an existing completed painting,
the matching Python environment, and a directory of JSON studio recipes.
Replace the illustrative paths with the actual selected archive and version.

### Recover real intermediate states

```bash
python -m tools.estuary_depth.history \
  --run /path/to/completed-estuary-run \
  --output /path/to/verified-history \
  --fractions 0.35 0.65
```

Replay rejects mismatched source/runtime/GPU identities before accepting a
history ledger. Interrupted or mismatched replays preserve their files and do
not certify them as complete.

### Prepare the maps

```bash
python -m tools.estuary_depth.prepare \
  --run /path/to/completed-estuary-run \
  --history /path/to/verified-history/history.json \
  --output /path/to/depth-bundle \
  --width 2048 --height 1536 \
  --mesh-width 768 --mesh-height 576
```

The command-line preparation path requests the 0.35 and 0.65 states when
`--history` is supplied. Omit `--history` for a final-state-only relief bundle.
The Python `build_bundle` API supports an explicit list of other verified times.

### Render one study

```bash
/path/to/blender --factory-startup -b --threads 4 \
  --python-exit-code 1 --python tools/estuary_depth/render.py -- \
  --bundle /path/to/depth-bundle \
  --recipe /path/to/studio-recipes/relief.json \
  --output /path/to/new-render
```

`render.py` requires an empty output directory. It imports `materials.py` from
its own directory, including when both files are frozen into an experiment.
Recipes reject unknown controls; omitted controls resolve to the renderer's
explicit defaults. The `name` labels the study and `family` selects the geometry.

For example, a minimal relief recipe is:

```json
{
  "name": "Porcelain current",
  "family": "relief",
  "relief_mm": 8,
  "camera": {"tilt_degrees": 12, "azimuth_degrees": -30},
  "render": {"resolution": [1024, 768], "samples": 64}
}
```

The render aspect must match the painting. The material recipe exposes
per-pigment roughness, blue/ivory transmission, subsurface distance in millimetres,
coat, vermilion metallic accent, and micro-bump distance in micrometres.

### Render a finite comparison

```bash
python -m tools.estuary_depth.experiment \
  --blender /path/to/blender \
  --render-script tools/estuary_depth/render.py \
  --bundle /path/to/depth-bundle \
  --recipes /path/to/studio-recipes \
  --output /path/to/new-experiment \
  --workers 2 \
  --baseline /path/to/completed-estuary-run/poster.png
```

The default is two simultaneous Blender processes, each limited to four CPU
threads and using OptiX. `--workers` accepts 1 through 3; memory use still depends
on map size, geometry, layers and output resolution. Begin with one worker when
changing those budgets. No unrelated process or global device setting is changed.

Repeat the identical command to verify and reuse completed studies. Changes to
recipes, source bundle, renderer/material code, Blender binary or motion settings
require a new output directory. Changing only worker count is allowed.
Incomplete attempts belonging to the experiment are moved intact to
`CASE.incomplete-TIMESTAMP` before replacement. Unowned output directories are
rejected. An advisory lock prevents duplicate controllers.

### Film a completed painting

Add these arguments to the experiment command:

```bash
--motion-frames 144 --fps 24 \
--ffmpeg /path/to/ffmpeg --ffprobe /path/to/ffprobe
```

The supported range is 1 through 1441 frames and 1 through 60 fps. The painting
and all historical layers remain **frozen** throughout this film. Only the
camera changes, interpolating the recipe's `camera.orbit_start` and
`camera.orbit_end` with a smooth start and finish. This is a camera examination,
not another simulation of growth. `camera.json` records every pose.

The film is H.264 at CRF 18 with four encoder threads. Encoding is followed by
resolution, cadence and frame-count checks and a complete decode. A partial or
failed movie is never advertised as complete. SIGTERM/interrupt stops the
controller's owned render and encoder processes and preserves finished work.

### Open or rebuild the gallery

Each finished case immediately updates `EXPERIMENT/gallery/index.html`.
The gallery presents the recipe names, family groups, native stills, optional
films and the frozen flat-painting reference.

```bash
python -m tools.estuary_depth.gallery --experiment /path/to/experiment
python tools/atelier/serve_review.py /path/to/experiment/gallery --port 8784
```

## Archive layout

```text
experiment-request.json       Immutable batch inputs and their hashes
status.json                   Current per-case state, process IDs and errors
inputs/
  render.py, materials.py     Exact renderer and material code
  bundle/                    Frozen NPZ and authenticated manifest
  recipes/                   Exact submitted recipes
  baseline.*                 Optional flat reference
logs/CASE.log                 Dedicated Blender output
CASE/
  request.json               Resolved recipe, source/code/Blender identities
  recipe.json                Resolved studio controls
  camera.json                Exact camera matrices and angles
  scene.blend                Editable, texture-packed scene
  render.png                 RGB16 sRGB display image
  render.exr                 RGB32 scene-linear image
  frames/000000.png ...       RGB16 motion frames, when requested
  receipt.json               Renderer completion, hardware and artifact hashes
  film.mp4                   Verified motion, when requested
  experiment-result.json     Experiment completion and verified movie identity
gallery/                     Portable published comparison
```

For motion, `render.png` and `render.exr` show the first camera pose. The linear
EXR is archived for that first pose; the remaining movie frames are display PNGs.
Fixed settings improve repeatability, but different GPU/driver/platform versions
are not promised to produce byte-identical pixels.

## Verification

The CPU tests exercise source and history provenance, conservative resampling,
material units, exact archive reuse, worker bounds, failure isolation, process
cleanup and movie-completion checks. Native material-graph, packed-float reload and camera-framing tests require
Blender and are skipped in an ordinary Python interpreter. The archive tests use
small data blocks and do not render on the GPU. They verify exact float32 texture
values after saving/reopening the scene, and the 0.40 m by 0.30 m overhead framing.

```bash
python -W error -m unittest discover \
  -s tools/estuary_depth -p 'test_*.py' -v
uv tool run --from ruff==0.15.10 ruff format --check tools/estuary_depth
uv tool run --from ruff==0.15.10 ruff check tools/estuary_depth
```

CI runs the CPU contracts using the pinned Estuary dependency environment.
GPU renders and visual review remain necessary to evaluate materials, light,
composition and the appearance of the camera movement.

## Curate a collection across experiments and seeds

A curation catalog can put selected native-resolution stills first, pair them
with separately rendered camera films, and retain all exploratory studies after
them. Catalog order is the gallery order; selection never rewrites source
archives. Every case is bound to its own archived recipe controls, bundle and
renderer hashes before reuse or publication, so swapping case directories cannot
silently give one render another study's label.

```json
{
  "studies": [
    {
      "still": "/path/to/native-experiment/selected-relief",
      "motion": "/path/to/orbit-experiment/selected-relief",
      "formation": "/path/to/verified-formation-and-orbit-film",
      "name": "Porcelain current"
    },
    {
      "still": "/path/to/proof-experiment/layered-study"
    }
  ],
  "baselines": {
    "0xb7f327f9f722": "/path/to/original-estuary/poster.png"
  }
}
```

Only `still` is required. Paths must be absolute. `motion` must name a completed
experiment case with its full frame archive and verified movie; its seed and
original Estuary render identity must match the still. Resolution and camera
pose may differ. `formation` optionally names a completed `film.py` output;
its verification function must certify the same original painting and source.
A seed by itself is insufficient to prove that two render histories match.

`baselines` maps each seed to its original flat painting. The selected study
controls which reference appears. Missing references are hidden rather than
borrowed from another seed. Labels are assigned through DOM text nodes, and
films begin only after a user presses **Camera orbit** or **Formation and orbit**.
The orbit caption describes a camera examination of the completed painting; the
formation edit is labeled separately.

```bash
python -m tools.estuary_depth.curate \
  --catalog /path/to/catalog.json \
  --output /path/to/curated-gallery
python tools/atelier/serve_review.py /path/to/curated-gallery --port 8785
```

Curation copies presentation media only into `study-001`, `study-002`, and so on,
plus shared baseline images. Hash-named assets preserve previously served media
when the catalog is revised. `curation.json` records the source experiment
identities, original painting identities, input result receipts and copied media
hashes. `collection.json` switches only after the selected assets pass their
receipt checks. The large simulation bundles, EXRs and editable scenes stay in
their original archives.
