# Estuary Studio

Three material interpretations of the same recorded three-body motion. Each can
produce a final painting and a film that shows its complete formation, pauses,
then moves the camera around the finished surface.

## The three directions

| Family | Material process | What the image records |
| --- | --- | --- |
| **Tidal Fresco** (`fresco`) | Mobile pigment is stirred, settles onto the support, dries and can be lifted back into motion. Blue stain, mineral white and oxide have different settling and remobilization rates. | Washes, deposits, interrupted banks and previous passages that survive later movement. |
| **Three-Body Monotype** (`monotype`) | Three partially loaded tools deposit, drag and lift paint. Broad and narrow tools leave different marks; contact and loading vary along the recording. | Overlapping strokes, exposed ground, transferred material and a persistent direction of application. |
| **Nocturne** (`nocturne`) | A dark palette and directional surface finish reveal marks as the viewing angle changes. Its recipe selects either Monotype tools or Fresco transport. | Dark passages whose roughness, relief and directional reflection distinguish their histories. |

`family` selects the visual interpretation. `dynamics` selects the simulation:
`fresco` or `monotype`. Fresco and Monotype use their matching dynamics; Nocturne
supports both. For example, `{"family": "nocturne", "dynamics": "fresco"}`
uses the Fresco flow and deposited material with Nocturne lighting and pigment
optics. Its directional field comes from the evolving flow, not brush contact.

These are authored material models. Fresco uses a prescribed incompressible
stirring field, rather than a three-dimensional free-surface fluid simulation.
Its mobile/sediment exchange conserves local pigment; the inherited limited
MacCormack transport does not guarantee global conservation, and deposition adds material.
Monotype uses semi-Lagrangian drag, which is not mass conserving. Its finite tool
load is a **loading fraction**, with recorded depletion and reloading rates—not
a resolved volumetric reservoir or a simulation of individual bristles.

## How the motion becomes marks

The original `.orbit` file is immutable and is not integrated again. A single
projection is fitted to the entire recording. Every time sample uses the same
translation, PCA axes, scale and framing. Original-knot arc lengths bound tool
travel and transport substeps; changing output frame cadence cannot change the
painting's canonical evolution.

Monotype also uses the actual third coordinate. The normal to the fixed PCA
plane gives each body's distance from that plane. One scale—the full recording's
90th percentile absolute distance—normalizes contact for the entire run. Nearer
tools press into the surface; farther tools lift. Projected speed modulates
pressure. The plane and normalization never refit per frame. Reloading occurs
while a tool is lifted; travel under contact depletes its loading fraction.

The small support tooth and bristle profiles are deterministic material
structure. They are anchored to the support or tool and persist through the
simulation; no new screen-space noise is generated for successive frames.

## Rendering and units

All three films use the same renderer during formation and the final camera
movement. The orbit freezes the completed material fields: it does not restart
the simulation or substitute a separately generated sculpture.

`Surface` intersects orthographic rays with a heightfield, then evaluates
deterministic area-light samples, anisotropic GGX reflection and heightfield
shadows. This is **ray marching with analytical surface lighting**, not Cycles
path tracing. It supports parallax and self-occlusion within a single-valued
surface; it does not model overhangs, refraction or multiple scattering.

Pigment reflectance uses the project's finite-layer **RGB Kubelka–Munk
approximation**. Palette values and scattering strengths are authored, not
measured spectral properties of named commercial pigments. Physical height is
derived from material amounts and recipe-specific volumes, never image
brightness. Its mechanical scale is an artistic interpretation.

Simulation API:

```python
engine = Monotype(source, config, backend="egl")  # or Fresco(source, config)
engine.advance_to(step)  # integer canonical step, increasing from 0 to steps
fields = engine.snapshot()
engine.close()
```

Snapshots are float32, bottom-up arrays covering the **full guarded domain**.
The central visible painting is 0.4 m wide by default. The renderer returns
top-down linear sRGB; PNG encoding happens once at the output boundary.

| Field | Shape | Meaning |
| --- | --- | --- |
| `pigment` | H × W × 3 | Nonnegative relative pigment amounts, in fixed channel order. |
| `height` | H × W | Surface elevation in metres. |
| `wetness` | H × W | Local wet fraction, 0–1. |
| `direction` | H × W × 2 | Axial direction `(cos(2θ), sin(2θ))`; subunit length permits lower directional coherence. |
| `roughness` | H × W | Surface roughness, 0–1. |
| `coverage` | H × W | Material coverage, 0–1. |

Fresco snapshots additionally preserve mobile and deposited pigment for
numerical accounting. The renderer receives only the common fields above.

## Running an experiment

Run from the repository root with the existing Estuary Python environment
(`tools/estuary/requirements.txt`). A hardware OpenGL 4.3 EGL context is required;
software rendering is rejected. Films also require `ffmpeg` and `ffprobe`.

```sh
# Full recording, final image, formation film and camera orbit.
python -m tools.estuary_studio.run \
  --source path/to/source.orbit \
  --recipe path/to/recipe.json \
  --output output/studio/selected-study

# Smaller still proof; this still simulates the complete recording.
python -m tools.estuary_studio.run \
  --source path/to/source.orbit \
  --recipe path/to/recipe.json \
  --output output/studio/selected-proof \
  --still-only --resolution 1024 768 --image-size 1024 768
```

Other overrides are `--video-size W H`, `--formation-frames N` and
`--orbit-frames N`. All image aspect ratios must match the simulation.
`steps` must be divisible by `formation_frames - 1`, so every formation exposure
lands on a canonical state. Simulation resolution and output resolution are
independent: rendering a larger image does not add simulation detail.

The experiment server uses `/home/user/estuary-studio`; its existing Python
runtime is `/home/user/estuary/venv/bin/python`. The comparison gallery is served
locally on port **8786**. The earlier Estuary depth gallery on **8785** is preserved.

## Archives and verification

Each completed output contains:

- `poster.png`: final 16-bit image, with `poster-linear.npy` before display encoding.
- `film.mp4`: formation, hold and camera orbit; omitted with `--still-only`.
- `final.npz`: the completed material fields.
- `recipe.json`, `request.json` and `receipt.json`: resolved controls, source and
  runtime identities, hardware, final step, field statistics and artifact hashes.
- `frame-ledger.json`: every film frame's canonical step, source fraction, camera
  pose and image hash; individual images remain in `frames/`.
- `inputs/`: preserved original recording and runtime source.

Movie completion requires the expected dimensions, frame count and frame rate,
plus a successful decode of the entire encoded file. A successful receipt must
reach source fraction 1 and the recipe's final canonical step. Reusing a
completed output verifies its identity and all recorded artifact hashes.

Outputs are immutable. Failed or interrupted work is retained with an incomplete
receipt; partial runs are **not automatically resumed or overwritten**. Use a new
output directory for a retry. Rendering code and source files must remain fixed
throughout a run; the pipeline checks their identities again before completion.

## Checks

```sh
python -W error -m unittest discover -s tools/estuary_studio -t . -p 'test_*.py'
ESTUARY_TEST_GPU=1 python -W error -m unittest discover \
  -s tools/estuary_studio -t . -p 'test_*.py'
```

The GPU tests cover actual material bounds and response, complete-source clocks,
output-cadence independence, context isolation and camera/material behavior.
CPU tests cover source-plane pressure, local material accounting, recipe bounds
and archive contracts. Exact image identity is expected on the same frozen
runtime and device; cross-device bit identity is not promised.
