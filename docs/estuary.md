# The Estuary

Three recorded bodies stir a persistent field of mineral pigment. A fixed camera
looks into a larger sheet of moving paint. Old marks stretch and fold; output
frames never reset the material or manufacture new source motion.

This is a **kinematic painting experiment driven by a physical orbit**. Its flow
is prescribed by a stream function. It is not a Navier–Stokes simulation, and
neither the colors nor fluid parameters are measurements of physical pigments.

## Components

| Module | Responsibility |
| --- | --- |
| `tools/estuary/source.py` | Validate and project the unchanged full orbit |
| `recipe.py` | Strict, bounded, versioned controls and canonical serialization |
| `flow_reference.py` | Independent float64 stream function and velocity oracle |
| `engine.py`, `shaders/` | GPU flow, adaptive characteristics and pigment transport |
| `optics.py`, `optics.glsl` | Finite-layer reflective pigment mixing |
| `run.py` | Source/code identity, checkpoints, archival images and film verification |
| `gallery.py` | Publish completed media for visual review |

The existing Rust generators and earlier artwork modes are not dependencies of
this renderer beyond their shared `.orbit` cache format. The new GPU adapter
uses ModernGL and NumPy in an isolated environment.

## What comes from the orbit

The reader checks the `CSORBIT1` header, payload size, finite coordinates,
positive masses and timestep, and hashes the complete source. PCA uses the
entire recording, with orientation fixed by ordered source anchors. Every
sample receives one translation, orthonormal projection and uniform fit.
No moving crop, new orbit integration, selection, or time warping is applied.

Positions are interpolated between original source knots. Velocities are
derivatives with respect to normalized source time. Cumulative projected travel
uses **every original segment**. Actual three-dimensional pair separation is
retained in the same uniformly scaled units, so a projected crossing does not
become a false close encounter.

The three positions and velocities drive local stirring fields; pair motion
drives rotational fields. Body velocities receive the explicit smooth artistic
response `v / (1 + |v| / 24)`. Pair spin uses projected cross product divided by
true 3D separation squared plus stir radius squared, then `20*tanh(spin/20)`.

## Flow and transport

For a body displacement `d = x - p`, conditioned velocity `u`, and radius `s`,
its stream function is:

```text
psi_body = flow_strength * (u.x*d.y - u.y*d.x) * exp(-|d|²/(2s²))
```

A pair contributes a Gaussian stream function with radius `1.7s`. The optional
carrier current contributes `u.x*y - u.y*x`. This current is an authored property
of the paint bath; it is not additional body motion.

The total stream function is multiplied by a squared boundary envelope. Its
full analytic curl, including the product-rule terms, gives a divergence-free
continuous field with zero boundary velocity. The simulation extends beyond the
visible canvas by `domain_scale`; the camera always shows the central fixed
canvas. Transported paint can enter and leave this view. All recorded body
positions remain within the configured source fit.

GPU grid interpolation and finite integration approximate that continuous flow.
Pigments use RK2 backtracing and a locally limited MacCormack correction. The
limiter prevents new donor-cell extrema and negative concentrations before new
paint is added. It does **not** guarantee exact global mass conservation or
preserve arbitrarily thin features. Numerical diffusion remains a real limit.

Each canonical step is subdivided until both conditions hold:

- Actual projected source travel in that individual subinterval is at most
  `0.4 * brush_radius`, including concentrated encounter bursts.
- The sampled velocity field travels at most `1.5` simulation pixels per
  characteristic step. Bilinear interpolation cannot exceed the sampled maximum
  vector magnitude used in this bound.

Output times land on canonical steps. Changing output frame cadence or reading
an image does not change the evolution. A cumulative work cap of 500,000
transport steps is retained across checkpoints. Nonfinite state, invalid bounds,
or exceeded limits fail the render rather than certifying a partial result.

## Material preparation and color

`initial_pattern="pools"` places finite pigment pools at the first recorded
positions. `"strata"` prepares a continuous blue ground, a broad white band,
two finer white seams, and one narrow earth-color seam. The initial triangle
sets their center and orientation. This is the **declared initial paint layout**,
not invented orbital prehistory. `initial_load` sets its concentration and
`load_radius` its scale.

Additional paint follows the three moving sources through compact swept brush
footprints. Amounts depend on actual traveled distance, `deposition`, and
`pigment_weights`. The supply decreases as `exp(-fade * source_fraction)`;
already deposited pigment is never deliberately faded. Pool loading also uses
the source pigment weights; strata have their own fixed material fractions.

Each channel is a pigment concentration. Authored sRGB pigment reflectances are
decoded to linear light and converted to Kubelka–Munk absorption/scattering
coefficients. Concentrations mix those coefficients, and the finite layer is
evaluated over the substrate. The RGB approximation is documented and checked
against an independent float64 implementation. The minute optional grain belongs
to the stationary substrate; it is not a claim of advected microscopic pigment.

## Run on the experiment server

Rendering requires Linux, a hardware OpenGL 4.3 context with EGL, Python 3.10 or
later, and FFmpeg/ffprobe for films. Software GL fallback is rejected. The tested
server has an RTX 5060 Ti; its complete GL driver identity enters each request.

```sh
python3 -m venv /path/to/estuary-venv
/path/to/estuary-venv/bin/pip install --require-hashes \
  -r tools/estuary/requirements.txt

OPENBLAS_NUM_THREADS=4 /path/to/estuary-venv/bin/python -m tools.estuary.run \
  --orbit /path/to/source.orbit \
  --recipe tools/estuary/recipes/ultramarine.json \
  --output /path/to/new-render --backend egl
```

Add `--still-only` for a complete-history still. Use `--resume` with the identical
request to verify an existing result or recover a film checkpoint. Unknown
recipes, changed source/code/hardware, missing checkpoints, and corrupted frames
are rejected. Interrupted partial files remain available for inspection.

For a finite collection, `tools.estuary.batch` resumes existing seed archives
and publishes each verified film as it finishes. Its concurrency is bounded to
three; choose the worker count for the available GPU memory. The selected
recipe used approximately 2.35 GiB per renderer on the experiment server.

```sh
OPENBLAS_NUM_THREADS=4 python -m tools.estuary.batch \
  --sources /path/to/orbit-caches \
  --recipe tools/estuary/recipes/ultramarine.json \
  --output /path/to/collection --workers 3 \
  --seeds 0x808861c25b6c 0xb7f327f9f722 0xbc53af1cd380
```

`batch-request.json` binds inputs; `status.json` reports per-seed progress and
failures. Cancellation stops only owned processes, including detached encoders.
The batch and gallery are outside the renderer's runtime identity, so scheduler
or presentation changes do not force physical state to be recomputed.

The selected high-resolution recipe uses a 6144 × 4608 state with a 1.6 guard
band, giving a native 3840 × 2880 visible field. It has 7,200 canonical steps and
901 frames at 30 fps: the full source interval in 30.033 seconds. Each movie
frame averages the last four canonical material states in **linear light**;
the opening uses state zero alone. These discrete trailing exposure points are
recorded explicitly. They do not alter integration or create time outside the
recording. The final still is a separate sharp exposure. The smaller
study recipe keeps the same art controls but is explicitly lower resolution.

## Artifacts and verification

A completed render archives the source cache, runtime code and shaders,
dependency pins, complete recipe, source projection and GPU identity. It writes:

- `poster.png`: tagged RGB16 sRGB, full-history final painting.
- `linear.npy`: lossless float32 linear-sRGB final image.
- `final-state.npy`: lossless float32 pigment concentrations, bottom-to-top rows.
- Films: RGB8 display frames, exact canonical frame steps, checked checkpoints,
  H.264 movie and a receipt verifying dimensions, rate and complete decoding.
- `receipt.json`: completion, artifact hashes, elapsed time and integration
  diagnostics. A partial receipt never qualifies for gallery publication.

Checkpoints retain the source step, complete state, existing frame hashes,
internal transport count and maximum Courant number. SIGTERM and interruptions
close the GPU context and terminate owned encoding processes. Gallery changes
do not invalidate simulation resume. Cross-device pixel identity is not promised.

```sh
OPENBLAS_NUM_THREADS=4 ESTUARY_TEST_GPU=1 python -m unittest \
  discover -s tools/estuary -t . -p 'test_*.py'
ruff check tools/estuary
ruff format --check tools/estuary

python -m tools.estuary.gallery --renders /path/to/render-a /path/to/render-b \
  --output /path/to/gallery
python tools/atelier/serve_review.py /path/to/gallery --port 8784
```

Tests cover source identity and invariance, actual source-travel bursts, analytic
curl/divergence, true 3D separation, GPU/reference agreement, constant-material
transport, output-cadence independence, exact checkpoint continuation, reflective
optics, PNG16 encoding, cancellation, and archive integrity. They establish
technical behavior; artistic selection still requires viewing the real stills
and full movies across different seeds.

## References

- Aubrey Jaffer, [Oseen Flow in Paint Marbling](https://arxiv.org/abs/1702.02106),
  for the connection between prescribed fluid motion and mathematical marbling.
  The stream function used here is the explicit artistic model above.
- [ModernGL context documentation](https://moderngl.readthedocs.io/en/5.12.0/reference/context.html).
