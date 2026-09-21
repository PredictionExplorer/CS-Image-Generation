# Fine folds: reconstructing the Folded Tide

The supplied navy, ivory and oxide reference matches the archived **The folded
tide**, seed `0xbc53af1cd380`, study `finished-bc53/21-deep`. It is a photograph of
the completed original Estuary pigment fields rendered as relief in Cycles.

## What made the lines

The original used **three pigment channels**, initialized as a blue field, a
broad ivory band, two fine ivory bands and a narrow oxide band. It did not start
with three separate circular pools. The band direction follows the initial body
triangle.

The new experiments separate three causes:

- **Stretching and folding:** removing the two fine starting bands still leaves
  the intricate central folds. Three broad circular pools also produce them.
  The initial bands influence the broad composition and peripheral accents;
  they are not required for the central ribbons to appear.
- **Material resolution:** the original paint evolves at 6144 × 4608. Matched
  2048 × 1536 material-grid probes use the same final image size but lose or
  merge some of the tight folds. Enlarging the output image cannot recover
  these missing concentration structures.
- **Relief and lighting:** a flat-surface control retains the color ribbons.
  Height and angled light make their banks easier to read as folded material.
  The effect is not an added line texture.

The base solver uses limited MacCormack transport under the prescribed
three-body current. There is no explicit pigment interdiffusion or moving
upper/lower layer split in this older model. Numerical interpolation and
transport error remain; this is not a claim of zero mixing or exact local
conservation.

## Controlled material studies

`filament_studies.py` defines the starting-band width controls, original and
broader three-pool starts, stronger pair-driven folding, a calmer carrier, and
the labeled coarse-grid diagnostics. All traverse the entire source recording
with 7200 canonical paint steps. The orbital recording contains one million
samples; paint steps and orbital samples are different quantities.

The BC53 control retains the original colors and physics. Other seeds keep
navy, ivory and oxide roles with small, bounded variations derived from all
256 seed bits. Cycles sampling is deterministic per seed; BC53 retains the
original sampling seed for the reference comparison.

The optional `simulation.strata_profile` changes only initial band widths:

```json
{
  "version": "strata-profile-v1",
  "main_width_scale": 1.0,
  "fine_width_scale": 0.5,
  "accent_width_scale": 1.0
}
```

Zero fine width removes the two fine ivory bands. Omitted, null and neutral
profiles normalize to omission, preserving legacy recipe identities and initial
pixels. Active profiles require `initial_pattern="strata"`. The opt-in
`render.initial_image=true` records the actual state before any simulation
advance; rendering that image does not alter the paint.

The reconstructed native BC53 control matches the original final-state file
exactly: `2a1edb0149b9b8ffba9be67b2528fa995eec0ad5804e2e808b44bb75d2de42f5`.
Its prepared material bundle also matches exactly:
`4111f51eb1d7cf9db6fff7c4427e5b4bca98dd09f7a82b0eabac05572649e40d`.

## Photographing the paint

The baseline uses the original `recipes-finish/21-deep.json`: 16 mm authored
relief scale, 0.35 mm height-only smoothing, a 28° camera tilt, and a restrained
area light. Actual height is calculated from pigment concentrations and the
specific volumes `[0.25, 1.0, 0.5]`; display RGB does not determine height.
The optical studies reuse exactly the same bundle and vary flat/shallow/deeper
relief or light elevation. They do not rerun the paint.

Comparison photographs use 2048 × 1536 pixels and 128 Cycles samples. Final-quality
photographs use 3840 × 2880 and 256 samples. A finished painting can be
rephotographed from its verified bundle without simulating it again.

## Run and review

Freeze the source checkout before launching a batch. The coordinator records
resolved recipes, source and executable hashes, package identities and per-case
completion. Existing outputs are immutable; failures keep their logs.

```sh
python -m tools.estuary_depth.filament_batch \
  --source-root /path/to/orbits --blender /path/to/blender \
  --seeds 0xbc53af1cd380 \
  --variants control three-broad-pools plain-bands fine-bands \
  --output /path/to/new-batch

python -m tools.estuary_depth.filament_gallery \
  --cases /path/to/completed-control /path/to/completed-variant \
  --lighting-root /path/to/verified-lighting-experiment \
  --output /path/to/new-review
```

The portable review shows starting paint, finished photographs and linked detail
inspection. Its copied provenance binds the batch plan, source, paint state,
prepared maps, exact Cycles recipe and published images. Lighting comparisons
require the exact same Control bundle. Verification works after the original
server directories are unavailable.
