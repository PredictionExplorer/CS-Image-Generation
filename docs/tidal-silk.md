# Tidal Silk

Tidal Silk expresses the original three-dimensional motion as three attachment
regions pulling a continuous fabric. The new `tidal_silk` executable uses CPU
cloth simulation and CPU ray tracing. It does not require Blender or a GPU.

## Independent stages

1. **Orbit:** reconstruct a selected system from existing generation metadata,
   import explicit initial conditions, or run the existing seeded selection.
   Store physical samples before projection, view rotation, or artistic drift.
2. **Bake:** choose a contiguous source interval, one spatial normalization and
   one playback scale. Simulate fixed cloth topology with slack, attachment
   regions, bending, stretching and self-contact. Store the moving mesh.
3. **Render:** choose material, lights, camera and sampling independently. Save
   16-bit PNG frames; compatible interrupted renders can resume.
4. **Encode:** turn rendered frames into browser H.264 or high-quality HEVC films
   using the repository's existing FFmpeg dependency.

## Example

```sh
cargo build --release --bin tidal_silk

target/release/tidal_silk orbit \
  --config tools/tidal_silk/orbit-b7f327f9f722.json \
  --output /path/to/results/source.orbit

target/release/tidal_silk bake \
  --orbit /path/to/results/source.orbit \
  --config tools/tidal_silk/memory.json \
  --output /path/to/results/study.silk

target/release/tidal_silk --threads 32 render \
  --bake /path/to/results/study.silk \
  --config tools/tidal_silk/indigo.json \
  --output /path/to/results/pearl

target/release/tidal_silk encode \
  --input /path/to/results/pearl \
  --output /path/to/results/tidal-silk.mp4
```

`render --frame 48` produces one still. `render --every 4` samples the full motion
at a lower frame rate for inexpensive previews; encoding preserves that timing.
`inspect --bake ...` reports geometry, the source recipe, and motion diagnostics.
`compare --first a.png --second b.png` reports differences between decoded RGB16
values. These numerical metrics assist visual inspection; they do not by
themselves establish perceptual identity.

The bundled orbit profile freezes the original initial conditions of the
selected `0xb7f327f9f722` artwork. `memory.json` is the expressive 64-resolution,
six-second fabric study; `calm.json` is the restrained 48-resolution, eight-second
study. `study.json` is a cheaper mesh for motion exploration. `indigo.json` and
`pearl.json` are the curated material and lighting treatments. Existing packages
can instead use `orbit --generation-record /path/to/metadata/generation.json`.
New seeds can use `orbit --seed 0x...`, with explicit search settings recorded.

The JSON config files accept the fields of their corresponding Rust config
types. CLI overrides are intentionally small; `--help` lists them. Existing
generation metadata overrides orbit defaults with the recorded source settings.
Unknown configuration keys are rejected. Binary caches have appended `.json`
sidecars; for example `study.silk.json`, without replacing their payloads.

## Cloth and material controls

The mesh is a concentric triangular disk with a genuinely circular edge. The
`rigid` attachment mode holds small patches in a fixed orientation; `centroid`
holds the area-weighted center of each patch while allowing it to rotate and
fold. Both follow the original three body positions. Changing modes changes the
material experiment and is recorded in its recipe.

Larger `bend_compliance` values make the fabric easier to bend. `slack` controls
its fixed material size. Neither rest lengths nor source paths are animated to
disguise infeasible movement. Optional long-range attachment bounds communicate
tension across the sheet using fixed material distances. Contact handling uses
vertex–triangle and edge–edge checks, bounded conservative advancement and
bounded step bisection. Unsafe configurations return a diagnostic error.

`subdivision_levels` smooths the rendered surface while preserving the original
bake. Light rotation and key/fill/rim strengths, pearl/indigo/rose palettes,
roughness, transmission and sheen are independent rendering controls. The
optional `denoise_passes` setting filters sample noise using primary-ray surface
normals, depth, material color and sampling variance; zero preserves unfiltered
light samples. It is ordinary deterministic CPU filtering, with no learned model.

## Reproducibility and artistic identity

The target is visually indistinguishable results on supported architectures.
Native double-precision geometry, stable constraint ordering and sample streams
independent of worker count avoid gratuitous differences. Bit-identical results
across arbitrary CPU architectures, compilers and encoders are not promised.

An orbit cache records source initial conditions and their exact floating-point
bits, the original seed and selection parameters. A cloth cache records its
recipe, mesh, full-precision frames and diagnostics. Rerendering a shared cache
and independently regenerating that cache are separate verification tasks.

Render manifests include the source-cache checksum, configuration and executable
checksum. Reusing a directory with a different recipe fails unless `--overwrite`
is supplied, preventing an interrupted render from mixing incompatible frames.

Encoding explicitly selects the RGB-to-YUV filter, rounding and dithering, and
uses CPU-independent H.264 decisions. Sidecars record encoder version and thread
count. This avoids relying on differing FFmpeg defaults across machines. Compare
decoded frames in a fixed color representation; file metadata and display color
management are separate from the underlying artwork.

Original core artwork generation remains available through `three_body_problem`.
The new executable does not alter existing seed-selection behavior or output
packages. A newly selected orbit explicitly records its own search settings.

## Artistic review

Judge broad fold silhouettes and delayed release before adding fine weave detail.
Inspect open, gathered and released states; watch their transitions. A good still
is not enough to establish a good film. Evaluate difficult contact regions and
highlights at full size and during playback.

Subdividing the render mesh only smooths its surface; resolving additional physical
folds requires refining the simulation mesh. Full-duration difficult contact must
be checked before expensive final rendering. No numerical method here makes an
impossible attachment span feasible: the fixed pattern must have enough material
distance between the attachments over the selected interval.

## Experiment server

Heavy experiments are run on `user@100.76.88.48` in `~/tidal-silk/`, independently
of previous visualization farms. Build sources live in `source/`; caches,
rendered frames, logs and films live in `results/`. Parallelize distinct studies
or independent frame rendering while budgeting CPU threads explicitly.
