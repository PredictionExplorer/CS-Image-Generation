# Tidal Silk validation

## Implementation and environment

Developed on `codex/tidal-silk`, based on `main` at `4dc6d64`. The existing
`three_body_problem` command remains the default Cargo executable. Tidal Silk
adds a separate CPU-only binary and reuses the existing dependencies; JSON float
round-trip parsing is enabled for initial-state fidelity.

Simulation and artwork experiments ran on the user's Linux x86-64 server:
Threadripper PRO 9985WX, 64 physical cores / 128 threads, approximately 503 GiB
RAM. Neither Blender nor GPU rendering was used. A macOS ARM64 machine performed
small portability checks; full-size artwork generation ran on the server.

## Automated checks

- Formatting and all-target Clippy pass with the repository's CI flags.
- 498 tests pass, including 33 Tidal Silk unit tests and 3 CLI integration tests.
- One existing test remains ignored.
- One existing scalar spectral NaN property test is excluded from the final
  successful full-suite run. Its failure was independently reproduced on
  unchanged `main` with the same minimal input.
- The server's native AVX2 build also exposes three existing spectral
  approximation tests that fail identically on unchanged `main`. These are
  outside Tidal Silk's renderer and were not changed in this work.

The new tests cover physical-orbit reconstruction, source-sample fidelity,
configuration errors, circular topology, bending gradients, attachment centers,
contact and tunneling cases, cache validation, per-thread render identity,
subdivision, filtering boundaries, sampling PDFs and estimator agreement,
RGB16 output, interrupted-render recovery, and frame integrity.

## Cross-architecture observations

These are measured test results, not a universal bit-identity promise:

| Stage | Test | Observed result |
|---|---|---|
| Original orbit | Independent replay of `0xb7f327f9f722`, one million samples of all three XYZ positions | All nine million coordinates exactly match |
| Cloth | Independent 12-frame, 127-vertex bake | Maximum coordinate difference approximately `4.01e-12` |
| Final renderer | Independently generated cloth, 256×208 RGB16 sequence, 12 frames | All decoded image channels exactly match |
| H.264 encoding | Same sequence, different installed FFmpeg builds; explicit conversion and CPU-independent encoder settings | All decoded RGB8 frames exactly match in the verification recipe |

The initial encoding comparison, before fixing conversion settings, showed
differences despite identical source images. Explicit bilinear color conversion,
accurate rounding, bit-exact paths, disabled dithering and CPU-independent x264
decisions removed those differences in the test. Encoder version and thread
count are now recorded in each film's sidecar.

Authoritative orbit and cloth caches remain useful for archival replay. Future
compiler, architecture or algorithm versions should repeat the tests. Image
metrics assist inspection; they do not replace visual checks of full motion.

## Selected physical studies

All figures below concern the simulated cloth before render-only subdivision.
Centroid attachments follow the source bodies; self-contact remains enabled.

| Study | Vertices | Frames / rate | Worst edge stretch | 95th-percentile stretch |
|---|---:|---|---:|---:|
| Calm, mesh48 | 1,801 | 192 / 24 fps | 3.25% | 0.24% |
| Calm, mesh64 | 3,169 | 192 / 24 fps | 6.63% | 0.26% |
| Memory, mesh48 | 1,801 | 144 / 24 fps | 7.18% | 0.327% |
| Memory, mesh64 | 3,169 | 144 / 24 fps | 4.66% | 0.248% |

The selected Memory64 and Calm48 studies have no edge/frame samples above 10%
stretch, and attachment-center errors remain below `7e-16` scene units. Difficult
earlier studies stopped with explicit collision diagnostics; their failures were
not bypassed by disabling collision in the finished studies.

Memory was selected for its taller, open, layered silhouette. Calm offers a
quieter unfolding. Their exact physical and rendering controls are supplied in
`tools/tidal_silk/`. The source-state profile allows reproduction without a fresh
candidate search or access to the earlier output archive.

## Practical limits

This is a specialized cloth/art renderer. The secondary cloth simulation does
not feed forces back into the original orbit. Finite-resolution XPBD and bounded
collision checks are numerical approximations; arbitrary extreme configurations
can fail safely. Mesh resolution can change fine folds, so each finished study
archives its actual geometry and settings. Render-only smoothing and optional
noise filtering are recorded artistic choices.
