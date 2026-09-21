# Paint material experiments

The reference is the preserved all-three-body RC1 painting. All experiments keep
its complete source recording, starting pigment amounts, seed-derived palette and
layout. The rejected one- and two-body studies remain historical archives and are
not selections in this material study.

## Four independent controls

1. **Fuller paint** changes displayed relief (`surface.height_scale`, 60 to 66).
   It does not add pigment or change transport. A 63 variant provides a restrained
   midpoint.
2. **Directional contact relief** reconstructs aggregate affinity along transported
   paint fabric, only across occupied, contacting material. Symmetric bounded
   exchanges smooth along the fabric before the existing conservative packing
   step. The displayed-height redistribution remains capped at 12%; pigment
   arrays remain unchanged. This is an authored surface reconstruction, not a
   measured buckling or stress model. No stripes are painted over the image.
3. **Seeded material variation** initializes two smooth, signed material traits
   from independent versioned streams of the full seed. They are transported with
   paint mass, rather than repeatedly sampled from averaged material origins.
   Their bounded influence changes aggregation, breakup and fabric response only
   at wet contacts. Coarse and fine scales use projected simulation world units.
   The existing RC1 nucleation law is preserved separately. When combined with
   resistance, the aggregation trait also mildly varies structural resistance,
   gated by the recorded contact dose. Its mass-weighted factor stays within
   `[1-amplitude, 1+amplitude]`; untouched material remains neutral.
4. **Resistance and memory** adds a bounded structure variable to both paint layers.
   Structure weakens under shear and recovers at rest; drying accelerates recovery.
   A coarse screened streamfunction response modifies the analytic prescribed
   flow through a native discrete curl. This is an authored two-dimensional
   quasistatic model, not a full non-Newtonian fluid, pressure or free-surface
   solver. Variable resistance can redistribute flow, so it does not promise that
   every pixel moves more slowly.

An optional `rheology.occupancy_mass_reference` separates the visible paint-amount
response from the numerical history-support threshold. When present, the coarse
resistance factor uses `mean_mass / (mean_mass + occupancy_mass_reference)`;
`minimum_concentration` still governs history support. When absent, the earlier
formula and normalized recipe remain exact. Values 0.01 and 0.02 test whether
letting thin paint offer less resistance improves the contours. A separate
response-length 0.08 study changes both the spatial response and its derived
coarse grid; it is not an isolated grid-resolution experiment.

Matte/satin variants use the existing small roughness bias. They retain the
interaction-driven variation already present in RC1; there is no extra canvas
noise layer.

## Numerical and reproducibility contracts

- Omitted or zero-amplitude extensions retain legacy configuration and material
  fields. Zero-feature reference renders must preserve the original material and
  image bytes on the qualified runtime/GPU.
- Traits (`trait_upper/lower`) are native float32 H×W×2 fields in [-1,1]. Structural
  memory (`structure_upper/lower`) is native float32 H×W in [0,1]. Optional fields
  are atomic pairs and are included in the complete archived material hash.
- New seed-derived traits initialize once and travel with the material. Capturing
  a frame or changing the camera cannot advance history or draw randomness.
- The response retains the old analytic velocity and subtracts a discrete curl;
  it does not finite-difference and replace the original velocity. Its correction
  is compactly supported within the original flow domain. Its measured velocity
  enters the existing adaptive stability check.
- The screened solve starts from zero for each forcing evaluation. Its fixed 64
  Jacobi iterations have an exact-arithmetic contraction bound below 6.3e-7;
  float32 roundoff is separately tested. A rejected timestep cannot secretly
  advance the response or structure.
- Structure reactions use an exponential solution with rates frozen per accepted
  substep. Rates use the existing normalized complete-recording clock, not video
  seconds. Structure is an authored property, not measured viscosity.
- Existing pigment advection still relies on canonical global pigment-budget
  restoration. These changes do not make pigment transport locally conservative.
- Camera holds and orbits reuse the final material. New material histories require
  native-grid capture and strict request/receipt metadata. Public review galleries
  bind copied media to verified archives; they do not claim to recompute simulation
  state from the images.

## Comparison method

Start with all 16 combinations of the four controls across three contrasting
seeds, using native 2048×1536 material. Inspect the complete painting and its fine
structure at full resolution. Follow with strength and finish studies, then apply
promising combinations to the remaining RC1 seeds. Changes in displayed relief
must be described separately from changes in actual paint motion.

Judgments are artistic and provisional. Prefer coherent forms, distinct material
structure, clear color relationships and intentional quiet areas. More visible
texture or more complicated motion is not by itself an improvement. Record which
settings succeed across seeds and where the original RC1 still works better.

## Reproduce and publish

```sh
# Four full-source smoke cases, then all 27 settings on three seeds.
python -m tools.estuary_confluence.paint_material_studies \
  --seeds 0xb7f327f9f722 --variants rc1 relief traits resistance \
  --source-root /path/to/orbits --output /path/to/smoke
python -m tools.estuary_confluence.paint_material_studies \
  --suite all --source-root /path/to/orbits --output /path/to/matrix

# Optional complete films: 937 frames, including the formation and camera sequence.
python -m tools.estuary_confluence.paint_material_studies \
  --all-seeds --variants rc1 resistance-light --film \
  --source-root /path/to/orbits --output /path/to/films

python -m tools.estuary_confluence.paint_material_gallery \
  --cases /path/to/matrix/cases/* --output /path/to/review
python -m tools.estuary_confluence.paint_material_gallery \
  --verify --output /path/to/review
```

Run qualification with `python -m unittest discover -s tools/estuary_confluence
-t .`. Set `ESTUARY_TEST_GPU=1` on the render host to include the actual OpenGL
transport, shading and ownership checks. Render from a frozen source checkout;
the study plan and every completed case bind the runtime files. Preserve the
original RC1 archives and never modify a completed case to adjust its appearance.

## What the initial visual comparisons show

The controls do not contribute equally. Structural resistance produces the largest
changes in silhouette and spacing. Strength 1 retains more of RC1's connected
ribbons than strengths 3 or 6. Stronger settings can introduce small openings,
fragmented strands and repeated contour irregularities. More variation is not
consistently more beautiful.

Directional relief and material variation alone are very subtle on the tested
RC1 material. The orientation and conservation tests demonstrate that the relief
operator works, but they do not establish a perceptual improvement in a finished
painting. Stronger surface-only probes on B7 also failed to reveal a convincing
new fine texture. These remain experimental controls, not a claimed upgrade.

The optional paint-amount gate changes the composition again; it does not
reliably remove the contour irregularities. The wider response also changes its
coarse grid and should not be described as a demonstrated numerical cure.

`gentle-worked-paint` is the restrained combined study:

| Control | Value |
| --- | --- |
| Displayed height | 63, compared with RC1's 60 |
| Directional relief | Strength 0.3, anisotropy 0.8 |
| Seeded property variation | Amplitude 0.10 |
| Structural resistance | Strength 1 |
| Paint-amount reference | 0.01 |
| Roughness bias | +0.04 |

“Gentle” describes the controls. The resulting shape can still change noticeably,
and the best choice varies by seed. The gallery's marked picks are subjective
comparisons with written reasons, not an automatic quality score. RC1 remains
available without changing its recipe, material or saved images.

The three native reference studies reproduce both RC1's complete material hashes
and its final PNG bytes exactly. Every film uses 721 formation frames, 72 hold
frames and 145 camera frames at 24 fps (39.0417 seconds). The entire recorded
trajectory is traversed during formation; video duration is not a truncated
source interval. Paint updates use the existing canonical/adaptive source clock,
independent of frame capture. The first completed combined film also has 72
identical hold-frame PNGs and passed browser playback and seeking checks.

## Completed collection

The published collection contains **95 native 2048×1536 images and 20 full films**:
27 treatments across the three main study seeds, followed by a matched RC1/gentle
comparison across all ten seeds. The 20 films cover RC1 and the gentle combination
for every seed. Other parameter studies are explicitly labeled as image studies.

The [release manifest](releases/paint-materials-v1.json) records every image and
film identity, batch plans, source hashes, and the ten visual picks. All 88 new
complete-source runs finished without a failed case. The review reuses the ten
preserved RC1 films and omits duplicate newly rendered RC1 controls.

My preferences are mixed: the gentle combination improves the openness and color
lanes of CEDD, 8088 and 6210, while RC1 remains the cleaner choice for the other
seven. These findings do not support replacing RC1 universally or claiming that
fine texture is now substantially more convincing. Both versions stay available
for the viewer's own judgment.

Validation included the 647-test Confluence suite on the GPU server (10 expected
skips), all 83 base-engine GPU tests, and ten publisher/browser-control tests
including the local Node checks. The final portable gallery was verified on both
hosts. All 95 image routes and 20 film routes responded correctly, including
byte-range requests for every film. Browser checks covered playback, seeking,
linked detail inspection, and the narrow app-panel layout.
