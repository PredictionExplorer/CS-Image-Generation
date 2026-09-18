# Convergence experiment record

## Selection method

Use the same three complete recorded sources throughout: `0xbc53af1cd380`,
`0x808861c25b6c`, and `0xb7f327f9f722`. Judge the actual images alongside per-color
movement, deformation, contact, image balance, and numerical material budgets.
Higher contact is not an artistic objective by itself: it can produce dull,
overmixed interiors. Final qualification uses the native material grid.

The archived experimental folders below live under
`/home/user/estuary-convergence/experiments` on the experiment host. Each complete
case contains its exact source recording, resolved recipe, runtime copies,
material state, design records, and verification receipt. Private prototypes use
their own frozen runtime; newer verifiers need not accept earlier private model
versions. The selected production recipes are versioned in Git.

## Screening and controlled checks

| Study | Scope | Finding |
| --- | --- | --- |
| `proof-02-matrix` | 36 full-trajectory five-color runs; three seeds; flow 0.85/1.1/1.4, diffusion 0/0.00025/0.0008/0.002, stirring radii 0.18/0.22/0.28; 768 × 576 material grid | Removing the fixed carrier reduced directional bias. Every pigment moved and stretched, but coarse contact was too optimistic and broad mixtures often became dull. |
| `native-proof-03` | Two matched 4096 × 3072 runs, plus identical-camera crops against coarse fields shaded at the same output size | Native resolution recovers openings and fine strands. The peripheral pink in seed 8088 still had only 5.35% native contact, despite its large motion. |
| `optical-mixing-study-bc53` | Six optical interpretations of one unchanged material state | Spectral color was clearer than RGB; thinner optical depth brightened mixtures. Changing scattering strengths alone did not recover lost strands. |
| `partial-mixing-study-bc53` | Controlled partial-mixing and independent-column diagnostics | Composing complete column reflection before averaging is the appropriate areal endpoint and improves color separation. Display-only mixedness overrides were not promoted. |
| `clarity-proof-04` | 12 full-trajectory five-color runs; three seeds; pool scale 0.20/0.28 and diffusion 0.00002/0.00008; 2048 × 1536 grid | Gentler mixing improves color clarity. Smaller pools help some compositions but can leave another seed's color peripheral. Neither diffusion value was a clear universal visual winner. |
| `bulk-pilot-v2-proof` | Three controlled placement comparisons at 2048 × 1536 | Area-distributed, directed tracer contact fixes the misleading single-edge proxy. Seed 8088's weakest final contact rises from 5.21% to 75.65%; pink becomes part of the central form. BC53 improves from 56.88% to 67.93%; B7F3 retains its existing candidate. |
| `budget-proof-05-bc53` | Full-trajectory corrected run at 2048 × 1536 | All five final/initial mass ratios are within 2.5e-8 of 1.0. The 600 recorded corrections change a channel by at most 0.2874% at one checkpoint. This is global budget restoration, not locally conservative advection. |
| `capture-comparison-bc53-v2`, `antialias-comparison-bc53-v2` | Same native fields, camera, and optics; compare material reduction and frame filtering | Full material capture avoids extra pigment mixing during export. Twice-resolution frame rendering followed by linear RGB averaging reduces dotted thin openings. |

A separate 7200/14400-step refinement is archived at
`/home/user/estuary-convergence/reviews/palette-step-refinement-v1`. At 512 × 384,
image linear-RGB mean absolute difference was 0.00626; per-pigment mass differences
were at most 1.82%. The test also exposed 32–44% uncorrected numerical paint growth.
Native uncorrected studies still showed material drift, motivating explicit global
budget restoration. Two temporal levels at a coarse spatial resolution do not
establish complete numerical convergence.

## Selected shared recipe

The three- and five-color production recipes share these controls across all
three seeds; they do not contain per-seed artistic overrides:

- Source-aware area-pilot placement; pool scale 0.28; fixed carrier `[0, 0]`.
- Flow strength 1.1, stirring radius 0.22, pair swirl 0.9.
- Gentle interdiffusion 0.00002, intimate mixing rate 0.55.
- Global pigment budgets restored every 12 canonical steps and at completion.
- Spectral independent-column optics, optical layer scale 12, crisp white ground.
- 4096 × 3072 simulation and full native movie capture; 3840 × 2880 stills.
- 50-second formation at 24 fps, followed by a hold and camera examination;
  movie frames use twice-resolution rendering and linear RGB area reduction.

B7F3 was the strongest visual candidate in screening: connected colored ribbons
and generous white openings. BC53 remains denser and more subdued. The recipe is
the preferred balance from these tests, not a claim of a universal optimum or
guaranteed artistic quality for every possible seed.

## Release qualification

Before publishing a cohort, render all selected seeds and both pigment counts at
native resolution. Inspect full paintings, detailed crops, and formation/camera
motion. Require complete source coverage, reproducible starting pools, certified
global budgets, clean empty channels, valid capture/filter metadata, and full
movie decoding. A separate still and film using identical physical controls must
produce the same final material hash. Preserve the previous gallery for direct
visual comparison.
