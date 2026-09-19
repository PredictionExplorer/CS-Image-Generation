# Color, shape, and blending studies

These experiments change the composition without replacing the complete
three-body recording with decorative noise. They extend the existing spectral,
two-layer paint renderer. Earlier recipes keep their released palettes and
zero-strain flow.

## Genuine starting pigment counts

All four palette modes support **1, 2, 3, and 5 chromatic pigments**. Every palette
resolves the same five candidates before selecting a prefix. Common pigments
retain their exact color, scattering, material properties, and initial layer
fractions. Source-aware starting locations use the same shared prefix. Unused
chalk remains exactly zero for these source-free experiments.

Removing pigments also removes their starting pools and material: this is a
comparison of simpler initial paint compositions, not a controlled equal-mass
recoloring of a fixed footprint. No extra colors or compensating paint are added.
A one-pigment painting can vary in tone through thickness, ordered overlap and
illumination. It does not mix with an invented second pigment.

## Pulling and folding

The optional `simulation.pair_strain` gain, default zero, adds a localized
extension/compression field for each actual body pair. The axis follows its
projected separation; its signed rate comes from conditioned relative motion,
with the real three-dimensional separation in the denominator. A coincident
projection has no strain axis and generates no artificial force.

A Gaussian quadrupole streamfunction is differentiated analytically, including
the product rule for the closed canvas boundary. Its curl is incompressible.
This is an authored kinematic response to the source, not a fluid-pressure
solver. The same field is used by the placement pilot and the GPU transport,
and is sampled at the solver's adaptive times rather than movie frame times.

The study controls reduce continuous pair rotation and test stronger localized
folding. Changed flow settings re-plan the initial layout so pigments remain
near active currents. Shape comparisons therefore test a complete flow/placement
recipe, not only the isolated strain parameter.

## Blending

Each multicolor simulation produces two interpretations of the same final
material and formation history: ordered layers and a homogeneous intimate
spectral mixture. They use the same synthetic 38-band pigment spectra. The
optical change does not change pigment quantities or transport. Additional
thinner-glaze and relief proofs use the same-state archive workflow described
in [BLEND_STUDIES.md](BLEND_STUDIES.md).

## Reproducible batches

`studies.py` freezes the complete recipes, source hashes and rendering code in
a plan. At most two renderer jobs run concurrently. Each completed case must
pass archive verification, including full source traversal, before it enters
the result list. Failed output remains inspectable and is never relabeled as
complete or overwritten. Run from a frozen source release:

```sh
python -m tools.estuary_confluence.studies \
  --counts 1 2 3 5 --flows original --width 1024 \
  --output /path/to/new-color-screen --execute

python -m tools.estuary_confluence.studies \
  --seeds 0xb7f327f9f722 0x808861c25b6c 0xceddf97909f39cc2 \
  --counts 3 --flows quiet fold open --width 1024 \
  --output /path/to/new-shape-screen --execute
```

Every case spans all one million recorded states through the existing 7,200-step
canonical paint clock, with extra adaptive transport steps where necessary.
The separate orbital warmup is excluded. A lower-resolution screen assesses
composition; it is not evidence of native fine detail. Still dimensions and
movie timing are archived explicitly. Frame cadence does not shorten the source.

## Validation

Tests cover old palette identities, count/pool prefixes, absent chalk, per-pigment
mass accounting, single-pigment contact diagnostics, CPU/GPU optical parity,
source-derived strain, analytic curl and numerical divergence, closed boundaries,
zero-gain legacy equality, archive association, and review controls. Full rendered
case verification complements the small numerical fixtures.

## Visual screening and film selection

The first frozen screen contains 40 physical cases: all ten cohort seeds at each
of 1/2/3/5 starting pigments. Multicolor cases render both optical interpretations,
for 70 images. A second screen tests quiet, folding, and broader strain-driven
currents on five seeds. Five additional calligraphic cases reduce initial pool
radius to 0.18, increase initial load to 0.28, and use radius 0.16, translation
strength 1.4, swirl 0.12, and strain 0.55. These are intentionally different
composition recipes, with their exact controls retained in the plans.

No global flow replacement won visually. The film study therefore uses explicit,
fixed selections rather than claiming that the new flow is always better:

| Seed | Selected flow | Visual reason |
| --- | --- | --- |
| `0xb7f327f9f722` | Calligraphic | More open, separated strokes |
| `0x808861c25b6c` | Quiet | Broader separated gestures and retained accent |
| `0xa0c78ebadfb75018` | Open-current study | Two linked lobes replace the compact disc |
| `0x2d01093da35729b3` | Quiet | More tapered form; calligraphic variant became a filled patch |
| Remaining six | Original | Stronger existing compositions or no comparative evidence for replacement |

Each seed keeps its selected flow, shared palette prefix and retained initial
pools across 1/2/3-pigment films. These are curated study selections, not an
automatic aesthetic ranking or a new default for arbitrary seeds. The separate
matched color-count gallery retains original currents for every seed.

The film plan uses a native 2048 × 1536 material grid and RGB16 final paintings,
1440 × 1080 movies with 2× linear-light frame filtering, and 937 frames at 24 fps.
Formation spans the entire recording in 30 seconds, followed by a three-second
hold and six-second camera study. The approximately 39-second duration is a
presentation choice, not a shorter source recording. Changing output cadence
leaves the canonical physical evolution unchanged.

Twelve five-pigment optical proofs and six monochrome proofs supplement the
material studies. Intimate mixing changes green/teal intersections more than it
changes near-analogous palettes. Thinner optical density gives the monochrome
6210 seed a pale jade appearance; it washes out B7 too much to be a lead choice.
Raised relief alone contributed little at the retained camera angle. These are
visual judgments, not objective museum-quality scores.

The original B7, 8088, and CEDD weighted five-pigment screening states were
regenerated with this implementation. Their complete material hashes match the
released states exactly, with zero strain and identical physical controls.
