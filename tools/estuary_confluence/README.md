# Confluence Fresco

Seeded pigment, selective deposition, and revealed underpainting driven by the
complete recorded three-body trajectory. This experiment extends Tidal Fresco
without changing the earlier Estuary, depth, or studio packages.

## Paint material experiments

See [PAINT_MATERIALS.md](PAINT_MATERIALS.md) for fuller relief, directional contact
structure, transported seeded properties, and paint resistance with memory. These
opt-in studies preserve RC1 as the all-three-body reference.

## Color and shape experiments

See [COLOR_SHAPE_STUDIES.md](COLOR_SHAPE_STUDIES.md) for one-, two-, three-, and
five-pigment compositions, optional source-driven pulling and folding, matched
spectral blend views, and reproducible experiment batches. Existing presets
retain their earlier defaults. [BLEND_STUDIES.md](BLEND_STUDIES.md) documents
same-material optical proofs.

## Layered paintings

`recipes/layered-five.json` opts into the new material study. All earlier
recipes, palette algorithms, and physical archives remain supported.

- `palette_mode="composed"` generates five color roles from the full seed:
  dominant, support, accent, deep anchor, and quiet bridge. Tonal, analogous,
  complementary, and split-accent relationships use different lightness/chroma
  ranges and authored scattering strengths. All five candidates are resolved
  before selecting a three-color prefix. Exact upper-layer mass fractions and
  spectral mixture checks travel with the palette. The spectra remain synthetic,
  not measured artist pigments.
- `simulation.initial_pigment_weights` optionally scales each starting pool's
  base load before layer splitting. The selected five-color recipe uses
  `[2.2, 0.9, 0.5, 0.6, 0.4]` in the palette's role order. The archived layout
  retains its base geometry/load, while the recipe records the exact multiplier.
  There is no count-dependent renormalization. Independent budget regeneration
  applies the same weights. `null` preserves the original initialization exactly.
- `simulation.material_model="laminate"` divides the existing initial paint into
  two moving layers. `mobile` stores the upper layer, `underpaint` the moving lower
  layer, and `deposit` remains empty. `pigment` is still their total. The lower
  layer follows the same trajectory-driven velocity at a bounded speed ratio;
  the upper layer follows the original velocity. This is an authored two-layer
  transport model, not a full three-dimensional fluid or a pressure solver.
- `interlayer_exchange_rate` controls positive, species-conserving exchange at
  occupied wet contacts. It preserves the local amount in each layer. Each layer
  also receives the existing within-layer interdiffusion. Both layers share the
  original transported wetness/mixedness field. The global mass correction now
  measures and scales both layers together. It does not make their limited
  MacCormack transport locally conservative; no flow-map solver is claimed.
- `surface.finish="glazed"` preserves the crisp real-mass silhouette and constant
  exterior ground. Inside it, actual optical mass is bounded by
  `glaze_min_mass_ratio` and `glaze_max_mass_ratio`, relative to
  `paint_mass_reference`. The actual ordered layer fractions are preserved.
  Concentrated paint banks carry most relief; wet upper-layer coverage and
  scattering vary surface sheen. These display controls never modify material
  arrays, and camera frames retain the completed material unchanged.
- An optional recipe-level `background="palette-night"` resolves and archives
  `background.json` from the generated palette and seed. Verification checks its
  derivation, exact saved colors, and agreement with the rendered surface.

The film review layout starts with the complete seed collection. Each card has
separate Film and Image actions that open a focused, keyboard-accessible viewer.
Closing the viewer stops playback and restores focus to the selected card.
Completed or failed playback restores the finished painting instead of leaving
an empty video surface. A continuous-play option retains native video controls.
The gallery pairs current/earlier images only when the source
recording, projection, aspect, color count, and optical view match. This comparison
intentionally changes colors and material; it does not claim identical physical
states. Small previews are derived from the published posters in linear light,
verified again after copying, and used for the grid and video posters. Full RGB16
posters and films remain downloadable. The five-color layered-view review is
explicitly selected with `--layout films`:

```sh
python -m tools.estuary_confluence.run \
  --source /path/to/seed.orbit \
  --recipe tools/estuary_confluence/recipes/layered-five.json \
  --output /path/to/new-complete-film

python -m tools.estuary_confluence.gallery \
  --layout films --title "The Estuary, in layers" \
  --cases /path/to/new-film /path/to/earlier-film \
  --output /path/to/new-review
```

Model checks cover local exchange conservation and positivity, real differential
layer motion, total pigment accounting, output-cadence independence, ordered-layer
optics, native GPU/CPU capture agreement, archive tampering, and old-render parity.

## Convergence studies

See [EXPERIMENTS.md](EXPERIMENTS.md) for the parameter sweeps, visual judgments,
numerical checks, and selected production settings.

`recipes/convergence-three.json` and `recipes/convergence-five.json` combine
source-aware starting pools, no imposed directional current, finite-rate pigment
interdiffusion, and spectral color. Earlier recipes retain their original models.

The `engaged-pigment-layout-v2` planner proposes separated, pure-color pools near
the actual stirring paths. A bounded particle pilot evaluates 32 candidate
layouts, considering the least active pigment's travel, stretching, proximity to
other colors, and composition. It evaluates both the three- and five-pool subsets
before selecting one shared layout. Its fixed pilot clock is independent of paint
resolution, diffusion, optics, and film cadence. The seed, source hash, projection,
flow settings, exact pool positions, and pilot diagnostics are archived. Pilot
scores are approximate placement diagnostics, not proof of final pigment contact
or artistic quality; unsuccessful eligibility is recorded explicitly.

The pilot distributes 17 tracers through each pool's area and measures the mean
nearest-other-color proximity across those tracers. A single touching edge cannot
stand in for participation by the whole pool. The original candidate RNG namespace
is retained explicitly, isolating the improved selection criterion from changes
to candidate randomness.

The earlier recipes impose a carrier velocity of `[2, 0.2]`. Convergence uses
`[0, 0]`, removing that systematic rightward current while retaining the
trajectory's own asymmetry. It does not recenter individual film frames.

Interdiffusion exchanges pigment fractions across neighboring wet paint cells.
Equal and opposite face fluxes conserve every pigment and local total thickness
in this operator; dry paint and empty space receive no flux. Substeps enforce an
explicit stability bound. This avoids diffusing the paint silhouette into a
colored haze. It is an authored mixing model, not measured molecular diffusion;
the separate advection operator still has its documented numerical mass error.

Convergence also enables `simulation.mass_budget_interval_steps=12`: after each
fixed block of canonical steps, and at the final step, a GPU reduction measures
each pigment's global amount and positive uniform channel scaling restores its
initial budget. This option is restricted to separated pure pools with no added
paint, settling, or underpaint. Empty channels remain empty. `mass-budget.json`
records every correction's step, amounts before and after, and exact applied
factors. Verification regenerates the initial sampled pools and independently
integrates the native final fields. This corrects global numerical paint growth;
it does **not** make advection locally conservative or recover misplaced material
and unresolved strands. Earlier recipes leave the option disabled.

`surface.optics_model="spectral"` integrates 38 wavelength samples, 380–750 nm,
using finite-layer Kubelka–Munk reflection and transmission. Display colors are
reconstructed into **synthetic** spectra; they are not measured artist pigments.
The reconstruction uses the permissively licensed Spectral.js bases and D65/CIE
integration data, with a documented colorimetric correction. Exact spectra,
coefficients, upstream revision, data hashes, and license attribution are saved in
`spectral.json`. Legacy RGB optics remain available and unchanged.

Spectral version 2 treats incomplete mixing as independent pigment columns over
the lower reflector: it averages their complete reflected spectra, then blends
with intimate-mixture reflection using the stored mixing field. It does not apply
an arbitrary saturation boost or reconstruct fine strands lost by simulation.

Optional `assessment` checkpoints record each pigment's mass, centroid, spread,
visible share, dominant area, and contact with another substantial pigment. The
history uses a reduced grid; the final assessment uses the native material grid.
Reduced grids can combine unresolved neighboring strands, so their contact
estimates are not interchangeable with native measurements. Movement, stretching,
and contact are separate diagnostics: maximizing contact can produce dull,
overmixed paintings. None of these measurements changes the simulation or rates
artistic quality. Image balance and actual solver substeps are also archived.

Convergence films use 1,201 formation frames, a 24-frame hold, and 144 camera
frames: 1,369 frames at 24 fps, about 57.04 seconds. Formation spans the entire
recorded trajectory in 50 seconds between its first and last frame. The separate
orbital warmup is not part of the recording. Simulation steps and orbital states
are different quantities; the receipt records both the source coverage and the
actual paint transport work.

The production Convergence recipes retain the full material grid for movie
capture, too: neighboring pigments are shaded before any image reduction. Each
film frame is rendered at twice its output dimensions and area-averaged in linear
display RGB before PNG encoding. This preserves fine color boundaries and reduces
thin-line aliasing without changing pigment state. The opt-in control is
`render.frame_supersampling=2`; its default of `1` preserves earlier frame output.

## Background and lighting studies

`appearance.py` renders eight presentations of an already verified painting:
White, Charcoal, Midnight Blue, Aubergine, Palette Night, Grazing Light, Raised
Paint, and Satin Reflection. The first five isolate the visible ground color.
The last three also change lighting, displayed relief, or surface reflection and
viewing angle. Pigment concentrations, trajectories, material coefficients, and
the pigment's optical backing remain unchanged. These are appearance studies,
not new physical simulations or a simulation of pigment on black primer.

Palette Night uses a principal pigment hue, restrained OKLCH chroma, and
full-seed-derived variations in darkness and hue. It preserves the shared ground
across three/five-color comparisons. Every view archives its exact sRGB and linear
ground colors, derivation, seed, palette binding, and identity. Background
validation retains those exact archived colors while allowing tiny numerical
roundoff when another platform regenerates the derivation. Image balance uses
contrast against the actual ground; its original white-ground result is retained.

```sh
python -m tools.estuary_confluence.appearance \
  --case /path/to/verified-physical-artwork \
  --output /path/to/new-appearance-study

python -m tools.estuary_confluence.appearance_gallery \
  --studies /path/to/study-a /path/to/study-b \
  --output /path/to/new-appearance-gallery
```

Appearance archives bind the parent request, receipt, and actual material hash;
their read-only fields are checked before and after rendering. They include all
resolved controls, background records, full-resolution images, linear rasters,
previews, and the rendering code. The parent physical archive remains the source
of the original recording and material arrays. The gallery copies images and
provenance without duplicating large linear rasters. Optional `--films` accepts
only verified films matching the physical state, source, pigment palette,
spectra, simulation, selected surface, and final still camera. Unmatched films
are recorded without being attached to the wrong presentation.

The expanded cohort is pinned in `recipes/ten-seeds.json`. All ten recordings use
the same production Yoshida4 f64 physics, G=9.8, dt=0.001, one million warm-up
steps, and one million recorded states. The first five trajectories were selected
from 30,000 candidates; the final five from 100,000. This difference is explicit
in the records: the cohort explores variety across seeds, not the effect of the
trajectory-selection algorithm. The five additional archived records were chosen
by stable SHA-256 ordering of canonical seeds. Exporter identity, original
generation records, initial-condition bits, and complete orbit files are retained.

## Native movie capture

`render.capture_pipeline="native-gpu"` keeps full-resolution material textures on
the GPU through geometry preparation, spectral optics, lighting and optional 2×
frame reduction. It requires capture resolution equal to simulation resolution.
Only a small validation summary and the final linear RGB frame return to the CPU;
the full material snapshot is still archived at completion. The default `"cpu"`
path remains available for comparison and older workflows.

`Engine.gpu_frame()` is a read-only, current-step view. Borrowed-context surfaces
do not own the simulation context; explicit stale views and closed owners are
rejected. Camera-only rendering uses the surface's retained material copy.
The final still deliberately uses the established CPU snapshot/render path.
The accelerated path was checked against that reference on a complete native
trajectory: physical-state and final-poster values matched exactly, and the tested
8-bit PNG frame was identical before movie encoding (maximum linear error 1.19e-7). These are measured
checks on the tested hardware, not a promise of cross-driver pixel identity.

## Scattered color studies

The new `scattered-three.json`, `scattered-five.json`, and `random-five.json`
recipes give each chromatic pigment its own separate starting pool. Three colors
means three pools; five means five. Positions, radii, and loads come from the
complete seed through a versioned layout generator. It resolves five pools before
selecting the count, preserving the first three exactly in count comparisons.
Placement does not depend on palette choice, resolution, camera, or frame rate.

`palette_mode="harmonic"` chooses a base hue over the full color wheel and builds
related hues with distinct lightness and chroma. `palette_mode="random"` chooses
the other hues independently. Both use bounded gamut mapping and checks on actual
pigment mixtures. These are constrained random colors, not uniform random RGB
triples; the checks preserve visible distinctions, not a promise of artistic
quality. Neither new mode chooses colors from the old six palette families.
`palette_mode="curated"` remains the default and preserves released palettes.

The scattered recipes introduce no additional paint along body paths and disable
settling and burial, so the moving paint leaves no stationary colored deposit.
The three bodies still determine the flow. Its active boundary is the visible
canvas, while the larger simulation guard remains available for camera movement.
This keeps starting colors from circulating outside the photographed region.
The shared chalk channel remains in
the material format for compatibility but starts empty and stays empty.

The opt-in `surface.finish="crisp"` is an intentional filled-pigment print
interpretation. A contour of actual pigment concentration defines the silhouette;
only a one-pixel antialiasing transition blends its edge. Inside, optical
concentrations are normalized to a common reference while preserving pigment and
phase proportions. Outside, the ground is exactly the requested constant color,
unaffected by wetness, surface grain, lighting, or shadows. This display treatment
does not alter the archived physical state or claim physically opaque thin paint.

Every scattered archive includes `layout.json` with exact initial conditions and
an `initial.png` for each optical view. For films, that image is byte-identical to
the first frame. The gallery's **Starting colors** control shows these actual
starting pools. Independent-color comparisons use the same layout and material
history as their harmonic counterparts; color-count comparisons add two pools.

## What is simulated

- **Three or five colored pigments plus shared chalk.** Six curated palette
  relationships receive bounded OKLCH variations from the complete 256-bit seed.
  Each pigment has an authored scattering strength, settling rate, release rate,
  granulation response, and specific volume. Body mixtures and amounts establish
  a dominant chromatic wash, countercurrent, and concentrated accent. A lighter
  mineral ground supports transparent passages, while restrained chalk scattering
  preserves the differences between colored washes and pale deposits.
- **Confluence blooms.** A fixed full-source analysis selects up to three
  distinct close approaches using actual 3D pair separation. Their projected
  locations receive integrated rewetting pulses. A source with no qualifying
  encounter receives no invented event.
- **Chromatic shorelines.** Mobile pigment and deposited material exchange at
  pigment-specific rates, modulated by wetness and the fixed mineral ground.
- **Hidden color.** An actual underpaint reservoir can receive buried material
  and release pigment during later wet passages. Its concentrations are retained
  separately from mobile paint and the upper deposit.

Two optical views can render **one simulation**:

| View | Interpretation |
| --- | --- |
| `layered` | Finite-layer reflection and transmission through underpaint, deposit, and mobile paint; incomplete mixing uses the recorded mixing field. |
| `homogeneous` | The same total pigment interpreted as one intimately mixed layer. |

The incomplete-mixing operator represents unresolved neighboring patches. It
does not invent resolved bristles or microscopic filaments. Legacy pigment optics
use an RGB Kubelka–Munk approximation with authored coefficients; spectral optics
are the explicit alternative described above. Neither uses measured paint.
Flow is Estuary's prescribed incompressible stirring field, not
a Navier–Stokes fluid simulation. Local material exchange is conservative;
limited MacCormack transport is not globally mass-conservative, and source tools
add material. Relief comes from pigment volume, never displayed brightness.

The surface renderer uses a single-valued heightfield with approximate direct
lighting, shadows, and directional reflection. It has no overhangs, volumetric
refraction, or full multiple-scattering light transport.

## Render

Run from the repository root with the existing Estuary Python environment and a
hardware OpenGL 4.3 context. `ffmpeg` and `ffprobe` are required for films.

The included `recipes/three.json` and `recipes/five.json` use a 4096 × 3072
material grid, 3840 × 2880 stills, and 1920 × 1440 films. Each film contains
469 frames at 24 fps (about 19.54 seconds), including complete formation, a quiet
hold, and a six-second camera movement. The three-color recipe also renders a
blended optical comparison from the same physical history. Run at most two of
these production recipes concurrently on the 16 GB experiment GPU.

```sh
python -m tools.estuary_confluence.run \
  --source /path/to/recording.orbit \
  --recipe /path/to/confluence.json \
  --output /path/to/new-artwork
```

Add `--still-only` for a complete-trajectory still. A compact comparison recipe:

```json
{
  "name": "Confluence Fresco",
  "chromatic_count": 3,
  "looks": ["layered", "homogeneous"],
  "simulation": {
    "resolution": [2048, 1536],
    "steps": 3600,
    "initial_pattern": "strata"
  },
  "render": {
    "capture_resolution": [1024, 768],
    "resolution": [1920, 1440],
    "still_resolution": [3840, 2880],
    "formation_frames": 301,
    "hold_frames": 24,
    "orbit_frames": 145,
    "fps": 24
  }
}
```

This example produces a 4K output from a 2048 × 1536 material grid. Increase
`simulation.resolution` to increase physical detail; increasing the output alone
does not create finer simulated marks. Movie capture must divide both simulation
dimensions by the same integer. It area-averages state without changing the
simulation. Final stills always use the full material grid. The final camera arc
examines the completed, frozen painting.

Changing `chromatic_count` from 3 to 5 adds two subordinate pigments while
preserving the primary colors, their material coefficients, chalk, ground, and
body amounts. It changes the material mixture, so the resulting physical state
is a separate experiment. Changing optical view or movie cadence does not change
the material history.

## Reproduction and archives

Palette identity uses canonical numeric seed equivalence: hexadecimal case and
leading zeroes do not change the result. Independently named SHA-256 streams use
all seed bits. Palette version, exact colors, physical parameters, and mixture
quality checks are saved in `palette.json`. These checks reject basic numerical
failures; they are not an artistic-quality score or a guarantee of perceptual
uniqueness across all possible seeds.

Each archive contains:

- `request.json`, `recipe.json`, `palette.json`, and `events.json`.
- For engaged or scattered pools, `layout.json`; for spectral optics,
  `spectral.json`; when requested, `assessment.json` with canonical checkpoints
  and a native-grid final report. These files are hash-bound by the receipt.
- When global mass restoration is enabled, `mass-budget.json` with the initial
  per-pigment budgets and every canonical correction.
- Original `inputs/source.orbit` and copies of all four runtime packages.
- `final.npz`: full-grid mobile, deposited, and underpaint concentrations plus
  geometry, wetness, mixing, direction, roughness, coverage, and total pigment.
- A 16-bit `poster.png` and bounded linear-sRGB `poster-linear.npy` for each look.
- For films, each look's frames, `film.mp4`, full-decode evidence in `movie.json`,
  and a shared `frame-ledger.json` containing exact source and camera timing.
- `receipt.json` binding files, runtime, full source traversal, and the common
  physical-state hash. The receipt remains incomplete after failures.

Completed output is verified before reuse. Incomplete output is preserved for
inspection and cannot silently be advertised as complete or overwritten. Palette
and event identities are independent of capture size, frame rate, and camera.
Pixel identity across different GPU drivers or devices is not promised.

## Comparison gallery

```sh
python -m tools.estuary_confluence.gallery \
  --output /path/to/new-gallery \
  --cases /path/to/seed-a-three /path/to/seed-a-five \
          /path/to/seed-b-three /path/to/seed-b-five
```

Cases are verified before publication and must include complete films unless
`--allow-stills` is explicitly supplied for proofs. Each physical case contributes
all of its optical views. Within each seed, the gallery orders three-color
layered, five-color layered, then three-color blended. The palette strip includes
the shared chalk and uses the exact archived display-sRGB pigment values.

The comparison control pairs the two three-color optical views only when their
source and physical-state hashes match. A five-color layered painting compares
against the three-color layered painting for the same trajectory; its caption
identifies the changed pigment count. Images and films can be downloaded at their
published resolutions. Content-addressed media and copied design records keep
the gallery verifiable if the original experiment folders move.

Pass `--earlier-gallery /path/to/previous-gallery` to include an independent
**Earlier version** comparison. It requires the same seed, pigment count, palette
mode, optical view, and source recording. Its label identifies the earlier
version; it does not imply identical physical history or layout. The copied
earlier image and design records remain available inside the new gallery.

## Texture from paint encounters

The optional [interaction material](INTERACTION_TEXTURES.md) records contact,
deformation-aligned fabric, and aggregation while the paint moves. Named optical
views compare the original finish, satin seams, and contact grain on the same
complete history. All random material variation derives from the archived seed.
The preserved Color & Form release remains the control for composition and
pigment transport.

Use `python -m tools.estuary_confluence.interaction_studies --help` for matched
native-grid experiments. The guide explains the numerical model, controls,
archive contract, limitations, and validation requirements.

## Body-position guides

[Body-position diagnostics](BODY_MARKERS.md) add optional red X1–3 to the images
and films. They follow the same projected source positions and camera as the
painting, without entering the paint simulation. The matched ten-seed experiment
reconstructs the saved RC1 recipes and requires identical complete material states.

## Starting shape and placement

[Six starting-paint compositions](INITIAL_COMPOSITIONS.md) compare random circles,
tapered ribbons, open crescents, facing shores, scattered commas, and body-centered
wedges across the ten RC1 seeds. Each preserves the original per-pigment amounts,
palette, source recording, flow, surface finish, and complete film timeline.
The comparison page pairs any two setups with shared playback and seeking, and
includes the saved RC1 artwork as a reference.

[Deliberate beginnings](CHOREOGRAPHED_STARTS.md) returns to the accepted RC1
appearance and tests trajectory-informed placement, unequal sizes, elongated
deposits and split components. Every pigment retains its original amount;
separate experiments vary its starting allocation between the existing layers.

## Selecting the bodies that influence paint


[Body-influence studies](BODY_INFLUENCE.md) compare each body and each pair against
RC1's all-three reference. The original trajectories, initial painting, palette,
and film timeline stay fixed. Body forcing, pair effects, wetting, and adaptive
travel bounds follow the selection without compensating for reduced strength.

## Validation

```sh
python -m unittest discover -s tools/estuary_confluence -t .
ESTUARY_TEST_GPU=1 python -m unittest discover -s tools/estuary_confluence -t .
```

CPU tests cover seed equivalence and full-bit influence, palette mixtures and
color-count invariance, real encounter selection, conservative phase exchange,
optical passivity, source-complete timelines, paired-view state identity,
artifact tampering, and incomplete-run behavior. Convergence tests also cover
source-aware layout determinism, conservative interdiffusion, dry/air boundaries,
spectral colorimetry, assessment timing independent of movie captures, native
assessment regeneration, global pigment-budget restoration, linear-light frame
filtering, and archived solver diagnostics. Hardware tests exercise
the actual GPU transport and optical kernels.

OKLab conversion uses the public-domain sRGB matrices published by
[Björn Ottosson](https://bottosson.github.io/posts/oklab/).

Spectral reconstruction derives from
[Spectral.js](https://github.com/rvanwijnen/spectral.js), revision
`bb2b05c9d1e65ae824d47e3b1cc17ea32c8ee68f` (MIT, Ronald van Wijnen).
See `spectral_data.py` and `licenses/spectral-js-MIT.txt` for the transformation
details and preserved attribution.
