# Confluence Fresco

Seeded pigment, selective deposition, and revealed underpainting driven by the
complete recorded three-body trajectory. This experiment extends Tidal Fresco
without changing the earlier Estuary, depth, or studio packages.

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
