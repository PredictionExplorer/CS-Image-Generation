# Starting paint studies

**Release draft:** the active family is `pattern-studies-v2`. Version two addresses
contrast lost in actual simulated thin folds, which the original pure-color
separation checks did not predict. The deployment below is the planned v2 campaign;
its final integration counts and rendering completion remain to be recorded.

`pattern-studies-v2` compares ten starting compositions on the **same ten recorded
three-body trajectories**: 100 image-and-film pairs. Each seed keeps one palette
across every motif. This makes differences between patterns easier to judge while
retaining different colors and trajectories across seeds.

The family uses the existing [paired-film pipeline](FILAMENT_FILMS.md). It changes
the initial pigment field and its seeded palette; it keeps the full source clock,
transport settings and photographic studio consistent across the ten patterns.

## The ten motifs

| Option                   | Starting composition                                               |
| ------------------------ | ------------------------------------------------------------------ |
| `lacuna-banks`           | Joined light and medium banks around two unequal dark openings.    |
| `interlocking-crescents` | Opposing crescents with open dark interiors.                       |
| `braided-ribbons`        | Broad crossing ribbons beside a fine returning strand.             |
| `split-fan`              | A medium stem between two tapered light tongues.                   |
| `river-confluence`       | Tributaries joining a divided, bending river.                      |
| `broken-terraces`        | Unequal contour shelves with deliberate interruptions.             |
| `meandering-fault`       | A broad winding seam between a light bank and a dark field.        |
| `folded-sash`            | An S-shaped, two-color band that narrows through its middle.       |
| `asymmetric-rosette`     | Unequal petals around an open dark aperture; labeled Open rosette. |
| `branching-channels`     | Forked dark channels separating light and medium peninsulas.       |

The v2 recipe entry point is [pattern_studies_v2.py](pattern_studies_v2.py). It
reuses the immutable catalog and geometry controls from
[pattern_studies.py](pattern_studies.py); the spatial construction lives in
[initial_patterns.py](../estuary/initial_patterns.py).
Lacuna banks is the comparison reference, and Folded sash is the default selected
option. Neither choice is a claim that it is the best result for every seed.

### Placement, paint amount and feeding

The initializer samples the projected body positions at 65 times spanning the
entire recording. Their median center, principal orientation and spatial extent
guide the motif's placement. Small rotation, phase, skew and stretch variations
come from the seed and pattern identifier. This is structured placement around
the trajectory, rather than unconstrained random placement on the canvas.

Every solver pixel starts with a partition of three pigments whose fractions sum
to one, within float32 roundoff. `initial_load=0.6` is applied once. Overlapping
motif regions therefore do not multiply the paint amount, and all patterns have
the same initial total loading over the full solver domain. **Individual pigment
amounts can differ between motifs.**

Continuing deposition uses `pigment_weights=[1,1,1]`. These are equal feeding
coefficients; the actual amount supplied by each body still depends on its travel
and the common fading rule. Constant total loading describes the initial field,
not a promise of exact mass conservation throughout transport and deposition.

### Coverage is not proof of movement

`coverage_stats` measures pigment shares in the visible canvas and in a central
ellipse derived from the source positions. The ellipse is called the active
corridor in the diagnostic. It is a **spatial heuristic**, not a fluid simulation
or a measurement showing that every painted pixel will move.

All three pigments having substantial initial coverage does not guarantee equal
participation, balanced final colors, fine folds, or an attractive finished image.
Full films and visual comparison remain necessary. The source-position sampling
used to arrange the motif is separate from the simulation's integration clock.

## Seeded, distinct colors

[pattern_palette_v2.py](pattern_palette_v2.py) defines `pattern-palette-v2`. It
retains the seed choices of [pattern_palette.py](pattern_palette.py), whose
versioned, named SHA-256 streams use all 32 seed bytes. Leading zero bytes are
retained; no ambient random generator or render-dependent reroll is used.

Seed-derived controls and palettes are deterministic. Render archives also pin
code, tools and inputs and record the execution hardware; they do not promise
bit-identical pigment states or images across different GPUs, drivers or numeric
platforms.

The three pigments retain dark, light and medium value roles. Seeded triadic or
split-complementary arrangements vary their hues and assign them to those roles.
Colors are constructed in OKLab/OKLCH, then brought into sRGB by reducing chroma
while keeping the intended lightness and hue. The generator verifies:

- At least 0.20 pairwise OKLab distance between the authored pigment colors.
- At least 75 degrees of hue separation.
- Minimum chroma of 0.04, 0.05 and 0.07 for the dark, light and medium roles.

Version two additionally keeps every pigment channel's **linear reflectance at
or above 0.003**. A nearly zero channel implied extreme absorption in the existing
RGB pigment model, allowing dark pigment to overwhelm intermingled light strands.
The new construction reduces OKLCH chroma at fixed lightness and hue until the
bound is satisfied; it does not clip RGB channels. The three separation checks
above remain in force.

The substrate remains seed-derived, usually dark, with occasional light grounds.
V2 uses relative scattering coefficients **[0.2, 8.0, 3.0]** for dark, light and
medium pigments, selected using actual completed pigment fields. These are
authored artistic parameters, not measured pigment properties. Subtractive
mixtures can still lose chroma; neither version promises that every mixture stays
vivid. V1 palettes and their original coefficients remain reproducible unchanged.

## Resolution and time

| Stage                | Settings                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| Native paint         | 6144 × 4608, float32 pigment state, 7200 canonical paint integration steps.                      |
| Source recording     | Complete interval of the 1,000,000-sample trajectory recording, including both endpoints.        |
| Master photograph    | 3840 × 2880, 256 Cycles samples.                                                                 |
| Formation film       | 721 frames at 24 fps, 1920 × 1440, four trailing canonical states per exposure after frame zero. |
| Finished-relief film | 96 frames at 24 fps, 1920 × 1440, 32 Cycles samples per frame.                                   |
| Published edit       | 817 frames at 24 fps, approximately 34.04 seconds.                                               |

The recording has one million trajectory samples; this is **not one million
fluid integration steps**. Its full interval is sampled onto 7200 fixed
source-clock paint steps, with bounded adaptive subdivisions, and 721 formation
frames. Changing output cadence does not change the canonical paint integration.
The sharp final material state is archived independently of the final formation
frame's temporal average.

After the complete formation, the editor dissolves into a camera examination of
the frozen completed paint. The camera ends at the paired photograph's pose.
Resolution, sampling and video compression differ, so the movie endpoint is not
promised to be pixel-identical to the master photograph. Relief is authored from
the pigment concentrations, not produced by a three-dimensional fluid solver.

One native four-channel state contains 28,311,552 pixels and requires
452,984,832 bytes (432 MiB), before the NPY header. The initializer works in row
tiles instead of allocating full-canvas intermediate geometry. The GPU solver
owns additional transport textures; 432 MiB is not the total GPU memory budget.

## Versioned runtime and archive compatibility

Pattern plans opt in with `study_family="pattern-studies-v2"` and
`paint_runtime_extensions=["initial_patterns.py"]`. Their formation recipe uses:

```json
{
  "initial_pattern": "composition",
  "initial_design": {
    "version": "starting-patterns-v1",
    "pattern": "folded-sash",
    "seed": "0x followed by 64 lowercase hexadecimal digits"
  }
}
```

This example shows the simulation fields; use the recipe factories to produce a
complete valid recipe and seed. `code_identity(recipe)` includes the initializer
only when the optional design is active. Offline verification selects the frozen
version-one core, archived shaders and explicitly declared extensions. It does
not reinterpret an old archive using today's dependency list.

Existing fine-fold plans omit the new family and extension fields. Their record
schemas and viewer remain unchanged, and old publications continue to verify.
Different study families use separate publications. Algorithm changes must keep
the versioned recipe and runtime contracts explicit.

Both pattern families remain selectable. `pattern-studies-v1` regenerates the
original palettes and recipes; `pattern-studies-v2` replaces only optical controls.
The starting geometry, placement seed, feeding, transport, camera and source clock
are unchanged. The dispatcher never silently upgrades a v1 plan. Omitting
`study_family` still selects the earlier fine-fold catalog.

## Plan, run, resume and publish

Use the pinned Estuary Python environment, Blender 4.5.14 with OptiX, and explicit
FFmpeg/ffprobe executables. Run from a frozen checkout: runtime changes invalidate
an executing plan. The commands below use `python` for that environment and
illustrative unused output paths; they do not replace the deployed campaign.

### Create the complete plan

`make_plan` verifies the completed source cohort, resolves every recipe and pins
the tool/runtime identities. `master_quality=True` upgrades unreferenced stills
only; formation and motion controls are unchanged. Explicit earlier reference
quality flags are honored, and conflicting upgrades are rejected.

```python
from pathlib import Path

from tools.estuary_depth.filament_film_batch import make_plan
from tools.estuary_depth.pattern_studies_v2 import DEFAULT_OPTION, OPTIONS, REFERENCE_OPTION
from tools.estuary_studio.common import read, write

source_root = Path("/home/user/estuary-depth/filament-films/new-cohort-v1")
cohort = read(source_root / "cohort.json")
order = [REFERENCE_OPTION, DEFAULT_OPTION]
order += [option for option in OPTIONS if option not in order]
selections = [(seed, option) for option in order for seed in cohort["seeds"]]

plan = make_plan(
    cohort,
    source_root=source_root,
    blender="/home/user/remaining-form/vendor/blender-4.5.14-linux-x64/blender",
    ffmpeg="/usr/bin/ffmpeg",
    ffprobe="/usr/bin/ffprobe",
    selections=selections,
    study_family="pattern-studies-v2",
    master_quality=True,
)
write(Path("/path/to/pattern-plan.json"), plan)
```

Explicit case order prioritizes the reference across all seeds, then Folded sash,
then the other options. A pilot and a remaining matrix can instead use disjoint
`selections` lists, while sharing the same source cohort and frozen runtime.

### Execute or resume

```sh
python -m tools.estuary_depth.filament_film_batch \
  --plan /path/to/pattern-plan.json \
  --output /path/to/pattern-batch --workers 2
```

Repeat the identical command to resume. The controller bounds GPU work to two
workers, retains the current and preceding paint checkpoints, verifies completed
stages, and preserves failed attempts. A case becomes complete only after its
still, camera movie and full-formation edit verify. New source, recipe or runtime
settings require a new plan and output archive.

### Publish and verify pairs

```sh
python -m tools.estuary_depth.filament_film_gallery \
  --batches /path/to/pattern-batch --output /path/to/pattern-review

python -m tools.estuary_depth.filament_film_gallery \
  --output /path/to/pattern-review --verify
```

The publisher infers the family from the plan. The Python `publish_review` API
also accepts a title, for example `title="The Estuary · Starting forms"`. Register
both batch roots for a split pilot/matrix campaign. Repeated publication copies
newly completed pairs and atomically updates `comparison.json`; it never reports
a still-only case as a completed pair.

Lacuna banks must be ready before a seed's other patterns become visible, so the
reference comparison is always available. Full-resolution images and full films
are preserved; source-bound 640 × 480 previews serve the small cards. Portable
verification works after the source batches are removed. For a source archive,
`filament_film_batch.verify_case(case_directory)` additionally verifies the
original material and stage records, returning its certificate and artifact paths.

## Deployment and qualification

The planned v2 frozen release directory is:

```text
/home/user/estuary-depth/pattern-studies/releases/pattern-atlas-v3
```

The campaign is split into `pilot-v2` (Lacuna banks and Folded sash for seed A709)
and `matrix-v2` (the remaining 98 pairs), with publication in `review-v3` at the
same [local port 8805](http://127.0.0.1:8805/). These are distinct archives;
the earlier v1 pilot and publications are retained. Consult `campaign-v2-status.json`,
the batch `progress.json` files and `publication-v2-health.json` for actual completion.

V2 palette checks cover 4096 seeds and retain the original OKLab distance, hue and
chroma thresholds while enforcing the new linear-reflectance floor. Full v2
integration and render qualification counts will be recorded after the release
is frozen. This section does not claim all 100 pairs have rendered.

### Initial prototype evidence

The initial qualified prototype used commit `93d79f3` at:

```text
/home/user/estuary-depth/pattern-studies/releases/pattern-atlas-v2
```

That prototype registered four pilot pairs in `pilot-v1` and 96 remaining pairs
in `matrix-v1`, with progressive publication in `review-v2`. The pilot is retained
for comparison while its optical response is assessed; the main campaign may use
new versioned plans and release directories. These paths are historical prototype
references, not a claim that the matrix has finished or that its settings are
final. Read `campaign-status.json`, each batch's `progress.json`, and
`publication-health.json` for current status; the gallery reports ready and pending
counts. The local review is [port 8805](http://127.0.0.1:8805/) when its server
tunnel is active.

Recorded prototype qualification results, to be rerun for the final calibration:

- **113 base server GPU tests passed**, with no skips.
- **182 depth tests ran:** 174 passed and eight were skipped on the server.
  Those six native Blender tests and two Node tests also passed in their
  appropriate runtimes.
- **4096 seeded palettes checked**, including pure colors and 50/50 mixtures
  through the existing finite-layer pigment optics.
- **100 actual seed–motif spatial combinations checked.** The smallest pigment
  share in the central source corridor was **7.898%**. This was a spatial coverage
  result, not a dynamic participation or aesthetic score.
- The native **6144 × 4608 state budget** was checked. Ruff passed across 203
  files, and the configured five-file strict Mypy target passed.

These checks establish implementation and archive contracts. They do not certify
artistic quality or imply that all 100 images and films have finished rendering.
