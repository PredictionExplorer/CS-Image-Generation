# Fine folds: paired paintings and films

For new artwork studies, a finished image and a full-trajectory film are a pair.
Use the paired batch pipeline by default. A photograph alone does not count as a
completed treatment. Earlier still-only archives remain reproducible historical
records; they are not silently changed into movie archives.

## What the film shows

`filament_motion.py` defines all fifteen options from the fine-fold study: eleven
paint treatments (including two labeled coarse-grid diagnostics) and four
lighting/relief comparisons. Lighting comparisons share the exact Control paint.

Every film includes:

1. **Complete formation:** 721 frames at 24 fps, sampling canonical steps 0 through
   7200 of the paint simulation. The entire one-million-sample orbit is traversed.
   Four trailing simulation states contribute to each exposure after the initial
   frame. The sharp final material state is also archived separately.
2. **Finished relief:** 96 camera frames at 24 fps, examining the completed paint
   without advancing its simulation. The camera gently approaches the matching
   still photograph's pose.

The existing editor dissolves from the completed formation into the relief view,
producing 817 frames, about 34 seconds, at 1920 × 1440. The dissolve is an edit
between views, not an additional paint interaction. The first part views pigment
from above; the final part photographs authored relief from its concentrations.
This is not a physically simulated three-dimensional liquid.

Comparison stills use 2048 × 1536 and 128 Cycles samples. Motion uses 1920 × 1440
and 32 samples. An explicitly selected master still uses 3840 × 2880 and 256
samples. The camera, material and lighting match at the endpoint; compression,
resolution and sampling differences mean the video is not pixel-identical to the
larger photograph. Native paint remains 6144 × 4608, except the named coarse-grid
diagnostics. Filming does not change the canonical simulation clock or physics.

## Sources and reproducibility

`filament_cohort.py` derives ten full-width 256-bit seeds from a versioned SHA-256
domain and counter. It rejects historical seed collisions, without inspecting
rendered results to discard an inconvenient seed. The exclusion inventory,
derivation, exporter binary and source-selection settings are archived.

Each fresh seed uses the pinned production exporter and a 30,000-candidate
trajectory search. The selected orbit has one million warm-up steps followed by
one million recorded steps at dt 0.001 and stride 1. The completed cohort manifest
binds each source recording, configuration and export log. Seed-derived palette
variations retain the reference's navy, ivory and oxide roles.

The batch plan freezes the cohort, recipes and runtime/tool hashes. Backfilled
films additionally require their final material hash to match the earlier
painting, and retain the earlier publication identity as evidence.

## Running and recovering

Prepare and validate a cohort with `filament_cohort`, then create the complete
plan using `filament_film_batch.make_plan`. Freeze the source checkout before
launching the durable worker:

```sh
python -m tools.estuary_depth.filament_film_batch \
  --plan /path/to/plan.json --output /path/to/batch --workers 2
```

The same command resumes an interrupted batch with its original plan and frozen
runtime. Material simulations are shared by dependent lighting cases. Each case
retains its formation, prepared maps, still, camera film, edited film and receipts.
GPU work is bounded; failed attempts and logs remain available. A case is only
marked complete after all its artifacts and the fully decoded movie verify.

The formation worker uses `--checkpoint-retention 2`. It atomically publishes a
new recovery checkpoint before retiring older owned recovery states; it retains
the current and preceding states. Movie frames and final archive artifacts are
not removed. Omitting this option preserves the earlier retain-all behavior.

Use `filament_film_gallery` for the portable progressive review. Published rows
always have both verified media types. Content-addressed entries are immutable;
the review's progress and comparison manifest are updated atomically as more
pairs finish. Pending work is reported separately from completed films.
New gallery entries include source-bound 640 × 480 previews for the small cards
and film posters. Full-resolution image links remain unchanged. Portable
verification checks both preview integrity and its source-derived pixels;
earlier entries without previews retain their original identities.

## The ten-new-seed campaign

The campaign under `/home/user/estuary-depth/filament-films` plans **171 pairs**:
150 treatments across ten new seeds, plus films for the 21 earlier study images.
The exact fresh seed derivation, production settings and frozen exclusion
inventory are in [the cohort plan](cohorts/filament-fresh-v1-plan.json).
All ten recordings are certified in [the completed cohort](cohorts/filament-fresh-v1.json).

- `new-cohort-v1/cohort.json` certifies the newly generated recordings.
- `backfill-v1/plan.json` binds the 21 earlier paintings to their known material
  hashes. `backfill-v1/progress.json` records their current stages.
- `fresh-plan-v1.json` is created after all ten new recordings verify;
  `fresh-v1/progress.json` records the 150-pair batch.
- `review-v1/comparison.json` is the atomic, portable gallery snapshot.
- `publication-health.json` records publication progress and any publisher error.
- `campaign-complete.json` is written only after all 171 pairs and the complete
  portable gallery verify. Its absence means completion is not yet certified.

The review is served on loopback port 8804 on the experiment server, with a
managed local SSH forward to the same port. Its “Check for completed films”
button loads the latest certified pairs. Source generation, rendering and
publication continue independently of the browser. The earlier still gallery
on port 8803 remains separate.

The pilot film reproduced the earlier broad-pool material hash exactly:
`d794cd8ef4371888ad077ee7d39cda297b0191471689878b20490942f60dde5a`.
Qualification includes 96 passing core GPU tests, the 155-test depth suite
(its six native Blender and two Node viewer tests also exercised in their
respective environments), and full playback of the reference and broad-pool
films in the browser. Repository Ruff and configured Mypy checks passed.
The source-derived preview and live/JSON double-precision compatibility checks
add six further regression tests; source hashes, physical checks and projection
tolerance remain unchanged.
