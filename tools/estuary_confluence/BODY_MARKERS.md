# Body-position review guides

Optional red X1–3 identify the three source bodies in their recorded order. They
are diagnostic guides burned into review images and films, not pigment sources
or part of the material simulation. Every body contributes to the shared stirring
field, so a marker need not lie on a painted region or identify a particular color.

## Rendering the experiment

The accepted checkpoint is Git tag `RC1`, also published on
`codex/estuary-rc1`. Marker work lives on `codex/estuary-body-markers`.

```sh
python -m tools.estuary_confluence.body_marker_studies \
  --output /path/to/new-body-position-study
```

This renders all ten RC1 seeds with their original resolution and complete film
timing. `--seeds` selects a subset; `--still-only` renders final-image proofs.
Every reconstructed recipe must match the RC1 record before annotations are
enabled, and every finished material state must match RC1's full fourteen-field
hash. Existing outputs are immutable; choose a new output directory.

For another confluence recipe, add `"body_markers": true` to its `render` object.
Omission, `null`, or `false` retain the unannotated contract. An object can
configure `labels`, `size_px_1080`, and `stroke_px_1080`; size and stroke scale
with image height. The default is a 36-pixel red X at 1080 pixels high, with a
contrasting outline and fixed vector labels. No font files or randomness are used.

## Coordinates and time

- Sample `Source.frame(canonical_fraction).positions`, which already contains the
  same fixed projection used by the paint engine. The flow's `tool_uniforms` does
  not alter these positions; it conditions only velocities.
- Convert projected simulation coordinates to metres using
  `canvas_width_m / (2 * aspect)`. Do not multiply by `domain_scale`: that expands
  the guard domain without changing the visible canvas scale.
- The guides sit at `z=0`, the canvas plane. Project with the renderer's
  orthographic `camera_basis` right/up columns, then invert image y for the
  top-down output. They are not depth-tested against raised paint or presented
  as marks on the heightfield.
- Each X shows the instantaneous source position at its captured state. Numerical
  flow integration samples adaptive **midpoint** times between captured states;
  the endpoint guide is therefore not the last substep's force center.
- Hold and camera-orbit frames retain the final source positions. Camera changes
  only reproject those frozen points; they never advance the simulation.

These are the fixed two-dimensional projections of three-dimensional paths.
Pair interactions also use the original three-dimensional separation, so
overlapping Xs do not necessarily indicate a close physical encounter.

## Provenance and publication

Annotated archives retain `body-markers.json`, including source positions,
projected pixel centers, and frame timing. The request and receipt bind marker
settings and source identity; full archive verification reconstructs positions
from the archived recording. Portable galleries retain the receipt-bound position
record and validate its timing and metadata associations.

Only annotated studies publish marker metadata or a **Body positions** record
link. Their gallery links back to the saved RC1 collection. Existing unannotated
studies keep their earlier metadata and collection link.

## Verified RC1 experiment

The [release record](releases/body-markers-rc1-v1.json) pins ten 2048 × 1536
marked paintings and ten complete 1440 × 1080 films. Each film contains 937 frames
at 24 fps, including formation, hold and camera movement. All 9,370 frames were
independently decoded. Every complete material state, unmarked final raster and
image-balance diagnostic matches RC1 exactly.

Both CPU and GPU suites ran 500 tests without failures, with 79 and seven skips
respectively. Browser review verified marked playback and progression between
seeds. The RC1 tag remains at `dda34ef`; the experiment is on its separate branch.
