# Six starting-paint compositions

This experiment changes the initial paint geometry and placement while retaining
RC1's source recording, palette, flow, layer behavior, lighting, camera and film
timeline. All ten seeds use three actual pigments. There is no five-pool search
and no future-trajectory participation pilot for these new arrangements.

| Setup | Initial geometry and placement |
| --- | --- |
| Random circles | Three independent uniformly sampled centers within each circle's fit bounds; overlaps allowed |
| Tapered ribbons | Curved, varying-width vector strokes with seeded orientation and bend |
| Open crescents | Open arcs with thick middles, narrow tips and seeded gap/orientation |
| Facing shores | Two curved paint banks facing a gap, with three accent-color pockets |
| Scattered commas | Three unequal curved patches per pigment; nine actual patches |
| Body-centered wedges | Wedge centroids at the initial projected body positions, heading along initial velocity |

The random variants share underlying placement quantiles. Each maps them into its
own admissible bounds, so centers can differ when geometry needs different canvas
margins. Shapes are not moved toward future activity. Body wedges alone read the
source's first frame; uniform shrink preserves the true body-centered anchor when
needed to fit. A stationary body's heading has a seeded fallback.

Geometry consists of bounded vector strokes and polygons, with narrow smooth
boundaries. Repeated same-color components accumulate before normalization.
Unresolved tips fail explicitly instead of silently disappearing at initialization.

## Matching the material

`releases/composition-rc1-inputs-v1.json` pins each seed's actual RC1 initial
pigment amounts, reference radii, source and palette identities, and the original
records' hashes. The amounts already include RC1's pigment load weights.

A `shaped` initializer takes `initial_composition` controls with version, setup,
three `target_mass` values, and three `reference_radii`. It requires source-free
laminate paint and `initial_pigment_weights: null`; the old loading controls are
not reapplied. Native-grid integration normalizes each complete color mask to its
reference amount. Chalk is zero. Upper/lower splitting uses the unchanged palette.

Float32 normalization and layer splitting introduce small rounding differences;
matching uses the existing 5e-6 relative mass tolerance. The shapes and resulting
material histories are expected to differ. Equal pigment amounts also preserve
integrated raw geometric volume to rounding; the existing nonlinear displayed
relief interpretation can change with concentration and is not claimed constant.

## Running complete studies

```sh
python -m tools.estuary_confluence.composition_studies \
  --output /path/to/new-composition-study
```

Run this where the original orbit archive recorded in `recipes/ten-seeds.json`
is available. Each input recording is checked against its pinned RC1 hash.

The default renders all six setups for all ten RC1 seeds: 60 complete films and
final paintings. `--seeds` and `--setups` can select a subset. `--body-markers`
enables diagnostic red Xs; clean artwork is the default. `--still-only` is for
individual implementation proofs, not the full-film comparison release.

Each film traverses the complete recording with RC1's 721 formation frames,
72 hold frames and final camera movement: 937 frames at 24 fps. Simulation is
native 2048 × 1536, final paintings are 2048 × 1536, and films are 1440 × 1080.
The runner preserves failed outputs and refuses to overwrite prior studies.

## Publishing and comparing

Publish only after all 60 cases have finished:

```sh
python -m tools.estuary_confluence.composition_gallery \
  --output /path/to/new-comparison \
  --reference-gallery /path/to/verified-rc1/films \
  --cases /path/to/new-composition-study/cases/*
```

`--cases` also accepts case paths from multiple independent batches. The publisher
requires the complete six-by-ten matrix and checks every setup against that seed's
saved RC1 controls and pigment amounts. It copies verified media and provenance
into a portable directory; the old RC1 gallery and numerical archives stay intact.
Moving the new directory does not break its relative media links.

The review offers starting paint, final paintings, and two synchronized films.
A shared seek control selects the same recorded instant in both panels. Changing
a setup or seed pauses both at that time; Restart both returns to the beginning.
Focus comparison keeps the complete paintings and controls in
view, and Escape restores the normal page. Only the two selected films load;
switching away releases their media buffers. Every image and film can be opened
or downloaded independently.

Unconstrained random placement intentionally permits quiet patches outside the
strongest stirring. Those are valid experimental outcomes and remain in the
comparison. No completed seed is discarded or moved to make its result look
better. Body-centered wedges provide the separate alternative tied directly to
the three initial body positions.

## Validation

Tests cover seed influence, geometry/support bounds, additive patches, no future
source dependence, true initial body anchors, multiple-grid mass integration,
archive tampering, actual GPU initialization/transport, and capture-cadence
independence. The preflight checks all 60 native initial conditions before the
full batch. Regression checks preserve legacy recipes, fields and rendered pixels.

## Verified experiment

The [release record](releases/initial-compositions-v1.json) binds all 60 final
paintings and full films to the frozen renderer, saved RC1 inputs, and independent
audit. Every film contains 937 frames, for 56,220 frames in the new collection.
The portable comparison includes ten saved RC1 references alongside these studies.

Both the CPU and server GPU suites ran 525 tests without failures, with 83 and
seven skips respectively. The final publisher's 11 tests passed locally; its
server run skipped the JavaScript test because Node was unavailable. The review
server passed 421 URL checks and byte-range requests for all 70 comparison films.
Browser review loaded every seed's images and previews and exercised distinct-film
playback, shared seeking, setup/seed changes, focus mode, and completed playback
returning to the final paintings. No browser errors or warnings were reported.
The largest measured final pigment-budget error was 9.01e-8 relative, below the
5e-6 limit. Four concurrent workers completed the full batch in 60.55 minutes;
peak sampled GPU memory was 7,711 MiB on the 16 GiB experiment GPU.

The numerical renderer is preserved at `cdf361b`, the comparison publisher at
`064c6ff`, and the work is on `codex/estuary-shape-studies`. RC1 remains at its
original tag. The release pins separate numerical and presentation revisions so
gallery refinements do not obscure which code produced the paintings.
