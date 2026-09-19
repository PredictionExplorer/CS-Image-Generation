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

The default renders all six setups for all ten RC1 seeds: 60 complete films and
final paintings. `--seeds` and `--setups` can select a subset. `--body-markers`
enables diagnostic red Xs; clean artwork is the default. `--still-only` is for
individual implementation proofs, not the full-film comparison release.

Each film traverses the complete recording with RC1's 721 formation frames,
72 hold frames and final camera movement: 937 frames at 24 fps. Simulation is
native 2048 × 1536, final paintings are 2048 × 1536, and films are 1440 × 1080.
The runner preserves failed outputs and refuses to overwrite prior studies.

## Validation

Tests cover seed influence, geometry/support bounds, additive patches, no future
source dependence, true initial body anchors, multiple-grid mass integration,
archive tampering, actual GPU initialization/transport, and capture-cadence
independence. The preflight checks all 60 native initial conditions before the
full batch. Regression checks preserve legacy recipes, fields and rendered pixels.
