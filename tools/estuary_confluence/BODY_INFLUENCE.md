# Which bodies move the paint?

This experiment keeps RC1's original three-body recording, palette, initial paint,
material settings, camera, lighting, and complete film timeline. A selection
controls which recorded bodies may affect the paint. All three trajectories still
come from the original gravitational simulation, and all starting pigments remain.

The seven comparisons are each individual body, each pair, and all three bodies.
The saved RC1 artwork supplies the all-three reference. Active bodies retain their
original strength; no multiplier compensates for removing other contributions.

## Influence contract

- Each selected body contributes its original conditioned stirring velocity,
  swept-track wetting, and any configured deposition. RC1 has zero deposition.
- Pair swirl and strain require both endpoints to be selected. Disabled body,
  pair, and strain descriptors are zeroed completely.
- Encounter wetting is planned only from eligible pairs, before ranking, event
  limits, and suppression of nearby events. A single body has no pair encounters.
  A pair can therefore have a different event schedule from the saved all-body
  painting, using the same selection rules and source-distance normalization.
- Inactive travel is excluded from adaptive brush-travel bounds as well as from
  paint/water doses. It cannot silently change the integration schedule.
- Diffusion, layer exchange, drying, and interaction texture continue to respond
  to the resulting paint, water, and velocity fields.

The full-source projection, scale, and original all-body initialization pilot are
frozen reference conditioning. Every experiment must reproduce RC1's initial
layout artifact and palette identity exactly. Reducing the initial-layout pilot
would relocate paint and confound the comparison.

## Configuration

```json
{
  "simulation": {
    "body_influence": {
      "version": "body-influence-v1",
      "bodies": [0, 2]
    }
  }
}
```

This is a fragment for an existing recipe. Source body indices are zero-based;
the gallery labels this selection **Bodies 1 + 3**. Lists must contain one to
three distinct indices. Their order is canonicalized. Omission, `null`, and the
explicit complete set normalize to the original all-body configuration without
adding a key to legacy recipes. The default numerical path and shaders remain
unchanged.

## Full matched studies

```sh
python -m tools.estuary_confluence.body_influence_studies \
  --output /path/to/new-body-influence-study
```

The default produces all six reduced selections for the ten RC1 seeds: 60 films
and 60 final paintings. `--variants` and `--seeds` select a subset; `--source-root`
locates the original orbit files when they are outside the saved archive path.
Inputs are checked against pinned RC1 hashes. `--still-only` supports individual
qualification proofs. The optional `all-bodies` variant requires exact equality
with the saved RC1 complete material state and is used to qualify the baseline.

Each complete film has 937 frames at 24 fps, including formation, hold, and camera
movement. Native material and final paintings are 2048 × 1536; films are
1440 × 1080. Existing archives and galleries are retained as separate releases.

## Validation

Validation covers strict selections, unchanged all-three behavior, isolated
forcing/wetting, active-pair event selection, adaptive scheduling, fixed starting
geometry, capture-cadence independence, artifact binding, and native pigment
budgets. Tests of inactive-body isolation hold the original projection and layout
fixed while changing the inactive body's sampled measurements. Recomputing the
projection or initial arrangement would be a different experiment.
