# Deliberate beginnings

These experiments retain the accepted RC1 appearance and change the **initial
condition**: where each pigment starts and the footprint it occupies. The aim is
to find more varied overall silhouettes without leaving isolated, inactive paint.

## Controlled comparison

`choreography_studies.py` reconstructs each released RC1 recipe and verifies its
source and recipe identity against the pinned release. Every treatment retains
the source recording, projection, colors, per-pigment amounts, three-body flow,
material evolution, surface finish, camera and complete timeline. Paint evolves
on the native 2048 × 1536 grid, and final images retain that resolution.

The first matrix has ten treatments, with two stronger shape studies added after
visual review:

| Treatment               | Initial change                                                         |
| ----------------------- | ---------------------------------------------------------------------- |
| Active pools            | Select three pools together using their predicted movement             |
| Compact pools           | Smaller footprints, unchanged pigment amounts                          |
| Broad pools             | Wider, shallower footprints                                            |
| Unequal pools           | Restrained small, medium and large hierarchy                           |
| Stretch ovals           | Elongated deposits oriented around measured currents                   |
| Crossing strokes        | Short tapered strokes across active currents                           |
| Facing banks            | Two banks and a third deposit sharing an active corridor               |
| Split lobes             | Divide the dominant pigment into two substantial deposits              |
| Long ribbons            | Longer tapered deposits spanning the active currents                   |
| Swept crescents         | Open curved deposits with negative space inside their starting outline |
| Ovals · balanced layers | Ovals with the first and second pigments redistributed between layers  |
| Ovals · accent mobility | Ovals with the first and third pigments redistributed between layers   |

RC1 already transports its upper and lower paint layers at different rates. The
last two treatments change only how much of each color **starts** in those
layers. They are not different physical viscosities. Their geometry is identical
to the ordinary ovals, their total pigment amounts remain unchanged, and the
palette record retains its original identity. Effective fractions are recorded
separately and invalid values are rejected rather than clipped.

## Placement and reproducibility

`choreography.py` builds a bounded set of candidates from the recorded motion and
measures their response using the same conditioned flow as RC1. It considers
only the three actual pigments. Every component of a split deposit receives its
own movement check, so one active component cannot conceal another inactive one.

The pilot uses both layer speeds and the original palette's layer fractions.
Candidate selection remains the same when a layer-allocation treatment is
enabled; this keeps that comparison controlled. A passing pilot is a prediction,
not a guarantee that all final paint will look involved, and is not a beauty
score. Failed placement gates are reported as failures, not promoted as winners.

Vector geometry is archived and rasterized by the shared native-grid utility.
Normalization independently matches each pigment's pinned RC1 world-area amount.
The former six shape experiments retain their original raster arithmetic. A
practical concentration limit is based on a mass-equivalent reference disk; it
is explicitly a proxy, not a measurement of the historical RC1 peak.

All choices derive from the source recording, its seed, the versioned algorithm
and explicit settings. There is no ambient random state. The archive binds the
complete source, projection, palette, vector geometry, pilot report, settings,
runtime and rendered media. Exact pixel identity across different GPU drivers
is not promised.

## Diagnostics and artistic review

The optional `launch_assessment` records an area-averaged view at source fractions
0, 0.25, 0.5, 0.75 and 1 using existing simulation checkpoints. It adds no force,
paint or random field. Reports distinguish visible pigment, contact, shape
moments and overlap with the initial footprint.

Overlap is an Eulerian measurement of concentration at canvas locations; it
cannot prove that specific paint parcels stayed still. Even a uniform moving
deposit may have little concentration change. Read these diagnostics alongside
the films. An elongated shape, a high contact score or many components does not
by itself make an artwork better.

The comparison publisher preserves the RC1 collection, verifies every recipe and
pigment amount, and supports starting images, final images, films, linked detail
inspection and per-seed visual picks. Stills without films are labeled honestly.
Publication is portable and cannot overwrite its source archives.

## Running studies

Use a frozen source checkout for every batch; runtime identities include the
renderer, runner and gallery source files.

```sh
# All treatments on three contrasting seeds, complete-source native-grid stills.
python -m tools.estuary_confluence.choreography_studies \
  --source-root /path/to/orbits --output /path/to/new-still-batch

# Extend selected treatments across all ten seeds with complete films.
python -m tools.estuary_confluence.choreography_studies \
  --all-seeds --variants stretch-ovals cross-strokes --film \
  --source-root /path/to/orbits --output /path/to/new-film-batch

python -m tools.estuary_confluence.choreography_gallery \
  --output /path/to/new-review --reference-gallery /path/to/preserved-rc1 \
  --cases /path/to/completed-case-a /path/to/completed-case-b

python -m tools.estuary_confluence.choreography_gallery \
  --verify --output /path/to/new-review
```

Select only one archive for each seed/treatment in a publication, preferring its
film archive when both still and film runs exist. Tests cover native mass budgets,
component engagement, seed determinism, layer allocations, archive tampering,
off-path RC1 compatibility and shared review controls.
