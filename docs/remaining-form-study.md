# The Remaining Form — first study

## Artistic selection

The selected object is an ivory porcelain shell with two opposing curls and a
dark S-shaped passage. It is the remainder of a fixed ellipsoid after cumulative
three-body excavation. The final recipe uses neither of the optional reveal cuts.
The opening therefore belongs to the moving cutters, rather than an added viewing hole.

The early triangle-envelope blank failed the artistic test: its sampled scallops
and broad planar cut looked anatomical, and small holes dominated the image.
Wider cutters exposed useful depth but did not fix that silhouette. A quieter
ellipsoid removed the scallops. The decisive change was the object's pose: the
initial rotation faced its main cavity toward the floor. Turning the same object
over revealed the opposing curls and a genuine passage through the interior.

The final material is restrained warm porcelain on charcoal. A modest global
rounding of the stock/carving intersection softens the rim. Mineral decoration
was omitted because the main opening should carry the image.

Close views exposed shading bands from area-weighted triangle normals. The final
export instead uses continuous field derivatives. A byte-level comparison of all
1,890,106 vertex positions and 3,780,216 triangles confirmed unchanged geometry.
The selected ellipsoid/cutter field supplied every normal analytically, with no
finite-difference fallbacks. This correction changes the lighting response, not
the sculpture's shape or history.

Eleven vertices on one tight rim have continuous-field normals more than 90° from
their polygonal area averages; their 57 incident faces cover approximately 0.003%
of the surface. This local discretization difference is retained in the comparison
record. Normals are not flipped heuristically to match the old approximation.

Three recorded sources were explored: `0xb7f327f9f722`, `0x808861c25b6c`, and
`0xbc53af1cd380`. The first was retained. Some stronger doses fragmented the stock;
those candidates were rejected rather than having detached components deleted.

## Evidence and limits

The selected geometry uses 2048 canonical time intervals. Its smallest cutter
support radius is 0.75 world units; the maximum source-center step is 0.03740034.
That ratio is a calibration indicator, not a proof of temporal convergence.

Spatial refinement at fixed source time and field settings gave:

| Longest grid axis | Signed volume | Boundary components | Euler characteristic |
| --- | ---: | ---: | ---: |
| 192 | 9.012313 | 1 | -16 |
| 384 | 9.021675 | 1 | -2 |
| 512 | 9.023010 | 1 | -2 |

The 384 and 512 grids agree in volume to approximately 0.015% and have the same
two-handle topology. The coarse grid retains small topology artifacts, so it
must not be treated as a manufacturing model. The hero mesh uses 384 nodes;
the separately labeled motion preview uses 176 nodes to explore the sequence.

A matched temporal refinement at the 384-node grid increased the canonical
interval count from 2048 to 4096. The volume changed from 9.021675358 to
9.021676955 (approximately 0.0000177%); the boundary remained connected with
Euler characteristic -2. This supports the selected temporal sampling for the
completed object. It does not certify every instantaneous topology transition.

Every exported mesh passes closed-manifold, orientation, and native-coordinate
precision checks. General self-intersection testing and physical fabrication
certification are outside this implementation. The recipe, source, binary,
meshes, images, and scenes have content hashes in their corresponding receipts.

## Reproduction

Use the [selected recipes](../tools/remaining_form/recipes/README.md) with the
[architecture and command guide](remaining-form.md). The movie spans the complete
recorded source interval in 96 excavation frames, then examines the unchanged
final mesh in 48 camera frames. At 24 fps with the declared holds, it is nine
seconds. This is an initial art study, with point-sampled motion rather than a
temporally supersampled production master.

The implementation was checked with the complete Rust suite: 760 passed, one
pre-existing test ignored, no test exclusions. All-target Clippy, formatting,
16 Python contract tests, and Ruff checks passed. These checks establish software
and numerical behavior; the artistic judgment comes from inspecting the stills,
multiple views, and the moving sequence.
