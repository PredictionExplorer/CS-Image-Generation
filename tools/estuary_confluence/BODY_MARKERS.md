# Body-position review guides

Optional red X1–3 identify the three source bodies in their recorded order. They
are diagnostic guides burned into review images and films, not pigment sources
or part of the material simulation. Every body contributes to the shared stirring
field, so a marker need not lie on a painted region or identify a particular color.

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

## Provenance and publication

Annotated archives retain `body-markers.json`, including source positions,
projected pixel centers, and frame timing. The request and receipt bind marker
settings and source identity; full archive verification reconstructs positions
from the archived recording. Portable galleries retain the receipt-bound position
record and validate its timing and metadata associations.

Only annotated studies publish marker metadata or a **Body positions** record
link. Their gallery links back to the saved RC1 collection. Existing unannotated
studies keep their earlier metadata and collection link.
