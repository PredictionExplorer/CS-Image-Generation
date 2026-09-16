# The Remaining Form — proposed next art study

Status: implemented as an experimental sculpture and photography pipeline.
The selected recipes and actual behavior are documented in [remaining-form.md](remaining-form.md).
The plan below records the original artistic goals; it is not a completion claim for every proposed production deliverable.

## Recommendation

Develop **Sculpture of Absence** into a genuinely three-dimensional porcelain
object. Keep **Mineral Memory** as a restrained, optional interior treatment after
the sculpture succeeds in plain clay.

The target is a memorable relationship between a deep chamber, its opening, and
the material that survives around it. The object should reward both a distant
view and looking through an opening. Its appearance should remain compelling
from multiple viewpoints.

Nocturne's strongest changes remained within a predetermined rounded outline.
Better numerics, spectral color, and reflections improved its finish without
substantially changing that visual premise. This study must establish the form's
artistic strength before expanding production infrastructure or surface effects.

## The intended image

A pale, asymmetric porcelain fragment occupies a quiet photographic scene.
A broad opening curls into shadow. Through it, a smaller passage catches cool
light at another depth. A thin lip admits warm light, while a narrow surviving
bridge gives the object a sense of fragility. Thick shoulders and delicate walls
create a deliberate rhythm.

Warm ivory is the initial material target. The exterior is softly matte; exposed
inner walls are slightly smoother. One large directional light reveals curvature
and cavity depth, supported by weak fill and a simple ground shadow. A three-quarter
view establishes the object's thickness. These are art-direction targets, not
guaranteed outcomes of the carving algorithm.

Light passing through a new opening can create a second, quieter composition on
the ground or the wall of another chamber. Explore this as a consequence of the
same geometry and lighting. A small pool of light appearing when two chambers
finally connect could carry the film's encounter more beautifully than a large
visual effect.

## How the motion becomes sculpture

### Preserve the physical source

Reuse the existing frozen `.orbit` recording and `OrbitSeries`. Preserve the
original three-dimensional positions, one fixed translation and uniform scale,
and stable transported frames. Evaluate pair proximity in original normalized
3D space. Output frame rate must not influence the generated object.

Begin with seed `0xb7f327f9f722` for comparison with the previous studies, then
test at least two additional recordings. Inspect actual depth variation before
choosing an orbit: a nearly planar source should not be stretched along one axis
to manufacture spatial richness.

### Define the starting material explicitly

The leading blank is a gently rounded, coarse envelope of the instantaneous
triangles swept through the complete recorded motion. This makes the initial
mass depend on the collective motion rather than imposing the same pebble on
every orbit. Freeze this envelope before the excavation animation begins.

Compare this with one simpler, controlled blank during the first experiment so
we can identify which geometric mechanism is responsible for the result.

A surrounding envelope can seal all interesting carving inside opaque material.
Include one deliberate oblique reveal cut as an art-direction parameter. Its
purpose is to establish a sightline into the sculpture. It must be visible in
the recipe; it is not an event claimed to arise from orbital physics. Start with
this opening present so the process can be judged. A final reveal cut is a later
cinematic variation.

### Carve with broad, gently flattened tools

Each body carries the same compact, smooth three-dimensional carving footprint.
A moderately flattened ellipsoid follows its position and stable trajectory
frame. Sweep these tools through the material with a nonnegative cumulative dose:

`C(x,T) = integral from 0 to T of the sum of the three tool contributions dt`.

Remove material where the dose exceeds a fixed threshold. This permits lingering
and repeat visits to enlarge chambers, and makes removal monotone in time.
A simple union of identical swept tubes would not deepen on exact revisits.

Initially hold cutter proportions and strength fixed. Only after seeing the
result, test a weak, bounded connecting contribution during actual close pair
approaches. That contribution can allow separate chambers to develop a shared
throat. Strong connecting fields can remove an entire triangular slab, so this
is a controlled second experiment rather than a default embellishment.

The final exposure field records cumulative occupation. It is not a reversible
encoding of temporal order. The animation and source provenance retain the
ordered history; any material chronology needs its own explicit definition.

### Keep interior strata optional

First prove the sculpture in neutral clay and then in unpatterned porcelain.
Afterward compare a restrained material-history version: closely related ivory
layers, perhaps one pale celadon or smoky-blue band inside a chamber.

If layers are described as chronological, derive them from a separate broad
deposition/arrival-time field established before excavation. Coloring arbitrary
stripes is an artistic material choice and must be described as such. First-removal
time alone is a poor independent stratification coordinate because it is closely
tied to the final carving threshold and last local exposure.

## Implementation sequence and deliverables

### 1. First sculpture experiment

- Establish a small visual reference board for form, material, and lighting.
  Treat references as targets, not evidence that our generator works.
- Build only the minimum continuous carving field, controlled blank, and mesh
  export needed for actual proofs. A roughly 192-cubed exploration grid is a
  starting point, subject to feature-size checks.
- Make six form studies: three carving doses and two cutter proportions.
- Render each in neutral clay using consistent lighting. Inspect promising
  objects from three substantially different views.
- Deliver a comparison sheet and a short account of why the strongest form
  deserves further work.

**Art checkpoint:** a distinctive silhouette, one dominant spatial idea, visible
depth, and a reason to look inside. Uniform holes, narrow drilled corridors,
unreadable lace, or an ordinary rounded lump do not advance simply because the
algorithm and export tests pass. If the construction keeps producing those
outcomes, change the carving rules or blank and rerun the bounded experiment.

### 2. Refine the selected form

- Adjust a small number of global controls: cutter proportions, total dose,
  blank rounding, encounter connection strength, and the reveal cut.
- Aim for a principal chamber, supporting passage, quiet exterior, and a delicate
  surviving connection. These are selection criteria, not hand-patched mesh parts.
- Test intermediate source times, including the first opening and chamber merger.
- Compare at doubled spatial and temporal resolution around the thinnest wall.
- Reject unwanted fragmentation or change the global recipe and replay. Do not
  silently refill walls or remove detached components independently every frame.

### 3. Material and photographic study

Keep geometry fixed while testing a small set of lighting and material treatments.
The preferred treatment is warm porcelain with thickness-dependent translucency,
soft cavity illumination, and selective grazing highlights. Compare at least one
lighting arrangement in which the cavity remains legible without looking flat.

Use a pinned Blender/Cycles CPU build for serious material proofs and final
photography. Keep the custom geometry and history engine in Rust. The existing
Silk renderer is useful for geometry debugging and inexpensive clay views, but
its thin-cloth transmission does not supply solid porcelain subsurface transport.
Do not invest in a new general renderer unless an actual tested limitation demands it.

The server reported 128 logical CPUs and 503 GiB memory during planning. Blender
and Mitsuba were not found on PATH; confirm availability and establish an isolated,
pinned renderer installation during implementation. No installation was performed
for this plan. Benchmark one representative still before estimating final cost.

### 4. Short motion proof

Render a short passage that includes meaningful change: a wall thinning, two
chambers meeting, or a new opening becoming visible. Preserve the cumulative
history before the excerpt starts.

The camera should provide a useful view of the carving. Hidden excavation is
not a successful film merely because the final object is attractive. Evaluate
the reveal cut, blank thickness, and view together. Check calm passages as well
as the encounter.

Changing topology requires deliberate exposure sampling. If a renderer's native
deformation blur assumes compatible meshes, render actual geometry at each
subframe and average scene-linear light; do not interpolate unrelated vertices.

### 5. Production and final examination

Render the entire recorded source once the form and motion proof meet the art
criteria. Hold the completed sculpture, then make a restrained camera move that
reveals a real passage or bridge previously hidden by the first view.

Deliver a high-resolution hero still, a second view of the same object, a detail
of the interior, the complete film, a short final-object examination, and the
surface mesh. Select output resolution and samples from measured convergence and
visual inspection. Preserve original linear-light masters and display versions.

## Code architecture

Reuse the source cache, source sampling, provenance, recipe validation, resumable
jobs, verified assembly, video encoding, and delivery infrastructure.

Keep new responsibilities small and separate:

1. **Source adapter:** fixed physical coordinates and source-time integration schedule.
2. **Excavation field:** compact kernels, positive dose integration, checkpoints, and queries.
3. **Surface extraction:** verified mesh generation and material attributes.
4. **Renderer adapter:** versioned scene export, physical scale, lights, materials, and camera.
5. **Artifact workflow:** input hashes, renderer version, scene/sample settings, frame receipts,
   resume checks, linear exposures, and complete movie verification.

Use established sparse-volume/meshing infrastructure where it improves reliability.
The older `viz-master-plan` capsule and marching-tetrahedra code offers useful
reference primitives, but its dimension caps and surface audits need independent
review before reuse. OpenVDB's scalar-grid meshing is a candidate, not an automatic
dependency commitment; first confirm the minimal prototype's actual needs.

### Required correctness properties

- Positive integration weights and a fixed source-time calibration. Never count
  output frames as erosion steps or divide the dose by currently elapsed time.
- Independent queries at a given time produce the same geometry regardless of
  frame order, worker count, restart, or output fps. Prefer spatial ownership of
  tiles and ordered accumulation over nondeterministic floating-point atomics.
- Cumulative dose and removed volume cannot decrease. Positive interpolation of
  canonical kernel samples can provide continuous partial-time evaluation.
- Exposure fields are not signed distance fields. Mesh them correctly or provide
  justified tracing bounds; never use arbitrary density as a sphere-tracing step.
- Validate oriented manifold edges and vertices, degeneracies, self-intersections,
  connected components, signed volume, and thin-feature resolution.
  A closed mesh includes the complete walls of intentional passages; a passage
  is not a missing polygon. Distinguish connected solid material from connected
  boundary surfaces, since an enclosed cavity can add a separate boundary shell.
- Confirm chamber connections survive spatial and temporal refinement. Detect
  visible topology pops and denoising instability in moving images.
- Preserve deterministic geometry and fixed rendering seeds/settings. Pin the
  external renderer and document its reproducibility limits; do not promise
  cross-platform pixel hashes that the renderer does not guarantee.
- Keep the existing Rust and Python checks green, and add tests for the new
  mathematical contracts rather than screenshots that merely freeze a weak look.

## Artistic acceptance is separate

The strongest candidate should be compelling in gray clay, coherent from more
than one view, visibly changed by its history, and materially convincing under
simple illumination. Fine detail should enrich a legible large-scale form.
Numerical correctness, sample counts, and 8K resolution do not establish those
qualities. The first implementation milestone is a genuinely persuasive sculpture
proof, not a finished production system.

## Reference capabilities checked

- Existing source/provenance: `src/atelier/source.rs`, `src/silk/orbit.rs`,
  `src/silk/cache.rs`.
- Existing CPU intersections and thin-cloth renderer: `src/silk/render/geometry.rs`
  and `src/silk/render.rs`.
- [Blender: subsurface scattering and closed-mesh requirements](https://docs.blender.org/manual/en/4.0/render/shader_nodes/shader/principled.html).
- [Blender: background rendering and explicit CPU selection](https://docs.blender.org/manual/en/3.5/advanced/command_line/render.html).
- [OpenVDB: meshing scalar grids](https://www.openvdb.org/documentation/doxygen/namespaceopenvdb_1_1v13__0_1_1tools.html).
