# Texture from paint encounters

`contact-microstructure-v1` adds an opt-in material history to the two moving
paint layers. Occupied contacts can develop a directional satin response and
irregular matte aggregate patches. The appearance comes from stored material
state; the camera does not generate texture.

This is an authored constitutive model. Its coefficients describe useful,
bounded behavior for these paintings, not measured pigment chemistry. The
aggregate variable represents an unresolved dispersed/aggregated partition of
the existing pigment. It does not simulate individual particles or add volume.

## Formation and transport

Each layer starts with its initial world coordinates and zero interaction state.
The coordinates are transported material origins, not current screen positions.
They allow two parcels of the same pigment to register an encounter after
following different paths.

Contact requires both layers to contain paint. Its strength combines the local
layer balance with either a difference in pigment composition or a separation
between the transported material origins. Empty space and two identical,
unseparated layers produce no contact. Wetness controls the local reactions:

- Contact dose accumulates toward a bounded limit.
- The symmetric, trace-free velocity gradient aligns an axial fabric tensor.
  Material spin rotates it; wet relaxation weakens it. Its magnitude never
  exceeds the contact dose. A velocity direction alone does not stand in for
  deformation.
- Aggregation grows at contact and competes with strain-driven breakup.
  Seed-derived nucleation propensity varies in transported material coordinates
  and affects the reaction, not a final-image overlay. Drying slows both rates.

Positive, mass-weighted characteristic interpolation carries these descriptors
with their respective paint layers. Empty neighbors contribute no history. The
default `linear` method uses a bilinear gather along the characteristic.

Set `simulation.interaction.advection` to `"maccormack"` to retain more of the
transported detail. This optional method makes forward and reverse gathers,
then applies their estimated error correction. Each corrected component is
limited to the extrema of occupied old donors that actually contribute to the
forward gather. Correction is skipped when the old target or reverse support
is absent. Contact and aggregate bounds and the coupled fabric/dose bound are
enforced after correction. The forward support is an interpolation guide for
history, not a replacement pigment-density field.

Both methods transport intensive material descriptors; neither is a separate
conservative particle solver. Existing pigment transport retains its existing
numerical properties and global mass-budget correction. The extension does not
change pigment or height, or make transport locally conservative. Omitting
`advection` retains linear transport and the earlier normalized interaction
dictionary exactly. Explicit `"linear"` and `"maccormack"` values are archived
as part of the recipe identity.

## Appearance controls

Two independent surface strengths interpret the stored state:

- `silk_strength` controls the contribution of fabric to directional satin
  reflection. Its axis comes from the transported axial tensor.
- `grain_strength` makes aggregated regions rougher and controls a white-albedo
  EON rough-diffuse kernel. Its roughness comes from the transported aggregate
  fraction. This changes how the existing pigment reflection is distributed
  over viewing angles. Grain is a scattering interpretation of the aggregate
  field; it is not displaced geometry.

Both are in `[0, 1]`. Set both to zero for an optical control on the exact same
material. Omit `surface.interaction` to retain the original rendering path.
Nonzero strengths require the real interaction fields. These two controls do
not alter pigment spectra, pigment totals, height, surface normals, or the outer
mass contour. The optional packing reconstruction below is a separate geometry
control. Existing substrate grain is a separate, older surface control;
set `grain_um` to zero when judging only the new contact-generated texture.

Optional `grain_contrast` in `[1, 8]` changes the optical response to the
recorded aggregate fraction `g` using
`g^c / (g^c + (1 - g)^c)`, where `c` is the configured contrast. This monotone
remapping preserves zero, one and the midpoint. It changes roughness contrast
without adding a texture field, changing pigment reflection spectra, or
claiming a measured chemical response. Omission and an explicit value of one
retain the exact linear response. `grain_strength` multiplies the remapped value
for optical roughness and EON. The raw archived aggregate remains unchanged
and still drives packing directly; packing does not consume the contrast-remapped
optical value.

For a fair comparison from one simulation, select
`"looks": ["control", "silk", "silk-grain"]`. These named views require an enabled
`simulation.interaction` and an explicit `surface.interaction` object. The
surface object may be empty to resolve the documented default strengths.

| View | Silk strength | Grain strength |
| --- | --- | --- |
| `control` | 0 | 0 |
| `silk` | Configured value | 0 |
| `silk-grain` | Configured value | Configured value |

When packing controls are explicitly present, `control` and `silk` set
`packing_strength=0`, while `silk-grain` retains its configured value. All three
retain the same `packing_length_um` in their manifests. Omitted packing keys
remain omitted from old view dictionaries.

All three use layered pigment optics with `mix_control=1`. They share the same
material at every frame, lighting, camera, final full-state hash and frame ledger.
Each receives its own initial image, final painting and complete film. Transport
runs only once. `layered` and `homogeneous` retain their original behavior and
can also be selected; up to five distinct views are supported.

Publish these named views with `gallery --layout studies`. The collection labels
are **Control**, **Satin seams**, and **Satin + grain**. Each textured view compares
with its own case's control. The control compares with Satin + grain when
available, otherwise Satin seams. Missing companion views remain unpaired;
another experiment with the same seed is not substituted.

### Scope of the optical model

The rough-diffuse kernel specializes the white-albedo case of
[Energy-preserving Oren–Nayar (EON)](https://arxiv.org/abs/2410.18026).
`scattering.py` contains an independent float64 reference and
`shaders/scattering.glsl` the GPU implementation. The scalar kernel multiplies
the existing pigment reflectance; it does not replace that reflectance, alter
pigment spectra, or introduce a colored coat. At zero roughness it returns the
original Lambertian diffuse response exactly.

The **isolated diffuse kernel** is reciprocal, nonnegative and has unit
directional-hemispherical reflectance at white albedo. Numerical tests check
the smooth and grazing limits, reciprocal direction swaps, and quadrature
convergence. The tested 256-node polar quadrature with 512 azimuth samples has
error below `2e-5` over the selected angle/roughness grid. Separate hardware tests compare
the GLSL evaluation with the independent reference.

This is not a claim that the complete image shader has a proven energy budget.
It retains the existing angle-dependent dielectric Fresnel coupling, GGX
specular term, approximate area-light sampling, ambient illumination, shadows
and display tone mapping. The white-kernel integral does not prove energy
conservation for all of those combined. It is also not a full OpenPBR material
or measured-pigment scattering model.

### Authored displayed thickness

`surface.height_scale` accepts `[0, 100]`, with its existing default of one.
This explicitly scales the displayed interpretation of the saved paint height.
It does not add simulation pigment, change the archived height array, or imply
that a thicker paint flow was simulated. The actual scaled geometry must still
remain below 15% of the visible canvas width and inside the guarded camera
volume; a numerically legal scale is not permission to violate those limits.

Larger scales are a separate appearance experiment, recorded alongside the
camera and lighting settings. A raised-paint comparison must use the same
authored thickness scale for both its control and its treatment. Packing
conserves the native-grid displayed volume after this explicit height
interpretation, rather than claiming equality with unscaled geometry.

### Optional packing reconstruction

`surface.interaction.packing_strength` in `[0, 1]` enables an authored displayed
height reconstruction from the frozen aggregate field. It does not change
the archived simulation height or feed back into pigment motion. This is a
material appearance experiment, not a claim that a fluid solver moved a
separately simulated binder phase.

For each pixel's existing displayed paint height `H`, the reconstruction holds
`0.88 H` fixed and assigns `b0 = 0.12 H` to a redistributable binder-height
component. Its affinity is `b / b0 - packing_strength × aggregate`, with local
capacities `(1 ± packing_strength) b0`. Pairwise transfers redistribute that
component with opposite signed contributions. The reconstructed height change
sums to zero up to numerical precision on the native grid. This preserves the
sum of displayed paint thickness over native pixels, not a proven continuous
volume integral after interpolation. It changes local relief and normals,
while pigment concentrations and their mass contour remain the source data.
The cached texture stores relative change `delta_H / H`; rendering multiplies
the legacy surface height by `1 + delta_H / H`. This retains the positive
skeleton and the exact quiet appearance. Camera frames sample that same frozen
reconstruction; camera movement does not evolve it.

`packing_length_um` sets the reconstruction scale in `[80, 2400]` micrometres.
If either packing key is supplied, the omitted partner resolves to
`packing_strength=0` or `packing_length_um=1200`. Supplying neither leaves old
normalized controls unchanged. A length by itself does not enable packing.
The native-grid plan uses four decreasing physical scales and 28 bounded
pair-exchange passes. It permits at most a 16-cell coupling radius and only
connects endpoints along fully occupied paint paths. Unsupported combinations
of physical canvas width, material resolution and packing length fail recipe
validation before simulation. This is a finite equilibrium approximation;
neither exact resolution independence nor fully converged equilibrium is claimed.

Nonzero packing requires **`simulation.substrate_um=0`**. CPU snapshots contain
combined paint/support height, so reconstructing them with an added substrate
would incorrectly redistribute support as paint. The runner rejects that
combination, and the native GPU path also checks its support-height contract.
Zero packing retains the existing height exactly. For a complete baseline
control, set all three strengths to zero.

## Enabling a study

Start with a source-free laminate recipe, such as a color-and-form study. Add
these optional sections to its existing simulation and surface objects:

```json
{
  "simulation": {
    "interaction": {
      "version": "contact-microstructure-v1",
      "contact_rate": 5.0,
      "origin_distance": 0.025,
      "composition_threshold": 0.03,
      "minimum_concentration": 0.00001,
      "fabric_rate": 3.0,
      "fabric_relaxation": 0.12,
      "aggregation_rate": 2.0,
      "breakup_rate": 0.25,
      "nucleation_scale": 0.008,
      "nucleation_contrast": 0.7
    }
  },
  "surface": {
    "interaction": {"silk_strength": 0.65, "grain_strength": 0.45}
  }
}
```

This fragment is not a complete recipe. `simulation.interaction: {}` resolves
the versioned defaults. Omission or `null` does not enable the simulation
extension, and does not add keys to earlier normalized simulation settings.
Unknown controls, nonfinite values, and unsupported versions fail validation.
To choose the optional transport method, add `"advection": "maccormack"` to the
interaction object above; all other controls remain explicit and reproducible.

Interaction films require `render.capture_resolution` to equal
`simulation.resolution`. Both the native GPU path and the native CPU reference
path are supported. An omitted capture size defaults to the native size for
these studies. Reduced snapshots may still serve pigment participation
diagnostics; area-averaged intensive history is not used to qualify the film.

For the native GPU film path, each new material capture keeps material arrays
on the GPU and reads an 8-byte validation/height summary. Active packing adds
a separate 4-byte height-bound summary, for **12 summary bytes total**, plus
the final image. This does not remove the full native snapshot needed for the
final material archive and reference still. Frozen camera frames reuse prepared
material and packing state. Packing also uses **12 additional GPU bytes per
native pixel** for its staging fields; that allocation is separate from the
small CPU summary and from the renderer's other textures.

### Prototype profiles

`interaction_studies.py` reconstructs each selected seed/count recipe from the
accepted Color & Form release and verifies its recipe hash before adding the
interaction extension. Its current profiles are experiments, not a ranking of
artistic quality:

| Profile | Transport | Intent |
| --- | --- | --- |
| `gentle` | Linear | Default contact, alignment and aggregation rates |
| `worked` | Linear | Faster alignment and aggregation, stronger nucleation contrast |
| `mineral` | Linear | Higher contact and aggregation rates, less breakup |
| `woven` | MacCormack | The worked rates with less diffusive history transport |
| `mineral-sharp` | MacCormack | The mineral rates with less diffusive history transport |
| `encounter` | MacCormack | Larger origin/composition contact thresholds and faster aggregation |

Every profile requests `silk-grain`, `silk`, and `control` from one physical
history. The configured appearance strengths are one; the named controls
select which responses are enabled. Stationary substrate grain remains off.
The plan records the source trajectory hash, exact resolved recipe, code
identity and accepted release hash. Native 2048-wide cases can compare the
base-material hash directly with the saved release. Other resolutions require
separate qualification rather than claiming the same numerical state.

## Persistent state and identity

Every enabled `final.npz` contains the original ten arrays plus this atomic
extension, all at the native grid and with `float32` storage:

| Field | Shape | Meaning |
| --- | --- | --- |
| `origin_upper` | H × W × 2 | Transported upper-layer world-coordinate origin |
| `origin_lower` | H × W × 2 | Transported lower-layer world-coordinate origin |
| `interaction_upper` | H × W × 4 | Contact, axial fabric x/y, aggregate fraction |
| `interaction_lower` | H × W × 4 | Contact, axial fabric x/y, aggregate fraction |

Coordinates are finite. Contact and aggregate fraction remain in `[0, 1]`;
fabric norm cannot exceed contact dose except for the validator's floating-point
tolerance. The GPU packs origins into RGBA32F textures with unused z/w channels;
the archive contains only meaningful x/y coordinates. Snapshots and archive
validation do not replace the raw state with normalized display data.

The request and receipt bind the extension's version, normalized seed,
simulation controls, initialization contract, field schema and native dimensions.
The capture descriptor identifies native CPU snapshots or borrowed GPU textures
from the same canonical step. The regular frame ledger binds the full formation
schedule, hold and camera orbit. Formation advances material; hold and camera
frames reuse the frozen final state.

Two hashes answer different questions:

- `physical_state_sha256` covers **all fourteen fields** and is shared by every
  optical view of the painting.
- `base_material_sha256` covers only the **original ten fields**. Matching it to
  an earlier baseline establishes unchanged pigment, wetness, geometry and
  original finish inputs. It does **not** mean the complete microstructured
  material is unchanged.

The geometry in that base hash is the archived simulation height. Optional
packing is an explicitly configured display reconstruction; the same base hash
does not imply identical displayed geometry when packing strength differs.

Field names, shape, dtype and raw values all contribute to these hashes. File
hashes separately bind the actual NPZ, images, films, recipe and archived code.
Disabled archives contain only the original fields and omit interaction
metadata and the extra base hash. Earlier archives remain verifiable.

Portable galleries expose `interaction_version` and `base_material_sha256` only
for enabled material. These values remain bound to the copied request and
receipt, including the material seed/settings and each optical view's controls.
Publication verifies the full source archive before copying its media. Portable
verification retains these provenance associations without needing the original
experiment directory or pretending to recover raw material from the images.

Verification checks identities and contracts; it does not replay the complete
simulation or independently re-render a film. A verified receipt is evidence of
internal consistency, not a claim that every pixel has been independently
recomputed.

## Validation and visual review

### Published comparison

The [contact-textures-v1 release record](releases/contact-textures-v1.json) pins
ten seeds with three starting pigments, using the `encounter` history profile
and `impasto-detailed` presentation. Each has a contact finish and a matching
control: twenty 2048 × 1536 paintings and twenty complete 1440 × 1080 films.
Each film contains 937 frames at 24 fps: formation, a hold, then camera movement.
All ten original-material hashes match the preserved Color & Form release.

The separate 8088 detail study uses a 4096 × 3072 material grid and produces a
3840 × 2880 painting with a 1440 × 1080 film. Its full fourteen-field state and
final PNG match its independently rendered still exactly. Its finer material
grid is a different simulation resolution; it is not claimed to have the same
base-material hash as the accepted 2048 study.

The clearest visible improvement is the thicker interpretation of paint. Fine
contact texture remains subtle, including under the more pronounced aggregate
response. Numerical correctness and comparison integrity do not establish
artistic quality. The saved Color & Form collection is retained for comparison.

Both CPU and GPU suites ran 466 tests without failures, with 79 and seven skips
respectively. The gallery suite passed after its label refinement. Browser
review confirmed all ten thumbnails, contact/control playback, paired full-size
images, and byte-range seeking. The release record includes actual runtime,
source, recipe, material and media hashes; Git references are recorded separately
and are not substituted for the archived runtime identity.

### Experiment record

The study history includes these matched experiments. They retain the accepted
base-material hashes; additional interaction state has its own complete hash.

| Batch | Scope | What the comparison established |
| --- | --- | --- |
| Initial linear-history proofs | 12 physical cases, three optical views each | Contact history can be added without changing the accepted pigment/height baseline; the roughness-only appearance was very restrained |
| EON lighting studies | Eight preserved histories, six views each | Angular redistribution produces a stronger lighting response, but broad shading differences alone do not establish visible mineral texture |
| Sharper-history proofs | Six physical cases | Limited history transport provides another way to retain encounter detail while preserving the accepted base material |
| Packing studies | Four preserved histories, ten views each | Matched optical controls isolate packing from existing satin/EON responses; the tested packing response remained very small at the original thickness scales |

Each appearance comparison reuses its parent's complete fourteen-field state;
changing the history transport is a separate physical case. The experiments
are preserved for review rather than presented as an artistic ranking. Stronger
numerical response does not by itself justify a selected look, and no preferred
final treatment is implied by these prototype names.

### Checks

The focused archive suite is:

```sh
python -W error -m unittest tools.estuary_confluence.test_interaction_archive
```

It checks raw native array persistence, full versus base hashes, missing and
malformed fields, rehashed metadata substitution, wrong native dimensions,
disabled-feature compatibility, complete-source cadence independence and
unchanged material during camera frames. It also checks that the three named
optical variants share one engine and the same material at every frame while
retaining their declared strength settings. Numerical and GPU tests separately
cover contact, kinetics, transport, initialization and CPU/GPU optical agreement.
Transport tests compare translating detail with an analytic reference, check
constant preservation with variable mass and empty neighbors, enforce the
coupled state bounds, and verify GPU/reference agreement. More retained
numerical detail is useful evidence about transport; it does not by itself
establish that a painting is more successful.

Review optical controls on the same material before comparing different
simulations: original appearance, silk only, grain only, and both together.
Inspect full compositions and native-resolution crops, then the formation and
camera phases of the film. Quiet paint should remain quiet, contact regions
should carry the detail, and monochrome collisions should remain eligible. A
larger strength is not automatically a better result.
