# Six art studies

## Objective and order

Create six distinct, exceptionally beautiful interpretations of the complete
recorded three-body motion of seed `0xb7f327f9f722`, one study at a time:

1. Tidal Calligraphy — translucent trajectory bands and fine fibers.
2. Gravity Loom — open structures woven from pairwise relationships.
3. Aurora Veils — broad, transparent curtains animated by the motion.
4. Light Cast by Gravity — moving optical forms expressed as projected light.
5. Orbital Engraving — detailed line fields with broad, changing interference.
6. Eclipse Garden — moving dark forms shaping luminous negative space.

The user explicitly prioritizes beauty over compute cost and elapsed time.
Rendering and significant builds run on `user@100.76.88.48`; artwork generation
and rendering use Rust on CPUs. All six use the same frozen orbit. These are
artistic mappings of the physical source, not claims that gravity literally
produces the depicted materials, light waves or optical installations.

## Output contract

- Target: 3840×2160, 1802 frames at 60 fps, approximately 30.033 seconds.
- First and last frames represent source fractions 0 and 1.
- Canonical RGB16 PNG frames remain on the experiment server.
- Each study delivers browser H.264 and a high-quality 10-bit HEVC movie.
- Each archives its recipe, executable/source hashes and per-frame receipts.
- Finished films are checked for complete decoding, correct timing and visual
  continuity; generated geometry and sampling are independent of worker count.
- A local review gallery will provide all films, synchronized comparison,
  full-screen viewing, close-encounter navigation and notes.

## Artistic standards

Require sweeping silhouettes and deliberate open space. Use several scales of
detail: a few dominant gestures, supporting curves, then fine material detail.
Judge motion during calm passages, approaches, release and the final interval.
Keep strong highlights selective and preserve a visible distinction between
transparent overlapping layers. Inspect high-resolution images and actual film
sections; successful numerical checks alone do not establish artistic success.

Detail is included from the start. Experiments select composition and materials;
they are not substitutes for the eventual complete high-resolution film.

## Working state

Branch: `codex/six-art-studies`, based on `e62cba5` in the `CS-Tidal-Silk`
worktree. Earlier cloth work remains available on `codex/tidal-silk`.

Server source: `/home/user/tidal-silk/source`.
Frozen orbit: `/home/user/tidal-silk/results/0xb7f327f9f722.orbit`.
Series output: `/home/user/tidal-silk/six-studies-b7`.
Local delivery: `CS-Image-Generation/output/six-art-studies-b7`.

| Study | State |
|---|---|
| Tidal Calligraphy | Complete: both 4K/60 films verified and copied to the local collection |
| Gravity Loom | Selected v08 spindle; full 4K/60 film rendering, motion review pending |
| Aurora Veils | Selected v11; complete 4K/60 film rendering, full motion review pending |
| Light Cast by Gravity | Implementing spectral phase lenses and conservative flux rendering |
| Orbital Engraving | Pending |
| Eclipse Garden | Pending |

## Implementation notes

The new `orbital_atelier` command is independent of the previous cloth pipeline.
It reuses the portable orbit cache. Stable frames carried along each source
curve prevent ribbon orientation flips. Open trajectory geometry replaces the
large freely gathered disk. A deterministic layered CPU renderer evaluates
front/reverse dyes, absorption, satin highlights and fine strands in linear light.

Update this document with selected recipes, observations, verification results,
completed films and the next outstanding study as work progresses.

### Calligraphy development log

- Shared source preprocessing, surface/fiber geometry, layered optics and the new
  CLI compile on the server. Initial focused coverage: 13 atelier tests pass.
- The `atelier-v01` binary is frozen in `/home/user/tidal-silk/bin/`.
- Eight actual 4K stills were rendered at source midpoint. Initial narrow
  principal-axis views read as pointed leaves or flat vector strokes. Wider
  views and longer histories give substantially more open overlapping loops.
- Selected exploration for further refinement: `calligraphy-v01-wide-airy.json`.
  It is a development recipe, not a finished selection. The provisional full
  motion render is `01-calligraphy/v01-motion`, logged in
  `logs/calligraphy-v01-motion.log`; it intentionally precedes the improvements
  below so the whole motion can be examined while they are implemented.
- Current problems: hard width limits create angular shoulders; fibers are too
  uniformly lit; A's shorter path makes it a small gesture beside B's long one.
- In progress: smooth width profiles, varied fine fibers, a slow roll anchored to
  source arc length, position-dependent strip lights and filtered microstructure.
- Final film pipeline will average multiple sub-frame samples in linear light
  before optical bloom and tone mapping. The default shutter spans half a frame.
- True prehistory is being reconstructed from the recorded initial-state bits
  and original warm-up integration. It must match the cached first visible state
  exactly. This lets history-based forms exist at the opening without fabricating
  source motion or changing the visible interval. Original source normalization,
  visible frame orientation and arc-length phase are to be preserved.
- A shared world-distance history budget is being added as an optional composition
  control. Body A travels about 16.284 normalized units over the source, B 32.324,
  and C 27.594; equal time windows do not make equal-length strokes.
- Fixed camera candidates and full-source extent checks are archived in
  `configs/cameras.json`; `camera-wide-sweep.json` is the current leading view.
- The standalone review gallery and local byte-range video server are implemented
  in `tools/atelier/`. Python lint passes; byte-range response/invalid-range cases
  and gallery JavaScript parsing were checked. Real playback UI QA remains.

### Selected Calligraphy film (v03)

- Recipe: `tools/atelier/recipes/01-calligraphy-b7.json`, mirrored as server
  `configs/01-calligraphy.json`. Renderer: immutable `bin/atelier-v03`.
- 3840×2160, 1802 frames at 60 fps; 3×3 spatial coverage and four sub-frame
  exposures integrated in linear light. Shutter spans half a frame.
- Geometry: up to 2048×64 surface intervals per stroke, 144 interior fibers and
  24 fringe fibers per side, with 768 intervals per fiber. Fine variation is
  stable in source arc length.
- A six-world-unit history budget balances the bodies' traveled distances;
  genuine half-interval prehistory is replayed and verified. Visible source
  positions, frames, normalization and arc phase are unchanged.
- The wider `filament` camera was preferred after independent reviews across
  seven source checkpoints. The depth view repeatedly merged pale strokes near
  the ending. Opening history is allowed to continue beyond the frame as a
  deliberate incoming stroke; the current body positions remain in view.
- Final refinement reduces fringe spread from 0.85 to 0.50 and the first body's
  width multiplier from 1.0 to 0.90, leaving more open space around crossings.
- Native 4K optical inspection confirmed distinct gold/blue/violet, visible
  overlaps, localized highlights and continuous fine fibers. No blocking
  optical defect was found. Encoded temporal stability still needs inspection.
- Four final ranges started 2026-09-15 06:14 UTC with 28 workers each:
  `01-calligraphy/final-chunk-0` (0..451), `final-chunk-1` (451..901),
  `final-chunk-2` (901..1352), `final-chunk-3` (1352..1802). End indices exclusive.
  Logs: `logs/calligraphy-final-{0,1,2,3}.log`; processes: `final-jobs.json`.
- A verified assembly command is being added. It must validate chunk identity,
  every PNG/receipt, and the complete frame union before publishing a full
  manifest. Pixels are unchanged; chunk originals remain intact.
- All-target Clippy passes. Combined v02 passed 22 atelier tests; the subsequent
  hue correction adds one renderer regression test (11 renderer tests pass).
- The private review-page fixture played the full development movie to 30.03s
  and its encounter/full-screen controls worked in the in-app browser. The
  final collection still needs two-distinct-film sync and notes/export checks.

### Packaging and review checks

- `atelier-v04` adds verified assembly; it leaves renderer `atelier-v03` intact.
  Six assembly test groups, all-target Clippy and formatting pass. Real v03
  chunks were assembled and encoded with v04, including a cross-filesystem copy
  test. Source PNGs, source receipts and original renderer identity were retained.
- The selected 4K frame 450 has identical PNG bytes with 36 and 112 workers:
  `bf30abd6f7829606f70835d16e569a92453754d584bac732fb960b506eb5caee`.
- `tools/atelier/finish_study.py` waits for complete ranges, assembles, encodes
  browser/10-bit films, decodes every frame through `verify_film.py`, and publishes
  recipe, verification and a canonical 4K poster. It is waiting on Calligraphy
  with `--executable /home/user/tidal-silk/bin/atelier-v04`.
  Finalizer log: `logs/calligraphy-finish.log`.
- Gallery playback, replay/end, mouse/keyboard seeking, quarter speed,
  fullscreen, distinct-film sync, notes/ratings persistence, timestamps and
  export passed Cua-only browser checks. Missing-master fallback preserves time.
- An optional original normal movie appears only in the right comparison
  selector. It does not count as a seventh study or enter study notes/exports.
- Production gallery readiness requires complete verification, correct study
  kind and full endpoints, matching web/master hashes and byte counts, and a
  poster. `--include-development` enables explicitly labeled private previews.
- Encounter seeking is corrected for 1802@60: source fraction
  `0.9274619274619275 × (1801/60) = 27.839315522648857` seconds, nearest frame 1670.
- Actual 10-bit-master playback remains to be checked when encoding completes.

### Calligraphy complete; Loom begins

Calligraphy completed on 2026-09-15 at approximately 07:26 UTC. Its four ranges
were assembled with v04; all 1802 frames in both 3840×2160 / 60 fps films decode
without errors. Both durations are 30.033333 seconds. The HEVC master is 10-bit.

Local result: `CS-Image-Generation/output/six-art-studies-b7/01-calligraphy/`.
Server result: `/home/user/tidal-silk/six-studies-b7/01-calligraphy/`.
Master SHA256: `68e9f2edea0568707a035427dc3c536de4bee1554ca78d2f7f85c1d2b52f43b3`.
The first implementation checkpoint is commit `87eeec5`.

Loom implementation starts only after that completed film. Its design is in
`docs/gravity-loom-design.md`; it has a stable S-curved chronological axis, a
positive teardrop radius, staggered apertures and actual over/under fiber lifts.
The next renderer adds optional metallic reflection and interpolated strand
tangents. Zero metallic remains omitted from serialized old materials, and
inactive study parameter blocks are omitted; a real archived v03 manifest/hash
fixture guards compatibility with Calligraphy artifacts.

### Loom look development

- v05 builds and passes 34 atelier tests, seven CLI tests and all-target Clippy.
  The v03 manifest hash remains unchanged under the extended schema.
- New optics add metallic reflection and continuous endpoint-interpolated fiber
  tangents, including closed braided loops. Zero metallic retains the original
  dielectric arithmetic and is omitted from serialized legacy materials.
- First actual 4K panel/conch/spindle proofs are in server `02-loom/*-v05-proof`;
  local copies are `output/atelier/loom-v05/`. They were rejected artistically:
  flat panel interiors, straight-cut ends and coarse regular cells read as an
  industrial wire-mesh tube. Numerical success was not treated as visual success.
- v06 revises the geometry to polar, rounded panel interpolation with matched
  seam normals, a fixed 180-degree twist along chronology, narrow ends and a
  fuller asymmetric belly. The weave now has 128×256 groups, two fine fibers per
  group and 3072/512 sampling intervals. Eight Loom tests cover the revised
  surface and its derivatives/crossings.
- v06 build/test is running; next step is new finished-light 4K proofs and further
  material/camera curation. No full Loom film has been started.
- The production review gallery is running locally on port 8767 (PID 9708,
  exec session 39971), with one verified completed film and an optional original
  reference. Actual 4K/60 HEVC master playback and mixed HEVC/H.264 synchronized
  comparison were verified. QA windows were closed and notes left untouched.

### Loom v08 selection

- Rounded polar panels, narrow spindle ends, and smooth terminal feathering
  replace the cut-tube appearance. The fixed 180-degree twist belongs to the
  material's history coordinate; the object does not spin with animation time.
- The original YZ source projection hid much of this seed's movement. A fixed
  plane fitted from the full source captures approximately 92% of sampled
  positional variance, versus 50% for YZ. Its two orthonormal axes are recorded
  in the recipe. No moving fit or change to physical coordinates is performed.
- Four actual v08 4K lighting proofs are in `02-loom/v08-*`. Current preferred
  candidate: `configs/loom-v08-luminous.json`, with gold, darker bronze and
  champagne; source influence 0.4, soft radius 1.0, pair bow response 0.35.
  It retains 128×256 yarn groups, 3072/512 curve intervals, 3×3 spatial samples,
  four temporal samples, and the original full-source timing.
- Immutable renderer: `bin/atelier-v08`. All 38 atelier tests and seven CLI
  tests pass. Native all-target Clippy passes after four syntax-only cleanups
  in the existing x86 spectral implementation. Three spectral approximation
  accuracy failures reproduce identically in baseline `e62cba5`. Independent
  native probes confirm bitwise equality of old/new constants, loads, 100,000
  exponential inputs and 1,000 spectra. Evidence: `qa-simd-e62cba5/results.log`
  on the server. This renderer does not call that spectral conversion path.
- v06 frame 900 is byte-identical with 8 and 112 workers and matches the original
  proof. Times were 57.99 and 9.91 seconds respectively; the smaller worker
  allocation used less total CPU per frame. Evidence is archived in server
  `logs/benchmark-v06-summary.json`.
- `tools/atelier/render_study.py` now runs one selected recipe in independent,
  resumable full-quality ranges and then invokes assembly, both encoders and
  complete-video verification. No spatial, temporal or material detail is
  reduced for parallel rendering.
- Selected recipe: `tools/atelier/recipes/02-loom-b7.json`, mirrored as server
  `configs/02-loom.json`. Full render started around 08:23 UTC with 16 ranges
  and seven workers each (112 total). Driver PID 97267; job record
  `02-loom/final-job.json`; progress log `logs/loom-final-v08.log`. The driver
  will finish both movies and verify every encoded frame. Source RGB16 images
  remain archived. Original v08 midpoint is the selected poster frame.
- After observing concurrent throughput, preserved ranges resumed with four
  active ranges of 28 workers; the 16 range boundaries and image settings are
  unchanged. Driver PID 98271 replaced 97267. Quarter-interval ranges run first
  so motion can be reviewed in several parts of the orbit before the full film
  finishes. Already verified frames are reused.
- Implementation proceeds to Aurora after Loom's selected design is frozen;
  Loom rendering and subsequent motion review continue independently through
  its immutable executable. The collection is not finished until every film
  has completed rendering, visual review, encoding and full verification.

### Aurora first implementation (v09)

- `src/atelier/aurora.rs` builds three open luminous sheets, each with
  1024×192 surface intervals and 3072 fine vertical rays of 160 intervals.
  Fixed principal source axes drive large lateral and lower-edge movement;
  positive heights, monotone chronology and separated depth lanes prevent
  crumpled or intersecting sheets. Camera remains fixed.
- A continuous optional emission profile supplies jade/cyan/violet height
  colors and smooth material density. Both glow and absorption disappear at
  feathered boundaries; ray ends taper geometrically. Curve coefficients are
  prepared once per material, with bounded smooth interpolation.
- Fine ray radius/length patterns are stable in material indices. The first
  source-arc fine pattern would have reached about 153 Hz near the fastest
  visible passage and was removed before rendering. Broad source-driven folds
  and light gatherings remain below about 1.35 Hz in the visible recording.
- The combined v09 binary passes 49 atelier tests, eight CLI tests and native
  all-target Clippy. Both archived Calligraphy and Loom recipe hashes remain
  unchanged. A nonblank Calligraphy fixture also has exactly identical RGB16
  pixels and PNG bytes between immutable v08 and v09. Regression evidence:
  `logs/qa-profile-none-v08-v09.json` on the server.
- Four first full-detail 4K proofs at frame 900: `03-aurora/v09-jade`,
  `v09-luminous`, `v09-mist`, and `v09-moon`, with matching recipes in `configs/`.
  Immutable renderer: `bin/atelier-v09`. No Aurora full film has started.
- Inspect the actual proofs next. Watch for overly flat filled shapes, uniform
  neon brightness, hard boundaries, or subtle ray/sheet depth-order flicker;
  a small forward ray offset is a possible refinement only if needed.

### Aurora refinement and current selection

- First v09 images were rejected as too much like a graphic equalizer: ruler-
  straight rays, broad bright green plateaus, and sharp lower hems. v10 adds a
  globally invertible world-height shear, curving the rays while retaining
  regular surfaces and separated layers. Chronology is monotone in the
  unsheared coordinate. The selected lean amplitude is 1.1 world units.
- Preferred v10 recipe: `configs/aurora-v10-dense.json`. It has 6144 fine rays
  per body (18432 total), radius 0.00014, quieter sheet emission 0.012,
  sheet/ray optical depths 0.012/0.13, lower/top feather 0.14/0.67, and a
  restrained opal/cyan/violet palette. Studio lighting is zero; emitted light
  supplies the image. 3×3 spatial and four temporal samples remain unchanged.
- Native 4K checkpoints at frames 0, 180, 450, 900, 1350, 1670 and 1801 show
  large source-driven shifts and changing open spaces with no observed crop.
  Local copies: `output/atelier/aurora-v10/`. The densest treatment was preferred
  independently over the more visibly striped and greener alternatives.
- v11 replaces the remaining periodic fine modulation with smooth seeded
  aperiodic value noise in fixed material indices. The existing
  `ray_detail_period` setting controls its characteristic scale; selected 13
  gives approximately 6.5, 13 and 24.7-ray scales. Temporal identity, bounded
  radii/lengths, source positions, camera, palette and broad motion are unchanged.
- v11 passes 52 atelier tests, eight CLI tests, native all-target Clippy and
  formatting. Immutable executable: `bin/atelier-v11`. Final texture proofs at
  900, 0 and 1670 are rendering under `03-aurora/v11-proof-*` using
  `configs/aurora-v11-selected.json`; no full Aurora film has started yet.

### Current motion review

- A verified Loom development excerpt contains 108 actual final-quality frames:
  0–53 followed by 901–954, native 3840×2160/60, duration 1.8 seconds. It has
  explicit excerpt labels and source-frame/hash provenance. Local copy:
  `output/atelier/loom-v08/motion-check.mp4`.
- The in-app browser's direct native-video speed menu crashed that temporary
  tab. A fresh tab using a simple local review page played to the end at quarter
  speed and reported native 3840×2160 media dimensions. Private page:
  `http://127.0.0.1:8768/loom-motion-check.html`. This verifies sample playback;
  longer full-source motion and shimmer review remain required when available.
- Full Loom rendering continues through driver PID 98271 and immutable v08.
  The completed collection still contains only Calligraphy. Later studies are
  not marked complete while rendering or look development is unfinished.
- Fourth-study design is prepared in `docs/light-cast-design.md`; implementation
  has not started. Use 12 wavelength groups from the existing spectral/CIE data
  for its initial high-detail optical render, rather than the proposal's original
  three-band starting point. Preserve energy and deterministic accumulation.

### Aurora film started; Light Cast implementation

- v11 final texture proofs at frames 0, 900 and 1670 were inspected and selected.
  Irregular fine rays remove the residual repeating comb while preserving the
  quieter opal/cyan/violet treatment. Full-source v10 composition checkpoints
  remain applicable because this final revision changes only fixed fine detail.
- Selected recipe: `tools/atelier/recipes/03-aurora-b7.json`, mirrored as
  `configs/03-aurora.json`. Immutable renderer `bin/atelier-v11` is archived with
  source commit `8276089` and `bin/atelier-v11-source.tar.gz` / build metadata.
- Full Aurora render started around 09:16 UTC, driver PID 106870, with 16 ranges,
  four active ranges and 32 workers total. Progress: `logs/aurora-final-v11.log`;
  job record: `03-aurora/final-job.json`. It will assemble and finish both movies
  after every range completes. Increase its worker allocation after Loom ends
  if useful; preserve the existing boundaries, binary and requested recipe.
- Loom renderer v08 is likewise archived with source commit `1a3097b`, a source
  tarball and build metadata. Its driver remains PID 98271, four active ranges
  of 28 workers. Around 09:23 it had 525 frames and four complete ranges; Aurora
  had 12 frames. These are progress counts, not completion claims.
- Light Cast implementation begins only after Aurora's implementation and
  selected recipe are frozen. Geometry/config/spectral grouping lives in
  `src/atelier/light.rs`; conservative CPU accumulation in `light/accumulate.rs`.
  Root integrates the new direct linear-frame path with existing encoding.
- Planned public API: `light::render_linear(source,time,config,camera,render)`
  returns `LightFrame { pixels, diagnostics }`. Frontal camera target, size and
  roll control the receiving plane; unsupported tilt is rejected. Config display
  is `LightDisplay::Ivory` or `DarkGain`. Twelve contiguous groups of the existing
  64-bin CIE XYZ weights preserve the existing observer normalization.
- Initial Light quality uses 2304×1536 source cells, 12 wavelength groups, 3×3
  receiving-plane sampling and **16** shutter samples. A conservative bound on
  the fastest lens center is about 31 final pixels per frame; four shutter
  samples could separate a caustic narrower than one pixel. No lower-detail
  final film is planned.
- Light receipts will preserve diagnostics for every exact shutter sample;
  assembly and encoding will reject missing/truncated, non-finite or mistimed
  optical records. Legacy receipt and recipe compatibility remains tested.

### September 15 continuation — current delivery and render state

This section supersedes the earlier progress snapshots above.

- Calligraphy and Gravity Loom are complete locally in the six-art-studies-b7
  collection. Both full 3840×2160, 1802-frame, 60 fps films passed complete decode
  verification. Loom's master SHA256 is
  `eff41917c85588ab9bfbd566ce264c6deeec49a34889926860de070a7cc0911e`.
  Full Loom playback, source seeking and synchronized comparison were checked;
  dense weave interference remains visible when reduced to a small display.
- The loopback gallery at `http://127.0.0.1:8767/` was restarted as local PID
  21254 after the older server lost its output pipe. Detached collection helper
  PID 25231 copies only verified complete packages, rebuilds the gallery as each
  arrives, and exits when all six are complete. Logs are `.review-server.log`
  and `.finish-collection.log` in the delivery root. The gallery currently has
  two complete films; unfinished studies are not presented as complete.
- Aurora resumed with immutable v11 and the identical frozen recipe and ranges,
  with 112 workers across four active ranges. Server driver PID 109919;
  `logs/aurora-final-v11.log`. At approximately 16:54 UTC it had 1288 PNGs;
  the count of render.json files includes active range manifests and is not a
  completed-range count. Automatic assembly, encoding and verification follow.
- Light Cast by Gravity is frozen as immutable v15, source commit `7b2ef6a`.
  The selected bright curved phase-lens recipe is now
  `tools/atelier/recipes/04-light-b7.json`, mirrored as `configs/04-light.json`.
  It uses 3072×2048 source cells, 12 wavelength groups, 3×3 receiver sampling,
  16 half-frame shutter samples, dark-gain exponent 2, scale .008, exposure .8
  stops and bloom .035. Four-kilopixel center/end proofs were inspected after
  rejecting the dimmer .0012 gain-scale treatment. Full film driver PID 126535
  uses 32 workers in four active ranges and immutable v15. Log:
  `logs/light-final-v15.log`. Full motion and convergence review remain pending.
- Engraving v16 passes 99 atelier and 13 CLI tests plus native all-target Clippy.
  It filters the complete combined engraving phase over space and exposure;
  receipts record every exact exposure cell and its adaptive residuals. Source
  bounds conservatively enclose the existing continuous Hermite trajectory
  without changing its evaluation. Native 4K proofs at 900 and 1670 were
  technically sound but too reminiscent of three broken vinyl records.
- Engraving refinement adds fixed radial twist to the lobe and open-mouth fields,
  with full analytic gradients and a strictly positive radial derivative bound.
  Proposed twists .8/-.65/1.0 and lobe amplitudes .10/.035 retain ordered cuts
  while bending the openings. Zero twist and original lobes are omitted from
  serialization to preserve old hashes; an actual v16 manifest is now archived
  as a compatibility fixture. New proofs must be reviewed before film selection.
- Eclipse Garden remains design-only in `docs/eclipse-garden-design.md` until
  Engraving's visual treatment is selected. All six films, motion review, and
  final collection delivery remain the task; it is not complete at two films.

### Engraving selection and sixth-study implementation

- Engraving v17 is frozen at source commit `5de6878`, with archived source,
  executable SHA and build metadata. Its 104 atelier tests, 14 CLI tests and
  native all-target Clippy passed. A nonzero-shutter legacy Engraving image has
  identical PNG bytes between v16 and v17; the low-resolution compatibility
  probe used depth eight to resolve its unusually coarse boundary pixels.
- The selected visual treatment is the finer `90/100/76` engraving with the
  curved mouths, exposure .70 stops and copper strength .65. Recipe:
  `tools/atelier/recipes/05-engraving-b7.json` / `configs/05-engraving.json`.
  Temporal refinement ceiling is six, retaining the original tolerances,
  sixteen exposure cells and 2×2 analytic spatial cells. The higher ceiling
  changes no result unless more integration is necessary.
- Native 4K first/last and quarter/three-quarter stills exist under
  `05-engraving/v17-luminous-*`; comparison proofs at 900 and 1670 use the same
  finer geometry with the slightly darker .35-stop exposure. All keep open
  centers and curved mouths; the full-source-independent crop bound remains
  positive even with the larger lobes. Full-quality opening 0–47 and encounter
  1646–1717 motion passages are rendering in four `v17-motion-*` ranges. The
  full Engraving film has not started before those motion checks.
- `examples/atelier_convergence.rs` compares actual linear renderer output in
  aligned native crops, with doubled spatial/temporal reference samples and
  optional doubled Light source grids. Its four semantic tests pass; native
  source tests and all-target Clippy passed after adding continuous velocity
  bounds. In Engraving's 192×192 fast-passage crop at (2120,800), the doubled
  reference differs by 0.0169% foreground-weighted relative luminance RMS;
  maximum absolute RGB difference is .000389. Both crops were inspected.
  Evidence: `05-engraving/qa-convergence-fast.json` and its PNG directory.
- Light's fast-cusp comparison doubles spatial resolution, shutter samples and
  both source-grid axes in a 256×256 crop at (1704,660), frame 1670. It is still
  running; do not treat that reference as a passed check yet.
- Eclipse implementation now lives in `atelier/eclipse.rs`, `eclipse/field.rs`
  and `eclipse/corona.rs`. It uses corrected closest-contour distances throughout
  the luminous bands, persistent base-pose arc-length corona anchors, finite
  Gaussian segments, symmetric opaque union, and joint light/mask integration.
  The main film path preserves legacy hashes and records every Eclipse exposure.
- The first combined Eclipse test run found a coarse synthetic-image refinement
  limit; 121 tests passed and one failed. Independent numerical review also
  found a center-vs-four-midpoint cancellation case for a Gaussian, so the
  quadrature is being strengthened before any Eclipse proof or full film.
  Neither Eclipse's implementation nor its artistic treatment is final yet.
- Live in-app gallery QA confirmed that entering Compare selects Calligraphy
  and Loom as distinct films. Loom's short description now matches its actual
  woven spindle treatment. The gallery still presents two finished films.


### Eclipse v18 compiled; first full-quality proof started

The strengthened joint quadrature resolves the reported Gaussian cancellation
cases. The complete current renderer passes 127 atelier tests, 15 CLI tests and
native all-target Clippy. V18 includes exact outside-distance optimization,
four-pixel curve bins with outward boundary inclusion, stable tanh-derivative
bounds and strict receipt-to-recipe integration checks. It preserves every
archived legacy recipe hash. Eclipse's first full 4K/128-exposure proof at frame
900 is running with 32 workers; local controller log is
`/tmp/eclipse-v18-pearl-900.log`. The source, shutter and all 1,536 hairs/body
remain at the planned detail from this first proof.

Whole-source Eclipse bounds: maximum 19.9971 pixels/frame, .078114 pixels between
128 exposure centers, and at least 60.699 pixels of conservative crop margin.
Worst interval1659 belongs to body C. The fixed recipe is
`configs/eclipse-v18-pearl.json`, with audit in `06-eclipse/v18-motion-audit.json`.
The first art proof is still pending; a compiled renderer is not a finished film.

Light's 8×-work reference comparison completed. Foreground-weighted linear
luminance RMS differs by2.457%, concentrated in the brightest HDR cusp; crop
integrated luminance is .906% higher in the reference. After the common tone
curve the PNG crops have SSIM .999904, and visual inspection showed the same
curves/fringes without a clear artistic improvement. Evidence remains in
`04-light/qa-convergence-fast.json` and its native/reference PNG directory.
This supports the tested crop; full-film motion review remains outstanding.

### Current continuation checkpoint — 2026-09-15 19:35 UTC

- Engraving is selected and its complete film is rendering. Immutable v17;
  `configs/05-engraving.json` / `tools/atelier/recipes/05-engraving-b7.json`.
  Detached server driver PID164853,48workers/four active ranges/16total.
  Log `logs/engraving-final-v17.log`, record `05-engraving/final-job.json`.
- Engraving opening frames0–47 and encounter1646–1717 are complete native
  4K/60 HEVC excerpts. All frames decode and exact source PTS were verified.
  Root reviewed opening/encounter/closest approach/release in the private IAB
  page, including quarter speed. File paths:
  `05-engraving/v17-motion-review/{opening,encounter}.mp4` with JSON provenance.
  Local copies are in delivery `.development/engraving/`. Broad interference
  is intentional; fine detail is less clear in a small display. Experimental
  1080p/720p linear-light proxies remain private; 720p loses too much detail.
- An additional curved-mouth Engraving crop at frame900,(1450,825),192×192
  agrees with doubled spatial/temporal reference to .0124% foreground-weighted
  luminance RMS, maximum linear channel difference .000366. Evidence:
  `05-engraving/qa-curved-mouth.json`. The earlier fast interior crop also passed.
- All six Rust CPU renderers are implemented. Full films remain complete only
  for Calligraphy and Loom. Aurora (PID109919) and Light (PID126535) continue;
  Aurora had1713 of1802 final frames at19:34. Count only final-chunk PNGs,
  excluding development proofs. The existing local collector still runs.
- Eclipse v18's initial pearl look was rejected after seeing its progressive
  image: three broad oval shapes with conspicuous straight lashes. V21's first
  thinner variants were too quiet. The chosen direction is now **Opal**, with
  slender asymmetric petals, thin cool pearl crescents, and sweeping fine fans.
  Recipe `configs/eclipse-v21-opal.json`, local selected copy
  `tools/atelier/recipes/06-eclipse-b7.json`. It remains a visual selection under
  validation; no full Eclipse film has started yet.
- Opal uses semi-axes(.44,1.70),(.68,1.30),(.36,1.62); shoulders .22/-.22/.20;
  light sigmas .0252/.0252/.02016 and reduced offsets; bends2.0/-1.6/1.8;
  1536hairs/body with12 long accents,64chords each; lengths .04–.20 with long
  .20–.32; radii .00035–.00070; corona fraction .25. Palette is cool pearl
  (.83,.92,1), muted rose(.68,.42,.52), copper(.48,.24,.12). AA3 and all128
  half-frame exposures stay unchanged. Fixed camera height7.4, zero bloom.
- Opal's continuous audit bounds motion at19.9957pixels/frame, .078109pixels
  per exposure step, and crop margin65.976pixels. Required count67, chosen128.
  Audit `06-eclipse/v21-opal-audit.json`. Progressive views at0,900,1659,1801
  were inspected; the merged contours reveal newly shaped luminous openings.
- Full 128-exposure Opal proofs continue at0,900,1659,1801 using immutablev21.
  Logs `logs/eclipse-opal-{0,1659,1801}.log` and
  `logs/eclipse-v21-opal-900.log`. Original v18, duplicatev19, warm Plume, and
  earlier Feather/Slender trials were deliberately stopped after art selection;
  their partial manifests/previews are retained, not finished deliverables.
- Renderer versions: v18=e6f4357; exact region caching v19=84bf5e1;
  labelled progressive CLI previews=375b8a9 (`atelier-v19-progress`);
  certified inside projections v20=090cb8e; broader permitted corona bends
  v21=d8fa800. Source archives and build metadata exist on the server for
  v18/v19/v20/v21. V21 passes30Eclipse tests and native all-target Clippy;
  previous CLI15 and full atelier127 checks remain recorded.
- Inside projection uses an outward-bounded rolling-disk certificate with
  explicit normal accuracy and the original fallback. It may improve rounding
  relative to the old search; tests require tight distance/normal/radiance
  agreement, not invented bit identity. Exact empty-region caching is separately
  checked bit for bit. An additional strict early-stop optimization for already
  converged stationary roots is under independent implementation/review; it is
  not built or selected yet.
- Progressive previews are `.progress-NNNNNN.png` plus explicit development
  sidecars. They are separate from canonical `frame_NNNNNN.png`, use a cloned
  accumulator, and cannot enter final assembly. They permit composition review
  while expensive complete exposures continue.
- The convergence helper now records its executable SHA. An update to publish
  native-pass PNGs early, separately from pending-reference status, is being
  built by the performance/QA agent. It will run Opal's shoulder crop at900,
  (1840,220),256×256,128native/256reference exposures and2×spatial resolution.
- Root's IAB is available (gallery tab1, private Engraving review tab2). The
  Aurora-review agent currently has only Brave and correctly avoided it;
  root must perform Aurora playback if that remains the case. The agent is
  watching Aurora until21:34 UTC and will notify when rendering ends.
