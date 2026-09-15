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
| Tidal Calligraphy | Final refined filament recipe selected; full 4K/60 film rendering in four verified ranges |
| Gravity Loom | Pending completion of Calligraphy |
| Aurora Veils | Pending |
| Light Cast by Gravity | Pending |
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
