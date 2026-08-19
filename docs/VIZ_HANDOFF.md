# Viz Project Handoff

Operational state of the visualization subsystem effort, for resuming work in
a fresh session. The authoritative implementation spec and progress ledger is
[docs/VIZ_MASTER_PLAN.md](VIZ_MASTER_PLAN.md) — start there for *what* to
build; start here for *where things stand*.

Last updated: 2026-08-19 ~00:10 UTC, after the Wave 9 commit (`1524c98`).

> **Standing lesson:** treat any non-green draft smoke as stop-the-line
> for server batches. The smokes have caught real bugs every wave (Wave
> 6: a NaN camera tangent spinning V02's marcher forever; Wave 8: xfade
> timebase mismatches and empty trailer cold-opens; Wave 9: witness
> metering against the wrong energy scale and vanitas veins saturating
> without the high-pass) — always run `viz-smokeN`, read the per-mode
> "projected" cost lines, and *look at the frames* before letting a wave
> near the server.

---

## Immediate next actions (fresh session, start here)

1. **THE CATALOG IS COMPLETE: 69 of 69 modes.** Wave 9 (grand combos)
   landed as `1524c98`. There are no more implementation waves; what
   remains is curation, the caption/cartouche backfill, deep-polish
   passes on chosen favorites, and batch logistics.
2. **Two batches are in flight** (topology + rationale in "The server";
   progress timestamps below are from the *previous* session — re-probe
   with `--status` first):
   - **Primary** (`--status` with defaults): seeds `0x1357` + `0xFACE`
     finishing max-quality Wave-1 turntables (at 2026-08-18 ~20:30 UTC:
     333/720 and 232/720 frames; ETA ~Aug 19 19:00 UTC and ~Aug 20 14:00
     UTC). Expected `WARN ... failed` lines for three other seeds in
     `viz_batch.log` — killed deliberately. When `COMPLETE`: fetch, then
     relaunch these two seeds from post-Wave-9 HEAD (`1524c98`+) with
     every implemented mode **except `turntable`** (no exclusion syntax —
     build the 68-flag comma list from `--viz-list`, pass via
     `--viz-flags`). Budget: core re-render ~4-5 h + modes per seed; read
     the projected lines; V67/V65/V64/V68 are the new heavy items.
   - **viz-batch2** (`--remote-dir viz-batch2/CS-Image-Generation`):
     seeds `0xBEEF 0xC0DE 0xCAFE` running all 36 modes of Waves 1-5 from
     `77b5c2d` (ETA roughly Aug 24-29; turntable-dominated). Fetch needs
     the same `--remote-dir`. Afterwards they need a top-up run with the
     **33 Wave 6-9 flags** via `--viz-flags` (budget the core re-render
     ~4-5 h/seed on top).
3. **Wave-1 partials already fetched** (2026-08-18 ~07:15 UTC) to
   `../CS-viz-results-20260818/` — 5 seeds' core packages + 5 cheap
   Wave-1 modes each, ready for curation now. `viz-results/` is
   gitignored; default fetches no longer block deploys.
4. **Caption/cartouche backfill** is unblocked (`common/text.rs` exists
   since Wave 8) — schedule with the curation pass (list in Open items).

## Where we are

- **Branch:** `viz-master-plan`. Wave 6 `47cf1e8`, Wave 7 `68c422f`,
  Wave 8 `95d7f9f`, Wave 9 `1524c98` (not yet pushed); every wave is one
  `feat:` commit plus this handoff kept in sync. All gates green
  (fmt, clippy pedantic, 616 tests, ruff/mypy).
- **Progress: 69 of 69 modes implemented.** `--viz-list` prints the live
  catalog; the ledger in the master plan is kept in sync by a unit test.
  - **Wave 0** — framework: catalog, `--viz` CLI, `VizContext` (lazy
    kinematics/events/energy-field), `ArtifactSink` + `viz/manifest.json`,
    frame tap, two-phase runner.
  - **Wave 1** (8): braid, gw-chirp, syzygy-wheel, slit-scan, winding-glass,
    plotter-svg, oscilloscope, turntable.
  - **Wave 2** (7, SPD family): alien-vision, spectral-centroid,
    prism-portrait, spectrum-card, thin-film, dwell-nebula, topo-contours.
  - **Wave 3** (9, re-accumulation family): triangle-centers,
    medial-recursion, chrono-grid, strobe, comet, three-shadows, ride-along,
    retarded-time, depth-pack.
  - **Wave 4** (5, fields & frames): field-lines, corotating, lensing,
    roche, reconnection — plus `common/fields.rs`.
  - **Wave 5** (7, particles/agents/media): dust-nebula, galaxy-collision,
    physarum, frost, lightning, marbling, light-echoes — plus
    `common/particles.rs`, `common/agents.rs`, `common/fluid.rs`,
    `common/wave.rs`, energy-field retention.
  - **Wave 6** (8, 3D scene family): worldtube, hyperspectral-flythrough,
    shape-sphere, aurora, neon, chandelier, sculpture-export, bullet-time
    — plus `common/tube_render.rs` and STL/PLY writers.
  - **Wave 7** (5, ensembles & cartography): epilogue, multiverse,
    editorial-retime, basin-map, terra — plus `common/resim.rs`,
    `events::drama()`.
  - **Wave 8** (12, sound & cinema, incl. V11): sonification, recurrence,
    chord-progression, epicycles, mission-control, trailer, webgl-viewer,
    tilt, hologram, ephemeris-poster, blueprint, celestial-atlas — plus
    `common/text.rs`, `common/style.rs`, `common/compositor.rs`, audio
    synthesis + LUFS + mux, the frame archive, the V53 viewer template.
  - **Wave 9** (8, grand combos — the final wave): instrument, broadcast,
    powers-of-fate, witness, rose-window, reliquary, pond, vanitas — plus
    Doppler kernel transport in the spectral rasterizer, `GlassBlock`
    refraction in `tube_render`, `fluid::add_velocity_field`,
    `resim::grid_cell_bodies`, the V57 instrument template in
    `assets/viewer/`, and `pub(crate)` promotion of every "exported for
    Vx" helper the combos consume. Deviations in the Wave 9 addendum
    (single-WAV broadcast score — the deferred `amix`/`adelay` graphs
    are closed as unnecessary; artifact-first L0 for the dive; in-mode
    projective finale for rose-window; single-stream vanitas acts).
- **Next up:** no further waves. The order of work is now: (1) curate the
  fetched Wave-1 partials + upcoming batch output, (2) caption/cartouche
  backfill with `text.rs`, (3) deep-polish passes on the shortlist
  (V37/V39 full-res stills queued, per the Wave 4 addendum), (4) batch
  top-ups per "Immediate next actions".

## The server

- **SSH:** `user@100.76.88.48` (passwordless key auth already set up).
  128 cores, 503 GB RAM, ~3.5 TB free disk, Linux x86_64, ffmpeg installed,
  rustup installed by our bootstrap.
- **Remote checkout:** `~/viz-batch/CS-Image-Generation` (deployed via
  `git archive HEAD` over ssh — always commit before deploying).
- **Batch orchestration:** [run_viz_batch.py](../run_viz_batch.py)
  (stdlib-only, ruff/mypy-strict clean):

  ```bash
  python3 run_viz_batch.py            # deploy HEAD + launch detached batch
  python3 run_viz_batch.py --status   # tail logs, per-seed progress, load
  python3 run_viz_batch.py --fetch viz-results   # download output/viz-*
  # options: --seeds 0x...,0x...  --viz-flags all  --host  --remote-dir
  ```

  The batch runs all seeds **concurrently**, each capped at
  `RAYON_NUM_THREADS = cores / seed_count`, fully detached via
  `setsid nohup` (safe to disconnect). Per-seed logs: `viz-<seed>.log`
  in the remote dir; batch log: `viz_batch.log`.

- **Batches in flight right now:** see "Immediate next actions" above
  (primary: 2 seeds finishing turntables; viz-batch2: 3 seeds on Waves
  1-5). **Turntable economics** (the standing lesson): V46 final = 720
  full production re-renders, measured 3.7-23 min/frame at max quality.
  Keep max quality; budget for it, or exclude `turntable` via an
  explicit `--viz-flags` list. Waves 2-5 add ~1.5-2.5 h/seed on top of
  the ~4-5 h core package; Waves 6-9 add the projected lines to read on
  the first final-quality run (see Open items).
- **Useful deep probe** (per-seed mode completion, beyond `--status`):

  ```bash
  ssh user@100.76.88.48 'cd viz-batch/CS-Image-Generation/output && \
    for d in viz-0x*; do echo "== $d"; ls "$d/viz" | tr "\n" " "; echo; done'
  ```

## Local workflow cheat sheet

```bash
# Quality gates (all must be green before committing):
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings     # pedantic is enabled
cargo test --release                          # ~616 tests
.venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/mypy

# Fast full-pipeline smoke of every mode (~8 min at 69):
./target/release/three_body_problem --seed 0xC0DE --sims 200 --steps 30000 \
  --resolution 640x414 --viz all --viz-quality draft --fast-encode \
  --output viz-smokeN
# artifacts land in output/viz-smokeN/viz/<flag>/ ; INSPECT THE FRAMES
# (ffmpeg -ss T -i file.mp4 -frames:v 1 out.png), not just the exit code

# Catalog:
./target/release/three_body_problem --viz-list
```

Smoke outputs (`output/…`) are gitignored. Local smoke dirs so far:
`viz-smoke` (Wave 1) through `viz-smoke8` (Wave 8), `viz-smoke9` (Wave 9
full run) plus `viz-smoke9b/c/d` (witness/vanitas retuning iterations).
CAUTION: never launch the smoke twice concurrently — two runs share
`output/<name>` and their encoders fight over the same mp4 paths.

## Code map (viz subsystem)

```text
src/viz/mod.rs        VizMode trait, VizPhase, VizSelection, VizStageState
src/viz/catalog.rs    all 69 modes (all implemented); ledger test
src/viz/context.rs    VizContext, FrameTapCollector, VizQuality
src/viz/sink.rs       ArtifactSink, viz/manifest.json writer
src/viz/common/       accum (production re-accumulation), display (graders,
                      SpdCanvas + decay/add_into/Doppler strokes), fields,
                      particles, agents (Physarum/DLA/streamer), fluid
                      (stable fluids + add_velocity_field), wave (leapfrog
                      + sponge), tube_render (capsule tracer + GlassBlock
                      refraction, camera rigs, SDF mesher), resim (replays,
                      perturbation grids, grid_cell_bodies, Kabsch), text,
                      style, compositor, kinematics, events (drama()), spd,
                      contours, raster, audio, vector_export
src/viz/modes/        one file per mode (69); Wave 9 combos consume the
                      pub(crate) exports of their upstream modes
assets/fonts/         bundled IBM Plex cuts + OFL license (include_bytes!)
assets/viewer/        V53 viewer.html + V57 instrument.html templates
src/render/drawing.rs shift_spectral_kernel (Doppler transport) + the
                      with-kernels rasterizer variant (V65)
```

Integration points in the core pipeline: `src/main.rs` (flags, SPD-phase
call inside the video branch, trajectory phase at the end),
`app::render_video` (optional `frame_tap` observer),
`render/mod.rs` (`pub(crate)` accumulation internals used by
`viz::common::accum`).

## Conventions (do not break)

- Every mode: seed-deterministic; any randomness via
  `ctx.fork_rng(flag)` = domain `viz/<flag>/v1` (Wave 9 combos reuse
  their upstream modes' domains where the spec says so).
- Stills 16-bit Display P3 via `save_image_as_png_16bit`; videos dual
  web H.264 + HQ HEVC (`fast_encode` respected); three grading paths in
  `common/display.rs` (auto-levels, run-levels, direct gamma encode).
- Ledger discipline: flip catalog `implemented`, ledger row, and spec
  `Status` line in the same commit; record any spec deviation in the wave
  addendum. A unit test enforces catalog/ledger agreement.
- Core-package safety: viz failures are collected and only surface in the
  exit code; never abort the production outputs.
- `--viz-quality draft` exists for development only; server batches run
  `final` (the default).

## Open items

- **Caption/cartouche backfill** (unblocked by `common/text.rs`):
  V03's annotated scale bar, V43's ring-gauge labels, V42's cartographic
  axis labels, V62's atlas cartouche, V24's cell labels — all deferred
  with JSON sidecars carrying the data. Schedule with the curation pass.
- `render_still_image` does not return the SPD, so `--image-only` skips
  SPD-phase modes (warning logged); V33/V34 fall back to a trajectory
  splat-density food map in that case.
- assets.json `viz` section (currently a separate `viz/manifest.json`).
- **Wave 5-8 draft-tuned constants** still queued for golden-seed review
  at final quality (physarum normalization, marbling knee, frost sparkle,
  dust stroke energy; worldtube/chandelier fog, aurora weights, neon
  albedo, V02 sigma; basin/terra palettes, epilogue margins, multiverse
  threshold; sonification levels, poster inks, CRT strengths, trailer
  shot lengths, hologram binning). Budget items to watch at final: V45
  hero 16 spp, V28 sweep, V42 768^2 lattice, V23 x8 sim, V55 8192^2
  grid, V48 barrel resample.
- **Wave 9 constants likewise draft-tuned:** witness steady-state meter
  bias (x1.15) and stroke energy, vanitas vein/ice gains (high-pass
  normalization; veins only resolve as filaments at final-res grids),
  pond moonlight/glint/coupling, dive hold lengths and Shepard voicing,
  rose-window choir levels, broadcast bed levels. Budget items to watch
  on the first final run (projected lines exist for every video): V67
  ~25-35 min, V65 ~15-20 min, V64's L1 micro-render sweep, V68's 9x
  refracted capsules. The V57 latency trace (< 30 ms) and file://
  cross-browser checks are curation-pass items; V49's cold-viewer
  comprehension test needs an actual cold viewer.
- The user curates favorites from batch outputs; deep-polish passes on
  chosen modes follow (V37/V39 full-res stills queued behind that
  shortlist, per the Wave 4 addendum).
