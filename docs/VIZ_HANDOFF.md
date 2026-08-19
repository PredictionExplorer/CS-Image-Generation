# Viz Project Handoff

Operational state of the visualization subsystem effort, for resuming work in
a fresh session. The authoritative implementation spec and progress ledger is
[docs/VIZ_MASTER_PLAN.md](VIZ_MASTER_PLAN.md) — start there for *what* to
build; start here for *where things stand*.

Last updated: 2026-08-19 ~04:10 UTC, after the random farm launch
(`5338c08` deployed).

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

1. **The catalog is complete: 69 of 69 modes.** There are no more
   implementation waves.
2. **A continuous random farm is running.** Check it with
   `python3 run_viz_batch.py --status`. Four workers independently choose
   one uniform random target from the live 69-mode Rust catalog and a
   random 64-bit seed. Artifact prerequisites are expanded automatically.
3. **Monitor every 20 minutes.** Treat repeated failures, panics, stale
   idle workers, OOMs, or low disk as stop-the-line. A healthy long job
   may have no completion for hours; require CPU/log/output progress.
4. Curate fetched results and schedule the caption/cartouche backfill and
   deep-polish pass listed under Open items.

## Where we are

- **Branch:** `viz-master-plan` (pushed to `origin`). Wave 9 completed at
  `1524c98`; the farm landed at `256387d`, with pilot fixes `cd90e76` and
  `5338c08`. All Rust and Python gates are green (including 12 farm tests).
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
- **Next up:** no further waves. Curate farm output, backfill captions,
  and deeply polish the shortlist (V37/V39 full-resolution stills remain
  queued behind curation).

## The server

- **SSH:** `user@100.76.88.48` (passwordless key auth already set up).
  128 cores, 503 GB RAM, ~3.5 TB free disk, Linux x86_64, ffmpeg.
- **Remote checkout:** `~/viz-farm/CS-Image-Generation`, deployed from
  committed HEAD via `git archive`. Deployed code: `5338c08`.
- **Legacy state:** `~/viz-batch` and `~/viz-batch2` were stopped, fetched
  one final time, and deleted on 2026-08-19. Their five packages are
  preserved in `../CS-viz-results-20260818/`.
- **Release build:** launcher runs `cargo build --release --locked`.
  `Cargo.toml` uses opt-level 3, fat LTO, one codegen unit, aborting
  panics, and stripped symbols.
- **Farm policy:**
  - 4 rolling workers, `RAYON_NUM_THREADS=30` each (8 cores reserved).
  - Every one of the 69 targets is eligible, including `turntable`.
    A turntable can occupy one slot for days; the other three continue.
  - Production defaults and `--viz-quality final`; no fast encode.
  - New jobs stop below 500 GB free; active jobs drain.
  - Five consecutive failures trip the circuit breaker.
  - Output names include UTC, sequence, target, and random seed; no two
    workers share an output directory.
- **Current session:** `f00e620a1a1330aa`, started 2026-08-19 04:08 UTC.
  First targets: `epicycles`, `roche`, `dust-nebula`, and `tilt`
  (`depth-pack` prerequisite included). Initial load/RSS/disk were healthy
  with no warnings or errors.
- **Orchestration:** [run_viz_batch.py](../run_viz_batch.py) and
  [viz_farm.py](../viz_farm.py) are stdlib-only and ruff/mypy-strict:

  ```bash
  python3 run_viz_batch.py                    # deploy/build/launch
  python3 run_viz_batch.py --status           # state, jobs, load, RSS, disk
  python3 run_viz_batch.py --stop             # graceful drain
  python3 run_viz_batch.py --force-stop       # terminate supervisor + children
  python3 run_viz_batch.py --fetch ../results # resumable rsync
  ```

- **Remote layout:**
  - `orchestrator/state.json`: atomic status snapshot.
  - `orchestrator/session.log`: supervisor lifecycle.
  - `orchestrator/jobs/<job-id>.json/.log`: provenance + full job log.
  - `output/random-.../`: complete Rust package and `viz/manifest.json`.

### Monitoring and remediation

Every 20-minute check should inspect:

- supervisor and exactly four live Rust jobs (unless draining);
- `state.json` failure streak, free disk, target/seed, elapsed/log age;
- system load, aggregate RSS, and output growth;
- recent `WARN`, `ERROR`, `FATAL`, panic, OOM, or failed lines.

Response policy:

- isolated random job failure: record it; the farm continues;
- dead supervisor with healthy disk and no orphan children: restart;
- clear repeatable code defect: drain, reproduce locally, add a regression
  test, fix, run all gates, commit/push, redeploy/rebuild, and continue;
- low disk, ambiguous hangs, SSH/auth failure, or any action requiring
  output deletion: remain paused and report rather than destroy data.

## Local workflow cheat sheet

```bash
# Quality gates (all must be green before committing):
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings     # pedantic is enabled
cargo test --release                          # ~616 tests
.venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/mypy
python3 -m unittest discover -s tests -p 'test_*.py' -v

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
viz_farm.py           continuous random scheduler, atomic state, safeguards
run_viz_batch.py      SSH deploy/build/lifecycle/fetch wrapper
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
