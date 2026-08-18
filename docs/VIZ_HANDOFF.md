# Viz Project Handoff

Operational state of the visualization subsystem effort, for resuming work in
a fresh session. The authoritative implementation spec and progress ledger is
[docs/VIZ_MASTER_PLAN.md](VIZ_MASTER_PLAN.md) — start there for *what* to
build; start here for *where things stand*.

Last updated: 2026-08-18 ~01:30 EST (2026-08-18 ~06:30 UTC).

---

## Immediate next actions (fresh session, start here)

1. **Check the server batch** (`python3 run_viz_batch.py --status`). A
   Wave-1-only batch over 5 seeds has been running since 2026-08-17
   23:54 UTC; as of ~06:15 UTC all five seeds were inside `turntable`
   (V46 — the full-res orbit re-render, by far the longest mode) with only
   `plotter-svg` and `oscilloscope` left after it. When `viz_batch.log`
   shows `COMPLETE`:
   - `python3 run_viz_batch.py --fetch viz-results` (downloads
     `output/viz-*` for curation; several GB), then
   - `python3 run_viz_batch.py` to relaunch the same 5 seeds from HEAD —
     they will regenerate with all **36** modes (fully seed-deterministic,
     so Wave-1 artifacts reproduce identically).
2. **Start Wave 6** (3D scene family) per the plan below — independent of
   the batch; local work never touches the server checkout.

## Where we are

- **Branch:** `viz-master-plan` (pushed to `origin`). Wave 5 landed as
  `27eee5e`; every wave is one `feat:` commit plus this handoff kept in
  sync. All gates were green at the Wave 5 commit.
- **Progress:** 36 of 69 modes implemented (`--viz-list` prints the live
  catalog; the ledger in the master plan is kept in sync by a unit test).
  - **Wave 0** — framework: catalog, `--viz` CLI, `VizContext` (lazy
    kinematics/events/energy-field), `ArtifactSink` + `viz/manifest.json`,
    frame tap on the main render, two-phase runner (`VizPhase::Spd` runs
    inside the render block while the SPD buffer is alive; `Trajectory`
    runs after the core package is written).
  - **Wave 1** (8): braid, gw-chirp, syzygy-wheel, slit-scan, winding-glass,
    plotter-svg, oscilloscope, turntable.
  - **Wave 2** (7, SPD family): alien-vision, spectral-centroid,
    prism-portrait, spectrum-card, thin-film, dwell-nebula, topo-contours.
  - **Wave 3** (9, re-accumulation family): triangle-centers,
    medial-recursion, chrono-grid, strobe, comet, three-shadows, ride-along,
    retarded-time, depth-pack.
  - **Wave 4** (5, fields & frames): field-lines, corotating, lensing,
    roche, reconnection — plus `common/fields.rs` (potential grids, asinh
    equipotentials, Jobard–Lefer streamlines, `RochePotential` with the L1
    saddle, all unit-tested against analytic two-body cases). Notable
    wave-wide deviation: the frame tap is *not* used (full-frame retention
    is memory-prohibitive at default res); ghosts/sources come from
    half-res re-accumulation and master.png. Details in the Wave 4
    addendum.
  - **Wave 5** (7, particles/agents/media): dust-nebula, galaxy-collision,
    physarum, frost, lightning, marbling, light-echoes — plus
    `common/particles.rs` (test-particle swarms + the banded parallel
    splatter), `common/agents.rs` (Physarum / batch-synchronous DLA /
    streamer growth), `common/fluid.rs` (stable fluids + MacCormack dye),
    `common/wave.rs` (leapfrog + sponge), and energy-field retention into
    the trajectory phase (`VizMode::needs_energy_field`). Deviations in
    the Wave 5 addendum. README gained a "Visualization Modes" section.
- **Next up:** **Wave 6 — 3D scene family** per the build order in the
  master plan Part IV.2: `common/tube_render.rs` (CPU sphere-traced capsule
  chains, II.8) first, then V43 `worldtube`, V02
  `hyperspectral-flythrough`, V08 `shape-sphere`, V40 `aurora`, V44 `neon`,
  V45 `chandelier`, V50 `sculpture-export`, V28 `bullet-time`.
- **Deviations from specs** are recorded in the "Wave N addendum" sections
  at the top of the master plan (single viz manifest instead of an
  assets.json section; typography/text.rs deferred to the posters wave;
  `--image-only` skips SPD-phase modes; per-mode notes in each addendum).

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
  `RAYON_NUM_THREADS = cores / seed_count` (25 on this server), fully
  detached via `setsid nohup` (safe to disconnect). Per-seed logs:
  `viz-<seed>.log` in the remote dir; batch log: `viz_batch.log`.

- **Batch in flight right now:** launched 2026-08-17 23:54 UTC from commit
  `e0066a9` (Wave 1 only, 8 modes), seeds `0xCAFE 0xBEEF 0xC0DE 0xFACE
  0x1357`, max quality (default resolution/sims/steps, HQ encodes,
  `--viz all`). Progress timeline: main videos finished ~03-05 UTC (HEVC
  `slower` encodes dominate the core package); as of ~06:15 UTC every seed
  sat inside `turntable` (V46), which re-renders the sculpture from 360
  angles at full res and dwarfs every other Wave-1 mode; `plotter-svg` and
  `oscilloscope` (both cheap) remain after it. Follow-through when
  `viz_batch.log` shows `COMPLETE` is step 1 of "Immediate next actions"
  above. Budget note for the relaunch: Waves 2-5 add roughly 1.5-2.5 h per
  seed at max quality (re-accumulation videos, the V31/V38 particle sims,
  V33 physarum, and V32's wave grid are the big contributors), on top of
  the Wave-1-only footprint this batch is finishing.
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
cargo test --release                          # ~500 tests
.venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/mypy

# Fast full-pipeline smoke of every implemented mode (~2-3 min):
./target/release/three_body_problem --seed 0xC0DE --sims 200 --steps 30000 \
  --resolution 640x414 --viz all --viz-quality draft --fast-encode \
  --output viz-smokeN
# artifacts land in output/viz-smokeN/viz/<flag>/ ; inspect the PNGs

# Catalog:
./target/release/three_body_problem --viz-list
```

Smoke outputs (`output/…`) are gitignored. Local smoke dirs so far:
`viz-smoke` (Wave 1), `viz-smoke2` (Wave 2), `viz-smoke3` (Wave 3),
`viz-smoke4` (Wave 4), `viz-smoke5` (Wave 5).

## Code map (viz subsystem)

```text
src/viz/mod.rs        VizMode trait, VizPhase, VizSelection, VizStageState
src/viz/catalog.rs    all 69 modes; ledger-consistency test (include_str!)
src/viz/context.rs    VizContext, FrameTapCollector, VizQuality
src/viz/sink.rs       ArtifactSink, viz/manifest.json writer
src/viz/common/       accum (production re-accumulation), display (graders,
                      SpdCanvas), fields (potential grids, streamlines,
                      Roche/L1), particles (test-particle swarms, banded
                      splatter), agents (Physarum/DLA/streamer), fluid
                      (stable fluids), wave (leapfrog grid), kinematics,
                      events, spd, contours, raster, audio, vector_export
src/viz/modes/        one file per implemented mode (36)
```

Integration points in the core pipeline: `src/main.rs` (flags, SPD-phase
call inside the video branch, trajectory phase at the end),
`app::render_video` (optional `frame_tap` observer),
`render/mod.rs` (`pub(crate)` accumulation internals used by
`viz::common::accum`).

## Conventions (do not break)

- Every mode: seed-deterministic; any randomness via
  `ctx.fork_rng(flag)` = domain `viz/<flag>/v1`.
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

## Open items beyond Wave 5

- `common/text.rs` (ab_glyph + bundled OFL font) — unblocks poster
  typography (Wave 8 per the build order; deferred captions noted in
  addendums).
- `render_still_image` does not return the SPD, so `--image-only` skips
  SPD-phase modes (warning logged); V33/V34 fall back to a trajectory
  splat-density food map in that case.
- assets.json `viz` section (currently a separate `viz/manifest.json`).
- **Wave 5 aesthetic constants were tuned on the draft smoke only** and
  should be re-reviewed on golden seeds at final quality during curation:
  physarum trail normalization (16x steady state), marbling dye knee /
  stir impulse, frost sparkle weights, dust-nebula stroke energy. The
  draft smoke also hits draft-only artifacts by design: frost's 50%
  fill-limit truncates growth on tiny grids, and dust barely drifts in a
  30k-step run — judge both at final only.
- The user curates favorites from batch outputs; deep-polish passes on
  chosen modes follow the wave completions (V37/V39 full-res stills are
  queued behind that shortlist, per the Wave 4 addendum).
