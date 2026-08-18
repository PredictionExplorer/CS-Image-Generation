# Viz Project Handoff

Operational state of the visualization subsystem effort, for resuming work in
a fresh session. The authoritative implementation spec and progress ledger is
[docs/VIZ_MASTER_PLAN.md](VIZ_MASTER_PLAN.md) — start there for *what* to
build; start here for *where things stand*.

Last updated: 2026-08-17 ~21:00 EST (2026-08-18 ~01:00 UTC).

---

## Where we are

- **Branch:** `viz-master-plan` (pushed to `origin`). All work happens here.
- **Progress:** 24 of 69 modes implemented (`--viz-list` prints the live
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
- **Next up:** **Wave 4 — fields & frames** per the build order in the
  master plan Part IV.2: V12 `field-lines`, V26 `corotating`, V30 `lensing`,
  V37 `roche`, V39 `reconnection`. Requires `common/fields.rs` (potential
  grids, equipotentials via the existing `common/contours.rs`, Jobard–Lefer
  streamlines) — see Part II.4 of the master plan.
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
  `--viz all`). As of ~01:51 UTC no seed had finished (full-res HEVC
  `slower` encodes dominate); `0xFACE` furthest along. When
  `viz_batch.log` shows `COMPLETE`, relaunch `python3 run_viz_batch.py`
  to regenerate the same 5 seeds against HEAD with all 24 modes
  (Wave 1 outputs reproduce identically — everything is seed-deterministic;
  budget roughly +40–50 min per seed over the Wave-1-only run).

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
`viz-smoke` (Wave 1), `viz-smoke2` (Wave 2), `viz-smoke3` (Wave 3).

## Code map (viz subsystem)

```text
src/viz/mod.rs        VizMode trait, VizPhase, VizSelection, VizStageState
src/viz/catalog.rs    all 69 modes; ledger-consistency test (include_str!)
src/viz/context.rs    VizContext, FrameTapCollector, VizQuality
src/viz/sink.rs       ArtifactSink, viz/manifest.json writer
src/viz/common/       accum (production re-accumulation), display (graders,
                      SpdCanvas), kinematics, events, spd, contours, raster,
                      audio (WAV), vector_export (SVG/RDP)
src/viz/modes/        one file per implemented mode (24)
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

## Open items beyond Wave 4

- `common/text.rs` (ab_glyph + bundled OFL font) — unblocks poster
  typography (Wave 8 per the build order; deferred captions noted in
  addendums).
- `render_still_image` does not return the SPD, so `--image-only` skips
  SPD-phase modes (warning logged).
- assets.json `viz` section (currently a separate `viz/manifest.json`).
- The user curates favorites from batch outputs; deep-polish passes on
  chosen modes follow the wave completions.
