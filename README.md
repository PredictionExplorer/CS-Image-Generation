# Three Body Problem

Seeded three-body simulation and renderer for generating a 16-bit PNG and H.265 MP4 from a single run.

## What It Does

- Simulates large batches of random three-body systems
- Selects the strongest orbit with a Borda-style score
- Renders spectral trails with SIMD acceleration
- Applies a curated post-processing pipeline
- Writes outputs to `pics/` and `vids/`

## Requirements

- Rust 1.93.1+ via `rustup`
- FFmpeg
- Python 3.10+ only if you use `run.py`

## Build

```bash
cargo build --release
```

The project is tuned for release builds with LTO, a single codegen unit, `panic = abort`, and native CPU flags from `.cargo/config.toml`.

## Run

```bash
./target/release/three_body_problem --seed 0xABC
```

Useful examples:

```bash
./target/release/three_body_problem --seed 0x46205528 --resolution 2560x1440
./target/release/three_body_problem --seed 0x123 --output piece-01
./target/release/three_body_problem --seed 0x123 --drift none
./target/release/three_body_problem --seed 0x123 --fast-encode
```

Current CLI surface:

- `--seed`
- `--output`
- `--sims`
- `--steps`
- `--resolution WIDTHxHEIGHT`
- `--drift {none|linear|brownian|elliptical}`
- `--fast-encode`
- `--log-level`

## Outputs

- `pics/<output>.png`
- `vids/<output>.mp4`
- `generation_log.json` for reproducibility metadata

## Automation

`run.py` is the deployment helper. It:

- fetches seeds from the CosmicSignature API,
- checks which assets are missing remotely,
- runs the Rust generator for missing seeds,
- uploads PNG and MP4 files,
- cleans up local artifacts.

Use `python3 run.py --preflight` before enabling the systemd timer.

## Project Layout

```text
src/main.rs              CLI entry point
src/app.rs               Pipeline orchestration
src/sim.rs               Physics simulation and selection
src/render/              Rendering, tonemapping, video
src/post_effects/        Post-processing effects
src/spectrum.rs          Spectral conversion
src/spectrum_simd.rs     SIMD spectral fast paths
src/oklab.rs             OKLab utilities
run.py                   Automated generation and upload
ci/                      Reference-image verification tooling
```
