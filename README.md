# Three Body Problem

**High-fidelity gravitational simulation turned into art.**

Simulates thousands of random three-body orbits, selects the most visually
interesting one via Borda voting, and renders it as a publication-quality image
and video with a deep post-processing pipeline.

## Features

- **Gravitational simulation** with energy-based escape detection
- **Borda voting** scores orbits on chaos and equilibrium to pick the best candidate
- **Spectral rendering** -- 16-bin wavelength SPD (380--700 nm) accumulated in OKLab color space
- **HDR pipeline** with ACES tonemapping and two-pass histogram leveling
- **15+ post-processing effects** -- bloom, chromatic bloom, color grading, opalescence, champlevé, aether, gradient mapping, atmospheric depth, and more
- **H.265 10-bit video** encoding via FFmpeg with optional hardware acceleration
- **Deterministic** -- SHA3-256-seeded RNG; every output is fully reproducible
- **Fast** -- parallel rendering (Rayon) and SIMD-optimized spectrum conversion

## Quick Start

```bash
cargo build --release
./target/release/three_body_problem --seed 0xABC
```

Output lands in `pics/` (PNG) and `vids/` (MP4).

### Prerequisites

| Dependency | Notes |
|---|---|
| **Rust 1.90.0+** | Auto-managed via `rust-toolchain.toml` -- just have [rustup](https://rustup.rs) installed |
| **FFmpeg** | Required for video encoding. `brew install ffmpeg` / `apt install ffmpeg` |

## Usage

```bash
# Custom resolution and seed
./target/release/three_body_problem --seed 0x46205528 --width 2560 --height 1440

# Render a single PNG frame (skip video)
./target/release/three_body_problem --seed 0x123 --test-frame

# Fast encode with hardware acceleration (macOS VideoToolbox)
./target/release/three_body_problem --seed 0x123 --fast-encode

# Disable all museum-quality enhancements
./target/release/three_body_problem --seed 0x123 --no-enhancements
```

**Batch A/B comparison** (enhanced vs. classic, runs continuously):

```bash
python3 run.py   # output + logs in pics/, vids/, run.log
```

Run `--help` for the full list of options covering simulation parameters, drift
modes, bloom settings, and individual effect toggles.

## Pipeline Overview

The renderer executes a seven-stage pipeline:

1. **Borda selection** -- simulate up to 100k random orbits, score each by chaos + equilibrium, pick the winner
2. **Re-simulation** -- rerun the selected orbit at full resolution
3. **Drift transform** -- apply camera motion (elliptical, Brownian, or linear)
4. **Color generation** -- assign spectral colors per body via OKLab
5. **Bounding box** -- compute aspect-aware viewport bounds
6. **Histogram & levels** -- two-pass render to determine black/white points and gamma
7. **Final render** -- draw all frames with the full effect chain, encode to H.265

## Project Layout

```
src/
  main.rs              CLI entry point and argument parsing
  app.rs               Pipeline orchestration (stages 1-7)
  sim.rs               Physics simulation, Borda selection, SHA3 RNG
  render/              Rendering engine (drawing, color, video, tonemapping)
  post_effects/        15+ composable visual effects
  spectrum.rs          Spectral rendering and wavelength-to-RGB conversion
  spectrum_simd.rs     SIMD-accelerated spectrum path
  oklab.rs             OKLab color space utilities
  drift.rs             Camera drift motion models
  analysis.rs          Orbit energy and angular momentum analysis
run.py                 Batch runner for A/B comparisons
ci/                    CI tooling and reference image regression tests
```

## Reproducibility

Every run is deterministic for a given `--seed`. All parameters are logged to
`generation_log.json` so any output can be reproduced exactly:

```bash
# Re-generate from a logged seed
./target/release/three_body_problem --seed 0x46205528
```

## Build Profile

Release builds use an aggressive optimization profile (see `Cargo.toml`):
LTO, `opt-level = 3`, single codegen unit, and `panic = abort` for maximum
throughput during rendering.
