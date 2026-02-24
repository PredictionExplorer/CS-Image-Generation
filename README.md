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
- **Automated asset sync** -- `run.py` monitors a token API, generates missing assets, and uploads them to a remote server

## Quick Start (Local Development)

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
| **Python 3.10+** | Required only for the automated asset sync script (`run.py`) |

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

Run `--help` for the full list of options covering simulation parameters, drift
modes, bloom settings, and individual effect toggles.

---

## Production Deployment

`run.py` is an automated sync script designed to run continuously on an Ubuntu
server. It periodically queries a token API for all minted CosmicSignature NFTs,
checks which images and videos are missing on a remote asset server, generates
them locally, and uploads them via SCP.

You need two pieces of information before starting:

1. **The API endpoint** that serves the CosmicSignature token list.
2. **The address of the remote server** where generated assets are stored, along with SSH credentials to write to it.

### Step 1: Install System Dependencies

```bash
sudo apt update
sudo apt install -y build-essential ffmpeg python3 openssh-client

# Install Rust (if not already present)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Step 2: Clone and Build

```bash
git clone <repo-url> ~/CS-Image-Generation
cd ~/CS-Image-Generation
cargo build --release
```

This compiles the Rust renderer in release mode with full optimizations (LTO,
single codegen unit). The build may take several minutes on the first run but
only needs to be repeated when the source code changes.

Verify the binary was built:

```bash
./target/release/three_body_problem --help
```

### Step 3: Configure the Environment

Copy the example environment file and fill in your deployment values:

```bash
cp .env.example .env
nano .env
```

The `.env` file contains four required settings:

```bash
# Remote server to upload generated assets to
COSMICSIG_SSH_HOST=your.server.com
COSMICSIG_SSH_USER=frontend

# CosmicGame API base URL (no trailing slash)
COSMICSIG_API_URL=http://your-api-host:8353

# Remote directory where PNG/MP4 assets are stored
COSMICSIG_REMOTE_DIR=/home/frontend/nft-assets/new/cosmicsignature
```

The `.env` file is excluded from version control via `.gitignore`. Configuration
can also be set via environment variables or CLI arguments (in increasing
priority).

### Step 4: Set Up SSH Key Authentication

The sync script uploads files via SCP, which requires passwordless SSH access
to the remote server. If you haven't already:

```bash
# Generate a key pair (if you don't have one)
ssh-keygen -t ed25519

# Copy the public key to the remote server
ssh-copy-id -i ~/.ssh/id_ed25519.pub frontend@your.server.com
```

### Step 5: Run Preflight Check

Before committing to lengthy generation runs, verify that everything is wired
up correctly:

```bash
python3 run.py --preflight
```

This tests four things in a few seconds:

1. **SSH connectivity** -- can we log in to the remote host?
2. **Write permissions** -- can we write to the remote asset directory?
3. **API reachability** -- is the token API responding?
4. **Generator binary** -- is the compiled Rust binary present?

All four checks must pass before proceeding. Fix any reported issues and re-run
until you see:

```
[preflight] All checks passed.
```

### Step 6: Dry Run

Once preflight passes, do a dry run to see how many assets are missing without
actually generating or uploading anything:

```bash
python3 run.py --dry-run
```

This fetches the full token list from the API, checks the remote server for
existing files, and reports what it *would* generate. Use this to get a sense of
the workload and confirm the script is reading the correct data.

### Step 7: Manual Test Run

Run the script once without any flags to generate and upload a real asset:

```bash
python3 run.py
```

Watch the output. The script will process missing seeds one at a time: generate
the image and video, upload both to the remote server, then move to the next.
You can stop it at any time with `Ctrl+C` -- it will finish the current seed
and exit gracefully.

### Step 8: Set Up the systemd Timer

Once you have confirmed everything works, set up the systemd timer so the script
runs automatically every 5 minutes.

**Edit the service file** to match your deployment. The two lines you need to
adjust are `User` and `WorkingDirectory`:

```bash
nano cosmicsig-sync.service
```

```ini
# Change these to match your setup:
User=ubuntu
WorkingDirectory=/home/ubuntu/CS-Image-Generation
```

The `EnvironmentFile` path will also need to match:

```ini
EnvironmentFile=/home/ubuntu/CS-Image-Generation/.env
```

**Install and enable the timer:**

```bash
sudo cp cosmicsig-sync.service cosmicsig-sync.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cosmicsig-sync.timer
```

**Verify the timer is active:**

```bash
systemctl list-timers | grep cosmicsig
```

You should see the timer listed with the next scheduled run.

### Monitoring

**View live logs:**

```bash
journalctl -u cosmicsig-sync -f
```

**View logs from the last run:**

```bash
journalctl -u cosmicsig-sync --since "5 minutes ago"
```

**Check if the service is currently running:**

```bash
systemctl status cosmicsig-sync.service
```

**View the detailed log file** (includes DEBUG-level output not shown in
journald):

```bash
tail -f ~/CS-Image-Generation/imgcheck.log
```

The log file rotates automatically at 10 MB with 5 backups (50 MB total).

### How It Works

The systemd timer fires every 5 minutes. If the previous run is still in
progress, systemd skips the invocation -- there is no risk of overlapping runs.

Each run follows this sequence:

1. Fetch all token seeds from the API
2. List existing files on the remote server (single SSH call)
3. Diff the two lists to find missing assets
4. For each missing seed:
   - Run the Rust generator (`--seed 0x... --file-name 0x...`)
   - Locate the output PNG and MP4
   - Upload both to the remote server via SCP
   - Clean up local files
5. Log a summary of results

If a seed fails at any stage, the error is logged and the script continues to
the next seed. Failed seeds are listed in the summary at the end of each run.

### Stopping the Service

```bash
# Temporarily stop (will resume on next timer tick)
sudo systemctl stop cosmicsig-sync.service

# Disable the timer entirely
sudo systemctl disable --now cosmicsig-sync.timer
```

---

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
run.py                 Automated asset sync script (generate + upload)
run-test-images.py     Batch A/B comparison runner (enhanced vs. classic)
.env.example           Template for deployment configuration
cosmicsig-sync.service systemd service unit
cosmicsig-sync.timer   systemd timer unit (every 5 minutes)
ci/                    CI tooling and reference image regression tests
```

## Reproducibility

Every run is deterministic for a given `--seed`. All parameters are logged to
`generation_log.json` so any output can be reproduced exactly:

```bash
./target/release/three_body_problem --seed 0x46205528
```

## Build Profile

Release builds use an aggressive optimization profile (see `Cargo.toml`):
LTO, `opt-level = 3`, single codegen unit, and `panic = abort` for maximum
throughput during rendering.
