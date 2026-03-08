# Three Body Problem

Seeded three-body simulation and renderer for generating a 16-bit PNG and H.265 MP4 from a single run.

## What It Does

- Simulates large batches of random three-body systems
- Selects the strongest orbit with a Borda-style score
- Renders spectral trails with SIMD acceleration
- Applies a curated post-processing pipeline
- Writes outputs to `pics/` and `vids/`

## Requirements

- Rust 1.94+
- FFmpeg (for video encoding)
- Python 3.10+ (only needed for `run.py` and `run-test-images.py`)
- Git

### Installing on Ubuntu

```bash
sudo apt update
sudo apt install -y build-essential ffmpeg python3 git curl

# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

The correct Rust toolchain version is pinned in `rust-toolchain.toml` and will be installed automatically on the first `cargo` command.

## Build

Clone the repository and build:

```bash
git clone <repo-url> CS-Image-Generation
cd CS-Image-Generation
cargo build --release
```

The project is tuned for release builds with LTO, a single codegen unit, `panic = abort`, and native CPU flags from `.cargo/config.toml`. On x86_64, AVX2 and FMA are enabled automatically.

> **Portability note:** `target-cpu=native` in `.cargo/config.toml` means the binary is optimized for the CPU it was built on and may not run on machines with older processors. Build on the same architecture you plan to deploy to.

## Run

```bash
./target/release/three_body_problem --seed 0xABCD
```

Useful examples:

```bash
./target/release/three_body_problem --seed 0x46205528 --resolution 2560x1440
./target/release/three_body_problem --seed 0x1234 --output piece-01
./target/release/three_body_problem --seed 0x1234 --drift none
./target/release/three_body_problem --seed 0x1234 --fast-encode
```

CLI reference:

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `0x100033` | Hex seed (with or without `0x` prefix, must have an even number of hex digits) |
| `-o, --output` | `output` | Base name for output files |
| `--sims` | `100000` | Number of orbits evaluated in the Borda search |
| `--steps` | `1000000` | Simulation steps per orbit |
| `-r, --resolution` | `1920x1080` | Output resolution as `WIDTHxHEIGHT` |
| `--drift` | `elliptical` | Camera drift mode: `none`, `linear`, `brownian`, `elliptical` |
| `--fast-encode` | off | Use faster (lower quality) video encoding |
| `--log-level` | `info` | Tracing log level (`error`, `warn`, `info`, `debug`, `trace`) |

## Outputs

- `pics/<output>.png`
- `vids/<output>.mp4`
- `generation_log.json` for reproducibility metadata

## Automation

`run.py` is the deployment helper that keeps a remote server in sync. Each run it:

1. Fetches the current list of token seeds from the CosmicSignature API.
2. Checks which `0x<seed>.png` / `0x<seed>.mp4` files already exist on the remote server (via SSH).
3. Generates missing assets locally with the Rust binary.
4. Uploads the new files to the remote server via SCP.
5. Deletes the local copies after a successful upload.

It also supports `--dry-run` (report what's missing without generating or uploading) and `--preflight` (test all external dependencies before committing to real runs).

### How the Two Machines Relate

```text
┌─────────────────────────┐         SSH / SCP         ┌──────────────────────────┐
│    Generator Machine    │ ──────────────────────────▶│     Remote Server        │
│  (Ubuntu, runs run.py)  │                            │  (any Linux with sshd)   │
│                         │                            │                          │
│  • Rust binary          │   uploads .png and .mp4    │  • Stores asset files    │
│  • run.py + systemd     │   to COSMICSIG_REMOTE_DIR  │  • Web server (nginx)    │
│  • All CPU-heavy work   │                            │    serves them to users  │
└─────────────────────────┘                            └──────────────────────────┘
```

The generator machine does all the compute. The remote server only stores and serves the finished files. They can be the same machine, but in a typical deployment they are separate.

### Setting Up from Scratch (Ubuntu)

The steps below assume you have a fresh Ubuntu generator machine and an existing remote server to upload assets to.

**1. Install dependencies and build**

Follow the [Requirements](#installing-on-ubuntu) and [Build](#build) sections above so that `./target/release/three_body_problem` exists.

**2. Set up SSH key access to the remote server**

The generator machine needs passwordless SSH access to the remote server. If you don't already have a key pair:

```bash
ssh-keygen -t ed25519 -C "cosmicsig-generator"
ssh-copy-id frontend@203.0.113.42
```

Replace `frontend` and `203.0.113.42` with your actual remote user and host. Verify it works without a password prompt:

```bash
ssh -o BatchMode=yes frontend@203.0.113.42 echo ok
```

You should see `ok` printed with no password prompt. If it asks for a password, the key was not copied correctly.

**3. Create the asset directory on the remote server**

SSH into the remote server and create the directory where assets will be stored:

```bash
ssh frontend@203.0.113.42 "mkdir -p /home/frontend/nft-assets/new/cosmicsignature"
```

This must match the `COSMICSIG_REMOTE_DIR` value you'll configure next. Make sure your web server (e.g. nginx) is configured to serve files from this directory.

**4. Create the `.env` file**

```bash
cp .env.example .env
```

Edit `.env` with your actual deployment values:

```dotenv
COSMICSIG_SSH_HOST=203.0.113.42
COSMICSIG_SSH_USER=frontend
COSMICSIG_API_URL=http://api.example.com:8353
COSMICSIG_REMOTE_DIR=/home/frontend/nft-assets/new/cosmicsignature
```

| Variable | What to put here |
|----------|------------------|
| `COSMICSIG_SSH_HOST` | IP address or hostname of the remote server |
| `COSMICSIG_SSH_USER` | SSH user on the remote server (must accept your key) |
| `COSMICSIG_API_URL` | CosmicGame API base URL, no trailing slash |
| `COSMICSIG_REMOTE_DIR` | Absolute path on the remote server where PNG/MP4 files are stored |

**5. Run the preflight check**

This tests SSH connectivity, remote write permissions, API reachability, and that the generator binary exists:

```bash
python3 run.py --preflight
```

All four checks should print `OK`. Fix any failures before continuing.

**6. Do a dry run (optional)**

See what `run.py` would generate and upload without actually doing it:

```bash
python3 run.py --dry-run
```

**7. Edit the systemd service file**

Open `cosmicsig-sync.service` and update the three values under `# --- Adjust these to match your deployment ---`:

```ini
User=ubuntu
WorkingDirectory=/home/ubuntu/CS-Image-Generation
EnvironmentFile=/home/ubuntu/CS-Image-Generation/.env
```

- `User=` — the Linux user on the generator machine that has the SSH key. This is the local user, not the remote one.
- `WorkingDirectory=` — absolute path to this repository on the generator machine.
- `EnvironmentFile=` — absolute path to the `.env` file you created in step 4. The default uses `%h` (systemd shorthand for the user's home directory), but you can use a full path instead.

**8. Install the systemd units**

```bash
sudo cp cosmicsig-sync.service cosmicsig-sync.timer /etc/systemd/system/
sudo systemctl daemon-reload
```

**9. Enable and start the timer**

```bash
sudo systemctl enable --now cosmicsig-sync.timer
```

This starts the timer immediately and ensures it survives reboots. The first run fires 2 minutes after boot; subsequent runs trigger every 5 minutes after the previous run finishes.

**10. Verify it's running**

```bash
# Timer schedule and next trigger time
systemctl status cosmicsig-sync.timer

# Logs from the most recent run
journalctl -u cosmicsig-sync.service -e

# Detailed run.py log (rotated, up to 5 x 10 MB)
cat imgcheck.log
```

**Trigger a manual run** outside the timer schedule:

```bash
sudo systemctl start cosmicsig-sync.service
```

**Disable the timer** when no longer needed:

```bash
sudo systemctl disable --now cosmicsig-sync.timer
```

## Batch Testing

`run-test-images.py` continuously generates images with random seeds, useful for visual QA and stress testing. It keeps 3 concurrent jobs running and logs progress to `run.log`.

```bash
python3 run-test-images.py
```

Press Ctrl+C to stop gracefully after the current jobs finish. Output lands in `pics/` and `vids/` as usual.

## Reference Image Verification

The `ci/` directory contains tooling for deterministic regression testing. A reference image is generated with a fixed seed and parameters, then future builds are verified against it by SHA256 hash.

Generate the reference baseline:

```bash
cd ci/reference
./generate_reference.sh
```

This creates `baseline_512x288.png` and a companion `.json` with the parameters and hash. To verify a test image against the baseline:

```bash
python3 ci/verify_reference.py pics/test.png
```

## Development

Format, lint, and test before pushing:

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

Formatting rules are defined in `rustfmt.toml` (100-char line width, 4-space indentation). The project treats all warnings as errors (`[lints.rust] warnings = "deny"`).

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
run-test-images.py       Batch random-seed test runner
ci/                      Reference-image verification tooling
.cargo/config.toml       Native CPU flags and SIMD features
cosmicsig-sync.service   Systemd service unit for run.py
cosmicsig-sync.timer     Systemd timer (every 5 minutes)
.env.example             Template for deployment secrets
```
