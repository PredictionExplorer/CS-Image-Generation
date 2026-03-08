# Three Body Problem

Seeded three-body simulation and renderer for generating a 16-bit PNG and H.265 MP4 from a single run.

## What It Does

- Simulates large batches of random three-body systems
- Selects the strongest orbit with a Borda-style score
- Renders spectral trails with SIMD acceleration
- Applies a curated post-processing pipeline
- Writes outputs to `pics/` and `vids/`

## Requirements

- Rust 1.94+ via `rustup`
- FFmpeg
- Python 3.10+ only if you use `run.py`

## Build

```bash
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
| `--seed` | `0x100033` | Hex seed for the simulation (with or without `0x` prefix) |
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

`run.py` is the deployment helper. It:

- fetches seeds from the CosmicSignature API,
- checks which assets are missing remotely,
- runs the Rust generator for missing seeds,
- uploads PNG and MP4 files,
- cleans up local artifacts.

### Systemd Timer Setup (Linux)

The repo ships two unit files that run `run.py` on a recurring schedule via systemd, similar to a cron job.

| File | Purpose |
|------|---------|
| `cosmicsig-sync.service` | One-shot service that executes `run.py` |
| `cosmicsig-sync.timer` | Timer that triggers the service every 5 minutes |

**1. Create the `.env` file**

Copy the example and fill in your deployment values:

```bash
cp .env.example .env
```

Then edit `.env` with your actual settings:

```dotenv
COSMICSIG_SSH_HOST=203.0.113.42        # IP address or hostname of the remote server
COSMICSIG_SSH_USER=frontend            # SSH user on the remote server
COSMICSIG_API_URL=http://api.example.com:8353  # CosmicGame API base URL (no trailing slash)
COSMICSIG_REMOTE_DIR=/home/frontend/nft-assets/new/cosmicsignature  # Remote asset directory
```

The `User=` in the service file must have passwordless SSH key access to `COSMICSIG_SSH_HOST`. Test with:

```bash
ssh -o BatchMode=yes COSMICSIG_SSH_USER@COSMICSIG_SSH_HOST echo ok
```

**2. Verify the environment**

Make sure the release binary is built and FFmpeg is installed. Then run the preflight check, which tests SSH connectivity, remote write access, and API reachability:

```bash
python3 run.py --preflight
```

**3. Review and adjust the service file**

Open `cosmicsig-sync.service` and confirm these values match your deployment:

- `User=` — the Linux user that will run the job (must have SSH key access for uploads).
- `WorkingDirectory=` — absolute path to the project checkout.
- `EnvironmentFile=` — path to the `.env` file with API keys and remote config.

**4. Install the unit files**

```bash
sudo cp cosmicsig-sync.service cosmicsig-sync.timer /etc/systemd/system/
sudo systemctl daemon-reload
```

**5. Enable and start the timer**

```bash
sudo systemctl enable --now cosmicsig-sync.timer
```

This starts the timer immediately and ensures it persists across reboots. The first run fires 2 minutes after boot; subsequent runs trigger every 5 minutes after the previous run finishes.

**6. Check status**

```bash
# Timer schedule and next trigger time
systemctl status cosmicsig-sync.timer

# Logs from the most recent run
journalctl -u cosmicsig-sync.service -e
```

**7. Run manually (optional)**

To trigger a one-off run outside the timer schedule:

```bash
sudo systemctl start cosmicsig-sync.service
```

**8. Disable the timer**

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
