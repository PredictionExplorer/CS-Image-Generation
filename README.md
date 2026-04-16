# Three Body Problem

Seeded three-body simulation and renderer for generating a 16-bit PNG and H.265 MP4 from a single run.

The Rust crate and binary are named **`three_body_problem`** (see `Cargo.toml`). Your checkout directory may use a different name (for example `CS-Image-Generation`).

## What It Does

- Simulates large batches of random three-body systems
- Selects the strongest orbit with a Borda-style score
- Renders spectral trails with SIMD acceleration
- Applies a curated post-processing pipeline
- Writes outputs to `output/<name>/`

## Requirements

- Rust 1.94.1+ (see `rust-version` in `Cargo.toml`)
- FFmpeg (for video encoding)
- Python 3.10+ for the helper scripts (`run.py`, `run-test-images.py`, `ci/verify_reference.py`). The scripts use only the standard library at runtime. Separate optional dev packages (Ruff, Mypy) apply when you run Python quality checks or CI; see [Development](#development).
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

From a Git checkout, clone then build (replace the URL with yours):

```bash
git clone <your-repo-url> CS-Image-Generation
cd CS-Image-Generation
cargo build --release
```

If you already have the source tree (for example from an archive or IPFS), open that directory and run `cargo build --release` there instead of cloning.

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
| `--chaos-weight` | random | Borda weight for chaos (FFT regularity); omit to sample from a curated range |
| `--equil-weight` | random | Borda weight for equilateralness; omit to sample from a curated range |
| `--fast-encode` | off | Use faster (lower quality) video encoding |
| `--log-level` | `info` | Tracing log level (`error`, `warn`, `info`, `debug`, `trace`) |

## Outputs

Under `output/<name>/` (default name `output`, so default paths look like `output/output/...` unless you pass `--output`):

- `image.png` — 16-bit still frame
- `video.mp4` — main H.265 trajectory video
- `spectral/` — 64 per-wavelength-bin 16-bit PNGs (`00_…nm.png` … `63_…nm.png`)
- `spectral_sweep.mp4` — spectral sweep through active bins (Gaussian blend + cosine easing)

`generation_log.json` is written in the **process working directory** (typically the repo root when you run the binary from there), not under `output/<name>/`. It records reproducibility metadata for each run.

## Automation

`run.py` is the deployment helper that keeps a remote server in sync. Each run it:

1. Fetches the current list of CosmicSignature token seeds from the CosmicGame HTTP API.
2. Checks which `0x<seed>.png` / `0x<seed>.mp4` files already exist on the remote server (via SSH).
3. Generates missing assets locally with the Rust binary.
4. Uploads the new files to the remote server via SCP.
5. Deletes the local copies after a successful upload.

It also supports `--dry-run` (report what's missing without generating or uploading) and `--preflight` (test all external dependencies before committing to real runs).

### How the Two Machines Relate

```text
┌─────────────────────────┐       SSH / SCP        ┌──────────────────────────┐
│    Generator Machine    │ ──────────────────────▶│     Remote Server        │
│  (Ubuntu, runs run.py)  │                        │  (any Linux with sshd)   │
│                         │                        │                          │
│  • Rust binary          │  uploads .png and .mp4 │  • Stores asset files    │
│  • run.py + systemd     │  to COSMICSIG_REMOTE_DIR │  • Web server (nginx)    │
│  • All CPU-heavy work   │                        │    serves them to users  │
└─────────────────────────┘                        └──────────────────────────┘
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
| `COSMICSIG_API_URL` | CosmicGame API base URL (CosmicSignature token list), no trailing slash |
| `COSMICSIG_REMOTE_DIR` | Absolute path on the remote server where PNG/MP4 files are stored |

**5. Run the preflight check**

This tests SSH connectivity, remote write permissions, API reachability, that the release generator binary exists, and that `ffmpeg` is on `PATH`:

```bash
python3 run.py --preflight
```

All five checks should report success in the log (`OK` lines for SSH, remote write, API, generator binary, and ffmpeg). Fix any failures before continuing.

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

Press Ctrl+C to stop gracefully after the current jobs finish. Output lands in `output/<seed>/`.

## Reference Image Verification

The `ci/` directory contains tooling for deterministic regression testing. A reference image is generated with a fixed seed and parameters, then future builds are verified against it by SHA256 hash.

Generate the reference baseline:

```bash
cd ci/reference
./generate_reference.sh
```

This creates `baseline_512x288.png` and a companion `.json` with the parameters and hash. From the **repository root**, verify a test image against the default baseline:

```bash
python3 ci/verify_reference.py output/test/image.png
```

Pass a second path if your reference image or JSON lives elsewhere. Run with no arguments to print a short usage line.

## Development

This section covers local checks before you push. **Continuous integration** on GitHub runs the same ideas in automated jobs; see [`ci/README.md`](ci/README.md) for the full job list (Rust fmt, Clippy, tests, benchmarks, docs, audit, coverage, and Python).

### Rust

Formatting and lint settings match CI (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Formatting rules live in [`rustfmt.toml`](rustfmt.toml) (100-character lines, 4-space indentation). The crate denies Rust warnings and missing public docs (`[lints.rust]` in [`Cargo.toml`](Cargo.toml): `warnings = "deny"`, `missing_docs = "deny"`).

If you use [just](https://github.com/casey/just): `just check` runs `fmt` + `clippy`; `just test` runs the release test suite; `just all` runs `check` then `test`.

### Python scripts (runtime)

These files are **stdlib-only**; you do not install anything from PyPI to execute them:

| Script | Role |
|--------|------|
| [`run.py`](run.py) | Sync CosmicSignature assets with a remote host (SSH/SCP + API). |
| [`run-test-images.py`](run-test-images.py) | Long-running random-seed generator for QA. |
| [`ci/verify_reference.py`](ci/verify_reference.py) | Compare a PNG to the CI reference hash. |

Use `python3 …` from the repository root (or `cd` as shown in each section). Deployment configuration for `run.py` is described under [Automation](#automation).

### Python quality (Ruff + Mypy)

Separate from *running* the scripts, the repo pins **developer** tools so formatting, lint, and static typing stay consistent:

| Tool | Role |
|------|------|
| [Ruff](https://docs.astral.sh/ruff/) | Lints and formats the four Python files (replaces a pile of flake8/isort/black-style checks in one fast binary). |
| [Mypy](https://mypy.readthedocs.io/) | Strict type-checking for the same files. |

Configuration is entirely in [`pyproject.toml`](pyproject.toml): Ruff target Python 3.10, line length **100** (same as Rust), rule sets **E, F, I, UP, B, SIM, PTH, RUF**; Mypy **`strict = true`** on `_utils.py`, `run.py`, `run-test-images.py`, and `ci/verify_reference.py`.

**Install the dev tools** (recommended: virtual environment so you do not fight [PEP 668](https://peps.python.org/pep-0668/) on Homebrew or Debian `externally-managed-environment`):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `[dev]` extra installs pinned **Ruff** and **Mypy** versions declared in `pyproject.toml`. The editable install also exposes the small `_utils` module the same way `run.py` expects when run from the repo root.

**Run checks** (same three steps as CI):

```bash
just py-check
```

That runs, in order: `ruff format --check .`, `ruff check .`, and `mypy`. To apply Ruff’s formatter without checking: `just py-fmt` (equivalent to `ruff format .`).

**Git hook:** If you use [`.githooks/pre-commit`](.githooks/pre-commit) (`git config core.hooksPath .githooks`), each commit runs **Rust** fmt + Clippy, then **Python** Ruff + Mypy. The hook calls `ruff` and `mypy` on your `PATH`, so activate the venv (above) in terminals where you commit, or install the tools into an environment that is always on your `PATH`.

**CI:** The workflow job **Python (ruff + mypy)** uses Ubuntu, **Python 3.12**, `pip install ".[dev]"`, then the same three commands as `just py-check`. Mypy is configured with `python_version = "3.10"` in `pyproject.toml`, so types stay compatible with the stated minimum interpreter.

## Algorithm

For a detailed description of the spectral pipeline (SPD buffer, accumulation, gallery, and spectral sweep video), see [docs/spectral-algorithm.md](docs/spectral-algorithm.md).

## License

This repository does not ship an SPDX `LICENSE` file. Before redistributing or pinning a build to IPFS for others to reuse, add a license you are comfortable with (for example MIT or Apache-2.0) so terms are explicit.

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
_utils.py                Shared helpers imported by `run.py` / `run-test-images.py`
run.py                   Automated generation and upload
run-test-images.py       Batch random-seed test runner
pyproject.toml           Python dev tooling (Ruff, Mypy) and optional `[dev]` deps
justfile                 `just` recipes (`check`, `test`, `py-check`, …)
ci/                      Reference-image verification tooling
docs/                    Long-form algorithm documentation
.cargo/config.toml       Native CPU flags and SIMD features
cosmicsig-sync.service   Systemd service unit for run.py
cosmicsig-sync.timer     Systemd timer (every 5 minutes)
.env.example             Template for deployment secrets
```
