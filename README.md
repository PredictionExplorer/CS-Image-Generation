# Three Body Problem

Seeded three-body simulation and renderer for generating a 16-bit PNG and H.265 MP4 from a single run, plus the **ember edition**: the same orbit drawn in sumi ink by the fluid it stirs.

The Rust crate and binary are named **`three_body_problem`** (see `Cargo.toml`). Your checkout directory may use a different name (for example `CS-Image-Generation`).

## What It Does

- Simulates large batches of random three-body systems
- Selects the strongest orbit with a Borda-style score plus an image-space aesthetic gate
- Renders the trajectory through a seed-composed **layer stack** of stroke vocabularies
  (webs, ribbons, chords, spokes, nebula veils, harmonic weaves, stipple constellations,
  tangent caustics), with rare phase-space projections and k-fold symmetry compositions
- Samples every palette from a continuous OKLCh genome (no presets) behind a
  deterministic perceptual beauty gate
- Renders spectral trails with SIMD acceleration
- Applies the crisp CosmicSignature visual profile with only the active finish
  traits: seed-gated halation, prism, diffraction spikes, and stardust
- Renders the **ember edition** of the same orbit (see [below](#the-ember-edition))
- Writes outputs to `output/<name>/`

## The Ember Edition

Each package also contains a second, very different picture of the selected orbit. The three
bodies are dragged as small discs through a two-dimensional Navier–Stokes fluid. Water that
sweeps past a body through its spinning boundary layer picks up black pine-soot sumi, and the
flow carries the ink into filaments. Where the fresh waters of two bodies meet, the ink turns vermilion. Everything is
shaded spectrally (36 bands, Kubelka–Munk) as sumi and cinnabar on a mottled kozo sheet:

- a 16-bit sRGB still of the orbit's final step (`images/source/ember.png`) with WebP derivatives;
- a 60 fps video in H.264 and HEVC whose frames follow `main.mp4` frame for frame and whose last
  frame is the still;
- a determinism certificate, `metadata/ember.json`.

The ember edition runs **on the CPU only** and is **bit-identical across CPU architectures**
(x86_64 with or without AVX2/FMA, aarch64) and thread counts. It uses only exactly rounded
arithmetic, a pinned pure-Rust `libm` for transcendentals, its own FFT and order-independent
parallelism. The certificate records the SHA-256 of the raw 16-bit frame stream and of the still,
plus every input they depend on, so anyone can re-render a seed on another machine and compare.
Encoded files (MP4, WebP) are outside that guarantee because their bytes depend on encoder versions.

The ember stage adds about 25 minutes to a default package on a 64-core Threadripper, for about
61 minutes in total (the fluid solver dominates; see
[docs/ember-edition.md](docs/ember-edition.md#runtime)). It adds about 0.4 GB to the package,
mostly the archival HEVC video (about 284 MB). That document
also covers the physics, the ink model, the look, the certificate fields, and how to verify a
package on two machines. Pass `--no-ember` to skip the edition. If the edition fails, the rest of
the package is still written and the run exits with status `3` (see [Exit status](#exit-status)).

## Requirements

- Rust 1.94.1+ (see `rust-version` in `Cargo.toml`)
- FFmpeg (for video encoding)
- Python 3.10+ for the helper scripts (`run.py`, `run-test-images.py`, `contact_sheet.py`, `ci/verify_reference.py`). The scripts use only the standard library at runtime. Separate optional dev packages (Ruff, Mypy) apply when you run Python quality checks or CI; see [Development](#development).
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
| `--steps` | `1000000` | Simulation steps per orbit. The ember edition needs an orbit that lasts more than 0.75 fluid time units, which short test runs usually do not (seeds tried: rejected at 20,000 steps, accepted from 100,000). Its preflight then rejects the orbit right after selection: the run renders the rest of the package and exits with status 3 (see [Exit status](#exit-status)); pass `--no-ember` for such runs |
| `-r, --resolution` | `3456x2234` | Output resolution as `WIDTHxHEIGHT` |
| `--drift` | `elliptical` | Camera drift mode: `none`, `linear`, `brownian`, `elliptical` |
| `--chaos-weight` | random | Borda weight for chaos (FFT regularity); omit to sample from a curated range |
| `--equil-weight` | random | Borda weight for equilateralness; omit to sample from a curated range |
| `--fast-encode` | off | Use faster (lower quality) video encoding. The ember edition always stays on software encoders: its HQ slot becomes H.264 10-bit |
| `--image-only` | off | Render only the stills and their WebPs: the master and the ember still with its certificate. Skip all videos, the spectral gallery, and the sweep |
| `--no-ember` | off | Skip the ember edition (no `ember.png`, ember WebPs, ember videos, or `metadata/ember.json`) |
| `--metadata-only` | off | Skip all rendering and write only `metadata/generation.json` and `metadata/nft_traits.json` |
| `--log-level` | `info` | Tracing log level (`error`, `warn`, `info`, `debug`, `trace`) |

### Exit status

| Status | Meaning |
|--------|---------|
| `0` | The package is complete (with the ember edition, unless `--no-ember`). |
| `1` | Any other failure, including an invalid `--seed` or a resolution above 16,384 per side; the package is incomplete. |
| `2` | Rejected by the argument parser (unknown flag or malformed value). |
| `3` | The package is complete **except for the ember edition**. |

A run killed by a signal (an out-of-memory kill, or an abort: release builds use `panic = abort`)
has no exit status of its own; `run.py` treats it like `1`.

Status `3` means the ember edition's preflight rejected the orbit, or its stage failed later (a
non-finite flow, an encoder error, a write error). A warning in the log names the error. The
generator then removes every ember file (`images/source/ember.png`, the two ember WebPs, both
`ember.mp4` and `metadata/ember.json`), including partial ones, and writes the rest of the
package exactly as with `--no-ember`: `metadata/assets.json` without ember entries,
`metadata/generation.json` and `metadata/nft_traits.json`. If an ember file cannot be removed,
the run exits with `1` instead, because a package must never contain ember files its manifest
does not list. For a new mint, `run.py` uploads a status-`3` package and retries the ember
edition later as a backfill (see [The ember backfill](#the-ember-backfill)). `--help` lists these
statuses too.

## Outputs

Under `output/<name>/` (default name `output`, so default paths look like `output/output/...` unless you pass `--output`):

- `images/source/master.png` — maximum-quality 16-bit Display P3 still frame
- `images/web/full.webp` — full-resolution WebP website image
- `images/web/preview.webp` — smaller same-aspect-ratio WebP preview/poster
- `videos/web/main.mp4` — browser-compatible H.264 trajectory video
- `videos/hq/main.mp4` — high-quality HEVC trajectory video
- `spectral/` — 64 per-wavelength-bin 16-bit PNGs (`00_…nm.png` … `63_…nm.png`)
- `videos/web/spectral_sweep.mp4` — browser-compatible spectral sweep video
- `videos/hq/spectral_sweep.mp4` — high-quality HEVC spectral sweep video
- `images/source/ember.png` — the ember edition's 16-bit sRGB still (the orbit's final step)
- `images/web/ember_full.webp` / `images/web/ember_preview.webp` — full-size and preview WebPs of the ember still
- `videos/web/ember.mp4` — browser-compatible H.264 ember video (sRGB, 60 fps, in step with `main.mp4`)
- `videos/hq/ember.mp4` — archival HEVC 4:2:2 10-bit ember video (software H.264 10-bit under `--fast-encode`)
- `metadata/ember.json` — the ember edition's determinism certificate: inputs, configuration, and SHA-256 of the raw frames and still (see [docs/ember-edition.md](docs/ember-edition.md))
- `metadata/generation.json` — per-package resolved generation parameters and randomization log
- `metadata/assets.json` — website asset manifest with paths, dimensions, codecs, byte sizes, and SHA-256 hashes (ember entries use the `ember_*` roles and `"color_space": "srgb"`)
- `metadata/nft_traits.json` — public NFT trait file: marketplace-ready attributes, physics analyses (syzygies, braid word, fate, chaos index), and the seed-resolved generation context (see [docs/augur-explorer-integration.md](docs/augur-explorer-integration.md))

`generation_log.json` is also appended in the **process working directory** (typically the repo root when you run the binary from there). It records the same reproducibility metadata across runs.

## Automation

`run.py` is the deployment helper that keeps a remote server in sync. Each run it:

1. Fetches the current list of CosmicSignature token seeds from the CosmicGame HTTP API.
2. Checks which per-seed asset packages already exist on the remote server (via SSH). If the
   listing fails, the run stops with status `1` instead of treating the remote as empty.
3. Generates incomplete packages locally with the Rust binary.
4. Uploads each package to the remote server via SCP: the media first, then `metadata/`, and
   `metadata/ember.json` last of all (any remote copy of it is deleted before the upload
   starts). The metadata files and the certificate are uploaded under temporary `.part` names
   and then renamed into place, `metadata/assets.json` after the other metadata files, so an
   interrupted upload never leaves a truncated metadata file. Before a whole package is uploaded
   (a new mint, or a `full` backfill), its remote `metadata/assets.json` is deleted too: until
   the new manifest lands, the package reads as new or incomplete, so an interrupted upload is
   regenerated and uploaded in full by a later run, whatever the backfill mode. Each transfer
   times out after 15 minutes, or after one second per MB when it carries more than 900 MB.
5. Deletes the local copies, whatever the outcome.

It also supports `--dry-run` (report what's missing without generating or uploading) and `--preflight` (test all external dependencies before committing to real runs).

At startup `run.py` runs `<generator> --help` and checks that it lists `--no-ember`. A binary
without it predates the ember edition (for example, the checkout was pulled but not rebuilt): every
run then logs an ERROR telling you to rebuild the generator, checks packages against the core
files only so that new mints are still uploaded (without the ember edition), and pauses the ember
backfill. `--preflight` reports such a binary as a failed check.

Remote files mirror the Rust output package under `COSMICSIG_REMOTE_DIR/0x<seed>/`:

```text
0x<seed>/
  images/
    source/
      master.png
      ember.png
    web/
      full.webp
      preview.webp
      ember_full.webp
      ember_preview.webp
  videos/
    web/
      main.mp4
      spectral_sweep.mp4
      ember.mp4
    hq/
      main.mp4
      spectral_sweep.mp4
      ember.mp4
  spectral/
    00_...nm.png
    ...
    63_...nm.png
  metadata/
    generation.json
    assets.json
    nft_traits.json
    ember.json
```

Only API-listed seeds are considered. If any required file of an API seed's remote package is missing, `run.py` treats the seed as incomplete. A seed whose package lacks a core file (a new mint, or a broken upload) is **urgent**: it is regenerated and its whole package uploaded. A seed whose package lacks only the six ember files is an **ember backfill** seed (see [below](#the-ember-backfill)).

**A failed ember edition never blocks a new mint.** When the generator exits with status `3` for an urgent seed (see [Exit status](#exit-status)), `run.py` deletes any stale ember file from the seed's remote directory (`rm -f` over SSH; if that fails, nothing is uploaded and the seed stays urgent), checks the local package against the core files only (everything except the six ember files, plus the 64 spectral bins), uploads it, and counts one failed ember attempt in `backfill_failures.json`. The token therefore gets its artwork and traits right away, and on later runs the seed is an ember backfill seed. Such seeds are counted as OK, and listed separately as "without ember", in the run summary. Any other non-zero status is a failure: nothing is uploaded, and `run.py` itself exits with `1` after the remaining seeds.

### The ember backfill

Packages uploaded before the ember edition existed, or after their ember edition failed, lack only its six files. `run.py` regenerates them as a backfill that yields to new mints.

**Scheduling and mint latency.** Each run plans its queue once, at the start: every urgent seed first, then at most `--max-backfill` backfill seeds (default 1, env `COSMICSIG_MAX_BACKFILL`; `0` pauses the backfill), those with the fewest failed runs first (see the retry cap below), then in API order. A token minted while a run is in progress is picked up by the next run. It waits for the rest of the current run (after that run's own new mints, at most `--max-backfill` backfill packages of about an hour each: 61 minutes measured on the production host), plus the timer's 5-minute pause after each run, and then for its own package.

**Backfill modes** (`--backfill-mode`, env `COSMICSIG_BACKFILL_MODE`):

| Mode | What is uploaded |
|------|------------------|
| `ember` (default) | Only the ember edition. The package is still generated in full locally, but before uploading `run.py` fetches the live `metadata/nft_traits.json` and `metadata/assets.json` over SSH and checks that the local render shows the **same orbit**: equal `simulation.masses` (compared as exact JSON numbers), `generation.borda.selected_index` and `generation.borda.retry_count`. If they match, it uploads the five ember media files, then a merged `metadata/assets.json` (the live manifest's other entries kept verbatim, the local `ember_*` entries added or replaced, `generated_at` from the local manifest), and `metadata/ember.json` last. The published main art, spectral files, `generation.json` and `nft_traits.json` are never touched. If the orbit differs, nothing is uploaded: an ERROR names the seed and the differing fields, and the seed counts one failed attempt. |
| `full` | The whole regenerated package replaces the remote one, main art and metadata included, in the upload order above. Its remote `metadata/assets.json` is deleted when the upload starts, so the package reads as incomplete until the new manifest has landed. Use it only to re-render published packages deliberately. |

In both modes, a backfill seed whose generator exits `3` uploads nothing (its core package is already live) and counts one failed attempt.

In `ember` mode, `run.py` also reads and checks the live `nft_traits.json` and `assets.json` before the render, so a live package it cannot use costs no render. If they cannot be read (for example, SSH fails), nothing is rendered and a later run tries again. If one of them is unusable (not valid JSON, a manifest without an `assets` list, or a traits file without the three orbit fields), nothing is rendered or uploaded, and an ERROR (`the live package cannot be used for an ember backfill: …`) says that the package needs repair on the asset host: restore the file, or delete the package's `metadata/assets.json` to have the seed regenerated and uploaded in full. Neither case is an ember attempt, so neither gives the seed up (see the retry cap below).

**Versions of backfilled tokens.** An `ember` backfill leaves the published `metadata/nft_traits.json` untouched, so a backfilled token keeps `pipeline_version` `1.0.0`, while its new `metadata/ember.json` records `crate_version` `1.1.0`. Only its `metadata/assets.json` changes: it gains the `ember_*` entries and a new `generated_at`. Consumers should detect the ember edition from the `ember_*` roles in `metadata/assets.json` (or from `metadata/ember.json`), never from `pipeline_version`.

**Retry cap.** A failed ember attempt is a generator exit `3`, an orbit that differs from the live package's (or regenerated metadata that cannot show its orbit), or a local package whose ember files are incomplete. After `--max-backfill-attempts` failed attempts (default 3, env `COSMICSIG_MAX_BACKFILL_ATTEMPTS`), a seed is left out of the plan, and every run logs the WARNING `0x<seed>: ember backfill given up after N attempts …`.

Any other failure of a backfill seed (the generator exits `1` or is killed by a signal, a timeout, an SSH or upload failure, a live package that cannot be used) is not an ember attempt: it never gives the seed up. It does move the seed back in the queue, because backfill seeds go in order of their failed runs of any kind, fewest first. A seed that always fails is therefore retried only once every other waiting seed has failed as often or is done: it costs one run per pass over the backlog and cannot stall the backfill. Nothing is counted in a run that is shutting down (for example after `systemctl stop`).

The counts live in `backfill_failures.json` in `run.py`'s working directory: `ember_failures` (toward the cap) and `other_failures` (for the order), together with the generator binary's identity (resolved path, size and modification time). Rebuilding the generator resets every count, so a fixed binary retries all the seeds it had given up. To retry one given-up seed with the same binary, delete its entry from `ember_failures`. In a run where a backfill seed fails, `run.py` exits with `1`, so the failure is visible to systemd (the service shows `failed`): until the seed is given up, or for as long as it fails for another reason.

**Duration and disk space.** After the upgrade to the ember edition, every existing token is a backfill seed. At the default of one package per run (about 61 minutes, then the upload and the timer's 5-minute pause, so about 66 minutes each), the 48 existing tokens take about 2 days. Each package grows by about 0.4 GB on the asset host (`ember.png` about 36 MB, the web `ember.mp4` about 84 MB at CRF 22, the HQ `ember.mp4` about 284 MB), so about 19 GB for 48 tokens.

### How the Two Machines Relate

```text
┌─────────────────────────┐       SSH / SCP        ┌──────────────────────────┐
│    Generator Machine    │ ──────────────────────▶│     Remote Server        │
│  (Ubuntu, runs run.py)  │                        │  (any Linux with sshd)   │
│                         │                        │                          │
│  • Rust binary          │  uploads asset packages│  • Stores asset files    │
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
COSMICSIG_ARBITRUM_RPC_URL=https://arb1.arbitrum.io/rpc
COSMICSIG_NFT_CONTRACT=0xbb84Be3500A63581d3F2d5AC3bdF8685AAedad25
COSMICSIG_REMOTE_DIR=/home/frontend/nft-assets/new/cosmicsignature
```

| Variable | What to put here |
|----------|------------------|
| `COSMICSIG_SSH_HOST` | IP address or hostname of the remote server |
| `COSMICSIG_SSH_USER` | SSH user on the remote server (must accept your key) |
| `COSMICSIG_API_URL` | CosmicGame API base URL (CosmicSignature token list), no trailing slash. Preferred, but the script can fall back to Arbitrum if this fails. |
| `COSMICSIG_ARBITRUM_RPC_URL` | Arbitrum One JSON-RPC URL used to verify/fallback seed reads. Defaults to the public Arbitrum RPC if omitted; a private provider is more reliable. |
| `COSMICSIG_NFT_CONTRACT` | Cosmic Signature NFT contract address on Arbitrum. Defaults to the official contract address. |
| `COSMICSIG_REMOTE_DIR` | Absolute path on the remote server where per-seed asset package directories are stored |
| `COSMICSIG_MAX_BACKFILL` | Optional. Packages missing only the ember edition to regenerate per run, after all new ones (default 1; 0 pauses the backfill). Same as `--max-backfill`. |
| `COSMICSIG_BACKFILL_MODE` | Optional. `ember` (default: upload only the ember edition, after the orbit check) or `full` (replace the whole package). Same as `--backfill-mode`; see [The ember backfill](#the-ember-backfill). |
| `COSMICSIG_MAX_BACKFILL_ATTEMPTS` | Optional. Failed ember attempts after which a backfill seed is given up until the generator binary changes (default 3). Same as `--max-backfill-attempts`. |

**5. Run the preflight check**

This tests SSH connectivity, remote write permissions, seed-source reachability (API and/or Arbitrum), that the release generator binary exists and supports the ember edition (its `--help` lists `--no-ember`), and that `ffmpeg` is on `PATH`:

```bash
python3 run.py --preflight
```

The seed-source check passes if either the API or the Arbitrum contract read works. If both API and blockchain reads work during a normal run, `run.py` verifies that they return the same unique seed set. A mismatch is treated as fatal and written to `seed_source_mismatch.json`.

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
- `EnvironmentFile=` — absolute path to the `.env` file you created in step 4. systemd's `%h` (the user's home directory) also works here, for example `%h/CS-Image-Generation/.env`.

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

### Upgrading a deployment

**Pushing to `main` deploys nothing.** The service runs `python3 run.py` and the prebuilt `./target/release/three_body_problem` in its working directory; it never pulls or builds. A new version reaches the generator machine only when you update the checkout and rebuild there. Do it in this order, so that no timer run starts between the pull and the end of the build:

1. Stop the timer, and wait for a running sync to finish. `systemctl is-active cosmicsig-sync.service` prints `activating` while a run is in progress; anything else (`inactive`, or `failed` after a run that exited with `1`) means no run is in progress. (`sudo systemctl stop cosmicsig-sync.service` aborts the run instead: the seed in flight is lost and is generated again later.)

   ```bash
   sudo systemctl stop cosmicsig-sync.timer
   systemctl is-active cosmicsig-sync.service
   ```

2. Update the checkout and rebuild. The build needs network access the first time, because the ember edition added the `libm` crate. If you edited the tracked `cosmicsig-sync.service` in setup step 7, `git pull` refuses to overwrite it: run `git checkout -- cosmicsig-sync.service cosmicsig-sync.timer` first (the installed copies in `/etc/systemd/system` keep your values) and re-apply your values in step 6.

   ```bash
   git pull --ff-only
   cargo build --release --locked
   ```

3. Confirm that the new binary has the ember edition:

   ```bash
   ./target/release/three_body_problem --help | grep -- --no-ember
   ```

4. Optionally, check the host's FFmpeg colour conversion for the ember videos. The test is ignored by default because it needs FFmpeg:

   ```bash
   cargo test --release --lib srgb_variants_round_trip -- --ignored
   ```

5. Run the preflight and a dry run:

   ```bash
   python3 run.py --preflight
   python3 run.py --dry-run
   ```

   For the upgrade to the ember edition, expect every existing token to be missing only the ember edition, for example `Found 48 seeds with incomplete asset packages (out of 48 total): 0 new or incomplete, 48 missing only the ember edition`, followed by `This run generates 1 of them; 47 ember backfill seeds wait for later runs`.

6. If `cosmicsig-sync.service` or `cosmicsig-sync.timer` changed, reinstall them (the upgrade to the ember edition changes both: the service's run ceiling is now `TimeoutStartSec`, and the timer now waits 5 minutes after each run finishes). Re-apply your `User=`, `WorkingDirectory=` and `EnvironmentFile=` values first.

   ```bash
   sudo cp cosmicsig-sync.service cosmicsig-sync.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

7. Start the timer:

   ```bash
   sudo systemctl start cosmicsig-sync.timer
   ```

8. Check the first backfilled package, about an hour later. The log shows `same orbit as the live package; uploading only its ember edition`, then a line of the form `OK  seed=0x…  (total …)  ember edition uploaded` (search for `ember edition uploaded`). On the asset host, the package's `metadata/assets.json` lists the five `ember_*` roles, `metadata/ember.json` exists, and `images/source/master.png` is unchanged: its `sha256sum` equals the `sha256` of the manifest's `source_master` entry.

The ember backfill then runs on its own for about 2 days (see [The ember backfill](#the-ember-backfill)). Make sure the asset host has about 20 GB free for it. If `run.py` is ever updated without rebuilding the binary, it logs `… predates the ember edition … rebuild the generator …` on every run, keeps uploading new mints without the ember edition, and pauses the backfill until the rebuild.

## Batch Testing

`run-test-images.py` continuously generates images with random seeds, useful for visual QA and stress testing. It keeps 3 concurrent jobs running and logs progress to `run.log`. Each finished render is scored with image-space aesthetic metrics (ink coverage, colorfulness, hue entropy, luminance spread — computed from a small ffmpeg-decoded proxy frame); low scores are flagged in the log.

```bash
python3 run-test-images.py
```

Press Ctrl+C to stop gracefully after the current jobs finish. Output lands in `output/<seed>/`.

## Contact Sheets and the Golden Gallery

`contact_sheet.py` renders a batch of seeds at preview quality (with `--no-ember`, since it only tiles the main master), scores each still with the same aesthetic metrics, and tiles everything into a single PNG for fast visual curation:

```bash
python3 contact_sheet.py --count 24        # random seeds -> contact_sheet.png
python3 contact_sheet.py --golden          # fixed seed set -> golden_gallery.png
```

The golden gallery re-renders the fixed seed list in [`ci/golden_seeds.txt`](ci/golden_seeds.txt); regenerate it after any tuning change and compare against the previous gallery side by side to catch look regressions. With `just` installed: `just contact-sheet` / `just golden-gallery`.

## Reference Image Verification

The `ci/` directory contains tooling for deterministic regression testing. A reference image is generated with a fixed seed and parameters, then future builds are verified against it by SHA256 hash.

Generate the reference baseline:

```bash
cd ci/reference
./generate_reference.sh
```

This creates `baseline_512x288.png` and a companion `.json` with the parameters and hash. From the **repository root**, verify a test image against the default baseline:

```bash
python3 ci/verify_reference.py output/test/images/source/master.png
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
| [Ruff](https://docs.astral.sh/ruff/) | Lints and formats the repository Python scripts (replaces a pile of flake8/isort/black-style checks in one fast binary). |
| [Mypy](https://mypy.readthedocs.io/) | Strict type-checking for the same files. |

Configuration is entirely in [`pyproject.toml`](pyproject.toml): Ruff target Python 3.10, line length **100** (same as Rust), rule sets **E, F, I, UP, B, SIM, PTH, RUF**; Mypy **`strict = true`** on `_utils.py`, `contact_sheet.py`, `run.py`, `run-test-images.py`, `ci/verify_reference.py`, and `tests/python/test_run.py`.

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

**Unit tests** for `run.py` live in [`tests/python/test_run.py`](tests/python/test_run.py). They use only the standard library (`unittest`) and no network: fake generator, `ssh` and `scp` scripts on `PATH` run against a temporary "remote" directory. From the repository root:

```bash
python3 -m unittest discover -s tests/python -v
```

**Git hook:** If you use [`.githooks/pre-commit`](.githooks/pre-commit) (`git config core.hooksPath .githooks`), each commit runs **Rust** fmt + Clippy, then **Python** Ruff + Mypy. The hook calls `ruff` and `mypy` on your `PATH`, so activate the venv (above) in terminals where you commit, or install the tools into an environment that is always on your `PATH`.

**CI:** The workflow's Python job uses Ubuntu, **Python 3.12**, `pip install ".[dev]"`, then the same three commands as `just py-check`, and then the unit tests. Mypy is configured with `python_version = "3.10"` in `pyproject.toml`, so types stay compatible with the stated minimum interpreter.

## Algorithm

For a detailed description of the spectral pipeline (SPD buffer, accumulation, gallery, and spectral sweep video), see [docs/spectral-algorithm.md](docs/spectral-algorithm.md).

For the ember edition (fluid, ink, look, spectral shading, the determinism contract and its certificate), see [docs/ember-edition.md](docs/ember-edition.md).

## License

This repository does not ship an SPDX `LICENSE` file. Before redistributing or pinning a build to IPFS for others to reuse, add a license you are comfortable with (for example MIT or Apache-2.0) so terms are explicit.

## Project Layout

```text
src/main.rs              CLI entry point
src/app.rs               Pipeline orchestration
src/sim.rs               Physics simulation and selection
src/render/              Rendering, tonemapping, visual profiles, video
src/ember/               The ember edition (fluid, ink, spectral sumi/cinnabar, certificate)
src/post_effects/        Active bloom, prism, and spectral-sweep post effects
src/spectrum.rs          Spectral conversion
src/spectrum_simd.rs     SIMD spectral fast paths
src/oklab.rs             OKLab utilities
_utils.py                Shared helpers imported by `run.py` / `run-test-images.py`
run.py                   Automated generation and upload
run-test-images.py       Batch random-seed test runner
tests/                   Rust integration tests; tests/python/ holds run.py's unit tests
contact_sheet.py         Visual contact sheet and golden gallery generator
pyproject.toml           Python dev tooling (Ruff, Mypy) and optional `[dev]` deps
justfile                 `just` recipes (`check`, `test`, `py-check`, …)
ci/                      Reference-image verification tooling
docs/                    Long-form algorithm documentation
.cargo/config.toml       Native CPU flags and SIMD features
cosmicsig-sync.service   Systemd service unit for run.py
cosmicsig-sync.timer     Systemd timer (5 minutes after each run)
.env.example             Template for deployment secrets
```
