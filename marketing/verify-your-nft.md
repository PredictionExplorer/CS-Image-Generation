# Verify Your NFT

## Reproduce Any Cosmic Signature Artwork from Its On-Chain Seed

Every Cosmic Signature NFT's art is **deterministic** -- the same seed always produces the exact same image and video, down to every pixel. Because the entire codebase is open source under CC0 1.0 (public domain), anyone can independently verify this.

This guide walks you through reproducing an NFT's artwork on your own machine.

---

## Prerequisites

You need three things installed:

| Requirement | Version | Purpose |
|-------------|---------|---------|
| **Rust** | 1.94+ | Compiles the renderer |
| **FFmpeg** | Any recent version | Encodes the 30-second video |
| **Git** | Any recent version | Clones the source code |

### macOS

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install ffmpeg git

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y build-essential ffmpeg python3 git curl

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Windows

Install [Rust](https://rustup.rs/), [Git](https://git-scm.com/), and [FFmpeg](https://ffmpeg.org/download.html). Ensure all three are on your PATH.

---

## Step 1: Clone and Build

```bash
git clone <repo-url> CS-Image-Generation
cd CS-Image-Generation
cargo build --release
```

The first build takes a few minutes as Rust compiles with LTO and native CPU optimizations. Subsequent builds are fast.

> **Note:** The build is optimized for your specific CPU (`target-cpu=native`). The binary may not run on machines with older processors. Always build on the machine you plan to run on.

---

## Step 2: Find Your NFT's Seed

Your NFT's seed is stored on-chain and is visible in the token metadata. It is a hex string like `0x46205528`. You can find it:

- On the Cosmic Signature gallery at **cosmicsignature.art**
- In the NFT's metadata on any Arbitrum block explorer
- Via the CosmicGame API

---

## Step 3: Generate the Art

```bash
./target/release/three_body_problem --seed 0xYOUR_SEED_HERE
```

This will:
1. Search 100,000 random orbits for the best candidate (~2--5 minutes)
2. Simulate the winning orbit at full resolution
3. Render the spectral image and 30-second video
4. Write output to `pics/output.png` and `vids/output.mp4`

To use the NFT's seed as the filename:

```bash
./target/release/three_body_problem --seed 0x46205528 --output 0x46205528
# Creates: pics/0x46205528.png and vids/0x46205528.mp4
```

### Optional: Higher Resolution

```bash
./target/release/three_body_problem --seed 0x46205528 --resolution 3840x2160
```

### Optional: Faster Encoding (Lower Quality Video)

```bash
./target/release/three_body_problem --seed 0x46205528 --fast-encode
```

---

## Step 4: Verify the Output

### Visual Comparison

Open `pics/output.png` and compare it with the NFT image displayed on cosmicsignature.art. They should be pixel-identical at the same resolution.

### Hash Verification

For cryptographic certainty, compare SHA-256 hashes:

```bash
# On macOS
shasum -a 256 pics/output.png

# On Linux
sha256sum pics/output.png
```

The hash should match the reference hash for that seed. The CI system uses this exact method to verify determinism across builds.

### Reproducibility Log

Every generation run writes a `generation_log.json` file containing:
- The seed used
- All resolved effect parameters (including which were randomized)
- The selected orbit index and Borda score
- Timing information

This log provides a complete audit trail for the generation.

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `0x100033` | Hex seed (with or without `0x` prefix) |
| `-o, --output` | `output` | Base name for output files |
| `--sims` | `100000` | Number of orbits in the Borda search |
| `--steps` | `1000000` | Simulation steps per orbit |
| `-r, --resolution` | `1920x1080` | Output resolution as `WIDTHxHEIGHT` |
| `--drift` | `elliptical` | Camera drift mode: `none`, `linear`, `brownian`, `elliptical` |
| `--fast-encode` | off | Use faster (lower quality) video encoding |
| `--log-level` | `info` | Log verbosity: `error`, `warn`, `info`, `debug`, `trace` |

---

## Why This Matters

**Provable art.** Unlike most NFT projects where the art is created separately and linked to a token, Cosmic Signature's art is a mathematical function of the on-chain seed. The relationship between your token and its artwork is not a database entry -- it is a deterministic computation that anyone can independently verify.

**No trust required.** You don't need to trust the project team, a centralized server, or a database. The code is public domain (CC0 1.0). The math is reproducible. The output is verifiable.

**Permanent.** Even if every server went offline, anyone with the source code and your seed could regenerate your NFT's exact artwork. The art lives in the algorithm, not on a server.

---

*The complete source code is released under CC0 1.0 Universal (public domain).*

*cosmicsignature.com | cosmicsignature.art*
