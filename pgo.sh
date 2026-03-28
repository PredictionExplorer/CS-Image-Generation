#!/usr/bin/env bash
set -euo pipefail

# Profile-Guided Optimization (PGO) build script.
#
# Runs a representative workload to collect branch/call profiling data,
# then rebuilds with that profile for 10-20% additional throughput on
# branch-heavy simulation and render code.
#
# Requirements: rustup component add llvm-tools-preview
# Usage:        ./pgo.sh [--seed 0xABCDEF]

SEED="${1:-0x100033}"
PROFDATA_DIR="target/pgo-profiles"
MERGED_PROF="target/pgo-merged.profdata"

echo "=== PGO Step 1/3: Instrumented build ==="
RUSTFLAGS="-Cprofile-generate=${PROFDATA_DIR}" \
    cargo build --release --target "$(rustc -vV | sed -n 's/host: //p')"

echo "=== PGO Step 2/3: Collecting profile data (seed=${SEED}) ==="
rm -rf "${PROFDATA_DIR}"
mkdir -p "${PROFDATA_DIR}"

# Run a representative workload: fewer sims/steps than production but
# exercises all hot paths (simulation, render, effects, video encode).
./target/release/three_body_problem \
    --seed "${SEED}" \
    --sims 500 \
    --steps 50000 \
    --resolution 640x360 \
    --fast-encode \
    --no-extras \
    --no-museum-prints \
    --no-cinematic-zoom \
    --output pgo-profile-run

# Merge raw profiles
LLVM_PROFDATA=$(find "$(rustc --print sysroot)" -name llvm-profdata -type f 2>/dev/null | head -1)
if [ -z "${LLVM_PROFDATA}" ]; then
    echo "ERROR: llvm-profdata not found. Run: rustup component add llvm-tools-preview"
    exit 1
fi
"${LLVM_PROFDATA}" merge -o "${MERGED_PROF}" "${PROFDATA_DIR}"

echo "=== PGO Step 3/3: Optimized build ==="
RUSTFLAGS="-Cprofile-use=${PWD}/${MERGED_PROF} -Cllvm-args=-pgo-warn-missing-function" \
    cargo build --release --target "$(rustc -vV | sed -n 's/host: //p')"

# Cleanup profile artifacts
rm -rf "${PROFDATA_DIR}" output/pgo-profile-run

echo ""
echo "=== PGO build complete ==="
echo "Binary: ./target/release/three_body_problem"
echo "Expected speedup: 10-20% over standard release build"
