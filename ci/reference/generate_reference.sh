#!/bin/bash
set -e

# Reference image generation script
# This script generates deterministic reference images for CI testing

echo "Generating reference images for CI..."

# Change to project root
cd "$(dirname "$0")/../.."

# Build the project in release mode
echo "Building project..."
cargo build --release

# Fixed parameters for reference image
SEED="0x46205528"
WIDTH=512
HEIGHT=288
RESOLUTION="${WIDTH}x${HEIGHT}"
NUM_STEPS=100000
DRIFT_MODE="brownian"

# Generate the reference image
echo "Generating reference image with parameters:"
echo "  Seed: $SEED"
echo "  Resolution: $RESOLUTION"
echo "  Steps: $NUM_STEPS"
echo "  Drift: $DRIFT_MODE"

./target/release/three_body_problem \
    --seed "$SEED" \
    --resolution "$RESOLUTION" \
    --steps "$NUM_STEPS" \
    --drift "$DRIFT_MODE" \
    --output "baseline"

# Move the generated files to reference directory
mv "pics/baseline.png" "ci/reference/baseline_${WIDTH}x${HEIGHT}.png"

# Generate JSON metadata
cat > "ci/reference/baseline_${WIDTH}x${HEIGHT}.json" << EOF
{
    "seed": "$SEED",
    "width": $WIDTH,
    "height": $HEIGHT,
    "num_steps": $NUM_STEPS,
    "drift_mode": "$DRIFT_MODE",
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "sha256": "$(shasum -a 256 ci/reference/baseline_${WIDTH}x${HEIGHT}.png | cut -d' ' -f1)"
}
EOF

echo "Reference image generated successfully!"
echo "Location: ci/reference/baseline_${WIDTH}x${HEIGHT}.png"
echo "Metadata: ci/reference/baseline_${WIDTH}x${HEIGHT}.json" 
