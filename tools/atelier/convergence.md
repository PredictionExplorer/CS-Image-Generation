# Comparing fine details without changing the film

`examples/atelier_convergence.rs` renders an aligned crop through the actual
Light, Engraving or Eclipse linear renderer. It preserves the original camera's pixel
coordinates, source clock, shutter endpoints and physical reconstruction width.
The default only prints a plan; `--run` performs the comparison on the CPU.

Run expensive comparisons on the rendering server. Example:

```sh
atelier_convergence --config configs/05-engraving.json \
  --orbit /home/user/tidal-silk/results/0xb7f327f9f722.orbit \
  --frame 1670 --crop 2120,800,192,192 --threads 16 \
  --output qa-convergence-fast.json --png-dir qa-convergence-fast --run
```

The default reference doubles spatial resolution and temporal cells. Set
`--spatial-factor 1` or `--temporal-factor 1` to isolate the other change.
`--double-light-source-grid` also doubles both Light source-grid dimensions.
Eclipse keeps the same world-space Gaussian footprint when spatial resolution
changes, along with the same curve identities and physical radii.
All reference spatial samples are averaged in linear light before comparison.
PNG previews use the native exposure and tone curve after downsampling; crop
bloom and film encoding are deliberately excluded from this comparison.

The JSON includes source/recipe identity, both pass plans, exact shutter times,
maximum/mean/RMS channel differences and foreground-weighted luminance metrics.
Read these together with the actual images. A small error in one crop supports
that specific passage and location; it does not certify the whole film. Changes
in source-grid density can affect Light's nonlinear artistic gain mapping, so
its returned linear-RGB comparison includes that sensitivity.
