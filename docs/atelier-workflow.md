# Atelier rendering and review

`orbital_atelier` reads an existing `.orbit` cache from the Tidal Silk pipeline.
The selected source motion is shared by every study. Geometry is generated in
Rust and all rendering runs on CPUs. FFmpeg supplies the final movie encoders.

The current study status and curated recipes are recorded in
[six-art-studies.md](six-art-studies.md). A preset is a starting point for look
development; a selected recipe is the record of a curated result.

## Freeze the renderer

Build on the render machine, then copy the executable to a unique filename
before starting a render. Each frame manifest records the executable's SHA256.
Do not replace that file during a render or resume with an unrelated binary.

```sh
cargo build --locked --release --bin orbital_atelier
mkdir -p bin
cp target/release/orbital_atelier bin/atelier-selected
bin/atelier-selected config --preset loom --output recipe.json
```

## Render a proof or full sequence

```sh
bin/atelier-selected --threads 28 render \
  --orbit source.orbit --config recipe.json --output proof --frame 900

bin/atelier-selected --threads 28 render \
  --orbit source.orbit --config recipe.json --output full-frames
```

Existing compatible, verified frames can be resumed. Configuration differences
are rejected instead of silently mixing outputs. `--overwrite` deliberately
replaces an existing output; use distinct directories for different experiments.

For parallel processes, assign disjoint ranges using `--start` and exclusive
`--end`. For the standard 1,802-frame film, four example ranges are 0..451,
451..901, 901..1352, and 1352..1802. All chunks must use the same frozen executable
and recipe. Worker counts may differ.

To render all ranges and finish one film with a single resumable command:

```sh
python3 tools/atelier/render_study.py \
  --orbit source.orbit --config recipe.json --output 02-loom \
  --executable bin/atelier-selected --workers 112 --chunks 8 --poster-frame 900
```

This preserves a copy of the requested recipe before starting the workers.
Keep the worker/range allocation and frozen renderer consistent when resuming.

## Finish and verify

The helper waits for complete manifests, verifies and assembles the exact full
frame union, encodes both movie versions, decodes every movie frame, and copies
the chosen poster and provenance into the study folder.

```sh
python3 tools/atelier/finish_study.py \
  --input chunk-0 --input chunk-1 --input chunk-2 --input chunk-3 \
  --output 02-loom --executable bin/atelier-selected \
  --poster-frame 900 --wait
```

The original chunk frames and receipts remain intact. Assembly uses hard links
when possible and verified copies across filesystems. The final film includes
both source endpoints. Spatial and temporal samples are integrated in linear
light before the display finish is applied.

## Build the review collection

Use the six folder names listed in `tools/atelier/build_review.py`. A finished
folder contains `web.mp4`, `master.mp4`, their JSON sidecars, `poster.png`,
`recipe.json`, `render.json`, `assembly.json`, and `verification.json`.

```sh
python3 tools/atelier/build_review.py /path/to/collection
python3 tools/atelier/serve_review.py /path/to/collection --port 8767
```

Open `http://127.0.0.1:8767/`. It offers full-screen playback, synchronized
comparison, slow motion, and private timestamped review notes. An optional
`reference/normal.mp4` adds the original presentation to the comparison menu.
Only verified complete studies count toward the six-film collection.

## Reproducibility

Fixed source samples, stable geometry ordering and deterministic per-pixel
work make results independent of render scheduling. Each artifact records
source, executable, configuration and pixel hashes. The intended guarantee
between supported architectures is visually indistinguishable output; strict
cross-architecture pixel identity is not asserted. Preserve the frozen source
cache and the complete recipe for reproduction.
