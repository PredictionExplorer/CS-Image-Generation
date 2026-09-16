# Selected porcelain study

These recipes belong to the **Sculpture of Absence** experiment, *The Remaining
Form*. They use recording `0xb7f327f9f722.orbit`, SHA-256
`a9864c974e31dce21aa80ff9edd6524b9d4f9141c5cf50de26b6fdc739cc1652`.

| Recipe | Purpose |
| --- | --- |
| `sculpture.json` | Final geometry, 384 nodes on the longest axis |
| `sculpture-film.json` | The same continuous field, 176-node motion preview |
| `studio.json` | 1920 × 1600 porcelain still, 256 Cycles samples |
| `studio-film.json` | 768 × 640 motion preview, 32 Cycles samples |

The field uses a fixed ellipsoid blank, broad transported cutters, cumulative
dose, and a small declared rounding of the intersection. Neither the planar
reveal nor the rounded aperture is enabled. The original physical trajectories
are unchanged. A rigid -90° X rotation presents the cavity upward; the camera
looks through the central passage. The ground remains fixed for every frame.

Both geometry recipes explicitly use continuous field-gradient shading normals.
The renderer preserves these custom normals so the light follows the carving
field rather than the tessellation pattern.

The geometry recipes differ only in sampling resolution. The photographic
recipes differ only in image dimensions and sample count. Lower motion resolution
is a preview tradeoff, not a different artwork or a print master. Both use
explicit denoising and archive the scene-linear EXR as well as the display PNG.

For the selected short film, use 96 excavation frames, 48 camera-examination
frames at -28°, 24 fps, and the default one-second opening/two-second closing
holds. This gives nine seconds and covers the complete recorded source interval.
The frames are instantaneous samples; changing topology is not vertex-interpolated.

See [the main guide](../../../docs/remaining-form.md) for commands, provenance,
numerical conditioning, and the limits of the checks. Artifact receipts record
the actual executable, scene adapter, renderer runtime, and resolved settings.
