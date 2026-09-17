# The Shell That Remembers

Selected recording: `0xb7f327f9f722.orbit`, SHA-256
`a9864c974e31dce21aa80ff9edd6524b9d4f9141c5cf50de26b6fdc739cc1652`.

| File | Purpose |
| --- | --- |
| `shell.json` | Frozen triangle mapping and ribbed shell geometry |
| `studio.json` | 1920 × 1600 hero photograph, 256 samples |
| `studio-film.json` | 960 × 800 motion study, 32 samples |
| `studio-still-4k.json` | 3840 × 3200 full-history still, 256 samples |
| `studio-film-4k.json` | 3840 × 3200 full-history film, 64 samples |
| `studio-second.json` | A view at the end of the camera movement |
| `studio-detail.json` | Close examination of the growth ribs |
| `source-times.json` | 120 monotone source times for uniform construction progress |

The selected material is ivory outside and celadon inside. The optional
thin-film coating was compared and left disabled. All views use the same
geometry, lighting, and material; the film retains the full mesh detail.

Use the shared `render.py` adapter with `--view front` and a named studio recipe.
Keep `build.json` next to `mesh.ply`: it supplies the verified shell-wall layout
for the interior glaze. Moving or altering that layout invalidates reuse.

See [the full guide](../../../docs/remembering-shell.md) for generation,
rendering, fixed ground placement, retiming, resume checks, and audit limits.
