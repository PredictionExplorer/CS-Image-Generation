# The Estuary, in layers

This study develops color composition and overlapping mobile paint. It uses the
same ten complete recordings pinned in `recipes/ten-seeds.json`; the original
Convergence films remain the comparison baseline.

## Controlled studies

The frozen `layers-proof-01` and `layers-proof-02` releases retain the runtime used
by each experiment. Every simulation reaches the final recorded state. The
screening material grid is 1024 × 768, so these images assess broad composition
and color, not native fine detail.

| Study | Cases | Main question |
| --- | ---: | --- |
| Layer speed and exchange | 18 | Lower speeds 0.92 / 0.82 / 0.70; exchange 0 / 0.55; B7, 8088, CEDD |
| Composed single-layer control | 3 | What changes come from the new palette alone? |
| Optical depth | 21 | Six variations plus an unchanged-state reference for each seed |
| Longer wet contact | 6 | Drying 1 / 2 with exchange 1.5, compared with drying 4.5 / exchange 0.55 |
| Matched optical references | 3 | Same physical parents under the selected optical settings |
| Pigment load hierarchy | 6 | Explicit role weights under the original and longer wet periods |

## Selection

The chosen recipe is `recipes/layered-five.json`:

- Composed palette, five pigments; initial load multipliers
  `[2.2, 0.9, 0.5, 0.6, 0.4]` for dominant, support, accent, anchor, and bridge.
- Two moving layers, lower speed 0.82, wet-contact exchange 0.55, drying 4.5.
- Existing within-layer diffusion 0.00002 and canonical mass restoration every
  12 steps across both layers.
- Glazed optical finish, minimum/maximum mass ratios 0.35 / 2.5, optical layer
  scale 24, exposure 0.95, displayed height scale 1.4.
- Seed-derived Palette Night ground, fixed across the film.

The slower 0.70 layer mainly widened parallel outer strokes and crowded some
negative spaces. A 0.82 ratio retained distinct overlaps without the stronger
doubling. Layering retained more chromatic separation than the matched composed
single-layer controls, particularly for CEDD.

Greater optical depth gave a modest improvement. Raising the minimum optical
mass to 0.9 effectively flattened almost all thickness variation. Longer wetness
and stronger exchange did not give a clear visual benefit; some warm bands became
softer. These findings favor restrained contact mixing.

Color roles initially had nearly equal quantities because the inherited pool
loads were independent of the palette. Anchor plus bridge contained roughly
41–43% of the pigment in the three screening seeds. Explicit loading reduced
that to roughly 21–25%, giving the dominant color about 43–50%. All pigments
continued to travel substantially. These diagnostics describe material behavior;
they are not automatic measures of artistic quality.

## Native qualification

B7 was independently rendered at 4096 × 3072 with a 3840 × 2880 RGB16 poster.
It covers all 1,000,000 recorded states, using 7,200 canonical and 10,332 actual
transport steps. The complete still pipeline took approximately 230 seconds on
the experiment server. One renderer peaked at 6,600 MiB of GPU memory, measured
at one-second intervals. This supports a bounded two-worker film queue on the
16-GiB device; it is not a guarantee for different resolutions or extra views.

The final pigment-budget error was below 3e-8 relative, and the image balance was
close to centered. Native detail confirms clean silhouettes and real ordered
color overlap. The interior remains smooth; this release does not add a flow-map
transport method, microscopic pigment texture, or measured pigment spectra.

## Film and review contract

Each new film contains 1,801 formation frames, 120 hold frames, and 240 camera
frames: 2,161 total at 24 fps, approximately 90.04 seconds. The 75-second formation
interval spans the entire recorded trajectory; the separate orbital warmup is
excluded. Output is 1920 × 1440, shaded from the full native material grid with
2× linear-light antialiasing. Every archive also contains a native final poster.

Final publication requires complete decode evidence and source/material/media
identity checks. B7's production film must reproduce its native qualification
state and final poster exactly. The review gallery exposes each film directly,
supports continuous play, and pairs the earlier painting by the same source and
projection. Frame cadence and presentation do not redefine simulation time.
