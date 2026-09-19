# Spectral blend studies

`blend_study.py` compares appearances of a completed, verified painting. Every
variant reads the same immutable pigment arrays and spectral coefficients.
Source coverage, physical-state identity, exact controls, images and renderer
source are bound into the archive. The original case is never modified.

The tool reuses the appearance study's locking, media capture and verification
pipeline. Existing appearance archives and public functions remain supported.

| Variant | Treatment | What remains fixed |
| --- | --- | --- |
| `ordered` | Parent layered glaze, unchanged | Everything; this is the control |
| `intimate` | All pigment interpreted as one intimate spectral mixture | Actual pigment amounts, material fields, spectra and lighting |
| `thin-glaze` | Ordered layers; optical mass floor 0.10, optical depth scale 16 | Transport, layer order, palette and lighting |
| `ink` | Optical mass floor 0.02, optical depth scale 4, revealing concentration as tone | Real-mass silhouette, layer order, pigment backing and lighting |
| `relief` | Height scale 4, stronger retained relief and grazing light | Archived geometry, pigment and spectra |

All views retain the parent's still camera and visible ground. A background must
be an exact archived background, or match a named background for older proof
cases. An arbitrary color is never silently changed to a preset. The parent
must have a spectral `glazed` layered view.

These treatments do not add physical color mixing to the simulation. The
intimate interpretation uses the existing finite-layer spectral mixture
operator; the ordered interpretation preserves actual upper/lower phase
amounts and the transported mixedness. The thinner glaze changes displayed
optical thickness, and raised paint changes displayed height. Neither adds
microscopic brushwork or reconstructs strands lost to a coarse simulation.

The `ink` treatment is intended especially for monochrome studies: a high
optical mass floor can make a single pigment nearly opaque throughout the shape,
concealing internal concentration changes. Lower optical depth exposes those
changes as tone. Thin passages transmit light to the existing pigment backing;
they do not fake transparency into the exterior ground. The actual mass cutoff
and crisp boundary remain unchanged.

```sh
python -m tools.estuary_confluence.blend_study \
  --case /path/to/verified-painting \
  --output /path/to/new-blend-study \
  --looks ordered intimate thin-glaze relief \
  --resolution 1280 960
```

Programmatic entry points are `presentation(parent, name)`,
`render_study(case, output, ...)`, and `verify_study(folder)`. The separate
`film_recipe(parent, name)` returns a validated recipe for the normal rendering
pipeline; it preserves simulation controls and film cadence. Any resulting film
must still be independently verified, including its final material identity.

## Initial screening

Twelve complete-material studies were rendered from the existing CEDD, B7 and
8088 weighted proof cases. Those parents used a 1024 × 768 material grid. The
1280 × 960 output assesses broad color and tone; it cannot establish native
strand detail. All three source histories were already complete before the
optical comparisons.

The intimate mixture creates visibly greener intersections in CEDD and deeper
teal passages in 8088, while B7 changes less. Thin glaze creates brighter,
pearl-like transitions, with a risk of looking too glossy. Increased relief
produces only a small change at the retained 8-degree camera angle. These are
visual observations, not automatic aesthetic scores. None changes the painting's
overall silhouette, so color-count and trajectory-driven shape experiments
remain the stronger way to broaden the collection.

Three later monochrome proofs compared the same material under the original
glaze and `ink`. In 6210, the lower optical depth produced pale jade passages
with emerald centers. CEDD shifted toward saffron and ivory with visible
concentration folds. B7 became nearly white, so the same treatment is not a
universal improvement. These checks retained the real-mass boundary and the
original pigment backing, without introducing background-colored haze.
