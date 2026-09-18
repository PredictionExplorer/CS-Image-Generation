# Into depth: study notes

## What is being tested

The entire existing Estuary trajectory is already present in each finished
painting. This experiment asks what a small camera orbit can reveal after
that painting settles. The original simulation and its flat rendering are
retained as the reference.

The first screen compares eighteen treatments of BC53: an illuminated flat
control, shallow and deep relief, enamel, raking light, buried pigment layers,
hybrids, and six changes to light and surface finish. Six closer views follow.
Six clearer glass interpretations then test whether older currents become
legible below the completed surface.

## Decisions from actual renders

- Broad lighting and a rough blue surface produced a gray veil. A quieter
  environment and a key light from the camera side restored the dark blue.
- Modest relief preserved the drawing but added little at a distance. The
  deeper **Folded Tide** made the right-hand wave read as folded material while
  preserving the quiet left side.
- Repeated dark, transmitting pigment sheets hid the earlier drawings. A
  separate glass lobe revealed them. At high transmission the drawing became
  congested; **Porcelain over Memory** retained a stronger foreground.
- Native 4K inspection revealed mesh-scale scalloping, particularly in the
  deeply raised ivory banks. A 0.35 mm geometric smoothing scale addresses it
  without blurring the color image or changing the simulation.

These are subjective art-direction decisions. A more dimensional image is not
automatically a better composition, and the original flat painting remains an
equally valid reading of the motion.

## Reproduction and comparison

The numbered `recipes/`, `recipes-refinement/` and `recipes-glaze/` directories
record the thirty screening studies. `recipes-finish/`,
`recipes-memory-finish/` and `recipes-orbit-selected/` contain the selected
finishing treatments. Every experiment archives its actual renderer and
material code alongside the submitted and resolved recipes; those archived
inputs, rather than the latest checkout, reproduce an earlier screen.

The three source seeds are `0xbc53af1cd380`, `0x808861c25b6c`, and
`0xb7f327f9f722`. Historical layers for BC53 use real states at 35%, 65%, and
100%, authenticated by a replay whose final state exactly matches the
original file. The depth is an artistic interpretation of those states.

Camera studies freeze every surface and every pigment field. Their changing
highlights, shadows and relative positions arise from a moving camera under
fixed lights. The longer formation film joins the complete original movie to
that examination with an explicit editorial dissolve after formation ends.
