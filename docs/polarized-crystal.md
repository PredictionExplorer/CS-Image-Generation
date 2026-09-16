# Polarized Crystal

## Artistic intention

A quiet transparent optical window contains broad, jewel-colored stress bands
and moving dark extinction seams. Every body loads the same material. A short
memory allows an encounter to gather, compress, and slowly release. Preserve
substantial dark space, restrained reflections, and a clear composition at both
thumbnail and native image size.

The first experiment uses the unchanged recorded orbit of `0xb7f327f9f722` for
direct comparison with the six art studies. Additional recorded orbits provide
a check that the material is useful beyond a single curated source.

## Model and boundaries

1. `OrbitSeries` supplies the raw three-dimensional motion after one common
   translation and uniform scale. No display drift, phase-space projection,
   retiming, or altered integration enters the source.
2. A fixed orthonormal pair of source axes maps positions into a two-dimensional
   sheet. Original three-dimensional pair separation controls a bounded load
   response. Each pair applies equal and opposite smooth loads. Optional elongated
   footprints follow each body's projected physical tangent; out-of-plane
   directions smoothly become circular footprints.
3. A linear plane-stress finite-element system computes displacement and stress.
   The rectangular sheet has a fixed remote perimeter. Its spatial resolution,
   Poisson ratio, force footprint, and solver tolerance are explicit controls.
4. Memory is a finite, deterministic average of past forcing. The compact kernel
   is `exp(-3u)(1-u)^2`, where `u` is age divided by the configured memory window.
   Both its value and slope vanish at the oldest endpoint. Fixed Gauss-Legendre
   quadrature makes independent frames reproducible. The average is
   solved using linearity of elasticity. It is an artistic relaxation model,
   not a claim to simulate a particular viscoelastic substance. Available history
   is truncated at source time zero; no unrecorded motion is invented.
5. The difference and orientation of the principal stresses determine ideal
   crossed-linear-polarizer transmission. Each of the existing 64 wavelength
   bins has its own optical phase delay. A converged one-dimensional lookup
   accelerates the spectral integral; out-of-table values use the direct model.
   Optional fixed Gaussian spectral illumination is normalized to unit incident
   luminance. It changes the light entering the specimen, not the calculated RGB
   hues after transmission.
6. A shallow, polished optical window introduces thickness, absorption, and
   restrained studio reflections. Exact analytic dome normals and dielectric
   Fresnel weights sample a fixed softbox environment. This studio approximation
   does not trace external scene geometry, cast shadows, or multiple scattering.
   One Snell refraction maps a viewing ray through the dome onto the internal
   stress plane. A local slab correction adjusts optical path length; this is
   not full multi-interface polarized transport. **This visible silhouette is not the mechanical
   boundary.** The mechanics assume a uniform sheet continuing beyond it; the
   thickness profile is optical only. This is a declared artistic approximation,
   not a free-standing crystal solved with traction-free surfaces.
7. Optional uniform prestress represents tension held by the remote supports.
   The finite-element solution gives increments about that reference; its
   diagnostics report the incremental field. Total stress, including the fixed
   baseline, enters the optical calculation.

The force mapping, optical sensitivity, outline, and material memory are artistic
choices. The elastic and polarization calculations within those choices retain
their mathematical meanings. No per-body RGB halos, automatic frame exposure,
or sensitivity changes manufacture an encounter.

## Source references

- [MIT, mechanics laboratory introduction to photoelasticity](https://ocw.mit.edu/courses/2-002-mechanics-and-materials-ii-spring-2004/09e8137971e754b1b7290a33a100bb05_lab_4_s04.pdf).
- [MIT, Jones matrices and photoelastic transmission](https://ocw.mit.edu/courses/2-71-optics-spring-2014/c683c04243df558f6048cc11924ff44f_MIT2_71S14_lec22_notes.pdf).

## Reproducibility and code structure

- `src/atelier/crystal/field.rs`: deterministic mechanics and force-memory model.
- `src/atelier/crystal/optics.rs`: spectral polarization with a direct reference.
- `src/atelier/crystal/surface.rs`: analytic glass shape, studio reflection, and viewing refraction.
- `src/atelier/crystal/presentation.rs`: bounded full-precision cache of fixed surface samples.
- `src/atelier/crystal/mod.rs`: recipe, fixed composition, diagnostics, and rendering.
- `orbital_atelier config --preset crystal --output recipe.json`: existing verified frame/receipt, assembly,
  and encoding workflow extended with optional Crystal fields.

Older recipes omit the new optional block and retain their serialized identity.
Prepared material state is immutable. Frame order and worker count cannot change
the intended field. Numerical failure is reported instead of producing a dark
or partially converged picture.

The solver uses IC(0)-preconditioned conjugate gradients, guarded diagonal shifts,
and an explicitly recorded Jacobi fallback. It certifies the residual using the
original stiffness operator. The presentation cache preserves spatial and temporal
sampling and f64 precision; a conservative four-GiB budget selects streaming for
larger images. Cache and streaming paths can differ by floating-point rounding,
so they are tested numerically rather than promised identical hashes.

## Render and experiment

```sh
cargo build --locked --release --bin orbital_atelier
target/release/orbital_atelier config --preset crystal --output nocturne.json
target/release/orbital_atelier --threads 4 render \
  --orbit source.orbit --config nocturne.json --output proof --frame 1670

python3 tools/crystal/experiment.py \
  --executable target/release/orbital_atelier --orbit source.orbit \
  --base-recipe nocturne.json --output material-comparison \
  --workers 4 --width 1280 --height 720 --aa 2 --temporal-samples 1
```

The helper labels its gallery as development, archives requested and resolved
recipes, hashes inputs, and rejects incompatible resumes. `resolve-crystal`
canonicalizes recipes with the exact defaults and omission rules used by rendering.
Use a new folder for each experiment. Its six variants intentionally change named
optical, memory, and presentation controls; they are not new physical orbits.

For the complete movie, use the existing `tools/atelier/render_study.py` driver
with an immutable executable, independent frame ranges, and an explicit worker
budget. It assembles verified frames, writes browser H.264 and 10-bit HEVC, and
decodes both films completely. A separate examination recipe sets
`freeze_source_fraction` and `polarizer_sweep_degrees`, preserving the mechanical
state while rotating both crossed polarizers together.

The server workspace is `/home/user/polarized-crystal/`. It is separate from
running six-study jobs. Immutable versioned binaries and recipes identify every
look-development result and final output.

## Required review

Mechanical checks cover equilibrium, solver residual, an affine patch, and mesh
refinement. Optical checks cover known extinction states, hydrostatic invariance,
polarizer periodicity, and lookup error against direct wavelength integration.
Integration checks cover invalid recipes, source timing, independent frames,
worker-count determinism, and verified receipt/resume behavior.

Art review includes the opening, calm passages, approach, closest encounter,
release, and ending. A finished still must be inspected at native resolution.
Final films must be decoded completely, checked for temporal stability, and
delivered in browser H.264 and 10-bit HEVC forms. High numerical accuracy does
not replace visual judgment.

## Experiment log

Implementation started from six-study commit `4f0a106` on the isolated branch
`codex/polarized-crystal`. Initial server capacity is limited because the other
studies are actively rendering. Look-development selections, measured costs,
quality evidence, and final artifact locations are recorded below as obtained.

- Initial circular-load studies were rejected as too diagrammatic. Tangent-aligned
  loads, restrained fixed prestress, and selective spectral illumination produced
  the selected Nocturne family. More than fifty proof frames explored material,
  exposure, polarity, encounter timing, and three different recorded orbits.
- A finer 257 × 193 mesh replaced 129 × 97 after comparison showed a 4.54% shift
  in peak encounter stress and visible motion of tight interference bands. Two
  grids establish sensitivity, not absolute convergence.
- IC(0) reduced the representative fine-grid solve from 1195 to 346 iterations,
  at unchanged 1e-9 residual tolerance. Its warmed 4K frame was about 23% faster
  before surface caching. Only 173 of 24.9 million RGB16 channels differed from
  Jacobi, each by one code value.
- Native 4K sampling comparison at frame 1670 retained AA2 and four temporal
  samples. Relative to that setting, AA3 changed encoded RGB16 RMSE by 0.000188;
  eight temporal samples changed it by 0.000244. The cheaper AA1/two-sample option
  changed RMSE by 0.001733 and was not selected. These are normalized display-code
  differences, not perceptual Delta E or a guarantee for every possible orbit.
- The original full-motion development film rendered and decoded completely and
  played to its ending in the browser. Final high-resolution film verification is
  recorded with the delivered files.
- Three pre-existing native SIMD accuracy tests fail identically in unchanged
  commit `4f0a106`: `test_avx2_vectorized_exp_accuracy`, `test_simd_matches_scalar`,
  and `test_simd_scalar_parity_exhaustive`. Crystal does not call that SIMD spectral
  accumulation path. Baseline and new-build failure logs are retained separately;
  they are not represented as passing tests.
