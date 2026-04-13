# Art Traits and Rarity

## Understanding the Visual DNA of Cosmic Signature NFTs

Every Cosmic Signature NFT is unique, but they are not all unique in the same way. The seed-driven randomization system creates natural categories of visual traits -- some common, some rare. This guide explains how those traits emerge and what makes certain pieces visually distinctive.

---

## How Traits Are Determined

When an NFT is generated, its hex seed drives a SHA3-256 random number stream. This stream determines everything: the orbit physics, color palette, camera behavior, and which post-processing effects are enabled. Each effect has an independent **enable probability** -- the chance that the seed's random stream activates it. These probabilities are tuned by hand to produce a curated distribution of visual qualities across the collection.

Traits are not assigned from a lookup table. They emerge naturally from the randomization of continuous parameters. Two NFTs might both have "glow" enabled, but with different strength, threshold, and radius values -- making every instance subtly different even within the same trait category.

---

## Visual Trait Categories

### Always Present (Core Traits)

These traits are part of every Cosmic Signature NFT:

| Trait | Description |
|-------|-------------|
| **Three-body orbit** | The fundamental gravitational dance. Orbit shape, complexity, and symmetry vary per seed. |
| **Spectral color** | 16-bin spectral rendering with OKLab color generation. Base hue, wave frequency, and drift all vary. |
| **DoG bloom** | Difference-of-Gaussians bloom is always applied (part of the default render pipeline). |
| **Two-pass tonemapping** | AgX-style exposure with Punchy outset matrix. Black/white points and exposure scale vary per seed. |
| **Velocity HDR** | Fast-moving bodies create intense flares (up to 8x brightness boost). |
| **Energy red-shift** | High-energy regions shift toward warmer wavelengths. |

### Camera Drift Modes

The camera drift mode determines how the virtual viewpoint moves during the 30-second video:

| Mode | Description | Selection |
|------|-------------|-----------|
| **Elliptical** (default) | Smooth elliptical orbit with randomized scale, arc fraction, and eccentricity | Default -- most common |
| **Brownian** | Organic, wandering camera path using Gaussian random walks | CLI override only |
| **Linear** | Slow steady pan across the scene | CLI override only |
| **None** | Fixed camera -- no movement | CLI override only |

At default settings, all NFTs use elliptical drift with per-seed randomized parameters. The eccentricity ranges from near-circular (0) to highly elongated (0.95), creating significant variation in how the camera orbits the scene.

### Post-Processing Effect Traits

Each of these effects is independently toggled per seed. The **enable probability** determines how common or rare the trait is across the collection:

| Effect | Enable Probability | Rarity | Visual Impact |
|--------|-------------------|--------|---------------|
| **Micro-contrast** | 85% | Very common | Sharpens fine trail structures, enhances detail in dense orbital regions |
| **Cinematic color grade** | 60% | Common | Vibrance, clarity, tone curve, cool shadow tint, warm highlight tint, vignette |
| **Glow** | 55% | Common | Broad-radius light diffusion creating an ethereal, atmospheric quality |
| **Edge luminance** | 55% | Common | Brightened edges where trails meet dark background, enhancing definition |
| **Fine texture** | 45% | Moderate | Subtle noise-based surface grain for organic quality |
| **Aether** | 35% | Uncommon | Volumetric scattering with filament density, flow alignment, and caustic patterns |
| **Gaussian bloom** | 28% | Uncommon | Soft glow halos around the brightest trail regions |
| **Champlevé** | 25% | Rare | Metal-inlay effect with cell-based interference, rim highlights, brushed-metal sheen |
| **Opalescence** | 25% | Rare | Iridescent pearlescent shimmer that shifts color with viewing angle |
| **Chromatic bloom** | 20% | Rare | Per-channel bloom with subtle RGB separation in highlights |
| **Gradient map** | 18% | Rare | Luminance-based palette mapping from one of 15 curated luxury palettes |
| **Atmospheric depth** | 18% | Rare | Distance-based desaturation and haze simulating atmospheric perspective |
| **Perceptual blur** | 5% | Very rare | OKLab-space Gaussian blur for a dreamy, soft-focus quality |

### Rarity Combinations

Because effects are independently toggled, combinations multiply rarity. Some examples:

| Combination | Approximate Probability |
|-------------|------------------------|
| Champlevé + Opalescence | ~6.25% |
| Gradient map + Atmospheric depth | ~3.24% |
| Chromatic bloom + Perceptual blur | ~1.0% |
| Champlevé + Opalescence + Gradient map | ~1.13% |
| All three material effects (champlevé + opalescence + aether) | ~2.19% |
| Perceptual blur + Chromatic bloom + Gradient map | ~0.18% |

---

## Orbit-Level Traits

Beyond post-processing, the orbit itself has distinctive qualities:

### Orbit Complexity

The Borda search rewards chaotic orbits -- but within that preference, the degree of chaos varies. Some orbits produce tight, intricate webs of overlapping trails. Others produce looser, more expansive patterns with wider sweeps. The chaos metric (FFT-based spectral analysis of the distance signal) determines where on this spectrum each piece falls.

### Triangle Balance

The equilateralness score (heavily weighted in selection at 11.0x) means all selected orbits maintain reasonable triangular balance. But the degree varies: some pieces feature nearly-equilateral triangles throughout, producing symmetrical, mandala-like patterns. Others have more elongated triangles that create directional, flowing compositions.

### Color Temperature

The base hue is random per seed, and the hue wave frequency varies from 1.8 to 4.0. This creates natural temperature families:
- **Cool pieces** -- dominated by blues, violets, and cyans
- **Warm pieces** -- dominated by golds, oranges, and reds
- **Full-spectrum pieces** -- seeds where the hue drift and wave frequency produce wide color traversal

### Density

The alpha denominator shuffle ({1/13M, 1/15M, 1/17M} per body) combined with orbit dynamics creates varying trail density. Some pieces are dense and luminous; others are sparse and delicate.

---

## Gradient Map Palettes

When the gradient map effect is enabled (18% probability), one of 15 curated luxury palettes is selected (palette index 0--14). Each palette maps luminance to a distinctive color scheme, producing families of tonal character. The palette index is randomized per seed.

---

## What to Look For as a Collector

1. **Rare effect combinations** -- Pieces with champlevé + opalescence + gradient map (roughly 1 in 90) have a distinctive metallic iridescence that most pieces lack.

2. **Perceptual blur** -- At only 5% enable rate, pieces with this dreamy soft-focus quality are genuinely scarce.

3. **Orbit character** -- Beyond effects, look at the orbit itself. Tight, symmetric mandalas vs. loose, asymmetric flows are both valid but represent different aesthetic families.

4. **Color range** -- Some seeds produce pieces that traverse the entire visible spectrum. Others settle into a narrow color band. Both are valuable but appeal to different tastes.

5. **The generation log** -- Every NFT's `generation_log.json` records exactly which effects were enabled and at what parameter values. This provides a definitive trait list for any piece.

---

## A Note on "Rarity"

In traditional PFP collections, rarity is determined by trait tables designed before minting. In Cosmic Signature, rarity is an emergent property of the SHA3 random stream interacting with curated probability distributions. The probabilities are fixed in the code, but no one -- not even the creators -- knows which specific seeds will activate which effects until generation runs.

This means rarity cannot be gamed or pre-selected. It is a mathematical consequence of the seed, just like the art itself.

---

*All enable probabilities and parameter ranges are defined in `src/render/parameter_descriptors.rs` and can be independently verified in the open-source codebase (CC0 1.0).*

*cosmicsignature.com | cosmicsignature.art*
