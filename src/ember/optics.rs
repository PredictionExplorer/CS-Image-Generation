//! Spectral shading of sumi on kozo, and the display encoding.
//!
//! # Reflectance (36 bands, λ = 380, 390, …, 730 nm)
//!
//! The ink sits *in* the paper's fibres (Duncan additivity of absorption `K` and scattering `S`,
//! per unit sheet thickness, semi-infinite sheet). For the pine-soot load `L`, paper mottle `m`
//! and ink gain `g` of the sample's pixel:
//!
//! ```text
//! k_ink = L·k_soot + (m - 1)·K_p                  (mottle: flocs carry more pulp absorbers)
//! s_ink = L·s_soot
//! K     = K_p + g·k_ink,   S = S_p + g·s_ink      (S_p = 1: the Kubelka–Munk unit)
//! R∞    = 1 / (1 + a + sqrt(a·(a + 2))),  a = K/S   (semi-infinite Kubelka–Munk, stable form)
//! R     = ks + (1 - k1)(1 - k2)·R∞ / (1 - k2·R∞)  (Saunderson surface; ks = matte black floor)
//! R    -= (ks - ks_film)·(1 - exp(-max(L, 0)/c_film))   (nikawa film over dense ink)
//! XYZ   = Σ_b R_b·W_b                               (gallery light, perfect diffuser Y = 1)
//! ```
//!
//! `k_soot = SOOT_PINE_K·STRENGTH_SOOT_PINE`, `s_soot` likewise: the strength scale makes one load
//! unit of pine soot optically as strong as one unit of neutral carbon (peak optical density 0.5
//! over kozo). The paper's `K_p` is derived once from its observed reflectance by the inverse
//! Saunderson correction and the Kubelka–Munk remission function,
//!
//! ```text
//! R' = (R_obs - ks)/((1 - k1)(1 - k2)),  R_int = R'/(1 + k2·R'),  K_p = S_p·(1 - R_int)²/(2·R_int)
//! ```
//!
//! so bare paper (`L = 0`, `m = g = 1`) reproduces `R_obs` to rounding.
//!
//! The Saunderson body term is evaluated in the algebraically identical, division-light form
//!
//! ```text
//! (1 - k1)(1 - k2)·R∞/(1 - k2·R∞) = (1 - k1)(1 - k2)·S / ((1 - k2)·S + K + sqrt(K·(K + 2S)))
//! ```
//!
//! (substitute `R∞ = 1/D`, `D = 1 + a + sqrt(a(a+2))`, and multiply through by `S > 0`): one
//! division and one square root per band instead of three divisions. It agrees with the
//! prototype's `saunderson(km_r_inf(K, S))` to a few ULP (the golden tests hold it to 1e-12).
//!
//! Preconditions: `L ≥ 0`, `g ≥ 0` and `K ≥ 0`. Since
//! `K = K_p·(1 - g·(1 - m)) + g·L·k_soot` with a non-negative table, `K ≥ 0` is
//! guaranteed by `g·(1 - m) ≤ 1` (sufficient, not necessary), which the kozo sheet ensures
//! (`m ≥ 0.3`, `g ≤ 1`, so `g·(1 - m) ≤ 0.7`). Then `S ≥ 1`, every band lies in `(0, 1)` and the
//! XYZ is finite.
//!
//! # Display encoding ([`Optics::encode_srgb16`])
//!
//! 1. **Black-point compensation** (ICC relative colorimetric with BPC, linear XYZ scaling): the
//!    medium black (pine soot at load 8 under a full nikawa film) maps to a display black of
//!    `Y = 0.004`, the paper-relative white stays. With `kb = (Y_black/Y_white)·white` and
//!    `kd = 0.004·white` the map `xyz' = (xyz - kb)·s + kd` has the uniform scale
//!    `s = (1 - 0.004)/(1 - Y_black/Y_white) ≈ 1.01115`.
//! 2. **Adaptation and primaries**: `lin = M_TOTAL·xyz'` (CAT16, D = 1, gallery white → D65, then
//!    linear sRGB; the gallery white maps to (1, 1, 1)).
//! 3. **Gamut map**, only when a channel leaves `[-1e-6, 1 + 1e-6]`: hue- and lightness-preserving
//!    chroma compression in `OKLab`. The colour goes to D65 XYZ with the inverse of the prototype's
//!    IEC sRGB matrix, to `OKLab` (`lab = M2·cbrt(max(M1·xyz, 0))`), and the chroma scale
//!    `μ ∈ [0, 1]` is bisected 24 times, keeping the largest in-gamut `lo`; the result is clipped
//!    to `[0, 1]`. (Neutral soot on warm paper stays inside sRGB; the map is a safeguard.)
//! 4. **Transfer function**: clip to `[0, 1]`, `v = x ≤ 0.0031308 ? 12.92·x : 1.055·x^(1/2.4) -
//!    0.055` (IEC 61966-2-1), clip to `[0, 1]`, and `round(v·65535)` as a 16-bit code.
//!
//! # Determinism
//!
//! Only exactly rounded `+ - * /` and `sqrt`, plus `exp`, `pow` and `cbrt` through [`math`].
//! The per-sample band sums use a fixed interleaved order (four partial sums, combined pairwise),
//! which LLVM may vectorise without changing a bit. Matrix inverses are computed once from the
//! embedded coefficients with explicit cofactor formulas.

use super::math::{self, clamp_unit};
use super::paper::PaperSample;

/// Spectral bands: `λ_b = 380 + 10·b` nm, `b = 0..36`.
const BANDS: usize = 36;

/// Diffuse (matte) first-surface reflectance that reaches the viewer: the sheet's black floor.
const KS_MATTE: f64 = 0.0164;
/// External (Fresnel) reflectance of the sheet surface.
const K1_FRESNEL: f64 = 0.04;
/// Internal reflectance of the sheet surface for diffuse light.
const K2_INTERNAL: f64 = 0.6;
/// Scattering of the paper per unit thickness (flat; the Kubelka–Munk unit `S_p = S0 = 1`).
const PAPER_S: f64 = 1.0;
/// Saunderson transmission factor `(1 - k1)(1 - k2)` of light entering and leaving the sheet.
const SAUNDERSON_GAIN: f64 = (1.0 - K1_FRESNEL) * (1.0 - K2_INTERNAL);
/// First-surface reflectance where dense ink's nikawa (animal glue) has dried to a smooth film:
/// its reflection is specular, so the matte floor falls from `KS_MATTE` to this value.
const KS_FILM: f64 = 0.005;
/// Total ink load (equal-strength units) at which `1 - 1/e` of the surface is filmed.
const C_FILM: f64 = 0.35;
/// Equal-strength scale of pine soot (its `k`, `s` per unit load are multiplied by this).
const STRENGTH_SOOT_PINE: f64 = 1.2440916174042291;
/// Luminance factor of the display black the medium black is mapped to.
const BPC_DEST_Y: f64 = 0.004;
/// Linear-sRGB tolerance of the gamut test: in gamut ⇔ every channel in `[-tol, 1 + tol]`.
const GAMUT_TOLERANCE: f64 = 1e-6;
/// Bisection steps of the `OKLab` chroma gamut map (chroma resolution `2^-24`).
const GAMUT_ITERATIONS: u32 = 24;
/// Interleaved partial sums of the band reductions.
const LANES: usize = 4;
const _: () = assert!(BANDS.is_multiple_of(LANES));

/// D65 XYZ → linear sRGB with the IEC 61966-2-1 seven-digit coefficients: the space of the
/// prototype's gamut map (`render._XYZ_TO_SRGB`), kept as is so the mapping matches it.
const XYZ_TO_SRGB_IEC: Matrix = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
];
/// `OKLab` `M1` (Ottosson 2020): D65 XYZ → approximate cone responses (LMS).
const OKLAB_M1: Matrix = [
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070],
];
/// `OKLab` `M2`: cube-rooted LMS → `(L, a, b)`.
const OKLAB_M2: Matrix = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
];

// ---------------------------------------------------------------------------------------------
// Embedded tables: exact decimal copies of the museum-lab porting spec's tables (provenance in
// docs/ember-design.md §7), generated from its text by a script and cross-checked against the
// prototype's dump; `table_checksums` verifies every table against checksums computed
// independently from that text.

/// Observed diffuse reflectance of the bare kozo sheet per band (dimensionless): a Jakob–Hanika
/// 2019 sigmoid-polynomial spectrum fitted to the warm cream sRGB (0.935, 0.915, 0.865).
const KOZO_R_OBS: [f64; BANDS] = [
    0.6028952013682464,
    0.6248332973498207,
    0.6454311023409005,
    0.6646764254139862,
    0.6825840513041871,
    0.6991897303241309,
    0.7145445824951332,
    0.7287101883661737,
    0.7417544867612845,
    0.7537484899258061,
    0.7647637568745721,
    0.7748705285472722,
    0.7841364145900698,
    0.7926255229405218,
    0.8003979333637014,
    0.807509430027454,
    0.8140114231093111,
    0.8199510035456798,
    0.8253710874884663,
    0.8303106175078078,
    0.8348047960848367,
    0.8388853336593205,
    0.8425806986896194,
    0.8459163611158373,
    0.8489150235418199,
    0.8515968365866181,
    0.8539795963826415,
    0.8560789232635413,
    0.8579084214054353,
    0.8594798196507374,
    0.8608030940228626,
    0.8618865725844836,
    0.8627370233399545,
    0.863359725862721,
    0.8637585272614621,
    0.8639358829993721,
];

/// Absorption `k` of one unit load of pine soot per band (per unit sheet thickness), before
/// [`STRENGTH_SOOT_PINE`]: Mie spheres, m = 1.95 + 0.79i in n = 1.5, log-normal `d_g` = 200 nm,
/// `σ_g` = 1.8; normalised so load 1 reads L* 19.87 over kozo (own floor L* 18.37 + 1.5).
const SOOT_PINE_K: [f64; BANDS] = [
    48.22555692694029,
    48.364490701622714,
    48.495578330513865,
    48.619063791482326,
    48.73518383303636,
    48.844166834971226,
    48.94623428885279,
    49.04160043303233,
    49.13047246228548,
    49.21305136082387,
    49.28953139618322,
    49.360100805426484,
    49.42494204743,
    49.48423160206743,
    49.538140564583145,
    49.5868347589424,
    49.63047468190692,
    49.66921593931529,
    49.70320938673229,
    49.73260111496786,
    49.75753274079298,
    49.7781416077312,
    49.7945608046991,
    49.806919324225994,
    49.815342282112276,
    49.8199510051326,
    49.8208631019258,
    49.81819262665012,
    49.81205022820052,
    49.80254322860209,
    49.78977570324128,
    49.77384861282292,
    49.75485992588728,
    49.73290470099757,
    49.70807515622802,
    49.68046076726936,
];

/// Scattering `s` of one unit load of pine soot per band, before [`STRENGTH_SOOT_PINE`].
const SOOT_PINE_S: [f64; BANDS] = [
    2.196709609871514,
    2.2212253310447614,
    2.245289019894825,
    2.2688907709575163,
    2.2920218174106,
    2.314674448666629,
    2.3368419308322514,
    2.358518496145772,
    2.379699235847246,
    2.4003800842311276,
    2.4205577643754825,
    2.440229719287305,
    2.4593940976734325,
    2.4780496955254567,
    2.496195909548939,
    2.513832718166546,
    2.5309606330024526,
    2.547580660687575,
    2.5636942827778437,
    2.579303418920353,
    2.5944103921552824,
    2.609017908645759,
    2.623129031403207,
    2.6367471496148167,
    2.6498759569503765,
    2.6625194325780237,
    2.6746818173343407,
    2.686367591216361,
    2.6975814563350142,
    2.708328320305146,
    2.7186132772685268,
    2.72844159014632,
    2.737818676359019,
    2.746750094525936,
    2.755241529924979,
    2.763298780012723,
];

/// Tristimulus weights `(W_X, W_Y, W_Z)` per band under the gallery light (CIE LED-V1, CIE 1931
/// 2° observer): `XYZ = Σ_b R_b·W_b`. Each weight is the exact 1 nm integral (360–830 nm) of the
/// piecewise-linear spectrum through the band values against the CMFs and the 1 nm illuminant,
/// normalised so a perfect diffuser has `Y = Σ_b W_Y = 1`. NOT `CMF(λ_b)·illuminant(λ_b)`.
const W_GALLERY: [[f64; 3]; BANDS] = [
    [3.220572556890139e-07, 9.238990561842059e-09, 1.5185418827604948e-06],
    [1.7030112003877992e-05, 4.794871256396754e-07, 8.062422619928824e-05],
    [0.0005781088848862577, 1.599991583207279e-05, 0.002746845526422996],
    [0.005021137643408888, 0.00014050110131308074, 0.023973820042645262],
    [0.009581803068306744, 0.0002922468319722502, 0.04606805682188442],
    [0.0069330315858795, 0.0002813923275548521, 0.03381590654793341],
    [0.0052254829993841605, 0.00035504933912017706, 0.02627103839688198],
    [0.006165526770669504, 0.0007326122547670515, 0.03268269155643491],
    [0.0070907471512355515, 0.0015474173821287142, 0.040915943381970694],
    [0.0063467376987765564, 0.00313507263959776, 0.04196195682785491],
    [0.0041599494529995645, 0.006257264508870385, 0.03528007671363636],
    [0.001838692026725997, 0.011746948381300346, 0.02574376239965469],
    [0.0004463796300519632, 0.020709998866820015, 0.017296027721327236],
    [0.0009100983654724335, 0.03370196100393101, 0.01063480867178693],
    [0.004664210931444603, 0.04854021350837797, 0.005640514653155896],
    [0.012087888289765494, 0.061711843150739104, 0.003084321995708537],
    [0.02240040778317001, 0.07257645209811164, 0.0016031883183961436],
    [0.035551488401141106, 0.08076937815136756, 0.0007503888072824324],
    [0.051693143342467086, 0.08585996251909286, 0.00035554019730103185],
    [0.07063730110780647, 0.08775983181382081, 0.0002039974208928247],
    [0.09213521584419881, 0.08720677875683143, 0.00016261838631293477],
    [0.11460077283243045, 0.0845196549695198, 0.00012882082882851635],
    [0.13302646707715438, 0.07927955712861275, 9.946776024447158e-05],
    [0.1392145547912804, 0.07026655328700897, 5.345689057190064e-05],
    [0.12892656962417534, 0.05786040867658159, 2.7088167652049136e-05],
    [0.10296407663121034, 0.04269930880681079, 9.527538544411301e-06],
    [0.07250469781897928, 0.02846466545356142, 3.2436920322984168e-06],
    [0.044805635622594556, 0.016971484700080402, 4.993745934368049e-07],
    [0.024324329636306166, 0.009021381829852456, 0.0],
    [0.011714219477211204, 0.004294817071293465, 0.0],
    [0.005333324410702663, 0.001940549819243241, 0.0],
    [0.0022238584966099736, 0.000804885209419629, 0.0],
    [0.0008924952136407394, 0.00032237599330033274, 0.0],
    [0.0003613779026569767, 0.0001305001706155076, 0.0],
    [0.00014232698524456073, 5.139687997250011e-05, 0.0],
    [8.59738356592957e-05, 3.104672646200899e-05, 0.0],
];

/// XYZ of the perfect diffuser under the gallery light, `Σ_b W_b` (the adapted white).
const WHITE_XYZ: [f64; 3] = [1.1246053835029066, 1.0000000000000002, 0.3495957514080328];

/// Gallery XYZ → linear sRGB: `M_RGB·M_ADAPT`, where `M_ADAPT` is the CAT16 von Kries transform
/// (degree of adaptation D = 1) from the gallery white to the D65 display white and `M_RGB` the
/// D65 XYZ → linear sRGB matrix derived from the sRGB primaries and white (white → (1, 1, 1)).
const M_TOTAL: [[f64; 3]; 3] = [
    [3.1103874704897123, -2.3116129361514917, -0.5330315291038245],
    [-0.9699677351819489, 2.1482741032024757, -0.16431311353578196],
    [0.05552066736131681, -0.12447787565832678, 3.0379060099290243],
];

/// Gallery XYZ of the medium's black point: pine soot at load 8 (× its strength scale) on plain
/// kozo (mottle 1, ink gain 1) under a complete nikawa film (film fraction exactly 1). See
/// `black_point_compensation_uses_the_medium_black` for its derivation from the tables.
const MEDIUM_BLACK_XYZ: [f64; 3] =
    [0.01697194186047163, 0.014982317318429677, 0.005014467410659896];

// ---------------------------------------------------------------------------------------------

/// A row-major 3×3 matrix.
type Matrix = [[f64; 3]; 3];

/// Tristimulus shading and display encoding.
#[derive(Debug)]
pub(crate) struct Optics {
    /// Paper absorption `K_p` per band (per unit thickness), derived from [`KOZO_R_OBS`].
    paper_k: [f64; BANDS],
    /// Pine-soot absorption per unit load, strength-scaled.
    soot_k: [f64; BANDS],
    /// Pine-soot scattering per unit load, strength-scaled.
    soot_s: [f64; BANDS],
    /// Tristimulus weights, structure of arrays: `weight[channel][band]` (X, Y, Z).
    weight: [[f64; BANDS]; 3],
    /// Black-point compensation.
    black_point: BlackPoint,
    /// `OKLab` chroma gamut map.
    gamut: GamutMap,
}

impl Optics {
    /// Builds the pigment and paper tables: derives `K_p` from the observed kozo reflectance,
    /// scales the pigment tables by their strength, transposes the weights to structure-of-arrays
    /// form, and prepares the black-point and gamut-map constants (inverting the three gamut-map
    /// matrices once).
    pub(crate) fn new() -> Self {
        let scaled = |table: [f64; BANDS], strength: f64| table.map(|value| value * strength);
        let mut weight = [[0.0; BANDS]; 3];
        for (band, row) in W_GALLERY.iter().enumerate() {
            for (channel, &value) in row.iter().enumerate() {
                weight[channel][band] = value;
            }
        }
        Self {
            paper_k: KOZO_R_OBS.map(paper_absorption),
            soot_k: scaled(SOOT_PINE_K, STRENGTH_SOOT_PINE),
            soot_s: scaled(SOOT_PINE_S, STRENGTH_SOOT_PINE),
            weight,
            black_point: BlackPoint::new(WHITE_XYZ, MEDIUM_BLACK_XYZ, BPC_DEST_Y),
            gamut: GamutMap::new(),
        }
    }

    /// Gallery-light XYZ (perfect diffuser `Y = 1`) of one ink sample on paper, nikawa film
    /// included. See the module docs for the model and its preconditions.
    pub(crate) fn reflect_xyz(&self, carbon: f64, paper: PaperSample) -> [f64; 3] {
        let film = if carbon > 0.0 { 1.0 - math::exp(-carbon / C_FILM) } else { 0.0 };
        self.reflect_with_film(carbon, paper, film)
    }

    /// Gallery-light XYZ of bare paper (`reflect_xyz` with zero loads).
    pub(crate) fn paper_xyz(&self, paper: PaperSample) -> [f64; 3] {
        self.reflect_xyz(0.0, paper)
    }

    /// Black-point compensation, CAT16 adaptation to D65, gamut mapping and the sRGB transfer
    /// function; returns 16-bit sRGB code values and whether gamut mapping was needed.
    ///
    /// `xyz` must be finite (a NaN channel encodes as 0 and reports a gamut mapping).
    pub(crate) fn encode_srgb16(&self, xyz: [f64; 3]) -> ([u16; 3], bool) {
        let linear = mul(&M_TOTAL, self.black_point.apply(xyz));
        let (linear, mapped) =
            if in_gamut(linear) { (linear, false) } else { (self.gamut.map(linear), true) };
        (linear.map(encode_channel), mapped)
    }

    /// [`Optics::reflect_xyz`] with an explicit filmed surface fraction `film ∈ [0, 1]`.
    ///
    /// The band loop has no branches and works on fixed-size arrays so it vectorises; the three
    /// band sums use [`dot_bands`]' fixed order.
    #[allow(clippy::needless_range_loop)] // indexed loops over fixed-size tables vectorise best
    fn reflect_with_film(&self, carbon: f64, paper: PaperSample, film: f64) -> [f64; 3] {
        let mottle = paper.mottle - 1.0;
        let gain = paper.ink_gain;
        let surface_drop = (KS_MATTE - KS_FILM) * film;
        let mut reflectance = [0.0f64; BANDS];
        for b in 0..BANDS {
            let k_ink = carbon * self.soot_k[b] + mottle * self.paper_k[b];
            let s_ink = carbon * self.soot_s[b];
            let k = self.paper_k[b] + gain * k_ink;
            let s = PAPER_S + gain * s_ink;
            let body =
                SAUNDERSON_GAIN * s / ((1.0 - K2_INTERNAL) * s + k + (k * (k + 2.0 * s)).sqrt());
            reflectance[b] = (KS_MATTE + body) - surface_drop;
        }
        [
            dot_bands(&reflectance, &self.weight[0]),
            dot_bands(&reflectance, &self.weight[1]),
            dot_bands(&reflectance, &self.weight[2]),
        ]
    }
}

/// Paper absorption `K_p` (per unit thickness, `S_p = 1`) of a semi-infinite sheet whose observed
/// reflectance is `r_obs`: inverse Saunderson, then the Kubelka–Munk remission function
/// `K/S = (1 - R)²/(2R)`.
fn paper_absorption(r_obs: f64) -> f64 {
    let r_prime = (r_obs - KS_MATTE) / ((1.0 - K1_FRESNEL) * (1.0 - K2_INTERNAL));
    let r_int = r_prime / (1.0 + K2_INTERNAL * r_prime);
    PAPER_S * ((1.0 - r_int) * (1.0 - r_int) / (2.0 * r_int))
}

/// `Σ_b r_b·w_b` in a fixed order: [`LANES`] partial sums `acc_l = Σ_{b ≡ l (mod 4)} r_b·w_b`
/// (bands ascending), combined as `(acc_0 + acc_1) + (acc_2 + acc_3)`.
#[inline]
fn dot_bands(r: &[f64; BANDS], w: &[f64; BANDS]) -> f64 {
    let mut acc = [0.0f64; LANES];
    for (r, w) in r.chunks_exact(LANES).zip(w.chunks_exact(LANES)) {
        for lane in 0..LANES {
            acc[lane] += r[lane] * w[lane];
        }
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// `m·v`, each row summed left to right.
#[inline]
fn mul(m: &Matrix, v: [f64; 3]) -> [f64; 3] {
    m.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2])
}

/// Inverse of a non-singular 3×3 matrix: transposed cofactors over the determinant (expanded
/// along the first row).
fn inverse(m: &Matrix) -> Matrix {
    let cofactor =
        |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let c00 = cofactor(1, 2, 1, 2);
    let c01 = -cofactor(1, 2, 0, 2);
    let c02 = cofactor(1, 2, 0, 1);
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    [
        [c00 / det, -cofactor(0, 2, 1, 2) / det, cofactor(0, 1, 1, 2) / det],
        [c01 / det, cofactor(0, 2, 0, 2) / det, -cofactor(0, 1, 0, 2) / det],
        [c02 / det, -cofactor(0, 2, 0, 1) / det, cofactor(0, 1, 0, 1) / det],
    ]
}

/// Whether every linear-sRGB channel lies in `[-GAMUT_TOLERANCE, 1 + GAMUT_TOLERANCE]` (false
/// for NaN).
#[inline]
fn in_gamut(linear: [f64; 3]) -> bool {
    linear.iter().all(|v| (-GAMUT_TOLERANCE..=1.0 + GAMUT_TOLERANCE).contains(v))
}

/// One linear-sRGB channel → 16-bit sRGB code: clip, IEC 61966-2-1 transfer function
/// (`math::pow`), clip, `round(v·65535)`.
#[inline]
fn encode_channel(linear: f64) -> u16 {
    let x = clamp_unit(linear);
    let encoded = if x <= 0.0031308 { 12.92 * x } else { 1.055 * math::pow(x, 1.0 / 2.4) - 0.055 };
    (clamp_unit(encoded) * 65535.0).round() as u16
}

/// Black-point compensation in XYZ (Adobe's linear scaling): `xyz' = (xyz - kb)·s + kd`.
#[derive(Clone, Copy, Debug)]
struct BlackPoint {
    /// `kb = (Y_black / Y_white)·white`: the medium black made neutral in the white's
    /// chromaticity.
    black: [f64; 3],
    /// `kd = dest_Y·white`: the display black it maps to.
    dest: [f64; 3],
    /// `s = (1 - dest_Y)/(1 - Y_black/Y_white)`, uniform over the channels because `kb` and `kd`
    /// are both multiples of the white (the prototype's per-channel `(white - kd)/(white - kb)`
    /// equals it to an ULP).
    scale: f64,
}

impl BlackPoint {
    /// Maps `black` to a display black of luminance factor `dest_y` and keeps `white`.
    fn new(white: [f64; 3], black: [f64; 3], dest_y: f64) -> Self {
        let black_ratio = black[1] / white[1];
        Self {
            black: white.map(|w| black_ratio * w),
            dest: white.map(|w| dest_y * w),
            scale: (1.0 - dest_y) / (1.0 - black_ratio),
        }
    }

    /// Compensated XYZ.
    #[inline]
    fn apply(&self, xyz: [f64; 3]) -> [f64; 3] {
        [
            (xyz[0] - self.black[0]) * self.scale + self.dest[0],
            (xyz[1] - self.black[1]) * self.scale + self.dest[1],
            (xyz[2] - self.black[2]) * self.scale + self.dest[2],
        ]
    }
}

/// Hue- and lightness-preserving `OKLab` chroma compression into the sRGB gamut.
#[derive(Clone, Copy, Debug)]
struct GamutMap {
    /// Inverse of [`XYZ_TO_SRGB_IEC`]: linear sRGB → D65 XYZ.
    srgb_to_xyz: Matrix,
    /// Inverse of [`OKLAB_M1`].
    m1_inverse: Matrix,
    /// Inverse of [`OKLAB_M2`].
    m2_inverse: Matrix,
}

impl GamutMap {
    /// Inverts the three matrices once.
    fn new() -> Self {
        Self {
            srgb_to_xyz: inverse(&XYZ_TO_SRGB_IEC),
            m1_inverse: inverse(&OKLAB_M1),
            m2_inverse: inverse(&OKLAB_M2),
        }
    }

    /// `OKLab` `(L, a, b)` of a linear-sRGB colour: `M2·cbrt(max(M1·xyz, 0))`.
    fn oklab(&self, linear: [f64; 3]) -> [f64; 3] {
        let lms = mul(&OKLAB_M1, mul(&self.srgb_to_xyz, linear));
        mul(&OKLAB_M2, lms.map(|v| math::cbrt(if v > 0.0 { v } else { 0.0 })))
    }

    /// Linear sRGB of an `OKLab` colour: `XYZ_TO_SRGB_IEC·M1⁻¹·(M2⁻¹·lab)³`.
    fn linear_srgb(&self, lab: [f64; 3]) -> [f64; 3] {
        let lms = mul(&self.m2_inverse, lab).map(|v| v * v * v);
        mul(&XYZ_TO_SRGB_IEC, mul(&self.m1_inverse, lms))
    }

    /// The in-gamut colour of the same `OKLab` lightness and hue with the largest chroma scale
    /// `lo ∈ [0, 1]` found by 24 bisection steps (clipped to `[0, 1]`: lightness beyond the
    /// gamut cannot be fixed by chroma).
    fn map(&self, linear: [f64; 3]) -> [f64; 3] {
        let [l, a, b] = self.oklab(linear);
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..GAMUT_ITERATIONS {
            let mid = 0.5 * (lo + hi);
            if in_gamut(self.linear_srgb([l, a * mid, b * mid])) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        self.linear_srgb([l, a * lo, b * lo]).map(clamp_unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    /// Pine-soot load (equal-strength units) of the medium black (`render.medium_black`).
    const MEDIUM_BLACK_LOAD: f64 = 8.0;

    /// The prototype's derived paper absorption `K_p` (docs/ember-design.md §7), to check the
    /// derivation in [`Optics::new`].
    const PAPER_K_P: [f64; BANDS] = [
        0.025858348945510038,
        0.02169581112182976,
        0.018296529394836867,
        0.015513462624865512,
        0.013227898715819454,
        0.011344348832592584,
        0.009786185663821056,
        0.008491991408056765,
        0.007412548050145172,
        0.006508387378069771,
        0.005747815367569607,
        0.0051053302373401,
        0.004560362076731597,
        0.004096272051540507,
        0.003699559312656095,
        0.0033592330505834546,
        0.0030663152869141305,
        0.0028134468764473117,
        0.002594574868548883,
        0.0024047039752102123,
        0.002239698573944866,
        0.0020961245932078296,
        0.0019711229292866087,
        0.0018623078498339165,
        0.0017676852531724385,
        0.0016855867579350992,
        0.0016146164616801877,
        0.001553607883062684,
        0.0015015891318437022,
        0.0014577547672430995,
        0.0014214431334693017,
        0.0013921182216256153,
        0.0013693553151665997,
        0.0013528298438444985,
        0.0013423090082660958,
        0.0013376458514404443,
    ];

    /// CAT16 (D = 1) gallery white → D65 adaptation (informational).
    const M_ADAPT: [[f64; 3]; 3] = [
        [0.9458703169913308, -0.20756459546129505, 0.2697105772857776],
        [-0.028292678490106443, 1.0358529292603638, -0.01154141805334534],
        [-0.002714574834329658, 0.09305746500338018, 2.857738136705947],
    ];

    /// D65 XYZ → linear sRGB from the primaries and white (informational).
    const M_RGB: [[f64; 3]; 3] = [
        [3.2409699419045226, -1.537383177570094, -0.49861076029300344],
        [-0.9692436362808798, 1.8759675015077206, 0.04155505740717561],
        [0.055630079696993635, -0.2039769588889765, 1.0569715142428784],
    ];

    /// The CAT16 cone-space matrix (informational).
    const CAT16: [[f64; 3]; 3] = [
        [0.401288, 0.650173, -0.051461],
        [-0.250268, 1.204414, 0.045854],
        [-0.002079, 0.048952, 0.953127],
    ];

    /// The D65 display white XYZ (`Y = 1`).
    const D65_WHITE: [f64; 3] = [0.9504559270516716, 1.0, 1.0890577507598784];

    /// `(sum, sum of squares, Σ (i+1)·x)` of every table in row-major order, each accumulated
    /// sequentially in `f64`: computed by the generator from its own parse of the porting spec's
    /// text.
    const CHECKSUMS: &[(&str, [f64; 3])] = &[
        ("KOZO_R_OBS", [28.538941910095794, 22.82858298689451, 554.0600386645627]),
        ("SOOT_PINE_K", [1779.5007030776358, 87969.51223229148, 33071.391180810824]),
        ("SOOT_PINE_S", [91.03687264964346, 231.24989075791925, 1746.9814980454573]),
        ("W_GALLERY", [2.474201134910939, 0.18242354394043814, 142.95137618120154]),
        ("WHITE_XYZ", [2.474201134910939, 2.386954458006267, 4.173392637727005]),
        ("M_TOTAL", [4.248685061351155, 30.132555234440837, 29.497663660568968]),
        ("PAPER_K_P", [0.20154892081411185, 0.0024785191092818, 1.8596639385632998]),
        ("M_ADAPT", [4.952116158407723, 10.259756803091186, 32.78181920848765]),
        ("M_RGB", [3.0618795617263372, 18.73835009270988, 12.69290312769651]),
        ("CAT16", [3.0, 3.0126066666040003, 15.798578999999998]),
        ("D65_WHITE", [3.0395136778115504, 3.089413253757818, 6.217629179331307]),
        ("MEDIUM_BLACK_XYZ", [0.036968726589561206, 0.0005376615261579172, 0.06197997872931067]),
    ];

    fn checksum(values: &[f64]) -> [f64; 3] {
        let (mut sum, mut squares, mut weighted) = (0.0f64, 0.0f64, 0.0f64);
        for (i, &x) in values.iter().enumerate() {
            sum += x;
            squares += x * x;
            weighted += (i + 1) as f64 * x;
        }
        [sum, squares, weighted]
    }

    fn table(name: &str) -> Vec<f64> {
        let flat = |m: &[[f64; 3]]| m.iter().flatten().copied().collect::<Vec<f64>>();
        match name {
            "KOZO_R_OBS" => KOZO_R_OBS.to_vec(),
            "SOOT_PINE_K" => SOOT_PINE_K.to_vec(),
            "SOOT_PINE_S" => SOOT_PINE_S.to_vec(),
            "W_GALLERY" => flat(&W_GALLERY),
            "WHITE_XYZ" => WHITE_XYZ.to_vec(),
            "M_TOTAL" => flat(&M_TOTAL),
            "PAPER_K_P" => PAPER_K_P.to_vec(),
            "M_ADAPT" => flat(&M_ADAPT),
            "M_RGB" => flat(&M_RGB),
            "CAT16" => flat(&CAT16),
            "D65_WHITE" => D65_WHITE.to_vec(),
            "MEDIUM_BLACK_XYZ" => MEDIUM_BLACK_XYZ.to_vec(),
            other => panic!("unknown table {other}"),
        }
    }

    fn rel(a: f64, b: f64) -> f64 {
        if a == b { 0.0 } else { (a - b).abs() / b.abs().max(a.abs()) }
    }

    fn max_rel(a: [f64; 3], b: [f64; 3]) -> f64 {
        (0..3).map(|c| rel(a[c], b[c])).fold(0.0, f64::max)
    }

    fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
        let mut out = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
            }
        }
        out
    }

    fn sample(carbon: f64, mottle: f64, ink_gain: f64) -> (f64, PaperSample) {
        (carbon, PaperSample { mottle, ink_gain })
    }

    /// Straightforward per-band evaluation of the prototype's formulas (R∞, then Saunderson),
    /// band sums sequential: the reference for the optimised kernel.
    #[allow(clippy::needless_range_loop)] // mirrors the kernel's per-band indexing
    fn reference_xyz(optics: &Optics, carbon: f64, paper: PaperSample, film: f64) -> [f64; 3] {
        let mut xyz = [0.0f64; 3];
        for b in 0..BANDS {
            let k_ink = carbon * optics.soot_k[b] + (paper.mottle - 1.0) * optics.paper_k[b];
            let s_ink = carbon * optics.soot_s[b];
            let k = optics.paper_k[b] + paper.ink_gain * k_ink;
            let s = 1.0 + paper.ink_gain * s_ink;
            let a = k / s;
            let r_inf = 1.0 / (1.0 + a + (a * (a + 2.0)).sqrt());
            let r = KS_MATTE
                + (1.0 - K1_FRESNEL) * (1.0 - K2_INTERNAL) * r_inf / (1.0 - K2_INTERNAL * r_inf)
                - (KS_MATTE - KS_FILM) * film;
            for (c, acc) in xyz.iter_mut().enumerate() {
                *acc += r * W_GALLERY[b][c];
            }
        }
        xyz
    }

    #[test]
    fn table_checksums() {
        assert_eq!(CHECKSUMS.len(), 12);
        for (name, expected) in CHECKSUMS {
            let values = table(name);
            assert_eq!(checksum(&values), *expected, "table {name} differs from the porting spec");
        }
        assert_eq!(KOZO_R_OBS.len(), BANDS);
        assert_eq!(W_GALLERY.len(), BANDS);
    }

    #[test]
    fn derived_paper_absorption_matches_the_prototype() {
        let optics = Optics::new();
        for (b, (&derived, &printed)) in optics.paper_k.iter().zip(&PAPER_K_P).enumerate() {
            assert!(rel(derived, printed) <= 1e-12, "band {b}: {derived} vs {printed}");
        }
    }

    #[test]
    fn weights_integrate_to_the_white() {
        let mut white = [0.0f64; 3];
        for row in &W_GALLERY {
            assert!(row.iter().all(|&w| w >= 0.0));
            for c in 0..3 {
                white[c] += row[c];
            }
        }
        assert!(max_rel(white, WHITE_XYZ) <= 1e-15, "{white:?}");
        assert!((white[1] - 1.0).abs() <= 1e-15);
        // Every table is positive where it should be.
        for b in 0..BANDS {
            assert!(KOZO_R_OBS[b] > KS_MATTE && KOZO_R_OBS[b] < 1.0);
            assert!(SOOT_PINE_K[b] > 0.0 && SOOT_PINE_S[b] > 0.0);
        }
    }

    #[test]
    fn display_matrix_is_cat16_then_srgb() {
        let product = mat_mul(&M_RGB, &M_ADAPT);
        for r in 0..3 {
            for c in 0..3 {
                assert!((product[r][c] - M_TOTAL[r][c]).abs() <= 1e-14, "{r},{c}");
            }
        }
        // M_ADAPT = CAT16⁻¹·diag((CAT16·d65)/(CAT16·white))·CAT16 (von Kries, D = 1).
        let source = mul(&CAT16, WHITE_XYZ);
        let target = mul(&CAT16, D65_WHITE);
        let mut scaled = CAT16;
        for (r, row) in scaled.iter_mut().enumerate() {
            for value in row.iter_mut() {
                *value *= target[r] / source[r];
            }
        }
        let adapt = mat_mul(&inverse(&CAT16), &scaled);
        for r in 0..3 {
            for c in 0..3 {
                assert!((adapt[r][c] - M_ADAPT[r][c]).abs() <= 1e-13, "{r},{c}");
            }
        }
        // White → (1, 1, 1): the D65 white through M_RGB, the gallery white through M_TOTAL.
        for v in mul(&M_RGB, D65_WHITE).into_iter().chain(mul(&M_TOTAL, WHITE_XYZ)) {
            assert!((v - 1.0).abs() <= 1e-13, "{v}");
        }
    }

    #[test]
    fn matrix_inverse_is_exact_to_rounding() {
        for m in [XYZ_TO_SRGB_IEC, OKLAB_M1, OKLAB_M2, M_TOTAL, CAT16] {
            let product = mat_mul(&m, &inverse(&m));
            for r in 0..3 {
                for c in 0..3 {
                    let identity = if r == c { 1.0 } else { 0.0 };
                    assert!((product[r][c] - identity).abs() <= 1e-14, "{product:?}");
                }
            }
        }
    }

    #[test]
    fn bare_paper_reproduces_its_observed_reflectance() {
        let optics = Optics::new();
        let mut expected = [0.0f64; 3];
        for b in 0..BANDS {
            for c in 0..3 {
                expected[c] += KOZO_R_OBS[b] * W_GALLERY[b][c];
            }
        }
        let paper = optics.paper_xyz(PaperSample::default());
        assert!(max_rel(paper, expected) <= 1e-13, "{paper:?} vs {expected:?}");
        assert_eq!(paper, optics.reflect_xyz(0.0, PaperSample::default()));
        // Without ink the ink gain is irrelevant.
        let gained = optics.paper_xyz(PaperSample { mottle: 1.0, ink_gain: 0.37 });
        assert!(max_rel(gained, paper) <= 1e-15);
    }

    /// The golden values of docs/ember-design.md §7 (computed by the prototype's float64 path).
    #[test]
    fn golden_table() {
        /// `(carbon, mottle, ink gain, XYZ, 16-bit code, sRGB8)`.
        type Golden = (f64, f64, f64, [f64; 3], [u16; 3], [u8; 3]);
        #[rustfmt::skip]
        let golden: [Golden; 8] = [
            (0.0, 1.0, 1.0, [0.9402834114004027, 0.8278798068379165, 0.25613089936975747], [61730, 60126, 56837], [240, 234, 221]),
            (1.0, 1.0, 1.0, [0.020521034654061254, 0.01814195130463335, 0.006133908911087946], [5487, 5201, 4910], [21, 20, 19]),
            (0.1, 1.0, 1.0, [0.055784391761169054, 0.04952957037451743, 0.017225604415632652], [14334, 14262, 14223], [56, 55, 55]),
            (0.01, 1.0, 1.0, [0.2097779838271311, 0.18654086630062808, 0.06522611495495041], [30034, 30036, 30039], [117, 117, 117]),
            (0.002, 1.0, 1.0, [0.4330375076985446, 0.38492773329308283, 0.1328277885679573], [42568, 42504, 42225], [166, 165, 164]),
            (0.0004, 1.0, 1.0, [0.671475105314176, 0.5961081472721036, 0.19987477811714788], [52265, 51966, 50917], [203, 202, 198]),
            (1.0, 1.2, 0.7, [0.02188044939077052, 0.019352530570086512, 0.006564053822515822], [6060, 5804, 5549], [24, 23, 22]),
            (0.01, 0.4, 0.5, [0.2973610590397358, 0.26442950408735566, 0.09234109757079008], [35599, 35601, 35579], [139, 139, 138]),
        ];
        let optics = Optics::new();
        for (carbon, mottle, ink_gain, xyz, code16, srgb8) in golden {
            let (loads, paper) = sample(carbon, mottle, ink_gain);
            let got = optics.reflect_xyz(loads, paper);
            // Contract: 1e-5; the kernel matches the float64 prototype to rounding.
            assert!(max_rel(got, xyz) <= 1e-12, "{loads:?} {paper:?}: {got:?} vs {xyz:?}");
            let (code, mapped) = optics.encode_srgb16(got);
            assert!(!mapped);
            for c in 0..3 {
                let eight = (f64::from(code[c]) / 257.0).round() as i32;
                assert!((eight - i32::from(srgb8[c])).abs() <= 1, "{loads:?}: {code:?}");
                assert_eq!(eight, i32::from(srgb8[c]), "{loads:?}: {code:?}");
                // 16-bit codes from an independent float64 evaluation of the same chain.
                assert!((i32::from(code[c]) - i32::from(code16[c])).abs() <= 1, "{code:?}");
            }
        }
    }

    #[test]
    fn optimised_kernel_matches_the_reference_formulas() {
        let optics = Optics::new();
        for &carbon in &[0.0, 0.0004, 0.01, 0.05, 0.3, 1.0, 8.0, 40.0] {
            for &(mottle, ink_gain) in &[(1.0, 1.0), (0.3, 1.0), (3.0, 0.2), (1.2, 0.7)] {
                let (loads, paper) = sample(carbon, mottle, ink_gain);
                for film in [0.0, 0.25, 1.0] {
                    let got = optics.reflect_with_film(loads, paper, film);
                    let want = reference_xyz(&optics, loads, paper, film);
                    assert!(max_rel(got, want) <= 1e-13, "{loads:?} {paper:?}");
                }
            }
        }
    }

    #[test]
    fn black_point_compensation_uses_the_medium_black() {
        let optics = Optics::new();
        let (loads, paper) = sample(MEDIUM_BLACK_LOAD, 1.0, 1.0);
        // The definition: full film (render.medium_black subtracts ks - ks_film outright).
        let black = optics.reflect_with_film(loads, paper, 1.0);
        assert!(max_rel(black, MEDIUM_BLACK_XYZ) <= 1e-13, "{black:?}");
        // The ordinary shading of load 8 is filmed to 1 - e^(-8/0.35) ≈ 1 - 1.2e-10.
        let shaded = optics.reflect_xyz(loads, paper);
        assert!(max_rel(shaded, MEDIUM_BLACK_XYZ) <= 1e-9, "{shaded:?}");

        let bpc = optics.black_point;
        assert!(rel(bpc.scale, 1.011149360576484) <= 4.0 * f64::EPSILON);
        for ((&white, &dest), &black) in WHITE_XYZ.iter().zip(&bpc.dest).zip(&bpc.black) {
            // The prototype's per-channel scale equals the uniform one.
            let proto = (white - dest) / (white - black);
            assert!(rel(proto, bpc.scale) <= 4.0 * f64::EPSILON);
        }
        // White stays white; the black's luminance goes to the display black.
        assert!(max_rel(bpc.apply(WHITE_XYZ), WHITE_XYZ) <= 1e-15);
        assert!((bpc.apply(MEDIUM_BLACK_XYZ)[1] - BPC_DEST_Y).abs() <= 1e-17);
    }

    #[test]
    fn white_and_black_encode_to_the_code_extremes() {
        let optics = Optics::new();
        assert_eq!(optics.encode_srgb16(WHITE_XYZ), ([65535; 3], false));
        // Brighter than white: lightness cannot be mapped, the clip saturates it.
        let (code, mapped) = optics.encode_srgb16(WHITE_XYZ.map(|v| 1.5 * v));
        assert!(mapped);
        assert_eq!(code, [65535; 3]);
        // The BPC black lands near display black (Y = 0.004 → sRGB ≈ 0.052).
        let (code, mapped) = optics.encode_srgb16(MEDIUM_BLACK_XYZ);
        assert!(!mapped);
        for v in code {
            assert!((2500..4500).contains(&v), "{code:?}");
        }
        // Linear toe of the transfer function and rounding.
        assert_eq!(encode_channel(0.0), 0);
        assert_eq!(encode_channel(-0.25), 0);
        assert_eq!(encode_channel(0.001), (0.001 * 12.92 * 65535.0f64).round() as u16);
        assert_eq!(encode_channel(1.0), 65535);
        assert_eq!(encode_channel(f64::NAN), 0);
    }

    #[test]
    fn encoding_is_monotone_along_a_grey_ramp() {
        let optics = Optics::new();
        let mut previous = [0u16; 3];
        for i in 0..=2000 {
            let t = f64::from(i) / 2000.0;
            let xyz = WHITE_XYZ.map(|w| t * w);
            let (code, mapped) = optics.encode_srgb16(xyz);
            if !mapped {
                assert!(code.iter().zip(previous).all(|(&c, p)| c >= p), "{t}: {code:?}");
                previous = code;
            }
        }
        assert_eq!(previous, [65535; 3]);
    }

    #[test]
    fn gamut_map_preserves_lightness_and_hue() {
        let optics = Optics::new();
        let gamut = optics.gamut;
        // Spectral colours (single bands under the gallery light) are far outside sRGB.
        let mut mapped_count = 0;
        for band in [4, 9, 12, 15, 20, 26, 30] {
            let xyz = W_GALLERY[band].map(|w| w * 6.0);
            let linear = mul(&M_TOTAL, optics.black_point.apply(xyz));
            let (_, mapped) = optics.encode_srgb16(xyz);
            assert_eq!(mapped, !in_gamut(linear));
            if !mapped {
                continue;
            }
            mapped_count += 1;
            let result = gamut.map(linear);
            assert!(result.iter().all(|&v| (0.0..=1.0).contains(&v)), "{result:?}");
            let before = gamut.oklab(linear);
            let after = gamut.oklab(result);
            if (0.02..0.98).contains(&before[0]) {
                // Same lightness and hue (the clip moves the colour by at most 1e-6).
                assert!((after[0] - before[0]).abs() <= 1e-4, "{before:?} → {after:?}");
                let chroma = |lab: [f64; 3]| (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
                assert!(chroma(after) < chroma(before));
                if chroma(after) > 1e-3 {
                    // sin and cos of the hue change.
                    let norm = chroma(after) * chroma(before);
                    let sin = (before[1] * after[2] - before[2] * after[1]) / norm;
                    let cos = (before[1] * after[1] + before[2] * after[2]) / norm;
                    assert!(sin.abs() <= 1e-3 && cos > 0.0, "{before:?} → {after:?}");
                }
                // Maximal: a slightly larger chroma scale leaves the gamut.
                let [l, a, b] = before;
                let scale = chroma(after) / chroma(before);
                let more = (scale + 1e-4).min(1.0);
                assert!(!in_gamut(gamut.linear_srgb([l, a * more, b * more])) || more == 1.0);
            }
        }
        assert!(mapped_count >= 3, "only {mapped_count} spectral colours needed mapping");
        // Soot on kozo is untouched by the encoder path at every load.
        for carbon in [0.0, 0.0004, 0.01, 0.1, 1.0, 8.0] {
            let (loads, paper) = sample(carbon, 1.0, 1.0);
            assert!(!optics.encode_srgb16(optics.reflect_xyz(loads, paper)).1, "{carbon}");
        }
    }

    #[test]
    fn ink_darkens_and_film_lowers_the_floor() {
        let optics = Optics::new();
        let paper = PaperSample::default();
        let mut previous = optics.paper_xyz(paper)[1];
        let mut carbon = 1e-4;
        for _ in 0..60 {
            carbon *= 1.25;
            let y = optics.reflect_xyz(carbon, paper)[1];
            assert!(y < previous, "load {carbon}");
            assert!(y > 0.0);
            previous = y;
        }
        // Infinitely dense soot bottoms out at the filmed floor plus its own volume scattering.
        let dense = optics.reflect_xyz(1e6, paper)[1];
        assert!(dense > KS_FILM && dense < 0.02, "{dense}");
        let white = optics.paper_xyz(paper);
        // Fibres shed ink: a lower ink gain is lighter.
        let full = optics.reflect_xyz(0.1, paper)[1];
        let shed = optics.reflect_xyz(0.1, PaperSample { ink_gain: 0.5, ..paper })[1];
        assert!(shed > full);
        // More paper absorbers (mottle > 1) darken bare paper.
        let mottled = optics.paper_xyz(PaperSample { mottle: 1.5, ink_gain: 1.0 })[1];
        assert!(mottled < white[1]);
    }

    /// Cross-architecture canary: the bits of `reflect_xyz` and `encode_srgb16` over a grid.
    #[test]
    fn golden_bits() {
        let optics = Optics::new();
        let mut hasher = Sha256::new();
        for &carbon in &[0.0, 0.0004, 0.003, 0.06, 0.4, 1.0, 3.0] {
            for &(mottle, ink_gain) in &[(1.0, 1.0), (0.7, 0.95), (1.3, 0.8)] {
                let (loads, paper) = sample(carbon, mottle, ink_gain);
                let xyz = optics.reflect_xyz(loads, paper);
                for v in xyz {
                    hasher.update(v.to_bits().to_le_bytes());
                }
                let (code, mapped) = optics.encode_srgb16(xyz);
                for c in code {
                    hasher.update(c.to_le_bytes());
                }
                hasher.update([u8::from(mapped)]);
            }
        }
        for row in &W_GALLERY {
            let (code, mapped) = optics.encode_srgb16(row.map(|w| 8.0 * w));
            for c in code {
                hasher.update(c.to_le_bytes());
            }
            hasher.update([u8::from(mapped)]);
        }
        assert_eq!(hex::encode(hasher.finalize()), GOLDEN_SHA256);
    }

    /// See [`golden_bits`]. A change means the shading or encoding changed on this machine. Last
    /// re-blessed for `ember-v2`: the shading takes the pine-soot load alone (no vermilion).
    const GOLDEN_SHA256: &str = "4d5efc2dab538bddbce0541f2614bde1b76b6669d8180dfe98eb4df57054a45a";

    /// Timing probe (not run by default):
    /// `cargo test --release --lib ember::optics::tests::timing -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing probe"]
    fn timing() {
        use std::hint::black_box;
        use std::time::Instant;
        let optics = Optics::new();
        let n = 4_000_000usize;
        let inputs: Vec<(f64, PaperSample)> = (0..n)
            .map(|i| {
                let t = (i % 1000) as f64 / 1000.0;
                sample(0.5 * t, 0.9 + 0.2 * t, 1.0 - 0.1 * t)
            })
            .collect();
        let start = Instant::now();
        let mut xyzs = Vec::with_capacity(n);
        for &(loads, paper) in &inputs {
            xyzs.push(optics.reflect_xyz(black_box(loads), black_box(paper)));
        }
        let reflect = start.elapsed().as_secs_f64() / n as f64;
        let start = Instant::now();
        let mut sum = 0u64;
        for &xyz in &xyzs {
            let (code, _) = optics.encode_srgb16(black_box(xyz));
            sum += u64::from(code[0]);
        }
        let encode = start.elapsed().as_secs_f64() / n as f64;
        black_box(sum);
        println!(
            "reflect_xyz: {:.1} ns/sample; encode_srgb16: {:.1} ns",
            reflect * 1e9,
            encode * 1e9
        );
    }
}
