//! Interactive exhibition page — a self-contained HTML file per NFT that
//! serves as a virtual gallery room with video, 3D viewer, audio, orbital
//! data, effect chain, and provenance information.

use crate::render::randomizable_config::ResolvedEffectConfig;
use crate::sim::TrajectoryResult;
use std::io::Write;
use tracing::info;

pub struct ExhibitionPageData<'a> {
    pub seed: &'a str,
    pub output_name: &'a str,
    pub result: &'a TrajectoryResult,
    pub config: &'a ResolvedEffectConfig,
    pub num_sims: usize,
    pub num_steps: usize,
}

pub fn generate_exhibition_page(
    data: &ExhibitionPageData<'_>,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating exhibition page...");

    let active_effects = collect_active_effects(data.config);
    let effects_html = build_effects_html(&active_effects);
    let survival_rate =
        (data.num_sims - data.result.discarded_count) as f64 / data.num_sims as f64 * 100.0;

    let seed_display = data.seed.strip_prefix("0x").unwrap_or(data.seed);

    let html = format!(
        r####"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cosmic Signature — 0x{seed}</title>
<meta name="description" content="Cosmic Signature NFT 0x{seed} — Three Body Problem visualization">
<meta property="og:image" content="../social/og_image.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@300;400&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0D0521;--bg2:#1A0B3E;--text:#F0EDFF;--muted:#8B7BAA;
  --accent:#6C3CE1;--teal:#00D4AA;--line:#2A1854;
  --serif:'Cormorant Garamond',Georgia,serif;
  --sans:'Inter',system-ui,sans-serif;
  --mono:'JetBrains Mono','Courier New',monospace;
}}
html{{scroll-behavior:smooth}}
body{{
  background:var(--bg);color:var(--text);
  font-family:var(--sans);font-weight:300;line-height:1.7;
  -webkit-font-smoothing:antialiased;
}}
.fade-in{{opacity:0;transform:translateY(30px);transition:opacity 0.8s ease,transform 0.8s ease}}
.fade-in.visible{{opacity:1;transform:translateY(0)}}
header{{
  text-align:center;padding:80px 24px 40px;
  border-bottom:1px solid var(--line);
}}
header h1{{
  font-family:var(--serif);font-weight:300;font-size:clamp(2rem,5vw,3.5rem);
  letter-spacing:0.06em;margin-bottom:8px;
}}
header .seed{{
  font-family:var(--mono);font-size:0.85rem;color:var(--accent);
  letter-spacing:0.1em;
}}
header .tagline{{
  font-family:var(--serif);font-style:italic;color:var(--muted);
  font-size:1.1rem;margin-top:16px;
}}
.hero{{
  max-width:1200px;margin:60px auto;padding:0 24px;text-align:center;
}}
.hero video{{
  width:100%;max-width:1080px;border-radius:4px;
  box-shadow:0 20px 80px rgba(108,60,225,0.15);
}}
section{{
  max-width:900px;margin:0 auto;padding:60px 24px;
  border-top:1px solid var(--line);
}}
section h2{{
  font-family:var(--serif);font-weight:300;font-size:1.8rem;
  letter-spacing:0.05em;margin-bottom:32px;color:var(--text);
}}
.data-grid{{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
  gap:24px;
}}
.data-item{{
  background:linear-gradient(135deg,rgba(26,11,62,0.6),rgba(13,5,33,0.8));
  border:1px solid var(--line);border-radius:8px;padding:20px;
}}
.data-item .label{{
  font-family:var(--mono);font-size:0.65rem;letter-spacing:0.15em;
  color:var(--muted);text-transform:uppercase;margin-bottom:6px;
}}
.data-item .value{{
  font-family:var(--mono);font-size:1.1rem;color:var(--teal);
}}
.effects-list{{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
  gap:12px;
}}
.effect-tag{{
  font-family:var(--mono);font-size:0.75rem;padding:8px 14px;
  background:rgba(108,60,225,0.15);border:1px solid var(--accent);
  border-radius:20px;text-align:center;color:var(--text);
}}
.viewer-container{{
  width:100%;aspect-ratio:16/9;background:#000;border-radius:4px;
  margin:24px 0;position:relative;overflow:hidden;
}}
.viewer-container canvas{{width:100%!important;height:100%!important}}
.provenance{{
  font-family:var(--mono);font-size:0.75rem;color:var(--muted);
  line-height:2;
}}
.provenance span{{color:var(--text)}}
.audio-toggle{{
  display:inline-flex;align-items:center;gap:8px;
  font-family:var(--mono);font-size:0.75rem;color:var(--muted);
  background:rgba(108,60,225,0.12);border:1px solid var(--line);
  border-radius:20px;padding:8px 20px;cursor:pointer;
  transition:border-color 0.3s;
}}
.audio-toggle:hover{{border-color:var(--accent)}}
footer{{
  text-align:center;padding:60px 24px;
  border-top:1px solid var(--line);
  font-family:var(--mono);font-size:0.65rem;color:var(--muted);
  letter-spacing:0.1em;
}}
</style>
</head>
<body>

<header class="fade-in">
  <h1>Cosmic Signature</h1>
  <div class="seed">0x{seed}</div>
  <div class="tagline">A visualization of the Three Body Problem</div>
</header>

<div class="hero fade-in">
  <video autoplay loop muted playsinline>
    <source src="../video.mp4" type="video/mp4">
  </video>
</div>

<section class="fade-in">
  <h2>Orbital Data</h2>
  <div class="data-grid">
    <div class="data-item">
      <div class="label">Chaos Score</div>
      <div class="value">{chaos:.6}</div>
    </div>
    <div class="data-item">
      <div class="label">Equilateralness</div>
      <div class="value">{equil:.6}</div>
    </div>
    <div class="data-item">
      <div class="label">Weighted Borda</div>
      <div class="value">{borda:.3}</div>
    </div>
    <div class="data-item">
      <div class="label">Selected From</div>
      <div class="value">{nsims} candidates</div>
    </div>
    <div class="data-item">
      <div class="label">Survival Rate</div>
      <div class="value">{surv:.1}%</div>
    </div>
    <div class="data-item">
      <div class="label">Integration Steps</div>
      <div class="value">{nsteps}</div>
    </div>
  </div>
</section>

<section class="fade-in">
  <h2>Effect Chain</h2>
  <div class="effects-list">
    {effects}
  </div>
</section>

<section class="fade-in">
  <h2>3D Orbital Viewer</h2>
  <div class="viewer-container" id="viewer3d"></div>
</section>

<section class="fade-in">
  <h2>Sonification</h2>
  <p style="color:var(--muted);font-size:0.9rem;margin-bottom:16px;">
    Orbital mechanics translated to sound — velocity maps to pitch, proximity maps to amplitude.
  </p>
  <button class="audio-toggle" onclick="toggleAudio()">
    <span id="audioIcon">&#9654;</span> Listen to this orbit
  </button>
  <audio id="orbitAudio" src="../audio/sonification.wav" preload="none"></audio>
</section>

<section class="fade-in">
  <h2>Provenance</h2>
  <div class="provenance">
    Seed: <span>0x{seed}</span><br>
    Resolution: <span>{width} &times; {height}</span><br>
    Pipeline: <span>SHA3-256 &rarr; Yoshida-4 &rarr; Borda &rarr; Spectral 16-bin</span><br>
    Render: <span>16-bit RGB &middot; H.265 10-bit &middot; 60fps</span>
  </div>
</section>

<footer>
  COSMIC SIGNATURE &middot; THREE BODY PROBLEM &middot; BUILT ON ARBITRUM
</footer>

<script type="importmap">
{{
  "imports": {{
    "three": "https://esm.sh/three@0.170.0",
    "three/addons/": "https://esm.sh/three@0.170.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

const container = document.getElementById('viewer3d');
if (container) {{
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0D0521);
  const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 10000);
  camera.position.set(0, 0, 500);
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.5;

  const loader = new GLTFLoader();
  loader.load('../3d/model.glb', (gltf) => {{
    const model = gltf.scene;
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    model.position.sub(center);
    const size = box.getSize(new THREE.Vector3()).length();
    camera.position.set(0, 0, size * 1.5);
    controls.update();
    scene.add(model);
  }});

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }}
  animate();

  window.addEventListener('resize', () => {{
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }});
}}
</script>
<script>
function toggleAudio() {{
  const audio = document.getElementById('orbitAudio');
  const icon = document.getElementById('audioIcon');
  if (audio.paused) {{ audio.play(); icon.textContent = '\u275A\u275A'; }}
  else {{ audio.pause(); icon.textContent = '\u25B6'; }}
}}
document.querySelectorAll('.fade-in').forEach(el => {{
  const obs = new IntersectionObserver(entries => {{
    entries.forEach(e => {{ if (e.isIntersecting) {{ e.target.classList.add('visible'); obs.unobserve(e.target); }} }});
  }}, {{ threshold: 0.1 }});
  obs.observe(el);
}});
</script>
</body>
</html>"####,
        seed = seed_display,
        chaos = data.result.chaos,
        equil = data.result.equilateralness,
        borda = data.result.total_score_weighted,
        nsims = data.num_sims,
        nsteps = data.num_steps,
        surv = survival_rate,
        effects = effects_html,
        width = data.config.width,
        height = data.config.height,
    );

    let mut file = std::fs::File::create(output_path)?;
    file.write_all(html.as_bytes())?;
    info!("   Saved exhibition page => {}", output_path);
    Ok(())
}

fn build_effects_html(effects: &[&str]) -> String {
    effects
        .iter()
        .map(|e| format!("    <div class=\"effect-tag\">{}</div>", e))
        .collect::<Vec<_>>()
        .join("\n")
}

fn collect_active_effects(config: &ResolvedEffectConfig) -> Vec<&'static str> {
    let mut effects = Vec::new();
    if config.enable_bloom { effects.push("DoG Bloom"); }
    if config.enable_glow { effects.push("Glow Enhancement"); }
    if config.enable_chromatic_bloom { effects.push("Chromatic Bloom"); }
    if config.enable_perceptual_blur { effects.push("Perceptual Blur"); }
    if config.enable_micro_contrast { effects.push("Micro Contrast"); }
    if config.enable_gradient_map { effects.push("Gradient Map"); }
    if config.enable_color_grade { effects.push("Color Grade"); }
    if config.enable_opalescence { effects.push("Opalescence"); }
    if config.enable_champleve { effects.push("Champlevé"); }
    if config.enable_aether { effects.push("Aether Weave"); }
    if config.enable_edge_luminance { effects.push("Edge Luminance"); }
    if config.enable_atmospheric_depth { effects.push("Atmospheric Depth"); }
    if config.enable_fine_texture { effects.push("Fine Texture"); }
    if config.nebula_strength > 0.0 { effects.push("Nebula Clouds"); }
    effects
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_effects_html_tags() {
        let effects = vec!["DoG Bloom", "Aether Weave"];
        let html = build_effects_html(&effects);
        assert!(html.contains("DoG Bloom"));
        assert!(html.contains("effect-tag"));
    }

    #[test]
    fn test_collect_active_effects_empty() {
        let config = make_test_config();
        assert!(collect_active_effects(&config).is_empty());
    }

    fn make_test_config() -> ResolvedEffectConfig {
        ResolvedEffectConfig {
            width: 1920, height: 1080,
            enable_bloom: false, enable_glow: false, enable_chromatic_bloom: false,
            enable_perceptual_blur: false, enable_micro_contrast: false,
            enable_gradient_map: false, enable_color_grade: false,
            enable_champleve: false, enable_aether: false, enable_opalescence: false,
            enable_edge_luminance: false, enable_atmospheric_depth: false,
            enable_fine_texture: false, nebula_strength: 0.0,
            blur_strength: 0.0, blur_radius_scale: 0.0, blur_core_brightness: 0.0,
            dog_strength: 0.0, dog_sigma_scale: 0.0, dog_ratio: 0.0,
            glow_strength: 0.0, glow_threshold: 0.0, glow_radius_scale: 0.0,
            glow_sharpness: 0.0, glow_saturation_boost: 0.0,
            chromatic_bloom_strength: 0.0, chromatic_bloom_radius_scale: 0.0,
            chromatic_bloom_separation_scale: 0.0, chromatic_bloom_threshold: 0.0,
            perceptual_blur_strength: 0.0, color_grade_strength: 0.0,
            vignette_strength: 0.0, vignette_softness: 0.0, vibrance: 0.0,
            clarity_strength: 0.0, tone_curve_strength: 0.0,
            gradient_map_strength: 0.0, gradient_map_hue_preservation: 0.0, gradient_map_palette: 0,
            opalescence_strength: 0.0, opalescence_scale: 0.0, opalescence_layers: 0,
            champleve_flow_alignment: 0.0, champleve_interference_amplitude: 0.0,
            champleve_rim_intensity: 0.0, champleve_rim_warmth: 0.0, champleve_interior_lift: 0.0,
            aether_flow_alignment: 0.0, aether_scattering_strength: 0.0,
            aether_iridescence_amplitude: 0.0, aether_caustic_strength: 0.0,
            micro_contrast_strength: 0.0, micro_contrast_radius: 0,
            edge_luminance_strength: 0.0, edge_luminance_threshold: 0.0,
            edge_luminance_brightness_boost: 0.0,
            atmospheric_depth_strength: 0.0, atmospheric_desaturation: 0.0,
            atmospheric_darkening: 0.0,
            atmospheric_fog_color_r: 0.0, atmospheric_fog_color_g: 0.0, atmospheric_fog_color_b: 0.0,
            fine_texture_strength: 0.0, fine_texture_scale: 0.0, fine_texture_contrast: 0.0,
            hdr_scale: 0.0, clip_black: 0.0, clip_white: 1.0,
            nebula_octaves: 0, nebula_base_frequency: 0.0,
        }
    }
}
