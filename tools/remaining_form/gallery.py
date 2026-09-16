#!/usr/bin/env python3
"""Build a portable, dependency-free gallery beside verified delivered media."""

import argparse
from pathlib import Path


def build(directory: Path) -> None:
    if not (directory / "hero.png").is_file():
        raise ValueError("A completed hero.png is required")
    film = ""
    if (directory / "film.mp4").is_file():
        film = """<section aria-labelledby="motion"><div class="heading">
<span>02 / The excavation</span><h2 id="motion">Time removes. A form remains.</h2></div>
<video controls playsinline preload="metadata" poster="hero.png">
<source src="film.mp4" type="video/mp4">Your browser cannot play this film.</video>
<p>The complete recorded motion, followed by a quiet examination of the surviving form.</p>
</section>"""
    views = []
    for name, caption in [
        ("second-view.png", "Another view of the same object"),
        ("detail.png", "Inside the chamber"),
    ]:
        if (directory / name).is_file():
            views.append(
                f'<figure><a href="{name}" target="_blank" rel="noopener">'
                f'<img src="{name}" alt="{caption}" loading="lazy"></a>'
                f"<figcaption>{caption}</figcaption></figure>"
            )
    studies = ""
    if (directory / "studies.jpg").is_file():
        studies = """<details><summary>The form studies</summary>
<p>Actual geometry proofs used to choose the blank, opening, and pose.</p>
<a href="studies.jpg" target="_blank" rel="noopener">
<img src="studies.jpg" alt="Comparison of the earlier clay studies" loading="lazy"></a>
</details>"""
    html = (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Remaining Form — a three-body sculpture</title>
<style>
:root{color-scheme:dark;font-family:Arial,Helvetica,sans-serif;background:#141514;color:#eae6db}
*{box-sizing:border-box}body{margin:0}main{max-width:1360px;margin:auto;padding:64px 5vw 80px}
header{display:flex;justify-content:space-between;align-items:end;gap:30px;margin-bottom:42px}
.eyebrow,.heading span{font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:#b3ad9e}
h1,h2{font-family:Georgia,'Times New Roman',serif;font-weight:400;line-height:1.08}
h1{font-size:clamp(42px,5.5vw,78px);margin:18px 0 0;letter-spacing:-.045em}
header p{max-width:300px;font-size:14px;line-height:1.7;color:#bdb8ad;margin:0 0 5px}
img,video{display:block;width:100%;height:auto;background:#20211f}a{color:inherit}
.hero{margin:0}.hero img{max-height:84vh;object-fit:contain}
.caption{display:flex;justify-content:space-between;gap:20px;padding-top:16px;color:#aaa699;font-size:12px}
.caption a{text-underline-offset:4px}.heading{margin:72px 0 26px}h2{font-size:36px;margin:14px 0 0}
section>p{max-width:600px;font-size:13px;line-height:1.8;color:#b3ad9e}
.views{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:70px}.views figure{margin:0}
figcaption{font-size:12px;color:#b3ad9e;padding-top:13px;line-height:1.6}
details{border-top:1px solid #383932;margin-top:66px;padding-top:24px;font-size:13px;color:#b3ad9e}
summary{cursor:pointer;color:#ddd8ca}details p{line-height:1.7}details img{margin-top:20px}
footer{margin-top:62px;max-width:710px;font-size:13px;line-height:1.9;color:#aaa699}
@media(max-width:650px){main{padding:32px 20px 52px}header{display:block;margin-bottom:26px}
header p{margin-top:24px}.views{grid-template-columns:1fr}.caption{flex-direction:column;gap:10px}
h2{font-size:29px}.heading{margin-top:48px}.hero img{max-height:none}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body><main>
<header><div><div class="eyebrow">An experiment in sculpture · 01</div>
<h1>The Remaining Form</h1></div>
<p>Three moving bodies carve a single volume. The sculpture is the material they leave behind.</p>
</header><figure class="hero"><a href="hero.png" target="_blank" rel="noopener">
<img src="hero.png" alt="Porcelain sculpture shaped by three-body motion"></a></figure>
<div class="caption"><span>01 / The surviving object · porcelain study</span>
<a href="hero.png" download>Download the still</a></div>
"""
        + film
        + '<div class="views">'
        + "".join(views)
        + "</div>"
        + studies
        + """
<footer>This is a photograph of generated three-dimensional geometry. Each body contributes
the same carving tool; lingering and returning deepen the excavation. The camera, light,
starting blank, and porcelain are deliberate artistic choices.</footer>
</main></body></html>"""
    )
    (directory / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build(parser.parse_args().directory.resolve())
