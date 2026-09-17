#!/usr/bin/env python3
"""Publish a portable gallery beside the finished shell images and film."""

import argparse
from pathlib import Path


def build(folder: Path) -> None:
    if not (folder / "hero.png").is_file():
        raise ValueError("A rendered hero.png is required")
    motion = ""
    if (folder / "film.mp4").is_file():
        motion = """<section><div class="eyebrow">02 / Growth</div>
<h2>A moment becomes a rim.</h2><video controls playsinline preload="metadata" poster="hero.png">
<source src="film.mp4" type="video/mp4">Your browser cannot play this film.</video>
<p>The shell grows through the complete recording, then the camera examines its final form.</p>
<a class="download" href="film.mp4" download>Download the film</a></section>"""
    views = ""
    for filename, caption in [
        ("second-view.png", "The open spiral, from another angle"),
        ("detail.png", "Growth lines in the porcelain"),
    ]:
        if (folder / filename).is_file():
            views += (
                f'<figure><a href="{filename}" target="_blank" rel="noopener">'
                f'<img src="{filename}" alt="{caption}" loading="lazy"></a>'
                f"<figcaption>{caption}</figcaption></figure>"
            )
    studies = ""
    if (folder / "studies.jpg").is_file():
        studies = """<details><summary>The form studies</summary>
<p>Rendered comparisons of the open fan, scroll, and spiral.</p>
<a href="studies.jpg" target="_blank" rel="noopener">
<img src="studies.jpg" alt="Six actual form and material studies" loading="lazy"></a></details>"""
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Shell That Remembers</title><style>
:root{color-scheme:light;
font-family:Arial,Helvetica,sans-serif;
background:#eeece4;
color:#263c34}

*{box-sizing:border-box}
body{margin:0}
main{max-width:1380px;
margin:auto;
padding:58px 5vw 80px}

header{display:grid;
grid-template-columns:1fr 290px;
gap:40px;
align-items:end;
margin-bottom:38px}

.eyebrow{font-size:11px;
letter-spacing:.18em;
text-transform:uppercase;
color:#59675d}

h1,h2{font-family:Georgia,'Times New Roman',serif;
font-weight:400;
line-height:1.03;
letter-spacing:-.045em}

h1{font-size:clamp(40px,5.6vw,76px);
margin:18px 0 0}
h2{font-size:40px;
margin:16px 0 28px}

p{font-size:14px;
line-height:1.8;
color:#526258}
header p{margin:0 0 4px}

img,video{display:block;
width:100%;
height:auto;
background:#55574c}
a{color:inherit}

figure{margin:0}
.hero img{max-height:88vh;
object-fit:contain;
background:#eeece4}

.caption{display:flex;
justify-content:space-between;
gap:20px;
padding-top:16px;
font-size:12px;
color:#59675d}

.download{font-size:12px;
text-underline-offset:5px}
section{margin-top:80px}
section p{max-width:650px}

.views{display:grid;
grid-template-columns:1fr 1fr;
gap:24px;
margin-top:72px}
figcaption{font-size:12px;
margin-top:14px;
color:#59675d}

details{margin-top:70px;
padding-top:24px;
border-top:1px solid #c9cec2;
font-size:13px}
summary{cursor:pointer}

details img{margin-top:24px}
footer{margin-top:50px;
max-width:720px;
font-size:13px;
line-height:1.9;
color:#59675d}

@media(max-width:700px){main{padding:30px 20px 50px}
header{grid-template-columns:1fr;
gap:22px}
.views{grid-template-columns:1fr}

.caption{flex-direction:column;
gap:12px}
section{margin-top:55px}
h2{font-size:30px}
.hero img{max-height:none}
}

</style></head><body><main><header><div><div class="eyebrow">Three-body study / Growth</div>
<h1>The Shell<br>That Remembers</h1></div><p>Three bodies become a growing spiral.
Every new rim keeps a moment of their motion.</p></header>
<figure class="hero"><a href="hero.png" target="_blank" rel="noopener">
<img src="hero.png" alt="An open ivory shell with a celadon interior and fine growth ribs">
</a></figure>
<div class="caption"><span>01 / Ivory, celadon, and the space between</span>
<a class="download" href="hero.png" download>Download the still</a></div>"""
    document += motion + '<div class="views">' + views + "</div>" + studies
    document += """<footer>The triangle's changing proportions shape the rim. Its turning rotates
the growing profile; the bodies' travel sets the spacing of the ribs. Earlier sections stay fixed
as the object grows. The growth axis, porcelain, and photography are artistic choices.</footer>
</main></body></html>"""
    temporary = folder / "index.partial.html"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(folder / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    build(parser.parse_args().folder.resolve())
