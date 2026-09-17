#!/usr/bin/env python3
"""Publish a small, portable gallery from verified Estuary render archives."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.estuary.run import commit_file, completed, encoded, read_json, write_json

DOCUMENT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estuary</title>
<style>
:root{color-scheme:light;
background:#e9e6dc;
color:#273035;
font-family:Arial,sans-serif}
*{box-sizing:border-box}body{margin:0}main{max-width:1640px;
margin:auto;
padding:42px 4vw 60px}
header{display:flex;
align-items:end;
justify-content:space-between;
gap:24px;
margin-bottom:30px}
h1{font-family:Georgia,serif;
font-size:clamp(42px,6vw,76px);
font-weight:400;
letter-spacing:-.05em;
margin:8px 0 0}
.eyebrow{font-size:10px;
letter-spacing:.2em;
text-transform:uppercase;
color:#6b716d}
p{font-size:13px;
line-height:1.75;
max-width:490px;
color:#606763;
margin:12px 0 0}
select{max-width:240px;
font:12px Arial,sans-serif;
border:1px solid #c8cbc2;
color:#37443e;
background:transparent;
padding:10px 30px 10px 12px;
border-radius:0}
label{display:block;
font-size:11px;
color:#606763;
margin-bottom:8px}
.art{background:#071021}img,video{display:block;
width:100%;
height:auto;
max-height:84vh;
object-fit:contain}video[hidden]{display:none}
.toolbar{display:flex;
gap:20px;
align-items:center;
justify-content:space-between;
padding:16px 0;
font-size:11px;
color:#606763}
nav{display:flex;
gap:18px;
align-items:center}a,button{color:inherit;
text-underline-offset:4px}
button{background:none;
border:0;
border-bottom:1px solid #53615a;
padding:0 0 3px;
cursor:pointer;
font:inherit}
footer{border-top:1px solid #c9cdc3;
margin-top:30px;
padding-top:18px;
display:flex;
justify-content:space-between;
gap:24px;
font-size:11px;
color:#69716b;
line-height:1.7}
@media(max-width:640px){main{padding:26px 18px 36px}
header{display:block}
header aside{margin-top:22px}
select{max-width:100%;
width:100%}.toolbar,footer{align-items:start;
flex-direction:column;
gap:14px}nav{flex-wrap:wrap}}
</style>
</head>
<body>
<main>
<header>
<div>
<span class="eyebrow">Three bodies / A painting in time</span>
<h1>Estuary</h1>
<p>The complete motion leaves a painting of mineral white, blue and earth.
Each body carries its own pigment;
 their encounters stir the field.</p>
</div>
<aside>
<label for="seed">Study</label>
<select id="seed" aria-label="Choose a rendered study">
</select>
</aside>
</header>
<div class="art">
<img id="poster" alt="An Estuary painting formed by three moving bodies">
<video id="film" controls playsinline preload="metadata" hidden>
</video>
</div>
<div class="toolbar">
<span id="caption">Loading paintings…</span>
<nav>
<button id="mode" hidden>Watch the painting grow</button>
<a id="download" download>Download the still</a>
<a id="movieDownload" hidden download>Download the film</a>
</nav>
</div>
<footer>
<span id="history">
</span>
<span>Pigment and light, shaped by a recorded three-body trajectory.</span>
</footer>
</main>
<script>
const $=id=>document.getElementById(id);
let studies=[],motion=false;

function choose(){const item=studies[Number($('seed').value)];
if(!item)return;

$('film').pause();
motion=false;
$('film').hidden=true;
$('poster').hidden=false;

$('poster').src=item.poster;
$('download').href=item.poster;
$('caption').textContent=item.seed+' · '+item.resolution.join(' x ');

$('mode').hidden=!item.film;
$('mode').textContent='Watch the painting grow';
$('movieDownload').hidden=!item.film;

if(item.film){$('film').src=item.film;
$('film').poster=item.poster;
$('movieDownload').href=item.film}else{$('film').removeAttribute('src')}
$('history').textContent=new Intl.NumberFormat().format(item.source_samples)
+' recorded moments · complete trajectory';
}
$('seed').onchange=choose;
$('mode').onclick=()=>{motion=!motion;
$('film').hidden=!motion;
$('poster').hidden=motion;

$('mode').textContent=motion?'Return to the still':'Watch the painting grow';
if(motion){$('film').play().catch(()=>{})}else{$('film').pause()}};

fetch('collection.json',{cache:'no-store'}).then(r=>{
if(!r.ok)throw Error('The collection is unavailable');
return r.json()}).then(data=>{
studies=data.studies;
for(const [i,item] of studies.entries()){const option=document.createElement('option');
option.value=String(i);
option.textContent=item.seed;
$('seed').append(option)}choose();

}).catch(error=>{$('caption').textContent=error.message});

</script>
</body>
</html>
"""


def build_gallery(directories, output):
    """Copy only media whose completed render archives still verify exactly."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    studies = []
    seen = set()
    for directory in directories:
        source = Path(directory).resolve(strict=True)
        request = read_json(source / "request.json")
        identity = hashlib.sha256(encoded(request)).hexdigest()
        if not completed(source, identity):
            raise ValueError(f"Cannot publish an incomplete render: {source}")
        if identity in seen:
            raise ValueError("The same render was supplied twice")
        seen.add(identity)
        receipt = read_json(source / "receipt.json")
        folder = output / identity[:16]
        folder.mkdir(exist_ok=True)
        names = ["poster.png", "receipt.json", "request.json"]
        if receipt["mode"] == "film":
            names += ["film.mp4"]
        for name in names:
            partial = folder / (name + ".partial")
            shutil.copyfile(source / name, partial)
            commit_file(partial, folder / name)
        studies.append(
            {
                "seed": receipt["seed"],
                "identity": identity,
                "poster": f"{folder.name}/poster.png",
                "film": f"{folder.name}/film.mp4" if receipt["mode"] == "film" else None,
                "resolution": receipt["resolution"],
                "source_samples": receipt["source"]["samples"],
                "source": receipt["source"],
            }
        )
    if not studies:
        raise ValueError("Supply at least one completed Estuary render")
    write_json(output / "collection.json", {"schema_version": 1, "studies": studies})
    partial = output / "index.html.partial"
    partial.write_text(DOCUMENT, encoding="utf-8")
    partial.replace(output / "index.html")
    return output / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_gallery(args.renders, args.output))
