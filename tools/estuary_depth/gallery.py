#!/usr/bin/env python3
"""Publish verified depth studies with their common, frozen painting reference."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.estuary_depth.experiment import encoded, finished, preserve, read, write

DOCUMENT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estuary / Into depth</title>
<style>
:root{font-family:Arial,sans-serif;
color:#3d443e;
background:#ece9e0;
color-scheme:light}
*{box-sizing:border-box}body{margin:0}main{max-width:1600px;
margin:auto;
padding:26px 4vw 60px}
header{display:flex;
justify-content:space-between;
gap:24px;
align-items:end;
margin-bottom:18px}
h1,h2{font-family:Georgia,serif;
font-weight:400;
letter-spacing:-.04em;
margin:0}
h1{font-size:clamp(34px,3.6vw,46px);
line-height:1.04}h2{font-size:30px}
.eyebrow{font-size:10px;
margin:0 0 9px;
line-height:1;
letter-spacing:.18em;
text-transform:uppercase;
color:#788074}
p{font-size:13px;
line-height:1.7;
max-width:500px;
color:#687266}
header p:not(.eyebrow){font-size:12px;
line-height:1.45;
margin:8px 0 0;
max-width:550px}
select{font:12px Arial,sans-serif;
max-width:min(100%,440px);
text-overflow:ellipsis;
color:inherit;
padding:10px 14px;
background:none;
border:1px solid #bbc1b5}
.artRow.comparing{display:grid;
grid-template-columns:minmax(0,1fr) minmax(0,1fr);
gap:18px;
align-items:start}
.art{background:#dedbd0;
min-width:0}img,video{display:block;
width:100%;
height:auto;
object-fit:contain}
.art>img,.art>video{max-height:max(240px,calc(100svh - 260px))}
[hidden]{display:none!important}.caption{display:flex;
justify-content:space-between;
align-items:baseline;
flex-wrap:wrap;
gap:12px;
padding:12px 0}
.caption span,.caption a,.caption button{font-size:12px;
color:#6c7668}
a{color:inherit;
text-underline-offset:4px}button{font:inherit;
color:inherit;
cursor:pointer}
.caption button{background:none;
border:0;
border-bottom:1px solid #798274;
padding:0 0 3px}
nav{display:flex;
flex-wrap:wrap;
gap:12px 16px}#motionCaption{font-size:11px;
margin:0 0 10px}.studies{margin-top:50px}.grid{display:grid;
grid-template-columns:repeat(3,1fr);
gap:24px;
margin-top:24px}
.card{text-align:left;
background:none;
border:0;
padding:0}.card img{aspect-ratio:4/3;
object-fit:cover}
.card strong{display:block;
font-size:13px;
font-weight:400;
margin-top:12px}.card small{font-size:11px;
color:#7a8177}
.reference{margin:0;
min-width:0}
.reference img,.artRow.comparing .art>img{max-height:max(240px,calc(100svh - 285px))}
.comparisonLabel{margin:6px 0 0;
font-size:11px;
line-height:1.25}
footer{margin-top:45px;
border-top:1px solid #cbd0c2;
padding-top:18px;
font-size:11px;
color:#727d6b}
@media(max-width:760px){main{padding:22px 18px 40px}
header{display:block}
.artRow.comparing{grid-template-columns:minmax(0,1fr);
gap:12px}
.art>img,.art>video,.reference img,
.artRow.comparing .art>img{max-height:max(240px,calc(100svh - 320px))}
header select{width:100%;
max-width:100%;
margin-top:12px}
.grid{grid-template-columns:repeat(2,1fr);
gap:18px}.caption{display:block}.caption nav{margin-top:10px}}
</style>
</head>
<body>
<main>
<header>
<div>
<p class="eyebrow">Estuary / Into depth</p>
<h1>Paint, given a body.</h1>
<p>Paintings, held at their completed moment.
Relief, translucent layers and raking light reveal their depth.</p>
</div>
<select id="choice" aria-label="Choose a depth study">
</select>
</header>
<div class="artRow" id="artRow">
<div class="art">
<img id="hero" alt="A rendered Estuary depth study">
<video id="film" controls playsinline preload="metadata" hidden>
</video>
</div>
<section class="reference" id="reference" hidden>
<img id="baseline" alt="The original flat Estuary painting">
<p class="comparisonLabel">Original painting</p>
</section>
</div>
<div class="caption">
<span id="caption">Loading the studies…</span>
<nav>
<button id="compare" hidden aria-pressed="false">Compare original</button>
<button id="motion" hidden>Camera orbit</button>
<button id="formation" hidden>Formation and orbit</button>
<button id="stillView" hidden>Still image</button>
<a id="fullSize" target="_blank" rel="noopener">Open full size</a>
<a id="download" download>Download the image</a>
</nav>
</div>
<p id="motionCaption" hidden>Camera orbit of the completed painting</p>
<section class="studies">
<h2>The studies</h2>
<div class="grid" id="grid">
</div>
</section>
<footer id="source">These studies explore material and light over complete
recorded trajectories.</footer>
</main>
<script>
const $=id=>document.getElementById(id);
let collection=null,comparing=false,referenceAvailable=false;
const families={relief:'Relief',layered:'Buried layers',hybrid:'Relief and glass'};
const seedLabel=seed=>seed?String(seed).replace(/^0x/i,'').slice(0,4).toUpperCase():'';
const studyDetails=study=>[families[study.group]||study.group,seedLabel(study.seed)]
.filter(Boolean).join(' · ');

function updateComparison(){
$('artRow').classList.toggle('comparing',comparing);
$('reference').hidden=!comparing;
$('compare').hidden=!referenceAvailable;
$('compare').textContent=comparing?'Hide original':'Compare original';
$('compare').setAttribute('aria-pressed',String(comparing));}

function select(index){const study=collection.studies[index];
$('choice').value=String(index);
$('stillView').hidden=true;

$('film').pause();
$('film').hidden=true;
$('hero').hidden=false;
$('hero').src=study.image;

$('hero').alt=study.name;
$('caption').textContent=study.name+' · '+studyDetails(study);
$('motionCaption').hidden=true;
const reference=study.baseline||collection.baseline;
referenceAvailable=Boolean(reference);
if(!referenceAvailable)comparing=false;
if(reference){$('baseline').src=reference}else{$('baseline').removeAttribute('src')}
updateComparison();

$('download').href=study.image;
$('fullSize').href=study.image;
$('motion').hidden=!study.film;
$('formation').hidden=!study.formation;
$('film').removeAttribute('src');
$('film').load();
$('film').poster=study.image;}
$('choice').onchange=()=>select(Number($('choice').value));
$('compare').onclick=()=>{const next=!comparing;
select(Number($('choice').value));
comparing=next&&referenceAvailable;
updateComparison();};

function playMotion(kind){const study=collection.studies[Number($('choice').value)];
if(!study[kind])return;
comparing=false;
updateComparison();
$('film').pause();
$('film').src=study[kind];
$('hero').hidden=true;
$('film').hidden=false;
$('stillView').hidden=false;
$('motionCaption').textContent=kind==='film'?'Camera orbit of the completed painting':
'Formation of the painting, followed by its camera orbit';
$('motionCaption').hidden=false;
$('film').play().catch(()=>{});}
$('motion').onclick=()=>playMotion('film');
$('formation').onclick=()=>playMotion('formation');
$('stillView').onclick=()=>select(Number($('choice').value));

fetch('collection.json',{cache:'no-store'}).then(r=>{
if(!r.ok)throw Error('The gallery is unavailable');
return r.json()}).then(data=>{
collection=data;
for(const [index,study] of data.studies.entries()){const option=document.createElement('option');
option.value=String(index);
option.textContent=study.name+(study.seed?' · '+seedLabel(study.seed):'');
$('choice').append(option);

const card=document.createElement('button');
card.className='card';
card.onclick=()=>{select(index);
$('hero').scrollIntoView({behavior:'smooth',block:'center'})};

const image=document.createElement('img');
image.src=study.image;
image.alt=study.name;
image.loading='lazy';

const title=document.createElement('strong');
title.textContent=study.name;
const group=document.createElement('small');
group.textContent=studyDetails(study);
card.append(image,title,group);
$('grid').append(card)}
if(data.studies.length)select(0)
}).catch(error=>{$('caption').textContent=error.message});

</script>
</body>
</html>
"""


def build_gallery(root, recipes, baseline=None):
    root = Path(root)
    request = read(root / "experiment-request.json")
    identity = hashlib.sha256(encoded(request)).hexdigest()
    output = root / "gallery"
    output.mkdir(exist_ok=True)
    studies = []
    for path in recipes:
        case = root / path.stem
        if not finished(case, identity):
            raise ValueError(f"Cannot publish an unfinished study: {path.stem}")
        recipe = read(root / "inputs/recipes" / path.name)
        result = read(case / "experiment-result.json")
        target = output / path.stem
        target.mkdir(exist_ok=True)
        preserve(target / "render.png", case / "render.png")
        movie = None
        if result.get("movie"):
            preserve(target / "film.mp4", case / "film.mp4")
            movie = f"{path.stem}/film.mp4"
        group = recipe.get("family", recipe.get("mode", recipe.get("kind", "Depth study")))
        studies.append(
            {
                "id": path.stem,
                "name": recipe.get("name", path.stem),
                "group": group,
                "seed": read(case / "receipt.json")["source"]["seed"],
                "image": f"{path.stem}/render.png",
                "film": movie,
            }
        )
    baseline_name = None
    if baseline:
        baseline = Path(baseline)
        baseline_name = "baseline" + baseline.suffix
        preserve(output / baseline_name, root / "inputs" / baseline_name)
    write(
        output / "collection.json",
        {
            "schema_version": 1,
            "studies": studies,
            "baseline": baseline_name,
            "history": "All studies use the same frozen paint history",
        },
    )
    partial = output / "index.html.partial"
    partial.write_text(DOCUMENT)
    partial.replace(output / "index.html")
    return output / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    request = read(args.experiment / "experiment-request.json")
    recipes = [Path(name) for name in request["files"] if name.startswith("recipes/")]
    baseline = next((Path(name) for name in request["files"] if name.startswith("baseline")), None)
    identity = hashlib.sha256(encoded(request)).hexdigest()
    recipes = [path for path in recipes if finished(args.experiment / path.stem, identity)]
    print(build_gallery(args.experiment, recipes, baseline))
