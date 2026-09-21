"""Explicit review options and film-only navigation over sparse film cohorts."""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from .review_page import MATERIALS, document


class ReviewPageTests(unittest.TestCase):
    def test_reference_identity_is_explicit_validated_and_defaults_to_rc1(self):
        self.assertEqual(MATERIALS.reference_variant, "rc1")
        for reference in ("rc1", "reference", "folded-tide"):
            page = document("Review", presentation=replace(MATERIALS, reference_variant=reference))
            settings = json.loads(re.search(r"const presentation = (.*?);", page).group(1))
            self.assertEqual(settings["reference_variant"], reference)
        for value in (None, True, 1, "", "RC1", "ref_one", "../reference", "a" * 81):
            with self.subTest(value=value), self.assertRaises(ValueError):
                document("Review", presentation=replace(MATERIALS, reference_variant=value))

    def test_film_only_setting_is_strict_and_legacy_default_is_false(self):
        self.assertFalse(MATERIALS.film_only_selection)
        for enabled in (False, True):
            page = document("Review", presentation=replace(MATERIALS, film_only_selection=enabled))
            settings = json.loads(re.search(r"const presentation = (.*?);", page).group(1))
            self.assertIs(settings["film_only_selection"], enabled)
        for value in (0, 1, None, "true", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                document("Review", presentation=replace(MATERIALS, film_only_selection=value))

    def data(self, *, reference_variant="rc1"):
        rows = []
        for seed, variants in (
            (
                "A",
                {
                    reference_variant: True,
                    "ovals": False,
                    "compact": True,
                    "ribbons": True,
                    "common": True,
                },
            ),
            ("B", {reference_variant: True, "ovals": False, "cross": True, "common": True}),
            ("C", {reference_variant: False, "ovals": False}),
        ):
            for variant, filmed in variants.items():
                stem = f"assets/{seed}/{variant}"
                rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "label": variant,
                        "description": "Study",
                        "image": stem + ".png",
                        "initial": stem + "-initial.png",
                        "preview": stem + "-preview.png",
                        "film": stem + ".mp4" if filmed else None,
                        "request": stem + ".json",
                        "resolution": [2048, 1536],
                        "initial_resolution": [1440, 1080],
                        "film_resolution": [1440, 1080],
                        "film_fps": 24,
                        "features": {},
                        "settings": {"treatment": variant},
                    }
                )
        return {
            "version": MATERIALS.version,
            "seeds": ["A", "B", "C"],
            "rows": rows,
            "picks": [
                {"seed": seed, "variant": variant, "note": "Visual pick"}
                for seed, variant in (
                    ("A", "ovals"),
                    ("A", "compact"),
                    ("B", "ovals"),
                    ("B", "cross"),
                )
            ],
        }

    def run_page(self, assertions, *, enabled, data=None, presentation=MATERIALS):
        page = document("Films", presentation=replace(presentation, film_only_selection=enabled))
        ids = re.findall(r'\bid="([^"]+)"', page)
        script = re.findall(r"<script>([\s\S]*?)</script>", page)[0]
        harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map(),events={},windowEvents={};
function element(tag='div'){return {tagName:tag.toUpperCase(),children:[],listeners:{},
 attributes:{},dataset:{},value:'',hidden:false,checked:false,pauses:0,loads:0,plays:0,
 style:{setProperty(){}},classList:{toggle(){}},
 get firstChild(){return this.children[0]},get selectedIndex(){
 return this.children.findIndex(option=>option.value===this.value)},
 set selectedIndex(index){this.value=this.children[index]?.value||''},
 setAttribute(k,v){this.attributes[k]=v},removeAttribute(k){delete this[k]},
 addEventListener(k,v){this.listeners[k]=v},
 replaceChildren(...items){this.children=items;if(this.tagName==='SELECT')this.value=items[0]?.value||''},
 append(...items){this.children.push(...items)},focus(){},pause(){this.pauses++},
 load(){this.loads++},play(){this.plays++}}}
for(const id of IDS)elements.set(id,element(['seed','left','right'].includes(id)?'select':'div'));
const get=id=>elements.get(id),emit=(id,event)=>get(id).listeners[event]?.();
const modes=['final','initial','film'].map(mode=>{
 const item=element('button');item.dataset.mode=mode;return item});
const videos=()=>['left','right'].flatMap(side=>
 get('art-'+side).children.filter(node=>node.tagName==='VIDEO'));
const document={getElementById:get,createElement:element,addEventListener(k,v){events[k]=v},
 body:element('body'),querySelectorAll(selector){
 return selector==='video'?videos():selector==='[data-mode]'?modes:[]}};
vm.runInNewContext(SCRIPT,{document,window:{addEventListener(k,v){windowEvents[k]=v},scrollTo(){}},
 fetch:async()=>({ok:true,json:async()=>DATA}),URL,location:'http://test/index.html',
 history:{replaceState(){}},matchMedia:()=>({matches:true})});
const options=id=>get(id).children.map(option=>option.value);
const mode=name=>modes.find(button=>button.dataset.mode===name).listeners.click();
const seed=value=>{get('seed').value=value;emit('seed','change')};
const choose=value=>{get('right').value=value;emit('right','change')};
setImmediate(()=>{
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "film-selections.js"
            path.write_text(
                "const IDS="
                + json.dumps(ids)
                + ";const DATA="
                + json.dumps(self.data() if data is None else data)
                + ";const SCRIPT="
                + json.dumps(script)
                + ";\n"
                + harness
                + assertions
                + "\n});\n"
            )
            result = subprocess.run(
                [shutil.which("node"), str(path)], capture_output=True, text=True, timeout=10
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Requires Node.js for film-only selection")
    def test_films_fall_back_per_seed_and_image_modes_restore_all_studies(self):
        self.run_page(
            r"""
assert.equal(get('left').value,'rc1');assert.equal(get('right').value,'ovals');
assert.equal(videos().length,0);assert.deepEqual(options('seed'),['A','B','C']);
mode('film');assert.equal(get('right').value,'compact');
assert.deepEqual(options('seed'),['A','B']);
assert.deepEqual(options('right'),['rc1','compact','ribbons','common']);
assert.equal(get('grid').children.length,4);
assert.equal(get('grid-title').textContent,'Available films for this seed');
assert.match(get('count').textContent,/4 available films/);
assert.ok(videos().every(video=>video.src&&video.plays===0));
const previous=[...videos()];emit('next','click');
assert.equal(get('seed').value,'B');assert.equal(get('right').value,'cross');
assert.deepEqual(options('right'),['rc1','cross','common']);
assert.ok(previous.every(video=>video.src===undefined&&video.pauses>0&&video.loads>0));
choose('common');seed('A');assert.equal(get('right').value,'common');
// A user's current filmed selection takes priority over the seed's visual pick.
mode('final');assert.deepEqual(options('seed'),['A','B','C']);
assert.deepEqual(options('right'),['rc1','ovals','compact','ribbons','common']);
assert.equal(get('grid').children.length,5);assert.equal(videos().length,0);
mode('initial');seed('C');assert.deepEqual(options('right'),['rc1','ovals']);
mode('film');assert.equal(get('seed').value,'A');assert.equal(get('right').value,'compact');
assert.equal(videos().length,2);assert.ok(videos().every(video=>video.src));
get('only-picks').checked=true;emit('only-picks','change');
assert.equal(get('grid').children.length,1); // Still-only picks are omitted.
get('grid').firstChild.listeners.click();assert.equal(get('right').value,'compact');
""",
            enabled=True,
        )

    @unittest.skipUnless(shutil.which("node"), "Requires Node.js for legacy selection")
    def test_disabled_option_keeps_legacy_still_only_selection_in_film_mode(self):
        self.run_page(
            r"""
assert.equal(get('right').value,'ovals');mode('film');
assert.equal(get('right').value,'ovals');assert.deepEqual(options('seed'),['A','B','C']);
assert.ok(options('right').includes('ovals'));
assert.match(get('art-right').firstChild.textContent,/has not been rendered/);
seed('C');assert.equal(videos().length,0);
assert.match(get('art-left').firstChild.textContent,/has not been rendered/);
""",
            enabled=False,
        )

    @unittest.skipUnless(shutil.which("node"), "Requires Node.js for empty film cohort")
    def test_no_movies_disables_film_mode_without_hiding_images(self):
        data = self.data()
        for row in data["rows"]:
            row["film"] = None
        self.run_page(
            r"""
assert.equal(modes.find(button=>button.dataset.mode==='film').disabled,true);
mode('film');assert.equal(videos().length,0);
assert.equal(get('art-left').firstChild.tagName,'IMG');
assert.deepEqual(options('seed'),['A','B','C']);
""",
            enabled=True,
            data=data,
        )

    @unittest.skipUnless(shutil.which("node"), "Requires Node.js for custom reference selection")
    def test_non_rc1_reference_drives_fallback_chips_and_film_controls(self):
        data = self.data(reference_variant="reference")
        data["picks"] = []
        # Put the baseline last to prove selection uses its identity, not order.
        data["rows"].sort(key=lambda row: row["variant"] == "reference")
        for row in data["rows"]:
            if row["seed"] == "C" and row["variant"] == "ovals":
                row["film"] = "assets/C/ovals.mp4"
        self.run_page(
            r"""
assert.equal(get('status').className,undefined);
assert.equal(get('left').value,'reference');assert.equal(get('right').value,'ovals');
assert.deepEqual(options('right'),['ovals','compact','ribbons','common','reference']);
const referenceCards=()=>get('grid').children.filter(card=>
 card.children[1].children[1].children.some(chip=>chip.textContent==='Reference'));
assert.equal(referenceCards().length,1);
assert.equal(referenceCards()[0].children[1].firstChild.textContent,'reference');
mode('film');assert.equal(get('left').value,'reference');
assert.equal(get('right').value,'compact');assert.equal(get('zoom').disabled,true);
assert.ok(videos().every(video=>video.controls===true&&video.playsInline===true&&
 video.preload==='metadata'&&video.src&&video.plays===0));
const previous=[...videos()];get('left').value='ribbons';emit('left','change');
seed('B');assert.equal(get('left').value,'reference');assert.equal(get('right').value,'cross');
assert.ok(previous.every(video=>video.src===undefined&&video.pauses>0&&video.loads>0));
// A still-only reference must not enter film-only selections.
seed('C');assert.deepEqual(options('left'),['ovals']);
assert.equal(get('left').value,'ovals');assert.equal(get('right').value,'ovals');
assert.equal(referenceCards().length,0);assert.equal(videos().length,2);
mode('initial');assert.deepEqual(options('left'),['ovals','reference']);
assert.equal(videos().length,0);assert.equal(get('zoom').disabled,false);
assert.equal(referenceCards().length,1);
assert.match(get('art-right').firstChild.src,/-initial\.png$/);
""",
            enabled=True,
            data=data,
            presentation=replace(
                MATERIALS, reference_variant="reference", default_variant="absent"
            ),
        )

    @unittest.skipUnless(shutil.which("node"), "Requires Node.js for reference data validation")
    def test_non_rc1_reference_is_required_even_if_rc1_exists(self):
        self.run_page(
            r"""
assert.equal(get('status').className,'error');
assert.equal(get('status').textContent,'Invalid gallery data.');
assert.equal(videos().length,0);assert.equal(get('grid').children.length,0);
""",
            enabled=True,
            presentation=replace(MATERIALS, reference_variant="reference"),
        )


if __name__ == "__main__":
    unittest.main()
