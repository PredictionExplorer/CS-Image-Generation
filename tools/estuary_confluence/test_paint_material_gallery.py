"""Pinned RC1 comparison provenance and accessible bounded media controls."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import unittest
from unittest.mock import patch

from tools.estuary_studio.common import artifact, encoded, read, write

from . import paint_material_gallery as gallery
from . import test_gallery as fixtures
from .backgrounds import generate_background
from .paint_material_studies import DEFAULT_SEEDS, material_recipe, references
from .run import frame_plan, interaction_metadata, rheology_metadata, surface_configs


class PaintMaterialGalleryTests(unittest.TestCase):
    def setUp(self):
        # Reuse receipt-bound tiny media. Numerical archive verification has its
        # own GPU tests; this fixture still uses real normalized RC1 recipes,
        # seeded palettes, spectral material and optional-history metadata.
        self.fixture = fixtures.GalleryTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.output = self.root / "material-review"
        self.inputs_file, self.release_file = self.root / "inputs.json", self.root / "release.json"
        self.cases = []
        bound = references()
        release = read(gallery.RELEASE)
        self.inputs = {**bound, "cases": []}
        self.release = {**release, "cases": []}
        for index, seed in enumerate(DEFAULT_SEEDS[:2]):
            accepted, reference, _ = material_recipe(seed, "rc1")
            baseline = self.case(seed, "rc1", film=index == 0)
            _, receipt = self.fixture.verified_case(baseline)
            pinned = copy.deepcopy(accepted)
            for view in pinned["views"]:
                view["image"].update(receipt["artifacts"][view["look"] + "/poster.png"])
            self.release["cases"].append(pinned)
            reference = copy.deepcopy(reference)
            reference["layout_artifact"] = receipt["artifacts"]["layout.json"]
            reference["mass_budget_artifact"] = receipt["artifacts"]["mass-budget.json"]
            self.inputs["cases"].append(reference)
            self.cases.append(baseline)
            self.cases.append(
                self.case(seed, "traits" if index == 0 else "fuller", film=index == 0)
            )
        write(self.release_file, self.release)
        self.inputs["reference_release_sha256"] = artifact(self.release_file)["sha256"]
        write(self.inputs_file, self.inputs)
        for name, path in (("INPUTS", self.inputs_file), ("RELEASE", self.release_file)):
            context = patch.object(gallery, name, path)
            context.start()
            self.addCleanup(context.stop)

    def case(self, seed, variant, *, film=False):
        accepted, reference, recipe = material_recipe(seed, variant)
        path = self.fixture.case(
            seed,
            3,
            recipe["looks"],
            palette_mode="composed",
            scattered=True,
            spectral=True,
            source_hash=reference["source_sha256"],
            film=film,
        )
        request, receipt = self.fixture.verified_case(path)
        request["recipe"] = recipe
        request["frames"] = frame_plan(recipe) if film else []
        request["background"] = generate_background(recipe["background"], request["palette"])
        receipt["final_step"] = recipe["simulation"]["steps"]
        receipt["base_material_sha256"] = accepted["base_material_sha256"]
        physical = accepted["physical_state_sha256"]
        if "material_variation" in recipe["simulation"]["interaction"]:
            physical = hashlib.sha256((seed + variant).encode()).hexdigest()
        receipt["physical_state_sha256"] = physical
        for look in recipe["looks"]:
            receipt["looks"][look]["physical_state_sha256"] = physical
            if film:
                receipt["looks"][look]["movie"] = {
                    "frames": 937,
                    "fps": 24,
                    "resolution": [1440, 1080],
                    "full_decode_verified": True,
                }
        write(
            path / "mass-budget.json",
            {
                "initial_mass": [*reference["target_mass"], 0.0],
                "interval_steps": recipe["simulation"]["mass_budget_interval_steps"],
            },
        )
        self.refresh(path, request, receipt)
        return path

    def refresh(self, path, request, receipt):
        request["surface_configs"] = surface_configs(request["recipe"])
        request["interaction"] = interaction_metadata(request["recipe"], request["source"]["seed"])
        receipt["interaction"] = copy.deepcopy(request["interaction"])
        rheology = rheology_metadata(request["recipe"])
        if rheology is not None:
            request["rheology"] = rheology
            receipt["rheology"] = copy.deepcopy(rheology)
        receipt["source"] = copy.deepcopy(request["source"])
        for name in ("layout", "events", "palette", "spectral", "background"):
            write(path / f"{name}.json", request[name])
            receipt["artifacts"][f"{name}.json"] = artifact(path / f"{name}.json")
        receipt["artifacts"]["mass-budget.json"] = artifact(path / "mass-budget.json")
        receipt["identity_sha256"] = hashlib.sha256(encoded(request)).hexdigest()
        write(path / "request.json", request)
        write(path / "receipt.json", receipt)

    def build(self, cases=None, **kwargs):
        return gallery.build_review(self.output, self.cases if cases is None else cases, **kwargs)

    def rehash(self, name):
        record = read(self.output / "publication.json")
        record["artifacts"][name] = artifact(self.output / name)
        if name == "comparison.json":
            record["comparison_sha256"] = hashlib.sha256(
                encoded(read(self.output / name))
            ).hexdigest()
        write(self.output / "publication.json", record)

    def test_portable_stills_and_films_are_bound_to_exact_recipes_and_rc1(self):
        data = self.build(
            picks=[{"seed": DEFAULT_SEEDS[0], "variant": "traits", "note": "Quiet surface detail."}]
        )
        self.assertEqual(len(data["rows"]), 4)
        self.assertEqual(len(data["seeds"]), 2)
        self.assertEqual(sum(row["film"] is not None for row in data["rows"]), 2)
        for row in data["rows"]:
            self.assertEqual(row["resolution"], [2048, 1536])
            self.assertEqual(row["initial_resolution"], [1440, 1080])
            for key in ("image", "initial", "preview", "request"):
                self.assertTrue(row[key].startswith("studies/"))
                self.assertTrue((self.output / row[key]).is_file())
        for path in self.cases:
            shutil.rmtree(path)
        self.inputs_file.unlink()
        self.release_file.unlink()
        self.assertEqual(gallery.verify_review(self.output), data)

    def test_missing_baseline_and_duplicate_variant_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires its RC1"):
            self.build(self.cases[1:])
        shutil.rmtree(self.output)
        duplicate = self.root / "duplicate"
        shutil.copytree(self.cases[1], duplicate)
        request, receipt = self.fixture.verified_case(duplicate)
        request["fixture_annotation"] = "another archive with the same recipe"
        self.refresh(duplicate, request, receipt)
        with self.assertRaisesRegex(
            ValueError, "Repeated seed and material|Ambiguous duplicate seed"
        ):
            self.build([*self.cases, duplicate])

    def test_cohort_cannot_consistently_replace_the_pinned_source(self):
        for path in self.cases[:2]:
            request, receipt = self.fixture.verified_case(path)
            request["source"]["sha256"] = "e" * 64
            self.refresh(path, request, receipt)
        with self.assertRaisesRegex(ValueError, "source or palette differs from pinned RC1"):
            self.build()

    def test_layout_mass_events_and_recipe_drift_are_rejected(self):
        path = self.cases[1]
        original_request, original_receipt = self.fixture.verified_case(path)
        original_budget = read(path / "mass-budget.json")
        for attack in ("layout", "mass", "events", "recipe", "metadata"):
            with self.subTest(attack=attack):
                request, receipt = copy.deepcopy(original_request), copy.deepcopy(original_receipt)
                write(path / "mass-budget.json", original_budget)
                if attack == "layout":
                    request["layout"]["pools"][0]["position"][0] += 0.1
                elif attack == "mass":
                    budget = copy.deepcopy(original_budget)
                    budget["initial_mass"][0] *= 1.1
                    write(path / "mass-budget.json", budget)
                elif attack == "events":
                    request["events"][0]["fraction"] += 0.01
                elif attack == "recipe":
                    request["recipe"]["simulation"]["flow_strength"] += 0.01
                self.refresh(path, request, receipt)
                if attack == "metadata":
                    receipt["interaction"]["version"] = "forged"
                    write(path / "receipt.json", receipt)
                with self.assertRaises(ValueError):
                    self.build()
                shutil.rmtree(self.output)

    def test_material_identity_and_preserved_rc1_image_are_required(self):
        for case_index, key, message in (
            (0, "physical_state_sha256", "complete RC1"),
            (1, "base_material_sha256", "original RC1"),
            (3, "physical_state_sha256", "complete RC1"),
        ):
            with self.subTest(case_index=case_index, key=key):
                path = self.cases[case_index]
                request, receipt = self.fixture.verified_case(path)
                original = copy.deepcopy(receipt)
                receipt[key] = "f" * 64
                if key == "physical_state_sha256":
                    for view in receipt["looks"].values():
                        view[key] = receipt[key]
                self.refresh(path, request, receipt)
                with self.assertRaisesRegex(ValueError, message):
                    self.build()
                shutil.rmtree(self.output)
                self.refresh(path, request, original)
        path = self.cases[0]
        request, receipt = self.fixture.verified_case(path)
        shutil.copyfile(path / "control/poster.png", path / "silk-grain/poster.png")
        receipt["artifacts"]["silk-grain/poster.png"] = artifact(path / "silk-grain/poster.png")
        self.refresh(path, request, receipt)
        with self.assertRaisesRegex(ValueError, "reference painting differs"):
            self.build()

    def test_visual_picks_need_known_distinct_identities_and_nonempty_notes(self):
        for picks in (
            [{"seed": "0x1", "variant": "rc1", "note": "Unknown"}],
            [{"seed": DEFAULT_SEEDS[0], "variant": "unknown", "note": "Unknown"}],
            [{"seed": DEFAULT_SEEDS[0], "variant": "rc1", "note": " "}],
            [{"seed": DEFAULT_SEEDS[0], "variant": "rc1", "note": "Fine"}] * 2,
        ):
            with self.subTest(picks=picks), self.assertRaises(ValueError):
                self.build(picks=picks)
            shutil.rmtree(self.output)

    def test_rehashed_catalog_cannot_change_media_paths_metadata_or_template(self):
        self.build()
        original = read(self.output / "comparison.json")
        for key, value in (
            ("image", "../outside.png"),
            ("initial", original["rows"][-1]["initial"]),
            ("features", {}),
            ("label", "Another painting"),
            ("initial_resolution", [2048, 1536]),
        ):
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed["rows"][0][key] = value
                write(self.output / "comparison.json", changed)
                self.rehash("comparison.json")
                with self.assertRaisesRegex(ValueError, "differs from its verified provenance"):
                    gallery.verify_review(self.output)
        write(self.output / "comparison.json", original)
        self.rehash("comparison.json")
        (self.output / "index.html").write_text("<p>Unbound replacement</p>")
        self.rehash("index.html")
        with self.assertRaisesRegex(ValueError, "qualified template"):
            gallery.verify_review(self.output)

    def test_publisher_refuses_overwriting_a_review_and_escapes_titles(self):
        self.build()
        before = artifact(self.output / "publication.json")
        with self.assertRaisesRegex(ValueError, "new immutable"):
            self.build()
        self.assertEqual(artifact(self.output / "publication.json"), before)
        page = gallery.document('<script>alert("title")</script>')
        self.assertEqual(len(re.findall(r"<script>", page)), 1)
        self.assertNotIn('<script>alert("title")</script>', page)

    @unittest.skipUnless(shutil.which("node"), "Requires Node.js for gallery controls")
    def test_media_cleanup_stale_errors_keyboard_and_touch_detail_controls(self):
        data = self.build()
        page = gallery.document("Controls")
        ids = re.findall(r'\bid="([^"]+)"', page)
        self.assertEqual(len(ids), len(set(ids)))
        script = re.findall(r"<script>([\s\S]*?)</script>", page)[0]
        harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map(),all=[],events={},windowEvents={};
function element(tag='div'){const node={tagName:tag.toUpperCase(),children:[],listeners:{},
 attributes:{},dataset:{},value:'',hidden:false,checked:false,pauses:0,loads:0,plays:0,
 style:{values:{},setProperty(k,v){this.values[k]=v}},
 classList:{values:new Set(),toggle(k,v){if(v)this.values.add(k);else this.values.delete(k)}},
 get firstChild(){return this.children[0]},setAttribute(k,v){this.attributes[k]=v},
 removeAttribute(k){delete this[k]},addEventListener(k,v){this.listeners[k]=v},
 replaceChildren(...items){this.children=items;if(this.tagName==='SELECT')this.value=items[0]?.value||''},
 append(...items){this.children.push(...items)},focus(){},pause(){this.pauses++},
 load(){this.loads++},play(){this.plays++},setPointerCapture(id){this.capture=id},
 getBoundingClientRect(){return {left:0,top:0,width:200,height:100}}};all.push(node);return node;}
for(const id of IDS)elements.set(id,element(['seed','left','right'].includes(id)?'select':'div'));
const get=id=>elements.get(id),emit=(id,event,value={})=>get(id).listeners[event]?.(value);
const modes=['final','initial','film'].map(mode=>{
 const b=element('button');b.dataset.mode=mode;return b});
const videos=()=>['left','right'].flatMap(side=>
 get('art-'+side).children.filter(n=>n.tagName==='VIDEO'));
const document={getElementById:get,createElement:element,addEventListener(k,v){events[k]=v},
 body:element('body'),
 querySelectorAll(selector){return selector==='video'?videos():selector==='[data-mode]'?modes:[]}};
const window={scrollY:143,addEventListener(k,v){windowEvents[k]=v},
 scrollTo({top}){this.scrollY=top}};
vm.runInNewContext(SCRIPT,{document,window,
 fetch:async()=>({ok:true,json:async()=>DATA}),URL,location:'http://test/index.html',
 history:{replaceState(){}},matchMedia:()=>({matches:true})});
emit('next','click');emit('zoom','click'); // Safe while data is still loading.
setImmediate(()=>{
 assert.equal(get('left').value,'rc1');assert.notEqual(get('right').value,'rc1');
 assert.equal(videos().length,0);assert.equal(all.filter(n=>n.tagName==='VIDEO').length,0);
 modes[1].listeners.click();assert.match(get('caption-left').textContent,/1440 \u00d7 1080/);
 assert.equal(get('links-left').children[0].href,DATA.rows.find(r=>r.seed===get('seed').value&&r.variant==='rc1').initial);
 emit('zoom','click');assert.equal(get('art-left').tabIndex,0);
 assert.equal(get('art-left').attributes['aria-describedby'],'mode-note');
 let prevented=0;emit('art-left','keydown',{key:'ArrowRight',preventDefault(){prevented++}});
 assert.equal(get('art-left').style.values['--x'],'55%');
 assert.equal(get('art-right').style.values['--x'],'55%');assert.equal(prevented,1);
 emit('art-right','keydown',{key:'Home',preventDefault(){}});
 assert.equal(get('art-left').style.values['--x'],'50%');
 emit('art-left','pointerdown',{pointerType:'touch',pointerId:7,currentTarget:get('art-left')});
 assert.equal(get('art-left').capture,7);
 emit('art-left','pointermove',{currentTarget:get('art-left'),clientX:180,clientY:10});
 assert.equal(get('art-right').style.values['--x'],'90%');
 assert.equal(get('art-right').style.values['--y'],'10%');
 modes[2].listeners.click();assert.equal(videos().length,2);assert.equal(get('zoom').disabled,true);
 const old=[...videos()];assert.ok(old.every(v=>v.src&&v.plays===0));
 old.forEach(video=>{video.currentTime=13.4});
 const mediaState=()=>videos().map(v=>[v.src,v.currentTime,v.loads,v.pauses,v.plays]);
 const beforeFocus=mediaState();
 emit('focus','click');assert.equal(document.body.classList.values.has('focus-paintings'),true);
 assert.equal(get('focus').attributes['aria-pressed'],'true');assert.equal(window.scrollY,0);
 assert.deepEqual(mediaState(),beforeFocus);assert.equal(videos()[0],old[0]);
 events.keydown({key:'Escape',preventDefault(){}});
 assert.equal(document.body.classList.values.has('focus-paintings'),false);
 assert.equal(get('focus').attributes['aria-pressed'],'false');assert.equal(window.scrollY,143);
 assert.deepEqual(mediaState(),beforeFocus);assert.equal(videos()[1],old[1]);
 get('right').value='rc1';emit('right','change');
 assert.ok(old.every(v=>v.src===undefined&&v.loads>0&&v.pauses>0));
 const caption=get('caption-right').textContent;old[1].listeners.error();
 assert.equal(get('caption-right').textContent,caption); // Detached source errors are stale.
 const current=[...videos()];windowEvents.pagehide();
 assert.ok(current.every(v=>v.src===undefined&&v.loads>0));
 windowEvents.pageshow({persisted:true});assert.equal(videos().length,2);
 assert.ok(videos().every(v=>v.src&&v.plays===0));
 get('seed').value=DATA.seeds[1];emit('seed','change');
 assert.equal(videos().length,0);
 assert.match(get('art-left').firstChild.textContent,/has not been rendered/);
 modes[0].listeners.click();get('only-picks').checked=true;emit('only-picks','change');
 assert.match(get('grid').firstChild.textContent,/No visual picks/);
 DATA.seeds.splice(1);DATA.rows=DATA.rows.filter(row=>row.seed===DATA.seeds[0]);
 get('seed').value=DATA.seeds[0];emit('seed','change');
 assert.match(get('status').textContent,/across 1 seed ·/);
 assert.doesNotMatch(get('status').textContent,/1 seeds/);
});
"""
        program = self.root / "material-controls.js"
        program.write_text(
            "const IDS="
            + json.dumps(ids)
            + ";const DATA="
            + json.dumps(data)
            + ";const SCRIPT="
            + json.dumps(script)
            + ";\n"
            + harness
        )
        result = subprocess.run(
            [shutil.which("node"), str(program)], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
