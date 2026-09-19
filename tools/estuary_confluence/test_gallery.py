"""Portable views, exact pigment swatches, and honest material comparisons."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.estuary.optics import srgb_to_linear
from tools.estuary.run import write_png
from tools.estuary_studio.common import artifact, encoded, read, write

from . import gallery
from .layout import plan_layout
from .palette import generate_palette, normalize_seed


class GalleryTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Requires Node.js to exercise film review")
    def test_film_review_playlist_skips_stills_and_switches_stop_old_playback(self):
        cases = [
            self.case(
                seed=seed,
                count=5,
                looks=["layered"],
                palette_mode="composed",
                simulation_updates={"material_model": "laminate"},
                film=film,
            )
            for seed, film in (("0xb7", True), ("0xbc", False), ("0x80", True))
        ]
        cases.append(self.case(seed="0xb7", count=5, looks=["layered"], palette_mode="harmonic"))
        gallery.build_gallery(self.output, cases, layout="films", allow_stills=True)
        data = read(self.output / "collection.json")
        script = re.findall(
            r"<script>([\s\S]*?)</script>", gallery.document("Films", layout="films")
        )[0]
        document = gallery.document("Films", layout="films")
        ids = re.findall(r'\bid="([^"]+)"', document)
        self.assertEqual(len(ids), len(set(ids)))
        harness = """
const vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();
let focused=null;
function element(){return {attributes:{},style:{},children:[],value:'',open:false,
 listeners:{},classList:{toggle(){}},setAttribute(k,v){this.attributes[k]=v},
 set id(v){this._id=v;elements.set(v,this)},get id(){return this._id},
 getAttribute(k){return this[k]},removeAttribute(k){delete this[k]},
 replaceChildren(){this.children=[]},append(...items){this.children.push(...items)},
 pause(){this.pauses=(this.pauses||0)+1},load(){},play(){return Promise.resolve()},
 focus(){focused=this.id},addEventListener(k,fn){this.listeners[k]=fn},
 showModal(){this.open=true},close(){this.open=false;this.listeners.close?.()},
 scrollIntoView(){}}}
for(const id of IDS){const item=element();item.id=id;}
const document={getElementById(id){return elements.get(id)||null},
 createElement:element,addEventListener(){},documentElement:{style:{overflow:''}}};
vm.runInNewContext(SCRIPT,{document,fetch:async()=>({ok:true,json:async()=>DATA})});
const get=id=>elements.get(id);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
get('next').onclick();get('modeFilm').onclick(); // Safe during loading.
setImmediate(async()=>{
 const current=DATA.studies.filter(s=>s.palette_mode==='composed');
 const earlier=DATA.studies.find(s=>s.palette_mode==='harmonic');
 assert.equal(get('viewer').open,false);
 assert.equal(get('film').src,undefined); // No hidden initial movie download.
 assert.equal(get('grid').children.length,3);
 get('film-'+current[0].seed).onclick();
 assert.equal(get('viewer').open,true);assert.equal(get('film').src,current[0].film);
 get('film').onended(); // Ordinary playback restores a visible finished image.
 assert.equal(get('film').hidden,true);assert.equal(get('film').src,undefined);
 assert.equal(get('hero').hidden,false);assert.equal(get('hero').src,current[0].image);
 get('playAllViewer').onclick();get('film').onended();
 assert.equal(get('film').src,current[2].film); // Skip the still-only seed.
 get('film').onended();assert.match(get('status').textContent,/All 2 films/);
 assert.equal(get('film').hidden,true);assert.equal(get('hero').src,current[2].image);
 get('closeViewer').onclick();assert.equal(get('viewer').open,false);
 assert.equal(document.documentElement.style.overflow,'');
 assert.equal(focused,'film-'+current[0].seed);
 get('image-'+current[1].seed).onclick();
 assert.equal(get('hero').src,current[1].image);assert.equal(get('modeFilm').disabled,true);
 get('closeViewer').onclick();
 get('playAll').onclick();assert.equal(get('film').src,current[0].film);
 get('modeCompare').onclick();assert.equal(get('reference').src,earlier.image);
 assert.equal(get('film').src,undefined);assert.equal(get('film').hidden,true);
 get('modeFilm').onclick();get('versionChoice').value=earlier.id;
 get('versionChoice').onchange();assert.equal(get('film').src,earlier.film);
 assert.equal(get('downloadFilm').href,earlier.film);assert.equal(get('openFilm').href,earlier.film);
 get('film').onerror();assert.equal(get('hero').hidden,false);
 assert.equal(get('film').src,undefined);assert.equal(get('error').hidden,false);
 get('film').play=()=>Promise.reject(Object.assign(new Error('blocked'),{name:'NotAllowedError'}));
 get('playAllViewer').onclick();await tick();
 assert.equal(get('film').hidden,true);assert.equal(get('hero').hidden,false);
 assert.match(get('playAll').textContent,/Play all/); // Failed autoplay cancels the queue.
 const rejects=[];get('film').play=()=>new Promise((resolve,reject)=>rejects.push(reject));
 get('modeFilm').onclick();get('modeFilm').onclick();
 rejects[0](Object.assign(new Error('old attempt'),{name:'NotAllowedError'}));await tick();
 assert.equal(get('film').hidden,false);assert.equal(get('film').src,earlier.film);
 get('closeViewer').onclick();
 rejects[1](Object.assign(new Error('closed'),{name:'NotAllowedError'}));await tick();
 assert.equal(get('film').src,undefined);assert.equal(get('viewer').open,false);
 assert.equal(focused,'playAll');assert.ok(get('film').pauses>1);
});
"""

        path = self.root / "film-controls.js"
        path.write_text(
            "const IDS="
            + json.dumps(ids)
            + ";\nconst DATA="
            + json.dumps(data)
            + ";\nconst SCRIPT="
            + json.dumps(script)
            + ";\n"
            + harness
        )
        result = subprocess.run(
            [shutil.which("node"), str(path)], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_film_review_compares_new_material_with_the_same_earlier_trajectory(self):
        baseline = self.case(count=5, looks=["layered"], palette_mode="harmonic")
        layered = self.case(
            count=5,
            looks=["layered"],
            palette_mode="composed",
            physical="e" * 64,
            simulation_updates={"material_model": "laminate"},
        )
        gallery.build_gallery(self.output, [layered, baseline], layout="films")
        collection, _ = gallery.verify_gallery(self.output)
        first, second = collection["studies"]
        self.assertEqual(first["material_model"], "laminate")
        self.assertEqual(first["comparison_id"], second["id"])
        self.assertEqual(second["comparison_id"], first["id"])
        self.assertNotEqual(first["physical_state_sha256"], second["physical_state_sha256"])
        first["material_model"] = "legacy"
        self.rehash_collection(collection)
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            gallery.verify_gallery(self.output)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for script validation")
    def test_film_review_script_parses_and_escapes_its_title(self):
        document = gallery.document('<script>alert("title")</script>', layout="films")
        self.assertNotIn('<script>alert("title")</script>', document)
        scripts = re.findall(r"<script>([\s\S]*?)</script>", document)
        self.assertEqual(len(scripts), 1)
        path = self.root / "review.js"
        path.write_text(scripts[0])
        subprocess.run(
            [shutil.which("node"), "--check", str(path)], check=True, capture_output=True
        )

    def test_rehashed_review_preview_cannot_depict_a_different_painting(self):
        from PIL import Image

        case = self.case(count=5, looks=["layered"])
        gallery.build_gallery(self.output, [case], layout="films")
        collection, curation = gallery.verify_gallery(self.output)
        name = collection["studies"][0]["preview"]
        with Image.open(self.output / name) as source:
            pixels = np.array(source)
        pixels[0, 0] = [255, 0, 0]
        Image.fromarray(pixels).save(self.output / name)
        curation["artifacts"][name] = artifact(self.output / name)
        write(self.output / "curation.json", curation)
        with self.assertRaisesRegex(ValueError, "does not depict"):
            gallery.verify_gallery(self.output)

    def experiment_cases(self):
        cases = []
        for count, label, flow in (
            (1, "One pigment", 0.9),
            (2, "Two pigments", 0.9),
            (2, "Two pigments · Tidal folds", 0.15),
            (3, "Three pigments", 0.9),
            (5, "Five pigments", 0.9),
        ):
            for seed in ("0xb7", "0xbc"):
                cases.append(
                    self.case(
                        seed,
                        count,
                        ["layered"] if count == 1 else ["layered", "homogeneous"],
                        palette_mode="composed",
                        recipe_name=label,
                        simulation_updates={"pair_swirl": flow},
                        physical=hashlib.sha256(f"{seed}/{count}/{flow}".encode()).hexdigest(),
                        scattered=True,
                    )
                )
        return cases

    def interaction_case(self, looks=None):
        """Complete interaction provenance over the lightweight publication fixture."""
        from .engine import validate_config as simulation_config
        from .run import interaction_metadata, surface_configs
        from .surface import validate_config as optical_config

        case = self.case(
            looks=["control", "silk", "silk-grain"] if looks is None else looks,
            recipe_name="Contact texture comparison",
            scattered=True,
        )
        request, receipt = self.verified_case(case)
        recipe = request["recipe"]
        recipe["simulation"] = simulation_config(
            {
                **recipe["simulation"],
                "material_model": "laminate",
                "settling_scale": 0,
                "underpaint_strength": 0,
                "underpaint_release": 0,
                "burial_rate": 0,
                "interaction": {},
            }
        )
        recipe["surface"] = optical_config({"interaction": {}})
        request["surface_configs"] = surface_configs(recipe)
        request["interaction"] = interaction_metadata(recipe, request["source"]["seed"])
        receipt["interaction"] = copy.deepcopy(request["interaction"])
        receipt["base_material_sha256"] = hashlib.sha256(b"unchanged base material").hexdigest()
        receipt["identity_sha256"] = hashlib.sha256(encoded(request)).hexdigest()
        write(case / "request.json", request)
        write(case / "receipt.json", receipt)
        return case

    def test_interaction_triplet_publishes_matched_views_and_portable_material_identities(self):
        case = self.interaction_case()
        gallery.build_gallery(self.output, [case], layout="studies")
        collection, _ = gallery.verify_gallery(self.output)
        by_group = {study["group"]: study for study in collection["studies"]}
        self.assertEqual(set(by_group), {"control", "silk", "silk-grain"})
        request, receipt = self.verified_case(case)
        for group, target in (
            ("control", "silk-grain"),
            ("silk", "control"),
            ("silk-grain", "control"),
        ):
            study, reference = by_group[group], by_group[target]
            self.assertEqual(study["comparison_id"], reference["id"])
            self.assertEqual(study["baseline"], reference["image"])
            self.assertEqual(study["case_id"], reference["case_id"])
            self.assertEqual(study["physical_state_sha256"], reference["physical_state_sha256"])
            self.assertEqual(study["base_material_sha256"], receipt["base_material_sha256"])
            self.assertEqual(study["interaction_version"], request["interaction"]["version"])
            self.assertEqual(
                study["comparison_caption"], f"{gallery.GROUPS[target]} · same material history"
            )
        shutil.rmtree(case)
        # Neither raw simulation arrays nor the original case directory are
        # required to retain the published provenance association.
        gallery.verify_gallery(self.output)

    def test_interaction_control_falls_back_to_silk_without_inventing_missing_comparisons(self):
        case = self.interaction_case(["control", "silk"])
        gallery.build_gallery(self.output, [case], layout="studies")
        collection, _ = gallery.verify_gallery(self.output)
        control, silk = collection["studies"]
        self.assertEqual(control["comparison_id"], silk["id"])
        self.assertEqual(silk["comparison_id"], control["id"])
        isolated = copy.deepcopy(silk)
        self.assertEqual(
            gallery._experiment_comparisons([isolated])[isolated["id"]], (None, None, None)
        )
        changed = copy.deepcopy(silk)
        changed["physical_state_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "different physical states"):
            gallery._experiment_comparisons([control, changed])

    def test_interaction_looks_require_studies_layout_for_publication(self):
        case = self.interaction_case()
        for layout in ("classic", "films"):
            with self.subTest(layout=layout), self.assertRaisesRegex(ValueError, "studies gallery"):
                gallery.build_gallery(self.output, [case], layout=layout)
        self.assertFalse(self.output.exists())

    def test_portable_interaction_collection_cannot_drop_its_required_layout(self):
        gallery.build_gallery(self.output, [self.interaction_case()], layout="studies")
        collection = read(self.output / "collection.json")
        collection.pop("layout")
        self.rehash_collection(collection)
        curation = read(self.output / "curation.json")
        curation.pop("layout")
        write(self.output / "curation.json", curation)
        with self.assertRaisesRegex(ValueError, "studies gallery"):
            gallery.verify_gallery(self.output)

    def test_rehashed_published_interaction_claims_remain_bound_to_portable_receipt(self):
        gallery.build_gallery(self.output, [self.interaction_case()], layout="studies")
        original = read(self.output / "collection.json")
        for key, value in (
            ("interaction_version", "invented-v1"),
            ("base_material_sha256", "f" * 64),
        ):
            with self.subTest(field=key, changed=True):
                changed = copy.deepcopy(original)
                changed["studies"][0][key] = value
                self.rehash_collection(changed)
                with self.assertRaisesRegex(ValueError, "metadata differs"):
                    gallery.verify_gallery(self.output)
            with self.subTest(field=key, missing=True):
                changed = copy.deepcopy(original)
                changed["studies"][0].pop(key)
                self.rehash_collection(changed)
                with self.assertRaisesRegex(ValueError, "missing published interaction"):
                    gallery.verify_gallery(self.output)

    def test_rehashed_portable_receipt_cannot_change_interaction_seed_or_version(self):
        gallery.build_gallery(self.output, [self.interaction_case()], layout="studies")
        curation = read(self.output / "curation.json")
        record = curation["sources"][0]["receipt"]
        original = read(self.output / record)
        for key, value in (("seed", "0x1"), ("version", "invented-v1")):
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed["interaction"][key] = value
                write(self.output / record, changed)
                curation["artifacts"][record] = artifact(self.output / record)
                write(self.output / "curation.json", curation)
                with self.assertRaisesRegex(ValueError, "Published interaction version, seed"):
                    gallery.verify_gallery(self.output)

    def test_disabled_publication_cannot_advertise_interaction_material(self):
        gallery.build_gallery(self.output, [self.case()], layout="studies")
        collection = read(self.output / "collection.json")
        self.assertTrue(
            all(
                not gallery.INTERACTION_PUBLIC_FIELDS.intersection(study)
                for study in collection["studies"]
            )
        )
        collection["studies"][0]["interaction_version"] = "contact-microstructure-v1"
        self.rehash_collection(collection)
        with self.assertRaisesRegex(ValueError, "Unbound or missing published interaction"):
            gallery.verify_gallery(self.output)

    def test_named_experiments_retain_collections_and_only_compare_the_same_case(self):
        gallery.build_gallery(self.output, self.experiment_cases(), layout="studies")
        collection, curation = gallery.verify_gallery(self.output)
        self.assertEqual(collection["layout"], "studies")
        self.assertEqual(curation["layout"], "studies")
        studies = {s["id"]: s for s in collection["studies"]}
        keys = {s["collection_key"] for s in studies.values()}
        self.assertEqual(len(keys), 9)
        for key in keys:
            group = [s for s in studies.values() if s["collection_key"] == key]
            self.assertEqual(len(group), 2)
            self.assertEqual({s["seed"] for s in group}, {"0xb7", "0xbc"})
            self.assertEqual(len({s["collection_label"] for s in group}), 1)
        for study in studies.values():
            self.assertIn(study["preview"], curation["artifacts"])
            self.assertIn(study["initial"], curation["artifacts"])
            self.assertEqual(
                study["collection_label"],
                study["experiment_label"] + " · " + gallery.GROUPS[study["group"]],
            )
            if study["chromatic_count"] == 1:
                self.assertIsNone(study["comparison_id"])
                self.assertIsNone(study["baseline"])
                self.assertEqual(study["name"], "Composed colors · 1 color")
            else:
                paired = studies[study["comparison_id"]]
                self.assertEqual(study["case_id"], paired["case_id"])
                self.assertEqual(study["physical_state_sha256"], paired["physical_state_sha256"])
                self.assertNotEqual(study["group"], paired["group"])
                self.assertIn("same material history", study["comparison_caption"])

    def test_experiment_labels_or_cross_flow_comparisons_cannot_be_rehashed_away(self):
        gallery.build_gallery(self.output, self.experiment_cases(), layout="studies")
        original, _ = gallery.verify_gallery(self.output)
        for key, value in (
            ("experiment_label", "Invented label"),
            ("collection_label", "A different experiment"),
            ("collection_key", "f" * 64),
        ):
            changed = copy.deepcopy(original)
            changed["studies"][0][key] = value
            self.rehash_collection(changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "experiment label"):
                gallery.verify_gallery(self.output)
        changed = copy.deepcopy(original)
        target = next(
            s for s in changed["studies"] if s["experiment_label"].endswith("Tidal folds")
        )
        first = next(s for s in changed["studies"] if s["experiment_label"] == "Two pigments")
        first["comparison_id"], first["baseline"] = target["id"], target["image"]
        self.rehash_collection(changed)
        with self.assertRaisesRegex(ValueError, "material-history association"):
            gallery.verify_gallery(self.output)
        self.rehash_collection(original)
        gallery.verify_gallery(self.output)

    def test_same_seed_and_named_trial_require_one_unambiguous_case(self):
        cases = [
            self.case(
                count=2,
                looks=["layered"],
                recipe_name="Two pigments",
                simulation_updates={"pair_swirl": swirl},
            )
            for swirl in (0.1, 0.9)
        ]
        with self.assertRaisesRegex(ValueError, "Ambiguous duplicate seed"):
            gallery.build_gallery(self.output, cases, layout="studies")

    @unittest.skipUnless(shutil.which("node"), "Requires Node.js to exercise study navigation")
    def test_study_controls_show_all_ten_seeds_per_trial_and_actual_starting_colors(self):
        cases = [
            self.case(
                hex(seed),
                count,
                ["layered"] if count == 1 else ["layered", "homogeneous"],
                palette_mode="composed",
                scattered=True,
                recipe_name=f"{count} starting pigments",
            )
            for count in (1, 2)
            for seed in range(10)
        ]
        # A deliberately partial trial must not borrow another experiment's
        # images for missing seeds or enqueue films from a different trial.
        cases.append(
            self.case("0x0", 3, ["layered"], recipe_name="Partial trial", palette_mode="composed")
        )
        gallery.build_gallery(self.output, cases, layout="studies")
        data = read(self.output / "collection.json")
        document = gallery.document("Pigment studies", layout="studies")
        ids = re.findall(r'\bid="([^"]+)"', document)
        script = re.findall(r"<script>([\s\S]*?)</script>", document)[0]
        harness = """
const vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();
function element(){return {attributes:{},style:{},children:[],value:'',open:false,
 listeners:{},classList:{toggle(){}},setAttribute(k,v){this.attributes[k]=v},
 set id(v){this._id=v;elements.set(v,this)},get id(){return this._id},
 getAttribute(k){return this[k]},removeAttribute(k){delete this[k]},
 replaceChildren(){this.children=[]},append(...items){this.children.push(...items)},
 pause(){},load(){},play(){return Promise.resolve()},focus(){},
 addEventListener(k,fn){this.listeners[k]=fn},showModal(){this.open=true},
 close(){this.open=false;this.listeners.close?.()}}}
for(const id of IDS){const item=element();item.id=id;}
const document={getElementById(id){return elements.get(id)||null},
 createElement:element,addEventListener(){},documentElement:{style:{overflow:''}}};
vm.runInNewContext(SCRIPT,{document,fetch:async()=>({ok:true,json:async()=>DATA})});
setImmediate(()=>{
 const get=id=>elements.get(id);
 assert.equal(get('viewer').open,false);
 assert.equal(get('grid').children.length,10);
 assert.equal(get('collectionChoice').children.length,4);
 const keys=[...new Set(DATA.studies.map(s=>s.collection_key))];
 for(const key of keys.slice(0,3)){
  const expected=DATA.studies.filter(s=>s.collection_key===key);
  get('collectionChoice').value=key;get('collectionChoice').onchange();
  assert.equal(get('grid').children.length,10);
  assert.equal(get('seedChoice').children.length,10);
  assert.match(get('playAll').textContent,/10 films/);
  for(const study of expected){
   get('image-'+study.seed).onclick();
   assert.equal(get('hero').src,study.image);
   get('modeInitial').onclick();assert.equal(get('hero').src,study.initial);
   assert.equal(get('fullSize').href,study.initial);
   assert.equal(get('downloadImage').href,study.initial);
   assert.equal(get('modeInitial').attributes['aria-pressed'],'true');
   get('modeFilm').onclick();assert.equal(get('film').src,study.film);
   get('modeInitial').onclick();assert.equal(get('film').src,undefined);
   if(study.comparison_id){
    get('modeCompare').onclick();
    assert.equal(get('reference').src,study.baseline);
    assert.equal(get('referenceLabel').textContent,study.comparison_caption);
   } else assert.equal(get('modeCompare').disabled,true);
   get('closeViewer').onclick();
  }
  get('playAll').onclick();
  for(const study of expected){
   assert.equal(get('film').src,study.film);get('film').onended();
  }
  assert.equal(get('hero').src,expected.at(-1).image);
  assert.equal(get('film').hidden,true);get('closeViewer').onclick();
 }
 const partial=DATA.studies.at(-1);
 get('collectionChoice').value=partial.collection_key;get('collectionChoice').onchange();
 assert.equal(get('grid').children.length,1);assert.equal(get('seedChoice').children.length,1);
 get('image-'+partial.seed).onclick();assert.equal(get('hero').src,partial.image);
 assert.equal(get('modeInitial').disabled,true);
 assert.equal(get('previous').disabled,true);assert.equal(get('next').disabled,true);
 assert.match(get('playAll').textContent,/1 film$/);
});
"""
        path = self.root / "study-controls.js"
        path.write_text(
            "const IDS="
            + json.dumps(ids)
            + ";\nconst DATA="
            + json.dumps(data)
            + ";\nconst SCRIPT="
            + json.dumps(script)
            + ";\n"
            + harness
        )
        result = subprocess.run(
            [shutil.which("node"), str(path)], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "gallery"
        self.verifier = patch.object(gallery, "verify_run", side_effect=self.verified_case)
        self.verifier.start()
        self.addCleanup(self.verifier.stop)

    @unittest.skipUnless(shutil.which("node"), "Requires Node.js to exercise focus restoration")
    def test_film_to_still_trial_close_restores_present_enabled_focus(self):
        cases = [
            self.case(
                "0x0",
                2,
                ["layered"],
                palette_mode="composed",
                scattered=True,
                recipe_name=name,
                film=film,
            )
            for name, film in (("Film trial", True), ("Still trial", False))
        ]
        gallery.build_gallery(self.output, cases, layout="studies", allow_stills=True)
        data = read(self.output / "collection.json")
        document = gallery.document("Focus review", layout="studies")
        ids = re.findall(r'\bid="([^"]+)"', document)
        script = re.findall(r"<script>([\s\S]*?)</script>", document)[0]
        harness = """
const vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();let focused=null;
function element(){return {attributes:{},style:{},children:[],value:'',open:false,
 listeners:{},classList:{toggle(){}},setAttribute(k,v){this.attributes[k]=v},
 set id(v){this._id=v;elements.set(v,this)},get id(){return this._id},
 getAttribute(k){return this[k]},removeAttribute(k){delete this[k]},
 detach(){if(this.id&&elements.get(this.id)===this)elements.delete(this.id);
  for(const child of this.children)child.detach()},
 replaceChildren(){for(const child of this.children)child.detach();this.children=[]},
 append(...items){this.children.push(...items)},pause(){},load(){},
 play(){return Promise.resolve()},focus(){if(!this.disabled)focused=this.id},
 addEventListener(k,fn){this.listeners[k]=fn},showModal(){this.open=true},
 close(){this.open=false;this.listeners.close?.()}}}
for(const id of IDS){const item=element();item.id=id;}
const document={getElementById(id){return elements.get(id)||null},
 createElement:element,addEventListener(){},documentElement:{style:{overflow:''}}};
vm.runInNewContext(SCRIPT,{document,fetch:async()=>({ok:true,json:async()=>DATA})});
setImmediate(()=>{
 const get=id=>elements.get(id),[film,still]=DATA.studies;
 get('film-'+film.seed).onclick();
 get('versionChoice').value=still.id;get('versionChoice').onchange();
 assert.equal(get('film-'+film.seed),undefined); // The old card really left the DOM.
 assert.equal(get('modeFilm').disabled,true);
 get('modeInitial').onclick();assert.equal(get('hero').src,still.initial);
 get('closeViewer').onclick();assert.equal(focused,'image-'+still.seed);
 assert.equal(get('viewer').open,false);assert.equal(get('film').src,undefined);
 assert.equal(document.documentElement.style.overflow,'');
 // A retained but disabled Play all opener is also not a focus destination.
 get('collectionChoice').value=film.collection_key;get('collectionChoice').onchange();
 get('playAll').onclick();
 get('versionChoice').value=still.id;get('versionChoice').onchange();
 assert.equal(get('playAll').disabled,true);
 focused=null;get('closeViewer').onclick();assert.equal(focused,'image-'+still.seed);
});
"""
        path = self.root / "focus-controls.js"
        path.write_text(
            "const IDS="
            + json.dumps(ids)
            + ";\nconst DATA="
            + json.dumps(data)
            + ";\nconst SCRIPT="
            + json.dumps(script)
            + ";\n"
            + harness
        )
        result = subprocess.run(
            [shutil.which("node"), str(path)], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @staticmethod
    def verified_case(case):
        request, receipt = read(case / "request.json"), read(case / "receipt.json")
        if not receipt["complete"]:
            raise ValueError("Incomplete source case")
        return request, receipt

    def case(
        self,
        seed="0xbc53af1cd380",
        count=3,
        looks=None,
        *,
        film=True,
        physical=None,
        source_hash=None,
        decode=True,
        orbit_frames=3,
        palette_mode="curated",
        scattered=False,
        simulation_updates=None,
        palette_updates=None,
        request_updates=None,
        spectral=False,
        assessed=False,
        formation_frames=6,
        recipe_name="Fixture",
    ):
        looks = ["layered", "homogeneous"] if looks is None else looks
        path = self.root / f"case-{len(list(self.root.glob('case-*')))}"
        path.mkdir()
        palette = generate_palette(seed, count, mode=palette_mode)
        if palette_updates:
            palette.update(copy.deepcopy(palette_updates))
            palette.pop("identity_sha256")
            palette["identity_sha256"] = hashlib.sha256(
                json.dumps(palette, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
        source_hash = source_hash or hashlib.sha256(normalize_seed(seed).encode()).hexdigest()
        physical = (
            physical or hashlib.sha256(f"{normalize_seed(seed)}/{count}".encode()).hexdigest()
        )
        request = {
            "source": {"seed": seed, "sha256": source_hash},
            "palette": palette,
            "events": [{"fraction": 0.3, "position": [0.1, -0.1], "strength": 0.7}],
            "recipe": {
                "name": recipe_name,
                "chromatic_count": count,
                "looks": looks,
                "simulation": {
                    "steps": 10,
                    "resolution": [128, 96],
                    "initial_pattern": "scattered" if scattered else "pools",
                    "load_radius": 0.28,
                    "initial_load": 0.18,
                    "initial_edge_width": 0.02,
                    "deposition": 0 if scattered else 0.025,
                },
                "render": {
                    "still_resolution": [16, 12],
                    "resolution": [16, 12],
                    "formation_frames": formation_frames,
                    "hold_frames": 2,
                    "orbit_frames": orbit_frames,
                    "fps": 24,
                },
            },
            "mode": "film" if film else "still",
        }
        request["recipe"]["simulation"].update(simulation_updates or {})
        if scattered:
            request["recipe"]["palette_mode"] = palette_mode
            simulation = request["recipe"]["simulation"]
            request["layout"] = plan_layout(
                seed,
                count,
                4 / 3,
                load_radius=simulation["load_radius"],
                initial_load=simulation["initial_load"],
                edge_width=simulation["initial_edge_width"],
            )
        elif palette_mode != "curated":
            request["recipe"]["palette_mode"] = palette_mode
        if spectral:
            from .spectral import build_spectral_material

            request["recipe"]["surface"] = {"optics_model": "spectral"}
            request["spectral"] = build_spectral_material(palette)
        if assessed:
            request["recipe"]["assessment"] = {
                "interval_steps": 5,
                "resolution": [128, 96],
                "share_threshold": 0.1,
            }
        request.update(copy.deepcopy(request_updates or {}))
        identity = hashlib.sha256(encoded(request)).hexdigest()
        files, views = {}, {}
        for index, look in enumerate(looks):
            (path / look).mkdir()
            pixels = np.full((12, 16, 3), 0.2 + 0.05 * index + 0.02 * count, dtype="f4")
            write_png(path / look / "poster.png", pixels)
            files[f"{look}/poster.png"] = artifact(path / look / "poster.png")
            if scattered:
                # Use actual seeded disjoint positions and pure display colors
                # for the publication fixture's starting image. This is not a
                # replacement for the engine's material-initialization tests.
                initial = np.ones((12, 16, 3), dtype="f4")
                x = ((np.arange(16) + 0.5) / 16 * 2 - 1) * (4 / 3)
                y = 1 - (np.arange(12) + 0.5) / 12 * 2
                for pool in request["layout"]["pools"]:
                    distance = np.hypot(
                        x[None, :] - pool["position"][0], y[:, None] - pool["position"][1]
                    )
                    initial[distance < pool["radius"]] = srgb_to_linear(
                        palette["pigments_srgb"][pool["pigment_index"]]
                    )
                write_png(path / look / "initial.png", initial)
                files[f"{look}/initial.png"] = artifact(path / look / "initial.png")
            if film:
                (path / look / "film.mp4").write_bytes(f"{identity}-{look}".encode())
                files[f"{look}/film.mp4"] = artifact(path / look / "film.mp4")
            views[look] = {
                "physical_state_sha256": physical,
                "movie": {
                    "full_decode_verified": decode,
                    "frames": formation_frames + 2 + orbit_frames - 1,
                    "fps": 24,
                    "resolution": [16, 12],
                }
                if film
                else None,
            }
            if assessed:
                views[look]["image_balance"] = {
                    "dominant_color_share": 0.7,
                    "effective_color_count": 2.5,
                }
        if spectral:
            write(path / "spectral.json", request["spectral"])
            files["spectral.json"] = artifact(path / "spectral.json")
        if assessed:
            report = {
                "version": "participation-fixture-v1",
                "settings": request["recipe"]["assessment"],
                "samples": [],
                "final": {"painted_fraction": 0.2},
            }
            write(path / "assessment.json", report)
            files["assessment.json"] = artifact(path / "assessment.json")
        mass_interval = request["recipe"]["simulation"].get("mass_budget_interval_steps", 0)
        if mass_interval > 0:
            initial_mass = [0.02] * count + [0.0]
            report = {
                "schema_version": 1,
                "version": "mass-budget-fixture-v1",
                "interval_steps": mass_interval,
                "initial_mass": initial_mass,
                "corrections": [
                    {
                        "step": mass_interval,
                        "mass_before": initial_mass,
                        "factors": [1.0] * (count + 1),
                        "mass_after": initial_mass,
                    }
                ],
            }
            write(path / "mass-budget.json", report)
            files["mass-budget.json"] = artifact(path / "mass-budget.json")
        if scattered:
            write(path / "layout.json", request["layout"])
            files["layout.json"] = artifact(path / "layout.json")
        receipt = {
            "complete": True,
            "identity_sha256": identity,
            "source": request["source"],
            "source_fraction": 1.0,
            "final_step": request["recipe"]["simulation"]["steps"],
            "physical_state_sha256": physical,
            "looks": views,
            "artifacts": files,
        }
        if assessed:
            receipt["solver_diagnostics"] = {
                "canonical_steps": receipt["final_step"],
                "actual_transport_substeps": receipt["final_step"],
                "diffusion_substeps": 0,
                "maximum_courant": 0.4,
                "maximum_diffusion_number": 0,
            }
        for name, value in (
            ("request", request),
            ("receipt", receipt),
            ("palette", palette),
            ("events", request["events"]),
        ):
            write(path / f"{name}.json", value)
        return path

    def rehash_collection(self, collection):
        write(self.output / "collection.json", collection)
        curation = read(self.output / "curation.json")
        curation["artifacts"]["collection.json"] = artifact(self.output / "collection.json")
        write(self.output / "curation.json", curation)

    def test_six_cases_publish_nine_views_in_seed_count_look_order(self):
        seeds = ["0xbc53af1cd380", "0x808861c25b6c", "0xb7f327f9f722"]
        cases = []
        for seed in seeds:
            cases.extend([self.case(seed, 3), self.case(seed, 5, ["layered"])])
        result = gallery.build_gallery(self.output, cases)
        self.assertEqual(result, (self.output / "index.html").resolve())
        collection, curation = gallery.verify_gallery(self.output)
        self.assertEqual(len(curation["sources"]), 6)
        self.assertEqual(len(collection["studies"]), 9)
        expected = [
            (seed, count, look)
            for seed in seeds
            for count, look in ((3, "layered"), (5, "layered"), (3, "homogeneous"))
        ]
        self.assertEqual(
            [(s["seed"], s["chromatic_count"], s["group"]) for s in collection["studies"]], expected
        )
        for study in collection["studies"]:
            self.assertTrue((self.output / study["image"]).is_file())
            self.assertTrue((self.output / study["film"]).is_file())
            self.assertEqual(study["film_caption"], gallery.FILM_CAPTION)
            self.assertEqual(len(study["swatches"]), study["chromatic_count"] + 1)
            self.assertEqual(study["swatches"][-1]["role"], "chalk")
        for case in cases:
            case.rename(case.with_name(case.name + "-moved"))
        gallery.verify_gallery(self.output)

    def test_comparisons_distinguish_same_physics_from_added_pigments(self):
        gallery.build_gallery(self.output, [self.case(), self.case(count=5, looks=["layered"])])
        collection, _ = gallery.verify_gallery(self.output)
        layered, five, blended = collection["studies"]
        self.assertEqual(layered["comparison_id"], blended["id"])
        self.assertEqual(layered["comparison_caption"], "Blended · same material history")
        self.assertEqual(blended["comparison_id"], layered["id"])
        self.assertEqual(blended["comparison_caption"], "Layered · same material history")
        self.assertEqual(five["comparison_id"], layered["id"])
        self.assertEqual(five["comparison_caption"], "Three colors · layered")
        self.assertNotEqual(five["physical_state_sha256"], layered["physical_state_sha256"])
        self.assertEqual(layered["palette_record"], blended["palette_record"])
        self.assertEqual(layered["swatches"], blended["swatches"])

    def test_palette_swatches_are_exact_generated_display_values(self):
        case = self.case()
        gallery.build_gallery(self.output, [case])
        collection, _ = gallery.verify_gallery(self.output)
        palette = read(case / "palette.json")
        study = collection["studies"][0]
        self.assertEqual(
            [item["rgba"] for item in study["swatches"]],
            [[*color, 1.0] for color in palette["pigments_srgb"]],
        )
        self.assertEqual([item["name"] for item in study["swatches"]], palette["pigment_names"])
        self.assertEqual(study["name"], "Mineral Tide · 3 colors")
        self.assertEqual(read(self.output / study["palette_record"]), palette)

    def test_no_comparison_is_invented_when_an_optical_pair_is_absent(self):
        gallery.build_gallery(self.output, [self.case(looks=["layered"])])
        collection, _ = gallery.verify_gallery(self.output)
        self.assertIsNone(collection["studies"][0]["comparison_id"])
        self.assertIsNone(collection["studies"][0]["baseline"])

    def test_different_physical_histories_cannot_be_presented_as_paired_looks(self):
        cases = [
            self.case(looks=["layered"], physical="a" * 64),
            self.case(looks=["homogeneous"], physical="b" * 64),
        ]
        with self.assertRaisesRegex(ValueError, "different physical states"):
            gallery.build_gallery(self.output, cases)
        self.assertFalse((self.output / "index.html").exists())

    def test_same_seed_different_source_recordings_cannot_be_compared(self):
        cases = [self.case(), self.case(count=5, looks=["layered"], source_hash="f" * 64)]
        with self.assertRaisesRegex(ValueError, "different source trajectories"):
            gallery.build_gallery(self.output, cases)

    def test_wrong_seed_comparison_rejected_even_with_updated_collection_hash(self):
        gallery.build_gallery(self.output, [self.case(), self.case(seed="0x808861c25b6c")])
        collection = read(self.output / "collection.json")
        first, other = collection["studies"][0], collection["studies"][2]
        first["comparison_id"], first["baseline"] = other["id"], other["image"]
        self.rehash_collection(collection)
        with self.assertRaisesRegex(ValueError, "Comparison"):
            gallery.verify_gallery(self.output)

    def test_caption_palette_and_physical_tampering_cannot_be_rehashed_away(self):
        gallery.build_gallery(self.output, [self.case()])
        original = read(self.output / "collection.json")
        for key, value in (
            ("chromatic_count", 5),
            ("physical_state_sha256", "0" * 64),
            ("seed", "0xffff"),
            ("name", "Different painting"),
        ):
            collection = read(self.output / "collection.json")
            collection["studies"][0][key] = value
            self.rehash_collection(collection)
            with self.subTest(key=key), self.assertRaises(ValueError):
                gallery.verify_gallery(self.output)
            self.rehash_collection(original)
        altered = read(self.output / "collection.json")
        altered["studies"][0]["swatches"][0]["rgba"][0] += 0.01
        self.rehash_collection(altered)
        with self.assertRaisesRegex(ValueError, "palette association"):
            gallery.verify_gallery(self.output)

    def test_proofs_require_explicit_still_publication(self):
        case = self.case(film=False)
        with self.assertRaisesRegex(ValueError, "needs its film"):
            gallery.build_gallery(self.output, [case])
        self.assertFalse(self.output.exists())
        gallery.build_gallery(self.output, [case], allow_stills=True)
        collection, _ = gallery.verify_gallery(self.output)
        self.assertTrue(all(study["film"] is None for study in collection["studies"]))

    def test_no_film_without_full_decode_evidence_can_be_published(self):
        with self.assertRaisesRegex(ValueError, "decode evidence"):
            gallery.build_gallery(self.output, [self.case(decode=False)])
        self.assertFalse((self.output / "index.html").exists())

    def test_all_cases_are_verified_before_publishing(self):
        good, bad = self.case(), self.case(count=5, looks=["layered"])
        receipt = read(bad / "receipt.json")
        receipt["complete"] = False
        write(bad / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            gallery.build_gallery(self.output, [good, bad])
        self.assertFalse(self.output.exists())

    def test_changed_input_during_copy_cannot_be_certified(self):
        case = self.case()

        def change_after_verification(path):
            request, receipt = self.verified_case(path)
            changed = {**request, "changed": True}
            write(path / "request.json", changed)
            return request, receipt

        with (
            patch.object(gallery, "verify_run", side_effect=change_after_verification),
            self.assertRaisesRegex(ValueError, "Source changed while copying"),
        ):
            gallery.build_gallery(self.output, [case])
        self.assertFalse((self.output / "curation.json").exists())

    def test_republication_is_stable_and_media_tampering_is_never_overwritten(self):
        case = self.case()
        gallery.build_gallery(self.output, [case])
        original = (self.output / "curation.json").read_bytes()
        gallery.build_gallery(self.output, [case])
        self.assertEqual((self.output / "curation.json").read_bytes(), original)
        collection, _ = gallery.verify_gallery(self.output)
        image = self.output / collection["studies"][0]["image"]
        image.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            gallery.verify_gallery(self.output)
        with self.assertRaisesRegex(ValueError, "Published media changed"):
            gallery.build_gallery(self.output, [case])
        self.assertEqual(image.read_bytes(), b"tampered")

    def test_duplicate_case_and_publication_inside_archive_are_rejected(self):
        case = self.case()
        with self.assertRaisesRegex(ValueError, "more than once"):
            gallery.build_gallery(self.output, [case, case])
        with self.assertRaisesRegex(ValueError, "outside"):
            gallery.build_gallery(case / "gallery", [case])

    def test_interface_uses_native_movies_safe_labels_and_dynamic_comparison_caption(self):
        document = gallery.document('<script>alert("title")</script>')
        self.assertIn("&lt;script&gt;", document)
        self.assertNotIn('<script>alert("title")</script>', document)
        self.assertIn("<video", document)
        self.assertIn(">Play film</button>", document)
        self.assertIn("comparisonCaption", document)
        self.assertIn("study.comparison_caption", document)
        self.assertIn("study.swatches", document)
        self.assertIn("study.film_caption", document)
        self.assertIn("colors + shared chalk", document)
        self.assertIn("[hidden]{display:none!important}", document)
        self.assertIn("http://127.0.0.1:8787/", document)
        self.assertNotIn("Compare original", document)
        self.assertNotIn("Earlier Estuary", document)

    def test_gallery_description_distinguishes_scattered_and_legacy_studies(self):
        document = gallery.document("Color studies")
        header = document.split("<script>", 1)[0]
        self.assertIn('<p id="description">', header)
        self.assertNotIn("Every color begins in its own pool", header)
        self.assertIn("$('description').textContent=study.initial", document)
        self.assertIn("Three colors or five, chosen by the seed", document)

    def test_template_is_independent_of_previous_gallery_prose_and_archived_as_runtime(self):
        from .run import runtime_identity

        expected = gallery.document("Confluence Fresco")
        with patch("tools.estuary_studio.gallery.DEPTH_DOCUMENT", "Different old interface"):
            self.assertEqual(gallery.document("Confluence Fresco"), expected)
        self.assertIn("gallery.html", runtime_identity()["estuary_confluence"])
        self.assertNotIn(gallery.TITLE_TOKEN, expected)

    def test_formation_only_movie_never_advertises_a_camera_orbit(self):
        gallery.build_gallery(self.output, [self.case(orbit_frames=1)])
        collection, _ = gallery.verify_gallery(self.output)
        for study in collection["studies"]:
            self.assertEqual(study["film_caption"], "Complete formation of the painting")
            self.assertEqual(study["film_frames"], 8)
            self.assertEqual(study["film_fps"], 24)
            self.assertEqual(study["film_resolution"], [16, 12])
            self.assertEqual(study["film_seconds"], 8 / 24)

    def test_rehashed_movie_caption_and_dimensions_still_match_original_recipe(self):
        gallery.build_gallery(self.output, [self.case(orbit_frames=1)])
        original = read(self.output / "collection.json")
        for key, value in (
            ("film_caption", gallery.FILM_CAPTION),
            ("film_resolution", [3840, 2880]),
            ("film_frames", 10),
            ("film_fps", 60),
            ("film_seconds", 100),
        ):
            collection = read(self.output / "collection.json")
            collection["studies"][0][key] = value
            self.rehash_collection(collection)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "formation caption"):
                gallery.verify_gallery(self.output)
            self.rehash_collection(original)

    def scatter_cases(self, seed="0xbc53af1cd380"):
        return [
            self.case(seed, 3, ["layered"], palette_mode="harmonic", scattered=True),
            self.case(seed, 5, ["layered"], palette_mode="harmonic", scattered=True),
            self.case(seed, 5, ["layered"], palette_mode="random", scattered=True),
        ]

    def test_three_scatter_versions_per_seed_publish_distinct_starting_colors(self):
        seeds = ["0xbc53af1cd380", "0x808861c25b6c", "0xb7f327f9f722"]
        cases = [case for seed in seeds for case in self.scatter_cases(seed)]
        gallery.build_gallery(self.output, cases)
        collection, curation = gallery.verify_gallery(self.output)
        self.assertEqual(len(collection["studies"]), 9)
        self.assertEqual(len(curation["sources"]), 9)
        self.assertEqual(
            [(s["seed"], s["palette_mode"], s["chromatic_count"]) for s in collection["studies"]],
            [
                (seed, mode, count)
                for seed in seeds
                for mode, count in (("harmonic", 3), ("harmonic", 5), ("random", 5))
            ],
        )
        for study in collection["studies"]:
            count = study["chromatic_count"]
            self.assertEqual(len(study["swatches"]), count)
            self.assertEqual(len({tuple(swatch["rgba"]) for swatch in study["swatches"]}), count)
            self.assertNotIn("chalk", [swatch["role"] for swatch in study["swatches"]])
            palette = read(self.output / study["palette_record"])
            self.assertEqual(
                [swatch["rgba"] for swatch in study["swatches"]],
                [[*rgb, 1.0] for rgb in palette["pigments_srgb"][:count]],
            )
            layout = read(self.output / study["layout_record"])
            self.assertEqual(layout["count"], count)
            self.assertEqual(
                [pool["pigment_index"] for pool in layout["pools"]], list(range(count))
            )
            self.assertTrue((self.output / study["initial"]).is_file())
            self.assertIn(study["initial"], curation["artifacts"])
            self.assertIn(study["layout_record"], curation["artifacts"])
        for case in cases:
            case.rename(case.with_name(case.name + "-moved"))
        gallery.verify_gallery(self.output)

    def test_random_palette_comparison_uses_same_harmonic_five_color_history(self):
        gallery.build_gallery(self.output, self.scatter_cases())
        collection, _ = gallery.verify_gallery(self.output)
        three, five, random = collection["studies"]
        self.assertEqual(random["comparison_id"], five["id"])
        self.assertEqual(random["baseline"], five["image"])
        self.assertEqual(
            random["comparison_caption"], "Seeded harmony · same starting pools and motion"
        )
        self.assertEqual(random["physical_state_sha256"], five["physical_state_sha256"])
        self.assertEqual(
            read(self.output / random["layout_record"]), read(self.output / five["layout_record"])
        )
        self.assertNotEqual(random["palette_identity_sha256"], five["palette_identity_sha256"])
        self.assertEqual(three["comparison_id"], five["id"])
        self.assertEqual(five["comparison_id"], three["id"])
        self.assertEqual(five["comparison_caption"], "Three colors · three starting pools")
        self.assertEqual(three["comparison_caption"], "Five colors · two additional starting pools")

    def test_count_comparison_rejects_changed_pool_sizes_loads_and_flow(self):
        for index, settings in enumerate(
            (
                {"load_radius": 0.3},
                {"initial_load": 0.25},
                {"flow_strength": 1.7},
                {"deposition": 0.02},
            )
        ):
            cases = [
                self.case(count=3, looks=["layered"], palette_mode="harmonic", scattered=True),
                self.case(
                    count=5,
                    looks=["layered"],
                    palette_mode="harmonic",
                    scattered=True,
                    simulation_updates=settings,
                ),
            ]
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "simulation"):
                gallery.build_gallery(self.root / f"different-controls-{index}", cases)

    def test_count_comparison_rejects_changed_primary_colors_and_material_coefficients(self):
        base = generate_palette("0xbc53af1cd380", 5, mode="harmonic")
        colors = copy.deepcopy(base["pigments_srgb"])
        colors[0][0] += 0.01
        scattering = list(base["scattering"])
        scattering[0] *= 1.1
        for index, changes in enumerate(
            (
                {"pigments_srgb": colors},
                {"scattering": scattering},
                {"substrate_srgb": [0.98, 0.98, 0.98]},
                {"substrate_seed": "0x" + "f" * 64},
            )
        ):
            cases = [
                self.case(count=3, looks=["layered"], palette_mode="harmonic", scattered=True),
                self.case(
                    count=5,
                    looks=["layered"],
                    palette_mode="harmonic",
                    scattered=True,
                    palette_updates=changes,
                ),
            ]
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "Comparison"):
                gallery.build_gallery(self.root / f"different-colors-{index}", cases)

    def test_count_comparison_binds_actual_pool_prefix_and_shared_process_records(self):
        for index, changed in enumerate(("pool", "projection", "events", "code")):
            cases = [
                self.case(count=3, looks=["layered"], palette_mode="harmonic", scattered=True),
                self.case(count=5, looks=["layered"], palette_mode="harmonic", scattered=True),
            ]
            request = read(cases[1] / "request.json")
            if changed == "pool":
                request["layout"]["pools"][0]["position"][0] += 0.01
                write(cases[1] / "layout.json", request["layout"])
            elif changed == "projection":
                request["recipe"]["projection"] = {"fill": 0.7}
            elif changed == "events":
                request["events"][0]["fraction"] = 0.6
                write(cases[1] / "events.json", request["events"])
            else:
                request["code"] = {"estuary_confluence": {"engine.py": "changed-runtime"}}
            write(cases[1] / "request.json", request)
            receipt = read(cases[1] / "receipt.json")
            receipt["identity_sha256"] = hashlib.sha256(encoded(request)).hexdigest()
            write(cases[1] / "receipt.json", receipt)
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Comparison"):
                gallery.build_gallery(self.root / f"different-inputs-{index}", cases)

    def test_random_and_harmonic_comparison_checks_inputs_even_if_final_hash_matches(self):
        cases = [
            self.case(count=5, looks=["layered"], palette_mode="harmonic", scattered=True),
            self.case(
                count=5,
                looks=["layered"],
                palette_mode="random",
                scattered=True,
                simulation_updates={"load_radius": 0.3},
            ),
        ]
        receipts = [read(case / "receipt.json") for case in cases]
        self.assertEqual(receipts[0]["physical_state_sha256"], receipts[1]["physical_state_sha256"])
        with self.assertRaisesRegex(ValueError, "simulation"):
            gallery.build_gallery(self.output, cases)

    def test_palette_comparison_rejects_different_physical_histories(self):
        cases = [
            self.case(count=5, looks=["layered"], palette_mode="harmonic", scattered=True),
            self.case(
                count=5, looks=["layered"], palette_mode="random", scattered=True, physical="f" * 64
            ),
        ]
        with self.assertRaisesRegex(ValueError, "different physical states"):
            gallery.build_gallery(self.output, cases)
        self.assertFalse((self.output / "index.html").exists())

    def test_initial_image_and_layout_tampering_are_rejected_after_file_rehash(self):
        gallery.build_gallery(self.output, self.scatter_cases())
        collection, original_curation = gallery.verify_gallery(self.output)
        study = collection["studies"][0]
        for key in ("initial", "layout_record"):
            path = self.output / study[key]
            original = path.read_bytes()
            if key == "initial":
                path.write_bytes(b"changed starting image")
            else:
                layout = read(path)
                layout["pools"][0]["position"][0] += 0.1
                write(path, layout)
            changed_curation = copy.deepcopy(original_curation)
            changed_curation["artifacts"][study[key]] = artifact(path)
            write(self.output / "curation.json", changed_curation)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "starting"):
                gallery.verify_gallery(self.output)
            path.write_bytes(original)
            write(self.output / "curation.json", original_curation)
        gallery.verify_gallery(self.output)

    def test_starting_image_layout_and_mode_association_cannot_be_rehashed_away(self):
        gallery.build_gallery(self.output, self.scatter_cases())
        original, _ = gallery.verify_gallery(self.output)
        first, _, random = original["studies"]
        for key, value in (
            ("initial", random["initial"]),
            ("layout_record", random["layout_record"]),
            ("palette_mode", "random"),
        ):
            changed = copy.deepcopy(original)
            changed["studies"][0][key] = value
            self.rehash_collection(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                gallery.verify_gallery(self.output)
            self.rehash_collection(original)
        self.assertNotEqual(first["initial"], random["initial"])

    def test_legacy_curated_case_retains_chalk_without_invented_starting_pools(self):
        case = self.case()
        request = read(case / "request.json")
        self.assertNotIn("layout", request)
        self.assertNotIn("palette_mode", request["recipe"])
        gallery.build_gallery(self.output, [case])
        collection, _ = gallery.verify_gallery(self.output)
        for study in collection["studies"]:
            self.assertEqual(study["palette_mode"], "curated")
            self.assertIsNone(study["initial"])
            self.assertIsNone(study["layout_record"])
            self.assertEqual(len(study["swatches"]), study["chromatic_count"] + 1)

    @unittest.skipUnless(
        shutil.which("node"), "Node.js is required for browser-script syntax validation"
    )
    def test_published_browser_script_has_valid_javascript_syntax(self):
        document = gallery.document("Scattered pigments")
        scripts = re.findall(r"<script>([\s\S]*?)</script>", document)
        self.assertEqual(len(scripts), 1)
        path = self.root / "gallery-script.js"
        path.write_text(scripts[0])
        result = subprocess.run(
            [shutil.which("node"), "--check", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def convergence_cases(self, seed="0xbc53af1cd380"):
        return [
            self.case(
                seed,
                count,
                ["layered"],
                palette_mode="harmonic",
                scattered=True,
                spectral=True,
                assessed=True,
                formation_frames=1201,
                simulation_updates={"initial_pattern": "engaged", "steps": 1200},
                physical=str(count) * 64,
            )
            for count in (3, 5)
        ]

    def test_spectra_assessment_and_exact_formation_duration_travel_with_publication(self):
        cases = self.convergence_cases()
        gallery.build_gallery(self.output, cases)
        collection, curation = gallery.verify_gallery(self.output)
        self.assertEqual(collection["publication_version"], 2)
        for study in collection["studies"]:
            self.assertEqual(study["optics_model"], "spectral")
            self.assertEqual(study["initial_pattern"], "engaged")
            self.assertEqual(study["formation_seconds"], 50)
            self.assertEqual(study["film_seconds"], 1205 / 24)
            self.assertIn(study["spectral_record"], curation["artifacts"])
            self.assertIn(study["assessment_record"], curation["artifacts"])
            self.assertEqual(
                read(self.output / study["spectral_record"])["palette_identity_sha256"],
                study["palette_identity_sha256"],
            )
            self.assertEqual(study["image_balance"]["dominant_color_share"], 0.7)
            self.assertEqual(study["solver_diagnostics"]["canonical_steps"], 1200)
        for case in cases:
            case.rename(case.with_name(case.name + "-moved"))
        gallery.verify_gallery(self.output)

    def test_rehashed_spectral_assessment_or_process_metadata_tampering_is_rejected(self):
        gallery.build_gallery(self.output, self.convergence_cases())
        original, curation = gallery.verify_gallery(self.output)
        for key, value in [
            ("optics_model", "rgb"),
            ("initial_pattern", "scattered"),
            ("formation_seconds", 12.5),
            ("image_balance", None),
            ("solver_diagnostics", None),
            ("spectral_record", None),
            ("assessment_record", None),
        ]:
            changed = copy.deepcopy(original)
            changed["studies"][0][key] = value
            self.rehash_collection(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                gallery.verify_gallery(self.output)
            self.rehash_collection(original)
        curation = read(self.output / "curation.json")
        for key in ("spectral_record", "assessment_record"):
            name = original["studies"][0][key]
            path = self.output / name
            previous = path.read_bytes()
            record = read(path)
            if key == "spectral_record":
                record["pigment_reflectance"][0][0] *= 0.5
            else:
                record["final"]["painted_fraction"] = 0.9
            write(path, record)
            changed = copy.deepcopy(curation)
            changed["artifacts"][name] = artifact(path)
            write(self.output / "curation.json", changed)
            with self.subTest(record=key), self.assertRaises(ValueError):
                gallery.verify_gallery(self.output)
            path.write_bytes(previous)
            write(self.output / "curation.json", curation)

    def test_earlier_comparisons_match_source_and_remain_portable_without_same_material_claim(self):
        earlier = self.root / "earlier-gallery"
        previous = [
            self.case(
                count=n,
                looks=["layered"],
                palette_mode="harmonic",
                scattered=True,
                physical="a" * 64,
            )
            for n in (3, 5)
        ]
        gallery.build_gallery(earlier, previous)
        cases = self.convergence_cases()
        gallery.build_gallery(self.output, cases, earlier_gallery=earlier)
        collection, curation = gallery.verify_gallery(self.output)
        self.assertEqual(len(curation["earlier_sources"]), 2)
        for study in collection["studies"]:
            self.assertEqual(study["earlier_caption"], "Earlier version · same trajectory")
            self.assertNotIn("material history", study["earlier_caption"])
            self.assertTrue((self.output / study["earlier_image"]).is_file())
            self.assertIsNotNone(study["comparison_id"])
            record = next(r for r in curation["earlier_sources"] if r["id"] == study["earlier_id"])
            self.assertEqual(record["chromatic_count"], study["chromatic_count"])
            self.assertEqual(record["palette_mode"], study["palette_mode"])
            receipt = read(self.output / record["receipt"])
            self.assertNotEqual(receipt["physical_state_sha256"], study["physical_state_sha256"])
        earlier.rename(self.root / "earlier-gallery-moved")
        for case in previous + cases:
            case.rename(case.with_name(case.name + "-moved"))
        gallery.verify_gallery(self.output)

    def test_earlier_gallery_wrong_trajectory_and_changed_image_are_rejected(self):
        earlier = self.root / "earlier-gallery"
        previous = self.case(
            looks=["layered"], palette_mode="harmonic", scattered=True, source_hash="f" * 64
        )
        gallery.build_gallery(earlier, [previous])
        current = self.convergence_cases()[0]
        with self.assertRaisesRegex(ValueError, "different source trajectory"):
            gallery.build_gallery(self.output, [current], earlier_gallery=earlier)
        self.assertFalse((self.output / "index.html").exists())
        image = read(earlier / "collection.json")["studies"][0]["image"]
        (earlier / image).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            gallery.build_gallery(self.output, [current], earlier_gallery=earlier)

    def test_earlier_caption_and_image_cannot_be_rehashed_away(self):
        earlier = self.root / "earlier-gallery"
        gallery.build_gallery(earlier, self.scatter_cases())
        gallery.build_gallery(self.output, self.convergence_cases(), earlier_gallery=earlier)
        original, curation = gallery.verify_gallery(self.output)
        for key, value in [
            ("earlier_caption", "Earlier version · same material history"),
            ("earlier_image", original["studies"][0]["image"]),
            ("earlier_id", original["studies"][1]["earlier_id"]),
        ]:
            changed = copy.deepcopy(original)
            changed["studies"][0][key] = value
            self.rehash_collection(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                gallery.verify_gallery(self.output)
            self.rehash_collection(original)
        name = original["studies"][0]["earlier_image"]
        path = self.output / name
        path.write_bytes(b"changed earlier image")
        curation = read(self.output / "curation.json")
        curation["artifacts"][name] = artifact(path)
        write(self.output / "curation.json", curation)
        with self.assertRaisesRegex(ValueError, "Earlier version image"):
            gallery.verify_gallery(self.output)

    def test_no_matching_earlier_study_does_not_invent_a_comparison(self):
        earlier = self.root / "earlier-gallery"
        gallery.build_gallery(earlier, self.scatter_cases("0x808861c25b6c"))
        gallery.build_gallery(self.output, self.convergence_cases(), earlier_gallery=earlier)
        collection, curation = gallery.verify_gallery(self.output)
        self.assertEqual(curation["earlier_sources"], [])
        self.assertTrue(all(s["earlier_image"] is None for s in collection["studies"]))

    @unittest.skipUnless(shutil.which("node"), "Requires Node.js to exercise gallery controls")
    def test_earlier_and_regular_controls_are_independent_and_show_actual_duration(self):
        earlier = self.root / "earlier-gallery"
        gallery.build_gallery(earlier, self.scatter_cases())
        gallery.build_gallery(self.output, self.convergence_cases(), earlier_gallery=earlier)
        collection = read(self.output / "collection.json")
        script = re.findall(r"<script>([\s\S]*?)</script>", gallery.document("Convergence"))[0]
        harness = """
const vm=require('node:vm');
const assert=require('node:assert/strict');
const elements=new Map();
function element(){return {attributes:{},style:{},children:[],value:'0',
 classList:{toggle(){}},setAttribute(k,v){this.attributes[k]=v},
 getAttribute(k){return this.attributes[k]},removeAttribute(k){delete this[k]},
 replaceChildren(){this.children=[]},append(...items){this.children.push(...items)},
 pause(){},load(){},play(){return Promise.resolve()},scrollIntoView(){}}}
const document={getElementById(id){
 if(!elements.has(id))elements.set(id,element());
 return elements.get(id)},createElement:element};
const context={document,fetch:async()=>({ok:true,json:async()=>DATA})};
vm.runInNewContext(SCRIPT,context);
setImmediate(()=>{
 const get=id=>elements.get(id); const study=DATA.studies[0];
 assert.match(get('description').textContent,/50 seconds/);
 assert.match(get('filmDuration').textContent,/50 s formation/);
 assert.match(get('filmDuration').textContent,/50.21 s complete film/);
 get('earlier').onclick();assert.equal(get('baseline').src,study.earlier_image);
 assert.equal(get('comparisonCaption').textContent,'Earlier version · same trajectory');
 assert.equal(get('earlier').attributes['aria-pressed'],'true');
 assert.equal(get('compare').attributes['aria-pressed'],'false');
 get('compare').onclick();assert.equal(get('baseline').src,study.baseline);
 assert.equal(get('earlier').attributes['aria-pressed'],'false');
 assert.equal(get('compare').attributes['aria-pressed'],'true');
 get('starting').onclick();assert.equal(get('hero').src,study.initial);
 assert.equal(get('fullSize').href,study.initial);
 assert.equal(get('reference').hidden,true);
 get('stillView').onclick();assert.equal(get('fullSize').href,study.image);
});
"""
        path = self.root / "controls.js"
        path.write_text(
            "const DATA="
            + json.dumps(collection)
            + ";\nconst SCRIPT="
            + json.dumps(script)
            + ";\n"
            + harness
        )
        result = subprocess.run(
            [shutil.which("node"), str(path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_enabled_mass_budget_report_is_portable_and_bound_to_case_bytes(self):
        case = self.case(
            looks=["layered"],
            palette_mode="harmonic",
            scattered=True,
            simulation_updates={"mass_budget_interval_steps": 5},
        )
        gallery.build_gallery(self.output, [case])
        collection, curation = gallery.verify_gallery(self.output)
        study = collection["studies"][0]
        name = study["mass_budget_record"]
        self.assertEqual(name, curation["sources"][0]["mass_budget"])
        self.assertEqual(
            (self.output / name).read_bytes(), (case / "mass-budget.json").read_bytes()
        )
        self.assertEqual(
            curation["artifacts"][name],
            read(case / "receipt.json")["artifacts"]["mass-budget.json"],
        )
        case.rename(case.with_name(case.name + "-moved"))
        gallery.verify_gallery(self.output)

    def test_enabled_mass_budget_without_receipt_binding_cannot_be_published(self):
        case = self.case(simulation_updates={"mass_budget_interval_steps": 5})
        receipt = read(case / "receipt.json")
        del receipt["artifacts"]["mass-budget.json"]
        write(case / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "bound mass-budget"):
            gallery.build_gallery(self.output, [case])
        self.assertFalse((self.output / "index.html").exists())

    def test_disabled_and_legacy_cases_do_not_invent_mass_budget_records(self):
        for index, settings in enumerate(({}, {"mass_budget_interval_steps": 0})):
            case = self.case(simulation_updates=settings)
            output = self.root / f"no-budget-{index}"
            gallery.build_gallery(output, [case])
            collection, curation = gallery.verify_gallery(output)
            self.assertTrue(
                all(study["mass_budget_record"] is None for study in collection["studies"])
            )
            self.assertNotIn("mass_budget", curation["sources"][0])

    def test_rehashed_mass_report_and_study_link_tampering_are_rejected(self):
        case = self.case(simulation_updates={"mass_budget_interval_steps": 5})
        gallery.build_gallery(self.output, [case])
        original, curation = gallery.verify_gallery(self.output)
        changed = copy.deepcopy(original)
        changed["studies"][0]["mass_budget_record"] = changed["studies"][0]["palette_record"]
        self.rehash_collection(changed)
        with self.assertRaisesRegex(ValueError, "records belong"):
            gallery.verify_gallery(self.output)
        self.rehash_collection(original)
        name = original["studies"][0]["mass_budget_record"]
        report = read(self.output / name)
        report["corrections"][0]["factors"][0] = 0.1
        write(self.output / name, report)
        curation = read(self.output / "curation.json")
        curation["artifacts"][name] = artifact(self.output / name)
        write(self.output / "curation.json", curation)
        with self.assertRaisesRegex(ValueError, "mass-budget report differs"):
            gallery.verify_gallery(self.output)


if __name__ == "__main__":
    unittest.main()
