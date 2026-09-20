"""Portable provenance and shared presentation for body-influence comparisons."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path

from tools.estuary_studio.common import artifact, encoded, read, write

from . import body_influence_gallery as comparison
from . import composition_gallery as composition
from . import test_composition_gallery as fixtures
from .body_influence import eligible_pairs
from .gallery import build_gallery
from .run import body_influence_metadata


class BodyInfluenceGalleryTests(unittest.TestCase):
    def setUp(self):
        # Keep the same receipt-bound media fixture as the original comparison.
        # Native archived-orbit regeneration is covered by the runner tests.
        self.fixture = fixtures.CompositionGalleryTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.output = self.root / "body-comparison"
        self.reference = self.fixture.reference
        self.inputs_file = self.fixture.inputs_file
        self.release_file = self.fixture.release_file
        self.cases = []
        release, inputs = read(self.release_file), read(self.inputs_file)
        origins = read(self.reference / "curation.json")["sources"]
        references = []
        for origin in origins:
            case = Path(origin["case"])
            request, receipt = self.fixture.fixture.verified_case(case)
            request["recipe"]["encounters"] = 3
            self.fixture.refresh(case, request, receipt)
            references.append(case)
            seed = request["source"]["seed"]
            accepted = next(row for row in release["cases"] if row["seed"] == seed)
            accepted["recipe_sha256"] = hashlib.sha256(encoded(request["recipe"])).hexdigest()
            accepted["request_identity_sha256"] = receipt["identity_sha256"]
            bound = next(row for row in inputs["cases"] if row["seed"] == seed)
            bound["rc1_recipe_sha256"] = accepted["recipe_sha256"]
            for setup, bodies in comparison.SETUPS.items():
                folder = self.root / f"{seed}-influence-{setup}"
                shutil.copytree(case, folder)
                req, rec = copy.deepcopy(request), copy.deepcopy(receipt)
                req["recipe"]["name"] = comparison.LABELS[setup]
                req["recipe"]["looks"] = ["silk-grain"]
                config = {"version": "body-influence-v1", "bodies": list(bodies)}
                req["recipe"]["simulation"]["body_influence"] = config
                req["events"] = [
                    {
                        "fraction": 0.45,
                        "position": [0.1, -0.1],
                        "radius": 0.1,
                        "duration": 0.02,
                        "strength": 0.7,
                        "pair": list(pair),
                    }
                    for pair in eligible_pairs(config)
                ]
                state = hashlib.sha256((seed + setup).encode()).hexdigest()
                rec["physical_state_sha256"] = state
                rec["looks"] = {"silk-grain": rec["looks"]["silk-grain"]}
                rec["looks"]["silk-grain"]["physical_state_sha256"] = state
                self.refresh(folder, req, rec)
                self.cases.append(folder)
        shutil.rmtree(self.reference)
        build_gallery(self.reference, references, layout="studies")
        write(self.release_file, release)
        inputs["reference_release_sha256"] = artifact(self.release_file)["sha256"]
        write(self.inputs_file, inputs)

    def refresh(self, folder, request, receipt, *, metadata=True):
        if metadata:
            request["body_influence"] = body_influence_metadata(
                request["recipe"], request["source"], request["events"]
            )
            receipt["body_influence"] = copy.deepcopy(request["body_influence"])
        self.fixture.refresh(folder, request, receipt)

    def build(self, cases=None):
        return comparison.build_comparison(
            self.output,
            self.cases if cases is None else cases,
            self.reference,
            inputs=self.inputs_file,
            reference_release=self.release_file,
        )

    def test_complete_matrix_preserves_initial_paint_and_is_portable(self):
        self.build()
        manifest, _ = comparison.verify_comparison(self.output)
        self.assertEqual(len(manifest["studies"]), 14)
        self.assertEqual([row["id"] for row in manifest["setups"]], ["rc1", *comparison.SETUPS])
        self.assertEqual(manifest["setups"][0]["label"], "All 3 · RC1")
        for seed in manifest["seeds"]:
            rows = [row for row in manifest["studies"] if row["seed"] == seed]
            self.assertEqual(
                {tuple(row["active_body_indices"]) for row in rows},
                {(0, 1, 2), *comparison.SUBSETS},
            )
            self.assertEqual(
                len({artifact(self.output / row["initial"])["sha256"] for row in rows}), 1
            )
            self.assertEqual(len({tuple(row["initial_pigment_mass"]) for row in rows}), 1)
            self.assertEqual(len({row["source_sha256"] for row in rows}), 1)
            self.assertEqual(len({row["palette_identity_sha256"] for row in rows}), 1)
            pair = next(row for row in rows if row["setup"] == "bodies-1-3")
            self.assertEqual(pair["body_influence"]["eligible_pairs"], [[2, 0]])
            self.assertTrue((self.output / pair["influence_record"]).is_file())
        page = (self.output / "index.html").read_text()
        self.assertIn('href="reference/"', page)
        self.assertIn("All starting paint colors", page)
        self.assertIn("a single selected body can move all three pigments", page)
        self.assertNotIn("http://127.0.0.1", page)
        for folder in (*self.cases, self.reference):
            shutil.rmtree(folder)
        comparison.verify_comparison(self.output)

    def test_missing_subset_cannot_be_published(self):
        with self.assertRaisesRegex(ValueError, "six complete body-influence films"):
            self.build(self.cases[:-1])

    def test_matched_controls_and_initialization_cannot_drift(self):
        case = self.cases[0]
        original_req, original_rec = self.fixture.fixture.verified_case(case)
        for name in ("flow", "layout", "palette", "mass", "initial-image", "source", "timeline"):
            with self.subTest(name=name):
                req, rec = copy.deepcopy(original_req), copy.deepcopy(original_rec)
                if name == "flow":
                    req["recipe"]["simulation"]["flow_strength"] += 0.1
                elif name == "layout":
                    req["layout"]["pools"][0]["position"][0] += 0.1
                elif name == "palette":
                    req["palette"]["pigments_srgb"][0][0] += 0.01
                elif name == "mass":
                    report = read(case / "mass-budget.json")
                    report["initial_mass"][0] += 0.01
                    write(case / "mass-budget.json", report)
                elif name == "initial-image":
                    # A different but correctly bound PNG must still be rejected.
                    shutil.copyfile(case / "silk-grain/poster.png", case / "silk-grain/initial.png")
                    rec["artifacts"]["silk-grain/initial.png"] = artifact(
                        case / "silk-grain/initial.png"
                    )
                elif name == "source":
                    req["source"]["sha256"] = "f" * 64
                else:
                    req["frames"][0]["fraction"] = 0.02
                self.refresh(case, req, rec)
                with self.assertRaises(ValueError):
                    self.build()
                if self.output.exists():
                    shutil.rmtree(self.output)
                # Restore mutable artifacts before the next independent attack.
                source_case = Path(read(self.reference / "curation.json")["sources"][0]["case"])
                shutil.copyfile(source_case / "mass-budget.json", case / "mass-budget.json")
                shutil.copyfile(
                    source_case / "silk-grain/initial.png", case / "silk-grain/initial.png"
                )

    def test_metadata_and_active_pair_eligibility_cannot_be_forged(self):
        case = self.cases[0]
        original_req, original_rec = self.fixture.fixture.verified_case(case)
        for name in ("missing-request", "receipt", "inactive-pair", "full-set"):
            with self.subTest(name=name):
                req, rec = copy.deepcopy(original_req), copy.deepcopy(original_rec)
                if name == "missing-request":
                    del req["body_influence"]
                elif name == "receipt":
                    rec["body_influence"]["config"]["bodies"] = [1]
                elif name == "inactive-pair":
                    pair_request, _ = self.fixture.fixture.verified_case(self.cases[3])
                    req["events"] = pair_request["events"]
                else:
                    # The all-three selection is canonically represented by omission.
                    req["recipe"]["simulation"].pop("body_influence")
                self.refresh(case, req, rec, metadata=False)
                with self.assertRaises(ValueError):
                    self.build()
                if self.output.exists():
                    shutil.rmtree(self.output)

    def test_rehashed_manifest_cannot_change_selected_bodies_or_media(self):
        self.build()
        original = read(self.output / "comparison.json")
        for key, value in (
            ("active_body_indices", [0]),
            ("film", original["studies"][-1]["film"]),
            ("initial_resolution", [4096, 3072]),
            ("label", "Body 1 only"),
        ):
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed["studies"][0][key] = value
                write(self.output / "comparison.json", changed)
                proof = read(self.output / "comparison-verification.json")
                proof["artifacts"]["comparison.json"] = artifact(self.output / "comparison.json")
                write(self.output / "comparison-verification.json", proof)
                with self.assertRaisesRegex(ValueError, "manifest differs"):
                    comparison.verify_comparison(self.output)

    def test_shared_presentation_is_explicit_and_safely_escaped(self):
        for kwargs in ({"default_setup": "not an id"}, {"intro": None}, {"selector_label": ""}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                composition.ComparisonPresentation(**kwargs)
        with self.assertRaises(ValueError):
            composition.document("Title", presentation={"default_setup": "body-1"})
        dangerous = '</script><script>alert("copy")</script>'
        document = composition.document(
            "__COMPARISON_INTRO__", presentation=replace(comparison.PRESENTATION, intro=dangerous)
        )
        self.assertEqual(len(re.findall(r"<script>", document)), 1)
        self.assertNotIn(dangerous, document)
        self.assertIn("__COMPARISON_INTRO__", document)  # Title is not recursively interpolated.
        self.assertIn('"default_setup": "body-wedges"', composition.document("Original"))

    @unittest.skipUnless(shutil.which("node"), "Requires Node.js for shared-player initialization")
    def test_shared_player_uses_body_defaults_and_does_not_preload_movies(self):
        self.build()
        page = (self.output / "index.html").read_text()
        script = re.findall(r"<script>([\s\S]*?)</script>", page)[0]
        ids = re.findall(r'\bid="([^"]+)"', page)
        harness = r"""
const vm=require('node:vm'), assert=require('node:assert/strict');
const elements=new Map();
function element(){return {attributes:{},children:[],value:'',currentTime:0,readyState:0,
 setAttribute(k,v){this.attributes[k]=v},getAttribute(k){return this[k]},
 removeAttribute(k){delete this[k]},
 append(...xs){this.children.push(...xs)},replaceChildren(){this.children=[]},
 pause(){},load(){},addEventListener(){}}}
for(const id of IDS)elements.set(id,element());
const get=id=>elements.get(id);
vm.runInNewContext(SCRIPT,{document:{getElementById:get,createElement:element,addEventListener(){}},
 window:{addEventListener(){}},fetch:async()=>({ok:true,json:async()=>DATA}),
 cancelAnimationFrame(){},requestAnimationFrame(){}});
setImmediate(()=>{
 assert.equal(get('leftSetup').value,'rc1');assert.equal(get('rightSetup').value,'body-1');
 assert.equal(get('leftVideo').src,undefined);assert.equal(get('rightVideo').src,undefined);
 assert.equal(get('leftLabel').textContent,'All 3 · RC1');
 assert.equal(get('rightLabel').textContent,'Body 1 only');
 assert.equal(get('grid').children.length,7);
 assert.match(get('status').textContent,/which bodies influence/);
 get('modeInitial').onclick();assert.match(get('status').textContent,/Identical starting paint/);
 assert.equal(get('leftDimensions').textContent,'16 \u00d7 12 pixels');
});
"""
        path = self.root / "body-controls.js"
        path.write_text(
            "const IDS="
            + json.dumps(ids)
            + ";const DATA="
            + json.dumps(read(self.output / "comparison.json"))
            + ";const SCRIPT="
            + json.dumps(script)
            + ";\n"
            + harness
        )
        result = subprocess.run(
            [shutil.which("node"), str(path)], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
