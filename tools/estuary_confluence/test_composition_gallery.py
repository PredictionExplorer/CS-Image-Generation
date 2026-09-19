"""Fair-comparison provenance and two-player synchronization regressions."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import unittest
from unittest.mock import patch

from tools.estuary.test_engine import SourceFixture
from tools.estuary_studio.common import artifact, encoded, read, write
from tools.estuary_studio.run import frame_plan

from . import composition_gallery as comparison
from . import test_gallery as gallery_fixtures
from .engine import validate_config as simulation_config
from .gallery import build_gallery
from .initial_composition import plan_layout
from .run import DEFAULT_RENDER, interaction_metadata, surface_configs


class CompositionGalleryTests(unittest.TestCase):
    def setUp(self):
        # Reuse the ordinary gallery's small, receipt-bound media fixtures. The
        # simulation/archive pipeline has its own native-grid integration tests.
        self.fixture = gallery_fixtures.GalleryTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.output = self.root / "comparison"
        self.reference = self.root / "rc1"
        self.inputs_file = self.root / "inputs.json"
        self.release_file = self.root / "rc1-release.json"
        self.cases, references, accepted, inputs = [], [], [], []
        base_case = self.fixture.case
        for seed in ("0xb7", "0xbc"):
            with patch.object(
                self.fixture,
                "case",
                side_effect=lambda *a, seed=seed, **kw: base_case(*a, seed=seed, **kw),
            ):
                case = self.fixture.interaction_case()
            request, receipt = self.fixture.verified_case(case)
            request["recipe"]["simulation"] = simulation_config(
                {
                    **request["recipe"]["simulation"],
                    "mass_budget_interval_steps": 5,
                    "initial_pigment_weights": [2.2, 0.9, 0.5],
                }
            )
            request["recipe"]["render"] = {**DEFAULT_RENDER, **request["recipe"]["render"]}
            request["recipe"]["projection"] = {"fill": 0.78, "rotation_degrees": 0.0}
            request["frames"] = frame_plan(request["recipe"])
            masses = [0.02, 0.03, 0.04]
            write(case / "mass-budget.json", {"initial_mass": [*masses, 0.0], "interval_steps": 5})
            self.refresh(case, request, receipt)
            references.append(case)
            radii = [pool["radius"] for pool in request["layout"]["pools"]]
            saved = {
                "id": seed + "-rc1",
                "seed": seed,
                "source_sha256": request["source"]["sha256"],
                "recipe_sha256": hashlib.sha256(encoded(request["recipe"])).hexdigest(),
                "request_identity_sha256": receipt["identity_sha256"],
                "physical_state_sha256": receipt["physical_state_sha256"],
                "views": [
                    {
                        "look": "silk-grain",
                        "image": receipt["artifacts"]["silk-grain/poster.png"],
                        "movie": receipt["artifacts"]["silk-grain/film.mp4"],
                    }
                ],
            }
            accepted.append(saved)
            inputs.append(
                {
                    "seed": seed,
                    "rc1_case_id": saved["id"],
                    "rc1_recipe_sha256": saved["recipe_sha256"],
                    "source_sha256": saved["source_sha256"],
                    "palette_identity_sha256": request["palette"]["identity_sha256"],
                    "target_mass": masses,
                    "reference_radii": radii,
                    "mass_budget_artifact": receipt["artifacts"]["mass-budget.json"],
                    "layout_artifact": receipt["artifacts"]["layout.json"],
                }
            )
            for setup, label in comparison.SETUPS.items():
                shaped = self.root / f"{seed}-{setup}"
                shutil.copytree(case, shaped)
                req, rec = copy.deepcopy(request), copy.deepcopy(receipt)
                spec = {
                    "version": "initial-composition-v1",
                    "setup": setup,
                    "target_mass": masses,
                    "reference_radii": radii,
                }
                req["recipe"]["name"] = label
                req["recipe"]["looks"] = ["silk-grain"]
                req["recipe"]["simulation"] = simulation_config(
                    {
                        **req["recipe"]["simulation"],
                        "initial_pattern": "shaped",
                        "initial_pigment_weights": None,
                        "initial_composition": spec,
                    }
                )
                req["layout"] = plan_layout(seed, 3, 4 / 3, spec, SourceFixture())
                state = hashlib.sha256((seed + setup).encode()).hexdigest()
                rec["physical_state_sha256"] = state
                rec["looks"] = {"silk-grain": rec["looks"]["silk-grain"]}
                rec["looks"]["silk-grain"]["physical_state_sha256"] = state
                self.refresh(shaped, req, rec)
                self.cases.append(shaped)
        build_gallery(self.reference, references, layout="studies")
        write(self.release_file, {"cases": accepted})
        write(
            self.inputs_file,
            {
                "schema_version": 1,
                "version": "composition-rc1-inputs-v1",
                "reference_tag": "RC1",
                "reference_release_sha256": artifact(self.release_file)["sha256"],
                "cases": inputs,
            },
        )

    def refresh(self, folder, request, receipt):
        request["surface_configs"] = surface_configs(request["recipe"])
        request["interaction"] = interaction_metadata(request["recipe"], request["source"]["seed"])
        receipt["interaction"] = copy.deepcopy(request["interaction"])
        receipt["source"] = copy.deepcopy(request["source"])
        for name, value in (
            ("layout", request["layout"]),
            ("palette", request["palette"]),
            ("events", request["events"]),
        ):
            write(folder / f"{name}.json", value)
            receipt["artifacts"][f"{name}.json"] = artifact(folder / f"{name}.json")
        receipt["artifacts"]["mass-budget.json"] = artifact(folder / "mass-budget.json")
        receipt["identity_sha256"] = hashlib.sha256(encoded(request)).hexdigest()
        write(folder / "request.json", request)
        write(folder / "receipt.json", receipt)

    def build(self, cases=None):
        return comparison.build_comparison(
            self.output,
            self.cases if cases is None else cases,
            self.reference,
            inputs=self.inputs_file,
            reference_release=self.release_file,
        )

    def rehash(self, name):
        proof = read(self.output / "comparison-verification.json")
        proof["artifacts"][name] = artifact(self.output / name)
        write(self.output / "comparison-verification.json", proof)

    def test_complete_matrix_uses_only_contact_finish_reference_and_remains_portable(self):
        self.build()
        manifest, _ = comparison.verify_comparison(self.output)
        self.assertEqual(len(manifest["studies"]), 14)
        self.assertEqual({row["setup"] for row in manifest["studies"]}, {*comparison.SETUPS, "rc1"})
        self.assertTrue(all(row["study_id"].endswith("-silk-grain") for row in manifest["studies"]))
        self.assertTrue(all(row["initial_resolution"] == [16, 12] for row in manifest["studies"]))
        self.assertIn("different starting compositions", manifest["comparison"])
        for folder in (*self.cases, self.reference):
            shutil.rmtree(folder)
        comparison.verify_comparison(self.output)

    def test_missing_compositions_cannot_be_published(self):
        with self.assertRaisesRegex(ValueError, "six complete composition"):
            self.build(self.cases[:-1])

    def test_publication_cannot_create_files_inside_immutable_source_archives(self):
        output = self.cases[0] / "comparison"
        with self.assertRaisesRegex(ValueError, "outside immutable"):
            comparison.build_comparison(
                output,
                self.cases,
                self.reference,
                inputs=self.inputs_file,
                reference_release=self.release_file,
            )
        self.assertFalse(output.exists())

    def test_setup_label_must_match_the_bound_layout(self):
        case = self.cases[0]
        request, receipt = self.fixture.verified_case(case)
        request["layout"]["setup"] = "body-wedges"
        self.refresh(case, request, receipt)
        with self.assertRaisesRegex(ValueError, "archived starting layout"):
            self.build()

    def test_optional_position_guides_do_not_change_the_matched_physical_controls(self):
        from .body_markers import validate_config as marker_config
        from .run import body_marker_metadata, make_body_marker_ledger

        case = self.cases[0]
        request, receipt = self.fixture.verified_case(case)
        request["recipe"]["render"]["body_markers"] = marker_config(True)
        request["body_markers"] = body_marker_metadata(request["recipe"], request["source"])
        receipt["body_markers"] = copy.deepcopy(request["body_markers"])
        source = SourceFixture()
        source.metadata = request["source"]
        ledger = make_body_marker_ledger(
            request["recipe"], source, request["frames"], has_initial=True
        )
        write(case / "body-markers.json", ledger)
        receipt["artifacts"]["body-markers.json"] = artifact(case / "body-markers.json")
        self.refresh(case, request, receipt)
        self.build()
        manifest, _ = comparison.verify_comparison(self.output)
        marked = [row for row in manifest["studies"] if row["body_markers"]]
        self.assertEqual(len(marked), 1)
        self.assertTrue((self.output / marked[0]["body_marker_record"]).is_file())

    def test_changed_flow_is_not_a_fair_initial_composition_comparison(self):
        case = self.cases[0]
        req, rec = self.fixture.verified_case(case)
        req["recipe"]["simulation"]["flow_strength"] += 0.2
        self.refresh(case, req, rec)
        with self.assertRaisesRegex(ValueError, "beyond starting paint"):
            self.build()

    def test_changed_target_mass_is_not_hidden_by_recipe_exclusion(self):
        case = self.cases[0]
        req, rec = self.fixture.verified_case(case)
        req["recipe"]["simulation"]["initial_composition"]["target_mass"][0] *= 1.1
        self.refresh(case, req, rec)
        with self.assertRaisesRegex(ValueError, "target amounts differ"):
            self.build()

    def test_bound_but_wrong_initial_mass_report_is_rejected(self):
        case = self.cases[0]
        req, rec = self.fixture.verified_case(case)
        report = read(case / "mass-budget.json")
        report["initial_mass"][0] *= 1.1
        write(case / "mass-budget.json", report)
        self.refresh(case, req, rec)
        with self.assertRaisesRegex(ValueError, "float32 tolerance"):
            self.build()

    def test_rehashed_manifest_cannot_swap_seeds_media_or_frame_timing(self):
        self.build()
        original = read(self.output / "comparison.json")
        for key, value in (
            ("seed", "0xbc"),
            ("film", original["studies"][-1]["film"]),
            ("film_seconds", 2),
            ("initial_resolution", [2048, 1536]),
        ):
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed["studies"][0][key] = value
                write(self.output / "comparison.json", changed)
                self.rehash("comparison.json")
                with self.assertRaisesRegex(ValueError, "manifest differs"):
                    comparison.verify_comparison(self.output)

    def test_rc1_mass_inputs_cannot_be_replaced_by_other_records(self):
        inputs = read(self.inputs_file)
        inputs["cases"][0]["mass_budget_artifact"]["sha256"] = "0" * 64
        write(self.inputs_file, inputs)
        with self.assertRaisesRegex(ValueError, "different RC1 records"):
            self.build()

    @unittest.skipUnless(shutil.which("node"), "Requires Node.js for player regressions")
    def test_two_player_clock_seek_switching_and_failures(self):
        self.build()
        manifest = read(self.output / "comparison.json")
        document = comparison.document("Comparison")
        ids = re.findall(r'\bid="([^"]+)"', document)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(re.findall(r"<video\b", document)), 2)
        script = re.findall(r"<script>([\s\S]*?)</script>", document)[0]
        harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();let sequence=0,callbacks=new Map();
function element(){return {attributes:{},listeners:{},children:[],value:'',hidden:false,
 readyState:0,seeking:false,currentTime:0,currentSrc:'',plays:0,pauses:0,
 setAttribute(k,v){this.attributes[k]=v},getAttribute(k){return this[k]},
 removeAttribute(k){delete this[k];if(k==='src')this.currentSrc=''},
 append(...xs){this.children.push(...xs)},replaceChildren(){this.children=[]},focus(){},
 pause(){this.pauses++},load(){this.readyState=0;this.currentTime=0;this.currentSrc=this.src||''},
 play(){this.plays++;return Promise.resolve()},addEventListener(k,f){this.listeners[k]=f}}}
for(const id of IDS)elements.set(id,element());
const get=id=>elements.get(id),emit=(id,event)=>get(id).listeners[event]?.();
const doc={getElementById:get,createElement:element,addEventListener(){},hidden:false};
vm.runInNewContext(SCRIPT,{document:doc,window:{addEventListener(){}},
 fetch:async()=>({ok:true,json:async()=>DATA}),
 requestAnimationFrame:f=>{callbacks.set(++sequence,f);return sequence},
 cancelAnimationFrame:id=>callbacks.delete(id)});
const tick=()=>new Promise(r=>setImmediate(r));
const ready=()=>{for(const side of ['left','right']){
 get(side+'Video').readyState=4;
 emit(side+'Video','loadedmetadata');emit(side+'Video','canplay')}};
setImmediate(async()=>{
 const l=get('leftVideo'),r=get('rightVideo');
 assert.equal(l.src,undefined);assert.equal(r.src,undefined);assert.equal(l.plays,0);
 assert.equal(get('grid').children.length,7);
 get('modeInitial').onclick();
 assert.equal(get('leftImage').src,DATA.studies.find(e=>e.seed===DATA.seeds[0]&&e.setup==='rc1').initial);
 get('modeFilm').onclick();assert.equal(l.plays,0);ready();
 get('play').onclick();await tick();assert.equal(l.plays,1);assert.equal(r.plays,1);
 r.readyState=1;emit('rightVideo','waiting');
 assert.match(get('status').textContent,/Waiting for both/);
 assert.ok(l.pauses>0&&r.pauses>0);
 r.readyState=4;emit('rightVideo','canplay');await tick();
 assert.equal(l.plays,2);assert.equal(r.plays,2);
 l.currentTime=.1;r.currentTime=.1;
 get('seek').value=.25;get('seek').oninput();
 assert.equal(l.currentTime,.25);assert.equal(r.currentTime,.25);
 get('seek').onchange();await tick();assert.equal(l.plays,3);assert.equal(r.plays,3);
 get('rightSetup').value='random-ribbons';get('rightSetup').onchange();
 assert.equal(get('play').textContent,'Play both films');assert.equal(l.readyState,0);
 ready();assert.equal(l.currentTime,.25);assert.equal(r.currentTime,.25);
 assert.equal(l.plays,3); // Changing a setup never autoplays.
 get('restart').onclick();await tick();assert.equal(l.currentTime,0);assert.equal(r.currentTime,0);
 get('play').onclick(); // Pause before exercising a rejected play.
 const rejects=[];l.play=()=>new Promise((resolve,reject)=>rejects.push(reject));
 get('play').onclick();get('rightSetup').value='facing-shores';get('rightSetup').onchange();
 rejects[0](new Error('old source'));await tick();assert.equal(get('error').hidden,true);
 ready();l.play=()=>Promise.reject(new Error('blocked'));get('play').onclick();await tick();
 assert.equal(l.src,undefined);assert.equal(r.src,undefined);assert.equal(get('leftImage').hidden,false);
 assert.equal(get('error').hidden,false);
 get('modeFilm').onclick();ready();l.play=()=>Promise.resolve();get('play').onclick();await tick();
 emit('leftVideo','ended');assert.equal(l.src,undefined);assert.equal(r.src,undefined);
 assert.match(get('status').textContent,/Films finished/);
});
"""
        program = self.root / "comparison-player.js"
        program.write_text(
            "const IDS="
            + json.dumps(ids)
            + ";const DATA="
            + json.dumps(manifest)
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
