#!/usr/bin/env python3
"""Archive seeded confluences and several optical views of one physical history.

One canonical simulation feeds every requested view. Movie capture may exactly
area-average the GPU state, without changing its evolution. Final stills retain
the full simulation grid. The original orbit, palette, encounters, source code,
all frames and complete decoding evidence remain inspectable together.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import math
import platform
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.estuary.optics import srgb_to_linear
from tools.estuary.run import encode_movie, write_array, write_png
from tools.estuary.source import Source
from tools.estuary_confluence.palette import SUPPORTED_CHROMATIC_COUNTS
from tools.estuary_studio.common import artifact, checked, digest, encoded, read, require, write
from tools.estuary_studio.run import dimensions, frame_plan, number, record_array

ROOT = Path(__file__).resolve().parent
PACKAGES = ("estuary", "estuary_depth", "estuary_studio", "estuary_confluence")
LABELS = {
    "layered": "Layered confluence",
    "homogeneous": "Blended comparison",
    "control": "Original surface",
    "silk": "Silk at the seams",
    "silk-grain": "Silk and mineral grain",
}
INTERACTION_LOOKS = frozenset({"control", "silk", "silk-grain"})
# Source PCA can differ in the last float64 bits across CPUs. These absolute
# tolerances are far below a visible pixel; all clocks/configuration stay exact.
BODY_MARKER_POSITION_ATOL = 1e-11
BODY_MARKER_PIXEL_ATOL = 1e-7
DEFAULT_RENDER = {
    "resolution": [1920, 1440],
    "still_resolution": [3840, 2880],
    "capture_resolution": None,
    "capture_pipeline": "cpu",
    "frame_supersampling": 1,
    "formation_frames": 301,
    "orbit_frames": 145,
    "hold_frames": 24,
    "fps": 24,
    "still_tilt_degrees": 8.0,
    "orbit_tilt_degrees": 12.0,
    "azimuth_start": -35.0,
    "azimuth_end": 35.0,
}


def runtime_identity():
    return {
        name: {
            str(p.relative_to(ROOT.parent / name)): digest(p)
            for p in sorted((ROOT.parent / name).rglob("*"))
            if p.is_file()
            and p.suffix in (".py", ".glsl", ".txt", ".html")
            and not p.name.startswith("test_")
        }
        for name in PACKAGES
    }


def verify_code(folder, code):
    require(set(code) == set(PACKAGES), "Incomplete runtime identity")
    for package, files in code.items():
        root = (Path(folder) / "tools" / package).resolve()
        for name, sha in files.items():
            path = (root / name).resolve()
            require(
                path.is_relative_to(root) and path.is_file() and digest(path) == sha,
                f"Archived runtime differs: {package}/{name}",
            )


def validate_recipe(raw):
    from tools.estuary_confluence.engine import reduction_factor
    from tools.estuary_confluence.engine import validate_config as simulation_config
    from tools.estuary_confluence.surface import interaction_enabled
    from tools.estuary_confluence.surface import validate_config as surface_config

    require(
        type(raw) is dict
        and not raw.keys()
        - {
            "schema_version",
            "name",
            "chromatic_count",
            "palette_mode",
            "looks",
            "encounters",
            "simulation",
            "surface",
            "projection",
            "render",
            "assessment",
            "background",
        },
        "Unknown confluence recipe keys",
    )
    require(
        type(raw.get("schema_version", 1)) is int and raw.get("schema_version", 1) == 1,
        "Unsupported recipe schema",
    )
    count = raw.get("chromatic_count", 3)
    require(
        type(count) is int and count in SUPPORTED_CHROMATIC_COUNTS,
        "Use one, two, three, or five chromatic pigments",
    )
    palette_mode = raw.get("palette_mode", "curated")
    require(
        type(palette_mode) is str and palette_mode in ("curated", "harmonic", "random", "composed"),
        "Choose a curated, harmonic, random, or composed palette",
    )
    name = raw.get("name", "Confluence Fresco")
    require(isinstance(name, str) and 0 < len(name) <= 100, "Painting needs a short name")
    looks = raw.get("looks", ["layered"])
    require(
        type(looks) is list
        and 1 <= len(looks) <= len(LABELS)
        and all(type(x) is str and x in LABELS for x in looks)
        and len(set(looks)) == len(looks),
        "Choose distinct known optical views",
    )
    encounters = raw.get("encounters", 3)
    require(type(encounters) is int and 0 <= encounters <= 3, "Use at most three encounter blooms")
    simulation = simulation_config(raw.get("simulation", {}))
    weights = simulation.get("initial_pigment_weights")
    require(
        weights is None or len(weights) == count, "Initial load weights differ from color count"
    )
    surface = copy.deepcopy(raw.get("surface", {}))
    require(type(surface) is dict, "Surface controls must be an object")
    require(
        surface.get("domain_scale", simulation["domain_scale"]) == simulation["domain_scale"],
        "Surface and simulation guard domains differ",
    )
    surface["domain_scale"] = simulation["domain_scale"]
    surface = surface_config(surface)
    require_interaction_looks(looks, simulation, surface)
    require_interaction_geometry(simulation, surface)
    require(
        not interaction_enabled(surface) or simulation.get("interaction") is not None,
        "Interaction optics require transported interaction material",
    )
    require(surface["tone_map"] == "reinhard", "Display output needs bounded tone mapping")
    projection = {"fill": 0.78, "rotation_degrees": 0.0}
    supplied = raw.get("projection", {})
    require(
        type(supplied) is dict and not supplied.keys() - projection.keys(),
        "Invalid projection keys",
    )
    projection.update(supplied)
    number(projection["fill"], "fill", 0.05, 0.98)
    number(projection["rotation_degrees"], "rotation", -360, 360)
    render = copy.deepcopy(DEFAULT_RENDER)
    supplied = raw.get("render", {})
    require(
        type(supplied) is dict and not supplied.keys() - (render.keys() | {"body_markers"}),
        "Invalid render keys",
    )
    render.update(supplied)
    from tools.estuary_confluence.body_markers import validate_config as marker_config

    markers = marker_config(render.pop("body_markers", None))
    if markers is not None:
        render["body_markers"] = markers
    sw, sh = simulation["resolution"]
    if render["capture_resolution"] is None:
        capture = [sw, sh]
        while (
            simulation.get("interaction") is None
            and capture[0] > 2048
            and all(n % 2 == 0 for n in capture)
        ):
            capture = [n // 2 for n in capture]
        render["capture_resolution"] = capture
    for key in ("resolution", "still_resolution", "capture_resolution"):
        w, h = dimensions(render[key])
        require(w * sh == h * sw, "Image and material aspects must match")
    reduction_factor((sw, sh), tuple(render["capture_resolution"]))
    require(
        simulation.get("interaction") is None or render["capture_resolution"] == [sw, sh],
        "Interaction capture requires the full native material grid",
    )
    require(render["capture_pipeline"] in ("cpu", "native-gpu"), "Unknown capture pipeline")
    require(
        render["capture_pipeline"] != "native-gpu" or render["capture_resolution"] == [sw, sh],
        "Native GPU capture requires the full simulation grid",
    )
    frame_raster_resolution(render)
    for key, lo, hi in (
        ("formation_frames", 2, 1801),
        ("orbit_frames", 1, 721),
        ("hold_frames", 0, 240),
        ("fps", 1, 60),
    ):
        require(type(render[key]) is int and lo <= render[key] <= hi, f"Invalid {key}")
    require(
        simulation["steps"] % (render["formation_frames"] - 1) == 0,
        "Movie captures must fall on canonical steps",
    )
    for key in ("still_tilt_degrees", "orbit_tilt_degrees"):
        number(render[key], key, 0, 30)
    for key in ("azimuth_start", "azimuth_end"):
        number(render[key], key, -180, 180)
    assessment = raw.get("assessment")
    if assessment is not None:
        require(
            type(assessment) is dict
            and not assessment.keys() - {"interval_steps", "resolution", "share_threshold"},
            "Unknown participation assessment controls",
        )
        assessment = {
            "interval_steps": 120,
            "resolution": None,
            "share_threshold": 0.1,
            **assessment,
        }
        require(
            type(assessment["interval_steps"]) is int
            and 1 <= assessment["interval_steps"] <= 40000,
            "Invalid assessment interval",
        )
        number(assessment["share_threshold"], "share_threshold", 0.001, 0.5)
        if assessment["resolution"] is None:
            size = [sw, sh]
            while size[0] > 512 and all(n % 2 == 0 for n in size):
                size = [n // 2 for n in size]
            assessment["resolution"] = size
        dimensions(assessment["resolution"])
        reduction_factor((sw, sh), tuple(assessment["resolution"]))
    result = {
        "schema_version": 1,
        "name": name,
        "chromatic_count": count,
        "palette_mode": palette_mode,
        "looks": looks,
        "encounters": encounters,
        "simulation": simulation,
        "surface": surface,
        "projection": projection,
        "render": render,
        "assessment": assessment,
    }
    if "background" in raw:
        from tools.estuary_confluence.backgrounds import NAMES

        require(
            type(raw["background"]) is str and raw["background"] in NAMES,
            "Unknown seeded background",
        )
        result["background"] = raw["background"]
    return result


def frame_raster_resolution(render):
    """Bound the intermediate frame raster before allocating GPU images."""
    factor = render.get("frame_supersampling", 1)
    require(type(factor) is int and factor in (1, 2), "frame_supersampling must be 1 or 2")
    width, height = render["resolution"]
    return dimensions([width * factor, height * factor])


def render_frame(surface, fields, render, *, tilt_degrees, azimuth_degrees):
    """Render a film frame or starting preview, then filter before PNG encoding.

    Surface returns display-bounded linear RGB after its tone map. The equal-area
    box integration happens in that linear space, before the PNG writer applies
    the sRGB transfer curve. Physical state and source cadence are unaffected.
    Native final stills deliberately do not pass through this helper.
    """
    factor = render.get("frame_supersampling", 1)
    if render.get("capture_pipeline", "cpu") == "native-gpu":
        return surface.render_gpu(
            fields,
            size=tuple(render["resolution"]),
            supersampling=factor,
            tilt_degrees=tilt_degrees,
            azimuth_degrees=azimuth_degrees,
        )
    pixels = surface.render(
        fields,
        size=tuple(frame_raster_resolution(render)),
        tilt_degrees=tilt_degrees,
        azimuth_degrees=azimuth_degrees,
    )
    if factor == 1:
        return pixels
    width, height = render["resolution"]
    return pixels.reshape(height, factor, width, factor, 3).mean(axis=(1, 3), dtype=np.float32)


def capture_metadata(recipe):
    render = recipe["render"]
    native = list(recipe["simulation"]["resolution"])
    capture = list(render["capture_resolution"])
    pipeline = render.get("capture_pipeline", "cpu")
    require(pipeline in ("cpu", "native-gpu"), "Unknown capture pipeline")
    require(
        pipeline != "native-gpu" or capture == native,
        "Native GPU capture requires the full simulation grid",
    )
    require(
        recipe["simulation"].get("interaction") is None or capture == native,
        "Interaction capture requires the full native material grid",
    )
    factor = render.get("frame_supersampling", 1)
    result = {
        "schema_version": 1,
        "material_sampling": (
            "full native material grid"
            if capture == native
            else "read-only GPU area averages before pigment optics"
        ),
        "simulation_resolution": native,
        "capture_resolution": capture,
        "material_reduction_factor": native[0] // capture[0],
        "frame_supersampling": factor,
        "frame_raster_resolution": frame_raster_resolution(render),
        "frame_output_resolution": list(render["resolution"]),
        "frame_filter": (
            "none"
            if factor == 1
            else "equal-area 2x2 mean in linear display RGB before sRGB encoding"
        ),
        "final_still": (
            "full native material grid at requested still resolution; no frame downsampling"
        ),
    }
    if "capture_pipeline" in render:
        result["capture_pipeline"] = pipeline
    if recipe["simulation"].get("interaction") is not None:
        from tools.estuary_confluence.interaction import FIELD_NAMES, VERSION

        result["interaction_material"] = {
            "version": VERSION,
            "fields": sorted(FIELD_NAMES),
            "capture": (
                "borrowed native textures from the same canonical material step"
                if pipeline == "native-gpu"
                else "read-only snapshots from the same canonical material step"
            ),
            "camera": "reuse the final material; no interaction update during hold or orbit",
        }
    return result


def capture_frame(engine, render):
    """Retain native GPU material, or use the explicit legacy CPU capture path."""
    if render.get("capture_pipeline", "cpu") == "native-gpu":
        return engine.gpu_frame()
    return engine.snapshot(resolution=tuple(render["capture_resolution"]))


def require_interaction_looks(looks, simulation, surface):
    """Named material comparisons need an explicit, reproducible optical recipe."""
    if INTERACTION_LOOKS.intersection(looks):
        require(
            simulation.get("interaction") is not None,
            "Interaction views require transported interaction material",
        )
        require(
            surface.get("interaction") is not None,
            "Interaction views require explicit surface.interaction strengths",
        )


def require_interaction_geometry(simulation, surface):
    """Packing redistributes paint height, never the independent substrate."""
    controls = surface.get("interaction") or {}
    require(
        controls.get("packing_strength", 0) == 0 or simulation["substrate_um"] == 0,
        "Packing relief requires zero simulation.substrate_um",
    )
    if controls.get("packing_strength", 0) > 0:
        from tools.estuary_confluence.packing import packing_plan

        # Validate the coupled scale/grid limit before an expensive simulation,
        # rather than discovering an unsupported plan during its final still.
        packing_plan(
            tuple(simulation["resolution"]),
            surface["canvas_width_m"] * surface["domain_scale"],
            controls["packing_length_um"],
        )


def surface_configs(recipe):
    """Resolve optical views while sharing one unchanged physical history.

    The three interaction comparisons vary only material appearance strengths.
    Their layered pigment optics use the same mixing control. Legacy view
    dictionaries retain their earlier resolution exactly.
    """
    require_interaction_looks(recipe["looks"], recipe["simulation"], recipe["surface"])
    configs = {}
    for look in recipe["looks"]:
        if look in INTERACTION_LOOKS:
            controls = copy.deepcopy(recipe["surface"]["interaction"])
            if look == "control":
                controls["silk_strength"] = 0.0
            if look != "silk-grain":
                controls["grain_strength"] = 0.0
                if "packing_strength" in controls:
                    controls["packing_strength"] = 0.0
            configs[look] = {
                **recipe["surface"],
                "mode": "layered",
                "mix_control": 1.0,
                "interaction": controls,
            }
        else:
            configs[look] = dict(
                recipe["surface"],
                mode=look,
                mix_control=0.0 if look == "homogeneous" else recipe["surface"]["mix_control"],
            )
    return configs


def resolved_layout(recipe, seed, source=None):
    """Resolve initial geometry without a GPU or any dependence on frame cadence."""
    from tools.estuary_confluence.layout import plan_layout

    settings = recipe["simulation"]
    if settings["initial_pattern"] == "engaged":
        from tools.estuary_confluence.participation_layout import plan_engaged_layout

        require(source is not None, "Engaged layout requires its complete source recording")
        return plan_engaged_layout(source, recipe["chromatic_count"], settings)
    if settings["initial_pattern"] != "scattered":
        return None
    width, height = settings["resolution"]
    return plan_layout(
        seed,
        recipe["chromatic_count"],
        width / height,
        load_radius=settings["load_radius"],
        initial_load=settings["initial_load"],
        edge_width=settings["initial_edge_width"],
    )


def equivalent_design(a, b):
    """Permit only f64 reconstruction roundoff in source-aware design verification."""
    if type(a) is dict and type(b) is dict:
        return a.keys() == b.keys() and all(equivalent_design(a[k], b[k]) for k in a)
    if type(a) is list and type(b) is list:
        return len(a) == len(b) and all(equivalent_design(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return (
            math.isfinite(a)
            and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        )
    return type(a) is type(b) and a == b


def assessment_steps(recipe):
    settings = recipe.get("assessment")
    if settings is None:
        return []
    end = recipe["simulation"]["steps"]
    return sorted({*range(0, end + 1, settings["interval_steps"]), end})


def measure(fields, recipe):
    from tools.estuary_confluence.assessment import assess

    return assess(
        fields["pigment"],
        recipe["chromatic_count"],
        recipe["simulation"]["domain_scale"],
        mass_threshold=recipe["surface"]["paint_mass_threshold"],
        share_threshold=recipe["assessment"]["share_threshold"],
    )


def verify_diagnostics(value, recipe, final_step):
    """Validate reported solver work without claiming to replay it during verification."""
    if value is None:
        return  # Older archives did not record these counters.
    require(
        type(value) is dict
        and set(value)
        == {
            "canonical_steps",
            "actual_transport_substeps",
            "diffusion_substeps",
            "maximum_courant",
            "maximum_diffusion_number",
        },
        "Invalid solver diagnostics schema",
    )
    for key in ("canonical_steps", "actual_transport_substeps", "diffusion_substeps"):
        require(type(value[key]) is int and value[key] >= 0, "Invalid solver work counter")
    require(
        value["canonical_steps"] == final_step
        and final_step <= value["actual_transport_substeps"] <= 500_000,
        "Solver diagnostics disagree with completed source steps",
    )
    number(value["maximum_courant"], "maximum_courant", 0, 1.50001)
    number(value["maximum_diffusion_number"], "maximum_diffusion_number", 0, 0.240001)
    if recipe["simulation"].get("diffusion_coefficient", 0) == 0:
        require(
            value["diffusion_substeps"] == 0 and value["maximum_diffusion_number"] == 0,
            "Disabled diffusion cannot report diffusion work",
        )
    else:
        require(
            value["diffusion_substeps"] >= value["actual_transport_substeps"]
            and value["maximum_diffusion_number"] > 0,
            "Enabled diffusion is missing its work diagnostics",
        )


def field_digest(fields):
    """Container-independent identity of the actual physical state."""
    sha = hashlib.sha256()
    for key in sorted(fields):
        value = np.ascontiguousarray(fields[key])
        sha.update(encoded({"field": key, "shape": list(value.shape), "dtype": value.dtype.str}))
        sha.update(memoryview(value).cast("B"))
    return sha.hexdigest()


def interaction_metadata(recipe, seed):
    """Bind an opt-in material extension without changing legacy request identity."""
    from tools.estuary_confluence.interaction import BASE_FIELDS, VERSION, validate_config
    from tools.estuary_confluence.palette import normalize_seed

    supplied = recipe["simulation"].get("interaction")
    settings = validate_config(supplied)
    if settings is None:
        from tools.estuary_confluence.surface import interaction_enabled

        require(
            not interaction_enabled(recipe["surface"]),
            "Interaction optics require transported interaction material",
        )
        return None
    require(supplied == settings, "Interaction settings must be normalized")
    from tools.estuary_confluence.engine import validate_config as simulation_config
    from tools.estuary_confluence.surface import validate_config as surface_config

    require(
        recipe["simulation"] == simulation_config(recipe["simulation"]),
        "Interaction simulation settings must be normalized",
    )
    require(
        recipe["surface"] == surface_config(recipe["surface"]),
        "Interaction surface settings must be normalized",
    )
    require_interaction_geometry(recipe["simulation"], recipe["surface"])
    return {
        "version": VERSION,
        "seed": normalize_seed(seed),
        "settings": settings,
        "resolution": list(recipe["simulation"]["resolution"]),
        "dtype": "float32",
        "initialization": {
            "origins": "initial simulation world coordinates",
            "contact_fabric_aggregate": "zero",
        },
        "fields": {
            "origin_upper": ["origin_x", "origin_y"],
            "origin_lower": ["origin_x", "origin_y"],
            "interaction_upper": ["contact", "fabric_x", "fabric_y", "aggregate"],
            "interaction_lower": ["contact", "fabric_x", "fabric_y", "aggregate"],
        },
        "base_material_fields": sorted(BASE_FIELDS),
    }


def validate_archived_material(fields, recipe):
    """Require the declared native material, including optional transported history.

    Optical validation checks array values without rewriting them. The archive
    binds the raw state returned by the engine, rather than a display projection.
    """
    from tools.estuary_confluence.interaction import BASE_FIELDS, FIELD_NAMES
    from tools.estuary_confluence.surface import validate_fields

    enabled = recipe["simulation"].get("interaction") is not None
    expected = set(BASE_FIELDS) | (set(FIELD_NAMES) if enabled else set())
    require(set(fields) == expected, "Archived material fields differ from interaction settings")
    validate_fields(fields, recipe["chromatic_count"] + 1)
    if enabled:
        width, height = recipe["simulation"]["resolution"]
        require(
            fields["height"].shape == (height, width),
            "Interaction archive requires the full native material grid",
        )


def base_material_digest(fields):
    """Identity of the original ten fields, excluding new material history.

    Equal base hashes demonstrate unchanged pigment and geometry inputs. They
    do not establish equality of the complete microstructured material.
    """
    from tools.estuary_confluence.interaction import BASE_FIELDS

    return field_digest({key: fields[key] for key in BASE_FIELDS})


def body_marker_metadata(recipe, source_metadata):
    """Describe the optional display annotation without changing legacy requests."""
    from tools.estuary_confluence.body_markers import VERSION, validate_config

    supplied = recipe["render"].get("body_markers")
    settings = validate_config(supplied)
    if settings is None:
        return None
    require(encoded(supplied) == encoded(settings), "Body-marker settings must be normalized")
    return {
        "version": VERSION,
        "config": settings,
        "source": copy.deepcopy(source_metadata),
        "projection": copy.deepcopy(recipe["projection"]),
        "anchor": "projected canvas plane z=0",
        "occlusion": "none; annotation after shading, tone mapping and frame filtering",
        "body_order": "source body index order",
        "pixel_coordinates": "top-left pixel-edge coordinates; pixel centers at half-integers",
        "verification_tolerance": {
            "source_position_absolute": BODY_MARKER_POSITION_ATOL,
            "pixel_center_absolute": BODY_MARKER_PIXEL_ATOL,
            "timing_and_configuration": "exact",
        },
    }


def _body_marker_record(positions, size, fraction, tilt, azimuth):
    from tools.estuary_confluence.body_markers import project_positions

    positions = np.asarray(positions)
    centers = project_positions(positions, tuple(size), tilt_degrees=tilt, azimuth_degrees=azimuth)
    return {
        "source_fraction": float(fraction),
        "resolution": list(size),
        "tilt_degrees": float(tilt),
        "azimuth_degrees": float(azimuth),
        "positions": positions.astype("f8").tolist(),
        "pixel_centers": centers.tolist(),
    }


def make_body_marker_ledger(recipe, source, frames, *, has_initial):
    """Sample actual source time, independently of simulation and output cadence."""
    metadata = body_marker_metadata(recipe, source.metadata)
    if metadata is None:
        return None
    render = recipe["render"]

    def record(fraction, size, tilt, azimuth):
        return _body_marker_record(source.frame(fraction).positions, size, fraction, tilt, azimuth)

    return {
        "metadata": metadata,
        "initial": record(0.0, render["resolution"], 0.0, render["azimuth_start"])
        if has_initial
        else None,
        "poster": record(
            1.0, render["still_resolution"], render["still_tilt_degrees"], render["azimuth_end"]
        ),
        "frames": [
            {
                "frame": index,
                "timing": frame,
                **record(
                    frame["source_fraction"],
                    render["resolution"],
                    frame["tilt_degrees"],
                    frame["azimuth_degrees"],
                ),
            }
            for index, frame in enumerate(frames)
        ],
    }


def validate_body_marker_records(request, receipt, ledger):
    """Verify portable source/config/timing/projection associations.

    This does not recover orbit samples from a gallery. Full archive verification
    separately regenerates every position from the archived recording.
    """
    expected = body_marker_metadata(request["recipe"], request["source"])
    if expected is None:
        require(
            "body_markers" not in request and "body_markers" not in receipt and ledger is None,
            "Disabled body markers cannot advertise annotation records",
        )
        return
    require(
        encoded(request.get("body_markers")) == encoded(expected)
        and encoded(receipt.get("body_markers")) == encoded(expected),
        "Body-marker source, version or settings differ",
    )
    require(
        type(ledger) is dict
        and set(ledger) == {"metadata", "initial", "poster", "frames"}
        and encoded(ledger["metadata"]) == encoded(expected),
        "Body-marker ledger metadata differs",
    )
    recipe, render = request["recipe"], request["recipe"]["render"]
    expected_frames = frame_plan(recipe) if request["mode"] == "film" else []
    require(
        encoded(request["frames"]) == encoded(expected_frames),
        "Body-marker canonical timeline differs",
    )

    def check(record, fraction, size, tilt, azimuth, extra=None):
        require(type(record) is dict and "positions" in record, "Missing body-marker positions")
        projected = _body_marker_record(record["positions"], size, fraction, tilt, azimuth)
        projected.update(extra or {})
        require(
            set(record) == set(projected)
            and encoded(
                {
                    key: value
                    for key, value in record.items()
                    if key not in ("positions", "pixel_centers")
                }
            )
            == encoded(
                {
                    key: value
                    for key, value in projected.items()
                    if key not in ("positions", "pixel_centers")
                }
            ),
            "Body-marker timing or camera differs",
        )
        require(
            _marker_coordinates_equal(
                record["pixel_centers"], projected["pixel_centers"], BODY_MARKER_PIXEL_ATOL
            ),
            "Body-marker pixel projection differs",
        )

    if request.get("layout") is None:
        require(ledger["initial"] is None, "Unbound initial body-marker record")
    else:
        check(ledger["initial"], 0.0, render["resolution"], 0.0, render["azimuth_start"])
    check(
        ledger["poster"],
        1.0,
        render["still_resolution"],
        render["still_tilt_degrees"],
        render["azimuth_end"],
    )
    require(
        type(ledger["frames"]) is list and len(ledger["frames"]) == len(expected_frames),
        "Body-marker frame count differs",
    )
    for index, (record, frame) in enumerate(zip(ledger["frames"], expected_frames, strict=True)):
        check(
            record,
            frame["source_fraction"],
            render["resolution"],
            frame["tilt_degrees"],
            frame["azimuth_degrees"],
            {"frame": index, "timing": frame},
        )


def _marker_coordinates_equal(actual, expected, tolerance):
    actual = np.asarray(actual)
    return (
        actual.shape == (3, 2)
        and actual.dtype.kind in "fiu"
        and np.isfinite(actual).all()
        and np.allclose(actual, expected, rtol=0, atol=tolerance)
    )


def _marker_records(ledger):
    initial = [] if ledger["initial"] is None else [ledger["initial"]]
    return [*initial, ledger["poster"], *ledger["frames"]]


def _annotate_body_markers(pixels, record, metadata):
    from tools.estuary_confluence.body_markers import annotate

    return annotate(
        pixels,
        record["positions"],
        tilt_degrees=record["tilt_degrees"],
        azimuth_degrees=record["azimuth_degrees"],
        config=metadata["config"],
    )


def verify_run(folder):
    from tools.estuary_confluence.palette import generate_palette, normalize_seed

    folder = Path(folder)
    request, receipt = read(folder / "request.json"), read(folder / "receipt.json")
    identity = hashlib.sha256(encoded(request)).hexdigest()
    require(
        receipt.get("complete") is True and receipt.get("identity_sha256") == identity,
        "Confluence archive is incomplete or identity differs",
    )
    required = {
        "inputs/source.orbit",
        "recipe.json",
        "palette.json",
        "events.json",
        "final.npz",
        "frame-ledger.json",
    }
    markers = body_marker_metadata(request["recipe"], request["source"])
    if markers is not None:
        required.add("body-markers.json")
        required.update(f"{look}/poster-unmarked-linear.npy" for look in request["recipe"]["looks"])
    else:
        validate_body_marker_records(request, receipt, None)
        require(
            "body-markers.json" not in receipt["artifacts"]
            and not (folder / "body-markers.json").exists()
            and all(
                f"{look}/poster-unmarked-linear.npy" not in receipt["artifacts"]
                for look in request["recipe"]["looks"]
            ),
            "Disabled body markers cannot contain annotation artifacts",
        )
    for look in request["recipe"]["looks"]:
        required |= {f"{look}/poster.png", f"{look}/poster-linear.npy"}
        if request.get("layout") is not None:
            required |= {"layout.json", f"{look}/initial.png"}
        if request["mode"] == "film":
            required |= {f"{look}/film.mp4", f"{look}/movie.json"}
    if request["recipe"]["surface"].get("optics_model", "rgb") == "spectral":
        required.add("spectral.json")
    if request["recipe"].get("assessment") is not None:
        required.add("assessment.json")
    if request["recipe"]["simulation"].get("mass_budget_interval_steps", 0):
        required.add("mass-budget.json")
    else:
        require(
            "mass-budget.json" not in receipt["artifacts"],
            "Disabled mass restoration cannot advertise a budget report",
        )
    if "background" in request["recipe"]:
        from tools.estuary_confluence.backgrounds import validate_background

        required.add("background.json")
        background = validate_background(request.get("background"), request["palette"])
        require(
            background["name"] == request["recipe"]["background"]
            and background["ground_srgb"] == request["recipe"]["surface"]["ground_srgb"]
            and read(folder / "background.json") == background,
            "Seeded background differs from its painting",
        )
    require(required <= receipt["artifacts"].keys(), "Required confluence artifacts are missing")
    for name, record in receipt["artifacts"].items():
        checked(folder, name, record)
    verify_code(folder / "inputs/code", request["code"])
    require(request["mode"] in ("film", "still"), "Unknown archive mode")
    expected_frames = frame_plan(request["recipe"]) if request["mode"] == "film" else []
    require(
        request["frames"] == expected_frames, "Film does not cover the canonical complete timeline"
    )
    require(
        read(folder / "recipe.json") == request["recipe"]
        and read(folder / "palette.json") == request["palette"]
        and read(folder / "events.json") == request["events"],
        "Archived design inputs differ",
    )
    bound_source = None
    if request["recipe"]["simulation"]["initial_pattern"] == "engaged" or markers is not None:
        w, h = request["recipe"]["simulation"]["resolution"]
        bound_source = Source.read(
            folder / "inputs/source.orbit", aspect=w / h, **request["recipe"]["projection"]
        )
        require(
            equivalent_design(bound_source.metadata, request["source"]), "Source projection differs"
        )
    expected_layout = resolved_layout(request["recipe"], request["source"]["seed"], bound_source)
    require(
        equivalent_design(request.get("layout"), expected_layout), "Seeded starting layout differs"
    )
    if expected_layout is not None:
        require(read(folder / "layout.json") == request["layout"], "Archived starting pools differ")
    marker_ledger = None
    if markers is not None:
        marker_ledger = read(folder / "body-markers.json")
        validate_body_marker_records(request, receipt, marker_ledger)
        expected_markers = make_body_marker_ledger(
            request["recipe"],
            bound_source,
            request["frames"],
            has_initial=expected_layout is not None,
        )
        for actual, expected in zip(
            _marker_records(marker_ledger), _marker_records(expected_markers), strict=True
        ):
            require(
                _marker_coordinates_equal(
                    actual["positions"], expected["positions"], BODY_MARKER_POSITION_ATOL
                )
                and _marker_coordinates_equal(
                    actual["pixel_centers"], expected["pixel_centers"], BODY_MARKER_PIXEL_ATOL
                ),
                "Body-marker positions differ from the archived trajectory",
            )
    if request["recipe"]["surface"].get("optics_model", "rgb") == "spectral":
        from tools.estuary_confluence.spectral import validate_spectral_material

        validate_spectral_material(request.get("spectral"), request["palette"])
        require(
            read(folder / "spectral.json") == request["spectral"], "Archived pigment spectra differ"
        )
    else:
        require(request.get("spectral") is None, "RGB archive cannot contain spectral inputs")
    require(
        request["recipe"].get("palette_mode", "curated")
        == request["palette"].get("mode", "curated"),
        "Palette algorithm differs from its recipe",
    )
    require(
        encoded(request["palette"])
        == encoded(
            generate_palette(
                request["source"]["seed"],
                request["recipe"]["chromatic_count"],
                mode=request["recipe"].get("palette_mode", "curated"),
            )
        ),
        "Palette is not derived from its seed and algorithm",
    )
    require(
        receipt["source"] == request["source"]
        and receipt["artifacts"]["inputs/source.orbit"]["sha256"] == request["source"]["sha256"]
        and normalize_seed(request["source"]["seed"]) == request["palette"]["seed"],
        "Palette and trajectory identity differ",
    )
    require(
        receipt["source_fraction"] == 1.0
        and receipt["final_step"] == request["recipe"]["simulation"]["steps"],
        "Painting does not contain the full trajectory",
    )
    verify_diagnostics(receipt.get("solver_diagnostics"), request["recipe"], receipt["final_step"])
    if type(request.get("capture")) is dict:
        require(
            request["capture"] == capture_metadata(request["recipe"]),
            "Capture resolution or frame filtering metadata differs",
        )
    else:
        require(
            "frame_supersampling" not in request["recipe"]["render"]
            and "capture_pipeline" not in request["recipe"]["render"]
            and request.get("capture")
            == "read-only GPU area averages; final still uses full physical state",
            "Invalid legacy capture metadata",
        )
    require(
        request["surface_configs"] == surface_configs(request["recipe"]),
        "Optical views differ from their recipe",
    )
    interaction = interaction_metadata(request["recipe"], request["source"]["seed"])
    if interaction is None:
        require(
            "interaction" not in request
            and "interaction" not in receipt
            and "base_material_sha256" not in receipt,
            "Disabled interaction cannot advertise microstructured material",
        )
    else:
        require(
            request.get("interaction") == interaction and receipt.get("interaction") == interaction,
            "Interaction version, seed, settings or field contract differs",
        )
    require(set(receipt["looks"]) == set(request["recipe"]["looks"]), "View selection differs")
    ledger = read(folder / "frame-ledger.json")
    require(len(ledger) == len(request["frames"]), "Frame ledger length differs")
    for i, (frame, timing) in enumerate(zip(ledger, request["frames"], strict=True)):
        require(
            frame["frame"] == i
            and frame["timing"] == timing
            and set(frame["images"]) == set(receipt["looks"]),
            "Frame ledger timing differs",
        )
        for look, record in frame["images"].items():
            require(
                record == receipt["artifacts"][f"{look}/frames/{i:06d}.png"],
                "Certified frame differs",
            )
            if i == 0 and expected_layout is not None:
                require(
                    record == receipt["artifacts"][f"{look}/initial.png"],
                    "Starting-pool image differs from the film's initial state",
                )
    with np.load(folder / "final.npz", allow_pickle=False) as archive:
        final = {key: archive[key] for key in archive.files}
    validate_archived_material(final, request["recipe"])
    actual_state = field_digest(final)
    require(actual_state == receipt["physical_state_sha256"], "Physical state identity differs")
    if interaction is not None:
        require(
            receipt.get("base_material_sha256") == base_material_digest(final),
            "Base material identity differs",
        )
    if request["recipe"]["simulation"].get("mass_budget_interval_steps", 0):
        from tools.estuary_confluence.mass_budget import validate_report

        validate_report(
            read(folder / "mass-budget.json"), request["recipe"], final, layout=expected_layout
        )
    if request["recipe"].get("assessment") is not None:
        from tools.estuary_confluence.assessment import VERSION as assessment_version

        report = read(folder / "assessment.json")
        require(report.get("version") == assessment_version, "Assessment version differs")
        require(report["settings"] == request["recipe"]["assessment"], "Assessment settings differ")
        require(
            [row["step"] for row in report["samples"]] == assessment_steps(request["recipe"])
            and all(
                row["source_fraction"] == row["step"] / receipt["final_step"]
                for row in report["samples"]
            ),
            "Assessment does not span its canonical checkpoint schedule",
        )
        require(
            report["final"] == measure(final, request["recipe"]),
            "Final participation metrics differ",
        )
    for look, result in receipt["looks"].items():
        require(
            result["physical_state_sha256"] == actual_state,
            "Optical views use different physical states",
        )
        require(
            result["poster"] == receipt["artifacts"][f"{look}/poster.png"], "View poster differs"
        )
        if markers is not None:
            plain = np.load(folder / look / "poster-unmarked-linear.npy", allow_pickle=False)
            annotated = np.load(folder / look / "poster-linear.npy", allow_pickle=False)
            width, height = request["recipe"]["render"]["still_resolution"]
            require(
                plain.shape == annotated.shape == (height, width, 3)
                and plain.dtype == annotated.dtype == np.float32,
                "Body-marker poster raster shape or dtype differs",
            )
            require(
                np.allclose(
                    annotated,
                    _annotate_body_markers(plain, marker_ledger["poster"], markers),
                    rtol=0,
                    atol=2e-6,
                ),
                "Body-marker poster differs from its unmarked painting and recorded positions",
            )
        if request["recipe"].get("assessment") is not None:
            from tools.estuary_confluence.assessment import image_balance

            pixels = (
                plain
                if markers is not None
                else np.load(folder / look / "poster-linear.npy", allow_pickle=False)
            )
            require(
                result.get("image_balance")
                == image_balance(
                    pixels, srgb_to_linear(request["recipe"]["surface"]["ground_srgb"])
                ),
                "Image balance differs",
            )
        if request["mode"] == "film":
            movie = read(folder / look / "movie.json")
            require(
                movie == result["movie"]
                and movie["full_decode_verified"] is True
                and movie["frames"] == len(ledger)
                and movie["fps"] == request["recipe"]["render"]["fps"]
                and movie["resolution"] == request["recipe"]["render"]["resolution"]
                and {k: movie["artifact"][k] for k in ("sha256", "bytes")}
                == receipt["artifacts"][f"{look}/film.mp4"],
                "View movie differs from its certified timeline",
            )
        else:
            require(result["movie"] is None, "Still-only view cannot advertise a film")
    return request, receipt


def run(args):
    from tools.estuary_confluence.engine import Engine
    from tools.estuary_confluence.events import plan_events
    from tools.estuary_confluence.palette import generate_palette
    from tools.estuary_confluence.surface import Surface

    recipe = read(args.recipe)
    for arg, key in (
        ("resolution", "resolution"),
        ("capture_resolution", "capture_resolution"),
        ("image_size", "still_resolution"),
        ("video_size", "resolution"),
    ):
        value = getattr(args, arg, None)
        if value is not None:
            section = "simulation" if arg == "resolution" else "render"
            recipe.setdefault(section, {})[key] = value
    recipe = validate_recipe(recipe)
    code = runtime_identity()
    sw, sh = recipe["simulation"]["resolution"]
    source = Source.read(args.source, aspect=sw / sh, **recipe["projection"])
    interaction = interaction_metadata(recipe, source.seed)
    palette = generate_palette(source.seed, recipe["chromatic_count"], mode=recipe["palette_mode"])
    background = None
    if "background" in recipe:
        from tools.estuary_confluence.backgrounds import generate_background

        background = generate_background(recipe["background"], palette)
        recipe["surface"]["ground_srgb"] = background["ground_srgb"]
    layout = resolved_layout(recipe, source.seed, source)
    spectral = None
    if recipe["surface"]["optics_model"] == "spectral":
        from tools.estuary_confluence.spectral import build_spectral_material

        spectral = build_spectral_material(palette)
    events = plan_events(source, recipe["encounters"])
    frames = [] if args.still_only else frame_plan(recipe)
    marker_ledger = make_body_marker_ledger(recipe, source, frames, has_initial=layout is not None)
    markers = None if marker_ledger is None else marker_ledger["metadata"]
    binaries = {}
    if frames:
        for name in ("ffmpeg", "ffprobe"):
            path = Path(shutil.which(name) or "").resolve(strict=True)
            require(path.is_file(), f"{name} is unavailable")
            binaries[name] = {"path": str(path), **artifact(path)}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = Engine(source, recipe["simulation"], palette, events)
        surfaces = {}
        try:
            require(
                equivalent_design(getattr(engine, "layout", None), layout),
                "Engine starting layout differs",
            )
            configs = surface_configs(recipe)
            surface_options = {"spectral": spectral} if spectral is not None else {}
            if recipe["render"]["capture_pipeline"] == "native-gpu":
                surface_options["gpu_frame"] = engine.gpu_frame()
            for look, config in configs.items():
                surfaces[look] = Surface(config, palette, **surface_options)
            request = {
                "schema_version": 1,
                "recipe": recipe,
                "surface_configs": configs,
                "code": code,
                "source": source.metadata,
                "palette": palette,
                "events": events,
                "layout": layout,
                "spectral": spectral,
                "mode": "film" if frames else "still",
                "frames": frames,
                "binaries": binaries,
                "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                "hardware": {
                    "simulation": engine.metadata,
                    "surfaces": {k: v.metadata for k, v in surfaces.items()},
                },
                "capture": capture_metadata(recipe),
            }
            if background is not None:
                request["background"] = background
            if interaction is not None:
                request["interaction"] = interaction
            if markers is not None:
                request["body_markers"] = markers
            identity = hashlib.sha256(encoded(request)).hexdigest()
            if (output / "request.json").exists():
                require(
                    read(output / "request.json") == request, "Existing run has different inputs"
                )
                verify_run(output)
                print(encoded({"status": "reused", "output": str(output)}).decode())
                return
            require(not any(p.name != ".lock" for p in output.iterdir()), "Output must be empty")
            write(output / "request.json", request)
            write(output / "receipt.json", {"complete": False, "identity_sha256": identity})
            for name, value in (("recipe", recipe), ("palette", palette), ("events", events)):
                write(output / f"{name}.json", value)
            if layout is not None:
                write(output / "layout.json", layout)
            if spectral is not None:
                write(output / "spectral.json", spectral)
            if background is not None:
                write(output / "background.json", background)
            if marker_ledger is not None:
                write(output / "body-markers.json", marker_ledger)
            (output / "inputs").mkdir()
            shutil.copyfile(args.source, output / "inputs/source.orbit")
            require(
                digest(output / "inputs/source.orbit") == source.sha256,
                "Source changed while archiving",
            )
            code_folder = output / "inputs/code/tools"
            for package, files in code.items():
                for name in files:
                    target = code_folder / package / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT.parent / package / name, target)
            verify_code(output / "inputs/code", code)
            started = time.monotonic()
            render = recipe["render"]
            artifacts = {
                name: artifact(output / name)
                for name in ("recipe.json", "palette.json", "events.json", "inputs/source.orbit")
            }
            if layout is not None:
                artifacts["layout.json"] = artifact(output / "layout.json")
            if spectral is not None:
                artifacts["spectral.json"] = artifact(output / "spectral.json")
            if background is not None:
                artifacts["background.json"] = artifact(output / "background.json")
            if marker_ledger is not None:
                artifacts["body-markers.json"] = artifact(output / "body-markers.json")
            measurements = []
            pending = iter(assessment_steps(recipe))
            next_checkpoint = next(pending, None)

            def advance(target):
                nonlocal next_checkpoint
                if recipe["assessment"] is None:
                    engine.advance_to(target)
                    return
                while next_checkpoint is not None and next_checkpoint <= target:
                    engine.advance_to(next_checkpoint)
                    sampled = engine.snapshot(resolution=tuple(recipe["assessment"]["resolution"]))
                    measurements.append(
                        {
                            "step": next_checkpoint,
                            "source_fraction": next_checkpoint / engine.steps,
                            "metrics": measure(sampled, recipe),
                        }
                    )
                    next_checkpoint = next(pending, None)
                if engine.step < target:
                    engine.advance_to(target)

            for look in surfaces:
                (output / look).mkdir()
                if frames:
                    (output / look / "frames").mkdir()
            if layout is not None and not frames:
                initial = capture_frame(engine, render)
                for look, surface in surfaces.items():
                    pixels = render_frame(
                        surface,
                        initial,
                        render,
                        tilt_degrees=0.0,
                        azimuth_degrees=render["azimuth_start"],
                    )
                    if markers is not None:
                        pixels = _annotate_body_markers(pixels, marker_ledger["initial"], markers)
                    path = output / look / "initial.png"
                    write_png(path, pixels, depth=8)
                    artifacts[f"{look}/initial.png"] = artifact(path)
            fields, ledger = None, []
            for i, frame in enumerate(frames):
                changed = fields is None or frame["step"] != engine.step
                if changed:
                    advance(frame["step"])
                    fields = capture_frame(engine, render)
                images = {}
                for look, surface in surfaces.items():
                    name = f"{look}/frames/{i:06d}.png"
                    path = output / name
                    if frame["phase"] == "hold":
                        shutil.copyfile(output / f"{look}/frames/{i - 1:06d}.png", path)
                    else:
                        pixels = render_frame(
                            surface,
                            fields if changed else None,
                            render,
                            tilt_degrees=frame["tilt_degrees"],
                            azimuth_degrees=frame["azimuth_degrees"],
                        )
                        if markers is not None:
                            pixels = _annotate_body_markers(
                                pixels, marker_ledger["frames"][i], markers
                            )
                        write_png(path, pixels, depth=8)
                    images[look] = artifacts[name] = artifact(path)
                    if i == 0 and layout is not None:
                        shutil.copyfile(path, output / look / "initial.png")
                        artifacts[f"{look}/initial.png"] = images[look]
                ledger.append({"frame": i, "timing": frame, "images": images})
                if i % 24 == 0 or i == len(frames) - 1:
                    write(
                        output / "progress.json",
                        {
                            "frames": i + 1,
                            "total": len(frames),
                            "source_fraction": frame["source_fraction"],
                            "phase": frame["phase"],
                        },
                    )
                    print(f"CONFLUENCE_FRAME {i + 1}/{len(frames)} {frame['phase']}", flush=True)
            if not frames:
                advance(recipe["simulation"]["steps"])
            final = engine.snapshot()
            validate_archived_material(final, recipe)
            if recipe["simulation"].get("mass_budget_interval_steps", 0):
                from tools.estuary_confluence.mass_budget import validate_report

                budget = engine.mass_budget_report
                validate_report(budget, recipe, final, layout=layout)
                write(output / "mass-budget.json", budget)
                artifacts["mass-budget.json"] = artifact(output / "mass-budget.json")
            record_array(output / "final.npz", final)
            final_identity = field_digest(final)
            if recipe["assessment"] is not None:
                from tools.estuary_confluence.assessment import VERSION as assessment_version

                write(
                    output / "assessment.json",
                    {
                        "version": assessment_version,
                        "settings": recipe["assessment"],
                        "samples": measurements,
                        "final": measure(final, recipe),
                    },
                )
                artifacts["assessment.json"] = artifact(output / "assessment.json")
            results = {}
            for look, surface in surfaces.items():
                pixels = surface.render(
                    final,
                    size=tuple(render["still_resolution"]),
                    tilt_degrees=render["still_tilt_degrees"],
                    azimuth_degrees=render["azimuth_end"],
                )
                unmarked = pixels
                if markers is not None:
                    name = f"{look}/poster-unmarked-linear.npy"
                    write_array(output / name, unmarked)
                    artifacts[name] = artifact(output / name)
                    pixels = _annotate_body_markers(pixels, marker_ledger["poster"], markers)
                write_png(output / look / "poster.png", pixels)
                write_array(output / look / "poster-linear.npy", pixels)
                for name in (f"{look}/poster.png", f"{look}/poster-linear.npy"):
                    artifacts[name] = artifact(output / name)
                results[look] = {
                    "poster": artifacts[f"{look}/poster.png"],
                    "movie": None,
                    "physical_state_sha256": final_identity,
                }
                if recipe["assessment"] is not None:
                    from tools.estuary_confluence.assessment import image_balance

                    results[look]["image_balance"] = image_balance(
                        unmarked, srgb_to_linear(recipe["surface"]["ground_srgb"])
                    )
            write(output / "frame-ledger.json", ledger)
            for name in ("final.npz", "frame-ledger.json"):
                artifacts[name] = artifact(output / name)
            for path in sorted((output / "inputs/code").rglob("*")):
                if path.is_file():
                    artifacts[str(path.relative_to(output))] = artifact(path)
            if frames:
                encoding_recipe = {
                    "render": {
                        "frames": len(frames),
                        "fps": render["fps"],
                        "resolution": render["resolution"],
                    }
                }
                for look in surfaces:
                    results[look]["movie"] = encode_movie(
                        output / look,
                        encoding_recipe,
                        binaries["ffmpeg"]["path"],
                        binaries["ffprobe"]["path"],
                    )
                    for name in (f"{look}/movie.json", f"{look}/film.mp4"):
                        artifacts[name] = artifact(output / name)
            require(
                runtime_identity() == code and digest(args.source) == source.sha256,
                "Runtime or original source changed",
            )
            write(
                output / "receipt.json",
                {
                    "schema_version": 1,
                    "complete": True,
                    "identity_sha256": identity,
                    "source": source.metadata,
                    "source_fraction": 1.0,
                    "final_step": engine.step,
                    "solver_diagnostics": getattr(engine, "diagnostics", None),
                    "physical_state_sha256": final_identity,
                    **({"body_markers": markers} if markers is not None else {}),
                    **(
                        {
                            "interaction": interaction,
                            "base_material_sha256": base_material_digest(final),
                        }
                        if interaction is not None
                        else {}
                    ),
                    "looks": results,
                    "seconds": time.monotonic() - started,
                    "artifacts": artifacts,
                },
            )
            print(
                encoded(
                    {
                        "status": "complete",
                        "output": str(output),
                        "seconds": time.monotonic() - started,
                    }
                ).decode()
            )
        finally:
            for surface in surfaces.values():
                surface.close()
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "recipe", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--still-only", action="store_true")
    for name in ("resolution", "capture-resolution", "image-size", "video-size"):
        parser.add_argument("--" + name, nargs=2, type=int)
    args = parser.parse_args()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == "__main__":
    main()
