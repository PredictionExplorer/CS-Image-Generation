"""Deterministic fresh full-width seeds and pinned production-orbit recordings.

Seeds are never selected by their rendered appearance. Only collisions with the
frozen historical seed inventory or an earlier digest are rejected. The original
production candidate search remains part of each seed's orbit generation; its
fixed budget, selected candidate and retries are retained in source provenance.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.estuary_studio.common import artifact, checked, encoded, read, require, write

VERSION = "fresh-filament-cohort-v1"
DOMAIN = "the-estuary/filament-films/new-cohort-v1"
COUNT = 10
EXPORTER_SHA256 = "2192e58590fb593e371232b871066031ebcb9fdc66abd81d00e949b4b24fc261"
SETTINGS = {
    "sims": 30000,
    "steps": 1000000,
    "sample_stride": 1,
    "selection_width": 1920,
    "selection_height": 1242,
    "chaos_weight": None,
    "equil_weight": None,
    "generation_record": None,
    "initial_conditions": None,
}
PROJECTION = {"aspect": 4 / 3, "fill": 0.78, "rotation_degrees": 0.0}
PHYSICS = {
    "integrator": "production-yoshida4-native-f64-v1",
    "gravitational_constant": 9.8,
    "integration_dt": 0.001,
    "warmup_steps": 1000000,
    "recording_steps": 1000000,
    "sample_stride": 1,
    "coordinate_frame": "original-Newtonian-center-of-mass",
}


def _sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _identity(value):
    return _sha({key: item for key, item in value.items() if key != "identity_sha256"})


def _projection_equal(left, right):
    """Allow only float64 PCA reconstruction roundoff when checking copied sources."""
    if type(left) is dict and type(right) is dict:
        return left.keys() == right.keys() and all(
            _projection_equal(left[key], right[key]) for key in left
        )
    if type(left) is list and type(right) is list:
        return len(left) == len(right) and all(
            _projection_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    # Source's PCA scale is a NumPy float64, a float subclass. JSON restores it
    # as a built-in float; both carry the same double-precision numeric value.
    if isinstance(left, float) and isinstance(right, float):
        return (
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
        )
    return type(left) is type(right) and left == right


def _seed_number(seed):
    if type(seed) is int:
        number = seed
    elif type(seed) is str and re.fullmatch(r"(?:0[xX])?[0-9a-fA-F]{1,64}", seed):
        number = int(seed, 16)
    else:
        raise ValueError("Seed must be an integer or hexadecimal string")
    require(0 <= number < 1 << 256, "Seed must fit in 256 bits")
    return number


def derive_seeds(excluded, *, count=COUNT, domain=DOMAIN):
    """Keep all 32 digest bytes, including leading zero bytes needed by Rust."""
    require(type(count) is int and 1 <= count <= 100, "Use one to 100 fresh seeds")
    require(
        type(domain) is str and 0 < len(domain) <= 256 and "\0" not in domain, "Invalid seed domain"
    )
    require(type(excluded) in (list, tuple, set), "Provide an explicit historical seed inventory")
    known = {_seed_number(value) for value in excluded}
    selected, attempts, seen = [], [], set()
    for counter in range(10000):
        digest = hashlib.sha256(
            VERSION.encode("ascii")
            + b"\0"
            + domain.encode("utf-8")
            + b"\0"
            + counter.to_bytes(8, "big")
        ).digest()
        seed, number = "0x" + digest.hex(), int.from_bytes(digest, "big")
        reason = (
            "historical-collision"
            if number in known
            else "cohort-collision"
            if number in seen
            else None
        )
        attempts.append(
            {"counter": counter, "seed": seed, "accepted": reason is None, "rejection": reason}
        )
        if reason is None:
            selected.append(seed)
            seen.add(number)
        if len(selected) == count:
            break
    require(len(selected) == count, "Fresh seed derivation exhausted its bounded counter")
    canonical = [hex(number) for number in sorted(known)]
    return {
        "version": VERSION,
        "domain": domain,
        "algorithm": (
            "SHA-256(version ASCII + NUL + domain UTF-8 + NUL + uint64 big-endian counter)"
        ),
        "encoding": (
            "0x followed by all 64 lowercase hexadecimal digest digits; never trim leading zeros"
        ),
        "historical_seeds": canonical,
        "historical_seeds_sha256": _sha(canonical),
        "attempts": attempts,
        "seeds": selected,
    }


def _fingerprint(record):
    require(
        type(record) is dict
        and type(record.get("sha256")) is str
        and re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        and type(record.get("bytes")) is int
        and record["bytes"] > 0,
        "Invalid cohort artifact fingerprint",
    )


def _validate_plan(plan):
    require(
        plan.get("version") == VERSION and plan.get("identity_sha256") == _identity(plan),
        "Cohort plan identity differs",
    )
    require(
        plan.get("settings") == SETTINGS and plan.get("projection") == PROJECTION,
        "Cohort generation settings differ",
    )
    history = plan["historical_inventory"]
    require(
        history.get("version") == "historical-art-seed-inventory-v1", "Unknown historical inventory"
    )
    derivation = derive_seeds(history["seeds"])
    require(
        plan["derivation"] == derivation and plan["seeds"] == derivation["seeds"],
        "Fresh cohort does not follow its declared seed derivation",
    )
    require(len(set(plan["seeds"])) == COUNT, "Fresh cohort requires ten unique seeds")
    generator = plan["generator"]
    _fingerprint(generator)
    require(
        generator["sha256"] == EXPORTER_SHA256
        and generator.get("path") == "tools/tidal_silk-full-v02"
        and generator.get("threads_per_export") == 8
        and generator.get("parallel_exports") == 2,
        "Fresh cohort exporter or bounded worker settings differ",
    )
    return plan


def _validate_row(row, plan):
    seed = row["seed"]
    require(
        seed in plan["seeds"] and re.fullmatch(r"0x[0-9a-f]{64}", seed),
        "Unexpected fresh source seed",
    )
    _fingerprint(row)
    require(row.get("path") == f"orbits/{seed}.orbit", "Fresh source path differs")
    metadata = row["source_metadata"]
    require(
        type(row.get("samples")) is int
        and type(metadata.get("samples")) is int
        and row["samples"] == metadata["samples"] == SETTINGS["steps"]
        and row.get("dt") == metadata.get("dt") == 0.001
        and row.get("samples_sha256") == metadata.get("samples_sha256")
        and metadata.get("seed") == seed
        and metadata.get("sha256") == row["sha256"]
        and type(metadata.get("source_first_step")) is int
        and metadata["source_first_step"] == 0
        and type(metadata.get("source_last_step")) is int
        and metadata["source_last_step"] == SETTINGS["steps"] - 1,
        "Fresh recording does not contain its complete declared source",
    )
    require(
        type(row["samples_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["samples_sha256"]),
        "Invalid physical sample identity",
    )
    projection = metadata["projection"]
    require(
        all(projection.get(key) == value for key, value in PROJECTION.items()),
        "Fresh source projection differs",
    )
    provenance = metadata["provenance"]
    require(
        all(provenance.get(key) == value for key, value in PHYSICS.items()),
        "Fresh source physics differs",
    )
    selected = provenance["selection"]
    require(
        selected.get("source") == "new-production-selection"
        and selected.get("candidate_count") == SETTINGS["sims"]
        and selected.get("selection_resolution")
        == [SETTINGS["selection_width"], SETTINGS["selection_height"]]
        and selected.get("sampling_bounds")
        == {"min_mass": 100.0, "max_mass": 300.0, "location": 300.0, "velocity": 1.0}
        and selected.get("escape_threshold") == -0.3,
        "Fresh source selection budget or physical sampling differs",
    )
    require(
        type(selected.get("candidate_index")) is int
        and 0 <= selected["candidate_index"] < SETTINGS["sims"]
        and type(selected.get("retry_count")) is int
        and selected["retry_count"] >= 0
        and all(
            type(selected.get(key)) in (int, float)
            and math.isfinite(selected[key])
            and selected[key] > 0
            for key in ("chaos_weight", "equil_weight")
        ),
        "Invalid selected candidate or production selection weights",
    )
    require(row.get("config") == {"seed": seed, **SETTINGS}, "Exported source config differs")
    require(row.get("returncode") == 0, "Source export did not complete")
    for name, expected in (
        ("config_artifact", f"configs/{seed}.json"),
        ("log_artifact", f"logs/{seed}.log"),
    ):
        _fingerprint(row[name])
        require(row[name].get("path") == expected, "Export evidence path differs")


def verify_cohort(manifest, *, source_root=None):
    """Verify offline identities, or also raw orbit bytes/projections at launch.

    Returned rows contain seed, relative ``path``, sha256, bytes and samples.
    All filenames preserve the full 32-byte seed. Absolute original paths are
    unnecessary for portable verification of the embedded manifest.
    """
    require(
        type(manifest) is dict
        and manifest.get("version") == VERSION
        and manifest.get("complete") is True,
        "Fresh cohort is incomplete",
    )
    require(
        manifest.get("identity_sha256") == _identity(manifest), "Cohort manifest identity differs"
    )
    plan = _validate_plan(manifest["plan"])
    require(
        manifest.get("seeds") == plan["seeds"] and not manifest.get("failures"),
        "Cohort selection or exports differ",
    )
    rows = manifest["sources"]
    require(
        type(rows) is list and [row.get("seed") for row in rows] == plan["seeds"],
        "Cohort source membership or order differs",
    )
    for row in rows:
        _validate_row(row, plan)
    if source_root is not None:
        from tools.estuary.source import Source

        root = Path(source_root)
        checked(
            root,
            plan["generator"]["path"],
            {key: plan["generator"][key] for key in ("sha256", "bytes")},
        )
        for row in rows:
            path = checked(root, row["path"], {key: row[key] for key in ("sha256", "bytes")})
            source = Source.read(path, **PROJECTION)
            actual, expected = source.metadata, row["source_metadata"]
            require(
                {key: value for key, value in actual.items() if key != "projection"}
                == {key: value for key, value in expected.items() if key != "projection"}
                and _projection_equal(actual["projection"], expected["projection"]),
                "Actual fresh source differs from its archived metadata",
            )
            for name in ("config_artifact", "log_artifact"):
                record = row[name]
                checked(root, record["path"], {key: record[key] for key in ("sha256", "bytes")})
            require(
                read(root / row["config_artifact"]["path"]) == row["config"],
                "Archived source config differs",
            )
    return {row["seed"]: row for row in rows}


def prepare_cohort(output, exporter, historical_inventory):
    output, exporter = Path(output).resolve(), Path(exporter).resolve(strict=True)
    history = read(historical_inventory)
    generator = {
        "path": "tools/tidal_silk-full-v02",
        **artifact(exporter),
        "threads_per_export": 8,
        "parallel_exports": 2,
    }
    require(generator["sha256"] == EXPORTER_SHA256, "Use the pinned production orbit exporter")
    derivation = derive_seeds(history["seeds"])
    plan = {
        "version": VERSION,
        "seeds": derivation["seeds"],
        "derivation": derivation,
        "historical_inventory": history,
        "generator": generator,
        "settings": copy.deepcopy(SETTINGS),
        "projection": copy.deepcopy(PROJECTION),
        "numeric_contract": (
            "Pinned native production exporter; "
            "cross-architecture identity requires separate verification."
        ),
        "selection_policy": (
            "All ten derived seeds are retained. "
            "Production candidate search is fixed for every seed. "
            "No visual seed filtering or rerolling."
        ),
    }
    plan["identity_sha256"] = _identity(plan)
    _validate_plan(plan)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        require(read(output / "plan.json") == plan, "Existing cohort has different inputs")
    else:
        require(
            all(path.name == "historical-seeds.json" for path in output.iterdir()),
            "Use a new cohort archive",
        )
        # Establish ownership before creating any derived inputs. An interrupted
        # preparation can then reconstruct missing files from this exact plan.
        write(output / "plan.json", plan)
    for name in ("tools", "configs", "logs", "orbits", "sources"):
        (output / name).mkdir(exist_ok=True)
    binary = output / generator["path"]
    fingerprint = {key: generator[key] for key in ("sha256", "bytes")}
    if not binary.exists():
        temporary = binary.with_name(f"{binary.name}.partial-{time.time_ns()}")
        shutil.copy2(exporter, temporary)
        require(artifact(temporary) == fingerprint, "Exporter changed during archival")
        temporary.replace(binary)
    checked(output, generator["path"], fingerprint)

    def ensure_input(name, value):
        path = output / name
        if path.exists():
            require(read(path) == value, f"Archived cohort input differs: {name}")
        else:
            write(path, value)

    ensure_input("historical-seeds.json", history)
    ensure_input("seeds.json", plan["seeds"])
    for seed in plan["seeds"]:
        ensure_input(f"configs/{seed}.json", {"seed": seed, **SETTINGS})
    return plan


def generate_cohort(output):
    from tools.estuary.source import Source

    output = Path(output).resolve()
    plan = _validate_plan(read(output / "plan.json"))
    generator = plan["generator"]
    binary = checked(
        output, generator["path"], {key: generator[key] for key in ("sha256", "bytes")}
    )
    with (output / ".generation.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "cohort.json").exists() and read(output / "cohort.json").get("complete"):
            manifest = read(output / "cohort.json")
            verify_cohort(manifest, source_root=output)
            return manifest
        started = time.monotonic()
        results, failures = {}, []

        def manifest():
            value = {
                "version": VERSION,
                "complete": False,
                "plan": plan,
                "seeds": plan["seeds"],
                "sources": [results[seed] for seed in plan["seeds"] if seed in results],
                "failures": failures,
                "seconds": time.monotonic() - started,
                "host": {"platform": platform.platform(), "machine": platform.machine()},
            }
            value["identity_sha256"] = _identity(value)
            write(output / "cohort.json", value)
            return value

        def export(seed):
            receipt_path = output / f"sources/{seed}.json"
            if receipt_path.exists():
                row = read(receipt_path)
                _validate_row(row, plan)
                checked(output, row["path"], {key: row[key] for key in ("sha256", "bytes")})
                return row
            destination, log = output / f"orbits/{seed}.orbit", output / f"logs/{seed}.log"
            require(
                not destination.exists(), "Uncertified orbit exists; preserved without overwriting"
            )
            config = output / f"configs/{seed}.json"
            require(read(config) == {"seed": seed, **SETTINGS}, "Source config changed")
            command = [
                str(binary),
                "--threads",
                "8",
                "orbit",
                "--config",
                str(config),
                "--output",
                str(destination),
            ]
            then = time.monotonic()
            print(json.dumps({"event": "source_started", "seed": seed}), flush=True)
            with log.open("w") as stream:
                subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=86400
                )
            require(
                artifact(binary)["sha256"] == generator["sha256"],
                "Exporter changed during generation",
            )
            source = Source.read(destination, **PROJECTION)
            row = {
                "seed": seed,
                "path": f"orbits/{seed}.orbit",
                **artifact(destination),
                "samples": source.samples,
                "samples_sha256": source.samples_sha256,
                "dt": source.dt,
                "source_metadata": source.metadata,
                "config": read(config),
                "config_artifact": {"path": f"configs/{seed}.json", **artifact(config)},
                "log_artifact": {"path": f"logs/{seed}.log", **artifact(log)},
                "returncode": 0,
                "seconds": time.monotonic() - then,
                "command": command,
            }
            _validate_row(row, plan)
            write(receipt_path, row)
            return row

        manifest()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(export, seed): seed for seed in plan["seeds"]}
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    results[seed] = future.result()
                    print(
                        json.dumps(
                            {
                                "event": "source_verified",
                                "seed": seed,
                                "sha256": results[seed]["sha256"],
                            }
                        ),
                        flush=True,
                    )
                except Exception as error:
                    failures.append({"seed": seed, "error": str(error)})
                    print(json.dumps({"event": "source_failed", **failures[-1]}), flush=True)
                manifest()
        completed = manifest()
        require(
            len(results) == COUNT and not failures,
            "Some source exports failed; preserve all logs and seed choices",
        )
        # Cached per-source receipts still require actual source/config/log
        # verification. Publish completion only after that verification succeeds.
        completed["complete"] = True
        completed["identity_sha256"] = _identity(completed)
        verify_cohort(completed, source_root=output)
        write(output / "cohort.json", completed)
        return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exporter", type=Path)
    parser.add_argument("--historical-seeds", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        rows = verify_cohort(read(args.output / "cohort.json"), source_root=args.output)
        print(json.dumps({"verified": True, "sources": len(rows)}))
        return
    if args.exporter is not None or args.historical_seeds is not None:
        require(
            args.exporter is not None and args.historical_seeds is not None,
            "Supply exporter and historical inventory together",
        )
        plan = prepare_cohort(args.output, args.exporter, args.historical_seeds)
        print(
            json.dumps(
                {
                    "prepared": True,
                    "seeds": plan["seeds"],
                    "plan_identity_sha256": plan["identity_sha256"],
                }
            ),
            flush=True,
        )
    if not args.prepare_only:
        result = generate_cohort(args.output)
        print(json.dumps({"complete": result["complete"], "sources": len(result["sources"])}))


if __name__ == "__main__":
    main()
