"""Replay an unchanged Estuary recording and authenticate real intermediate states.

The original solver, recipe, source, runtime and GPU identity must match. A ledger
is published only when the replayed final NPY bytes match the archived final state.
Failed intermediates remain available; a partial replay is never reused as proof.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import math
import platform
import signal
import sys
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.estuary.engine import Engine
from tools.estuary.run import (
    artifact,
    code_identity,
    digest,
    encoded,
    read_json,
    write_array,
    write_json,
)
from tools.estuary.source import Source
from tools.estuary_depth.prepare import history_records, require, verified_run


def canonical_steps(fractions, steps):
    require(1 <= len(fractions) <= 6, "Replay requires one to six historical fractions")
    result = []
    for fraction in fractions:
        require(
            type(fraction) in (int, float) and math.isfinite(fraction) and 0 < fraction < 1,
            "History fractions must be finite numbers inside (0, 1)",
        )
        index = round(fraction * steps)
        require(
            abs(fraction * steps - index) <= 1e-9 and 0 < index < steps,
            "History fraction must land exactly on a canonical source step",
        )
        result.append(index)
    require(all(a < b for a, b in pairwise(result)), "History fractions must increase strictly")
    return result


def replay(run_path, output_path, fractions=(0.35, 0.65)):
    folder, output = Path(run_path).resolve(strict=True), Path(output_path).resolve()
    request, identity, recipe, _state, records = verified_run(folder)
    require(
        code_identity() == request["code"], "Installed Estuary runtime differs from the original"
    )
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "glcontext": version("glcontext"),
    }
    require(runtime == request["runtime"], "Replay dependency runtime differs from the original")
    total = recipe["simulation"]["steps"]
    steps = canonical_steps(fractions, total)
    replay_request = {
        "schema_version": 1,
        "render_identity": identity,
        "steps": steps,
        "runtime": runtime,
        "hardware": request["hardware"],
        "code": request["code"],
        "history_code_sha256": digest(Path(__file__)),
        "original_request_sha256": digest(folder / "request.json"),
        "original_receipt_sha256": digest(folder / "receipt.json"),
    }
    replay_identity = hashlib.sha256(encoded(replay_request)).hexdigest()
    if output.exists():
        require(
            (output / "history.json").is_file(),
            "Existing replay is incomplete; intermediates preserved",
        )
        ledger = read_json(output / "history.json")
        require(
            ledger.get("complete") is True
            and ledger.get("replay_identity_sha256") == replay_identity,
            "Existing replay is incomplete or has a different identity",
        )
        history_records(output / "history.json", request, identity, records["final-state.npy"])
        require(
            [item["step"] for item in ledger["checkpoints"]] == steps, "Replay schedule differs"
        )
        return ledger
    output.mkdir(parents=True)
    with (output / ".replay.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(output / "replay-request.json", replay_request)
        write_json(output / "status.json", {"complete": False, "source_fraction": 0.0})
        source = Source.read(
            folder / "inputs/source.orbit",
            aspect=recipe["simulation"]["resolution"][0] / recipe["simulation"]["resolution"][1],
            **recipe["projection"],
        )
        engine = None
        try:
            engine = Engine(source, recipe, backend=request["backend"])
            require(engine.metadata == request["hardware"], "Replay GPU or driver identity differs")
            checkpoints = []
            for target in [*steps, total]:
                while engine.step < target:
                    engine.advance_to(min(target, engine.step + 120))
                    progress = {
                        "complete": False,
                        "source_fraction": engine.step / total,
                        "step": engine.step,
                        "internal_steps": engine.internal_steps,
                    }
                    write_json(output / "status.json", progress)
                    print(encoded(progress).decode().strip(), flush=True)
                path = output / (
                    "final-state.npy" if target == total else f"state-{target:06d}.npy"
                )
                write_array(path, engine.read_state())
                if target != total:
                    checkpoints.append(
                        {
                            "step": target,
                            "source_fraction": target / total,
                            "state": artifact(path, output),
                        }
                    )
            final_record = artifact(output / "final-state.npy", output)
            require(
                final_record["sha256"] == records["final-state.npy"]["sha256"],
                "Replay final-state.npy does not exactly match the original",
            )
            require(
                digest(folder / "final-state.npy") == final_record["sha256"],
                "Original state changed",
            )
            require(
                digest(source.path) == source.sha256 and code_identity() == request["code"],
                "Source or runtime code changed during replay",
            )
            require(
                digest(Path(__file__)) == replay_request["history_code_sha256"],
                "History tool changed during replay",
            )
            for name in ("request", "receipt"):
                require(
                    digest(folder / f"{name}.json") == replay_request[f"original_{name}_sha256"],
                    "Original render metadata changed during replay",
                )
            ledger = {
                "schema_version": 1,
                "complete": True,
                "identity_sha256": identity,
                "replay_identity_sha256": replay_identity,
                "source_sha256": source.sha256,
                "code": request["code"],
                "runtime": runtime,
                "hardware": engine.metadata,
                "final_state": final_record,
                "checkpoints": checkpoints,
                "verification": "Replayed final NPY file is byte-identical to the original",
                "diagnostics": {
                    "internal_steps": engine.internal_steps,
                    "maximum_courant": engine.maximum_courant,
                },
            }
            write_json(output / "history.json", ledger)
            write_json(
                output / "status.json",
                {
                    "complete": True,
                    "source_fraction": 1.0,
                    "replay_identity_sha256": replay_identity,
                },
            )
            return ledger
        except BaseException as error:
            write_json(
                output / "status.json",
                {"complete": False, "error": str(error), "replay_identity_sha256": replay_identity},
            )
            raise
        finally:
            if engine is not None:
                engine.close()


def interrupted(_signal, _frame):
    raise KeyboardInterrupt("Replay interrupted; intermediate states preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.35, 0.65])
    args = parser.parse_args()
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        print(encoded(replay(args.run, args.output, args.fractions)).decode())
    except KeyboardInterrupt as error:
        print(str(error), file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
