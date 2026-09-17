#!/usr/bin/env python3
"""Run a finite, resumable collection: one bounded CPU lane and one OptiX lane.

Every full-history still is photographed before the GPU lane starts the films.
The CPU lane independently replays the original rendering. Publication requires
comparison.py to verify both source histories, native media and complete decodes.
Restart with the identical command; changed inputs are rejected. Failed normal
renders are preserved before retry, never overwritten. No jobs run on import.
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

INPUTS = (
    "builder",
    "normal_renderer",
    "blender",
    "render_script",
    "film_script",
    "comparison_script",
    "ffmpeg",
    "ffprobe",
    "geometry_recipe",
    "studio_still",
    "studio_film",
)
SCRIPTS = {"render_script", "film_script", "comparison_script"}
RECIPES = {"geometry_recipe", "studio_still", "studio_film"}
NORMAL_FILES = {"normal.mp4", "normal-hq.mp4", "master.png", "full.webp", "preview.webp"}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read(path):
    value = json.loads(path.read_text())
    encoded(value)
    return value


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def preserve(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Archived input changed: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)


def quarantine(folder):
    """Keep the whole incomplete attempt intact; never merge old and new media."""
    if folder.exists():
        target = folder.with_name(f"{folder.name}.incomplete-{time.time_ns()}-{os.getpid()}")
        folder.rename(target)
        return target
    return None


def normal_complete(folder):
    path = folder / "normal.json"
    if not path.exists() or read(path).get("complete") is not True:
        return False
    record = read(path)
    artifacts = record.get("artifacts", [])
    if {item["path"] for item in artifacts} != NORMAL_FILES or len(artifacts) != len(NORMAL_FILES):
        raise ValueError("Completed normal render lacks its exact artifact manifest")
    for item in artifacts:
        artifact = folder / item["path"]
        if artifact.stat().st_size != item["bytes"] or digest(artifact) != item["sha256"]:
            raise ValueError(f"Completed normal artifact changed: {artifact}")
    return True


def run_process(command, log, cancel, env=None):
    if cancel.is_set():
        raise InterruptedError("Collection interrupted")
    with log.open("ab") as stream:
        stream.write(("\nCommand: " + json.dumps([str(x) for x in command]) + "\n").encode())
        stream.flush()
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        try:
            while True:
                try:
                    code = child.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    if cancel.is_set():
                        raise InterruptedError("Collection interrupted") from None
        except BaseException:
            # A dedicated group includes encoders and Blender grandchildren.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
    if code:
        raise RuntimeError(f"Child exited with status {code}; see {log}")


class Collection:
    def __init__(self, args):
        self.args, self.output = args, args.output.resolve()
        self.cancel, self.mutex = threading.Event(), threading.Lock()
        self.paths, self.inputs, self.archived = {}, {}, {}
        for name in INPUTS:
            path = getattr(args, name).resolve(strict=True)
            if not path.is_file() or (
                name not in SCRIPTS | RECIPES and not os.access(path, os.X_OK)
            ):
                raise ValueError(f"Invalid explicit input: {name}")
            self.paths[name] = path
        self.paths["runner"] = Path(__file__).resolve()
        for seed in args.seeds:
            for kind, filename in (("orbit", f"{seed}.orbit"), ("record", f"source-{seed}.json")):
                self.paths[f"{seed}/{kind}"] = (args.sources / filename).resolve(strict=True)
        # Limit only our normal subprocess and its native encoders, not other jobs.
        if not hasattr(os, "sched_getaffinity") or not shutil.which("taskset"):
            raise ValueError("This bounded batch requires Linux CPU affinity and taskset")
        self.cpus = sorted(os.sched_getaffinity(0))[:12]
        if not self.cpus:
            raise ValueError("No CPUs available to the normal rendering lane")
        self.paths["taskset"] = Path(shutil.which("taskset")).resolve(strict=True)
        self.inputs = {
            name: {"path": str(path), "sha256": digest(path)} for name, path in self.paths.items()
        }
        self.request = {
            "schema_version": 1,
            "inputs": self.inputs,
            "seeds": args.seeds,
            "execution": {
                "normal_cpus": self.cpus,
                "normal_threads": len(self.cpus),
                "gpu_workers": 1,
                "gpu_backend": "OPTIX",
                "device_ids": sorted(args.device_id),
            },
        }
        self.state = {
            "schema_version": 1,
            "complete": False,
            "execution": self.request["execution"],
            "seeds": {seed: {} for seed in args.seeds},
        }

    def check_inputs(self):
        for name, info in {**self.inputs, **self.archived}.items():
            if digest(Path(info["path"])) != info["sha256"]:
                raise ValueError(f"Collection input changed: {name}")

    def status(self, seed=None, phase=None, **values):
        with self.mutex:
            target = self.state if seed is None else self.state["seeds"][seed].setdefault(phase, {})
            target.update(values, updated_unix=time.time())
            write(self.output / "status.json", self.state)

    def command(self, command, log):
        self.check_inputs()
        env = os.environ.copy()
        env["PATH"] = str(self.paths["ffmpeg"].parent) + os.pathsep + env.get("PATH", "")
        env["OMP_NUM_THREADS"] = "4"
        run_process([str(x) for x in command], self.output / "logs" / log, self.cancel, env)
        self.check_inputs()

    def attempt(self, seed, phase, work):
        if self.cancel.is_set():
            return False
        self.status(seed, phase, status="running", error=None)
        try:
            work()
            self.status(seed, phase, status="complete")
            return True
        except Exception as error:
            self.status(seed, phase, status="failed", error=str(error))
            return False

    def prepare(self):
        request = self.output / "request.json"
        if not request.exists() and any(
            p.name != ".collection.lock" for p in self.output.iterdir()
        ):
            raise ValueError("Existing collection has no archived request; use a new output")
        preserve(request, encoded(self.request))
        for name in ("inputs", "logs", "gallery"):
            (self.output / name).mkdir(exist_ok=True)
        for name in SCRIPTS | RECIPES:
            source = self.paths[name]
            target = self.output / "inputs" / (name + source.suffix)
            preserve(target, source.read_bytes())
            if digest(target) != self.inputs[name]["sha256"]:
                raise ValueError(f"Input changed during archival: {name}")
            self.paths[name] = target
            self.archived["archive/" + name] = {"path": str(target), "sha256": digest(target)}
        if read(self.paths["geometry_recipe"]).get("source_fraction") != 1.0:
            raise ValueError("Final shell geometry must include the entire source fraction 1")
        for name in ("studio_still", "studio_film"):
            render = read(self.paths[name])["render"]
            if render["resolution"] != [3840, 3200] or render["threads"] != 4:
                raise ValueError(
                    "Both studios require native 3840x3200 resolution and four threads"
                )
        if read(self.paths["studio_still"])["render"]["samples"] != 256:
            raise ValueError("The full-resolution still requires 256 samples")
        self.status(status="running")
        self.command(
            [
                sys.executable,
                self.paths["comparison_script"],
                "init",
                "--collection",
                self.output / "gallery",
                "--seeds",
                *self.args.seeds,
            ],
            "init.log",
        )
        for seed in self.args.seeds:
            folder = self.output / seed
            folder.mkdir(exist_ok=True)
            self.command(
                [
                    sys.executable,
                    self.paths["comparison_script"],
                    "plan",
                    "--output",
                    folder / "plan",
                    "--samples",
                    "1000000",
                ],
                f"{seed}-plan.log",
            )
            plan = read(folder / "plan" / "alignment.json")
            if len(plan["frames"]) != 901 or plan["fps"] != 30:
                raise ValueError("Expected the complete 901-frame, 30 fps comparison plan")

    def normal(self, seed):
        folder = self.output / seed / "normal"
        if normal_complete(folder):
            self.publish_stills(seed)
            return
        quarantine(folder)
        self.command(
            [
                self.paths["taskset"],
                "-c",
                ",".join(map(str, self.cpus)),
                self.paths["normal_renderer"],
                "--threads",
                len(self.cpus),
                "normal",
                "--orbit",
                self.paths[f"{seed}/orbit"],
                "--generation-record",
                self.paths[f"{seed}/record"],
                "--width",
                "3840",
                "--height",
                "2484",
                "--output",
                folder,
            ],
            f"{seed}-normal.log",
        )
        if not normal_complete(folder):
            raise ValueError("Normal renderer exited without a verified complete archive")
        self.publish_stills(seed)

    def still(self, seed):
        folder = self.output / seed
        geometry, still = folder / "geometry", folder / "still"
        if geometry.exists() and not (geometry / "build.json").exists():
            quarantine(geometry)
        self.command(
            [
                self.paths["builder"],
                "--threads",
                "4",
                "build",
                "--orbit",
                self.paths[f"{seed}/orbit"],
                "--config",
                self.paths["geometry_recipe"],
                "--output",
                geometry,
                *(["--resume"] if geometry.exists() else []),
            ],
            f"{seed}-geometry.log",
        )
        if still.exists() and not (still / "request.json").exists():
            quarantine(still)
        command = [
            self.paths["blender"],
            "--factory-startup",
            "--background",
            "--threads",
            "4",
            "--python-exit-code",
            "1",
            "--python",
            self.paths["render_script"],
            "--",
            "--mesh",
            geometry / "mesh.ply",
            "--recipe",
            self.paths["studio_still"],
            "--output",
            still,
            "--view",
            "front",
            "--device",
            "OPTIX",
        ]
        for device_id in self.args.device_id:
            command += ["--device-id", device_id]
        self.command(command + (["--resume"] if still.exists() else []), f"{seed}-still.log")
        if read(still / "receipt.json").get("complete") is not True:
            raise ValueError("Still renderer exited without a complete receipt")
        self.publish_stills(seed)

    def film(self, seed):
        folder, film = self.output / seed, self.output / seed / "film"
        if film.exists() and not (film / "request.json").exists():
            quarantine(film)
        command = [
            sys.executable,
            self.paths["film_script"],
            "--builder",
            self.paths["builder"],
            "--blender",
            self.paths["blender"],
            "--render-script",
            self.paths["render_script"],
            "--ffmpeg",
            self.paths["ffmpeg"],
            "--orbit",
            self.paths[f"{seed}/orbit"],
            "--geometry-recipe",
            self.paths["geometry_recipe"],
            "--studio-recipe",
            self.paths["studio_film"],
            "--output",
            film,
            "--source-times",
            folder / "plan" / "source-times.json",
            "--frames",
            "901",
            "--fps",
            "30",
            "--workers",
            "1",
            "--device",
            "OPTIX",
            "--start-hold",
            "0",
            "--end-hold",
            "0",
        ]
        for device_id in self.args.device_id:
            command += ["--device-id", device_id]
        self.command(command + (["--resume"] if film.exists() else []), f"{seed}-film.log")
        if read(film / "film.json").get("complete") is not True:
            raise ValueError("Film runner exited without a complete receipt")

    def lane(self, gpu):
        if gpu:
            ready = [
                seed
                for seed in self.args.seeds
                if self.attempt(seed, "still", lambda s=seed: self.still(s))
            ]
            for seed in ready:
                self.attempt(seed, "film", lambda s=seed: self.film(s))
        else:
            for seed in self.args.seeds:
                self.attempt(seed, "normal", lambda s=seed: self.normal(s))

    def publish_stills(self, seed):
        folder = self.output / seed
        receipt = folder / "still" / "receipt.json"
        if not receipt.exists() or read(receipt).get("complete") is not True:
            return
        self.command(
            [
                sys.executable,
                self.paths["comparison_script"],
                "publish-stills",
                "--collection",
                self.output / "gallery",
                "--still",
                folder / "still",
                "--geometry",
                folder / "geometry",
                "--orbit",
                self.paths[f"{seed}/orbit"],
                "--normal",
                folder / "normal",
            ],
            f"{seed}-publish-stills-{threading.current_thread().name}.log",
        )

    def publish(self, seed):
        folder = self.output / seed
        self.command(
            [
                sys.executable,
                self.paths["comparison_script"],
                "publish",
                "--collection",
                self.output / "gallery",
                "--normal",
                folder / "normal",
                "--shell",
                folder / "film",
                "--still",
                folder / "still",
                "--orbit",
                self.paths[f"{seed}/orbit"],
                "--ffmpeg",
                self.paths["ffmpeg"],
                "--ffprobe",
                self.paths["ffprobe"],
            ],
            f"{seed}-publish.log",
        )
        if read(self.output / "gallery" / seed / "comparison.json").get("complete") is not True:
            raise ValueError("Publisher exited without a verified complete comparison")

    def publish_ready(self):
        # The coordinator is the only final publisher; neither render lane waits
        # for the other lane's later seeds or launches competing video encoders.
        with self.mutex:
            ready = [
                seed
                for seed, phases in self.state["seeds"].items()
                if "publish" not in phases
                and all(
                    phases.get(name, {}).get("status") == "complete"
                    for name in ("normal", "still", "film")
                )
            ]
        for seed in ready:
            self.attempt(seed, "publish", lambda s=seed: self.publish(s))

    def run(self):
        self.prepare()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            pending = {pool.submit(self.lane, gpu) for gpu in (False, True)}
            while pending:
                done, pending = concurrent.futures.wait(pending, timeout=1)
                for future in done:
                    future.result()
                self.publish_ready()
            complete = all(
                phases.get("publish", {}).get("status") == "complete"
                for phases in self.state["seeds"].values()
            )
            self.status(status="complete" if complete else "failed", complete=complete)
            if not complete:
                raise RuntimeError("Collection has failed seeds; inspect status.json and logs")
        except BaseException as error:
            self.cancel.set()
            self.status(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error),
            )
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUTS, "sources", "output"):
        result.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    result.add_argument("--seeds", nargs="+", required=True)
    result.add_argument("--device-id", action="append", default=[])
    return result


def main():
    args = parser().parse_args()
    if (
        not 1 <= len(args.seeds) <= 12
        or len(set(args.seeds)) != len(args.seeds)
        or any(not re.fullmatch(r"0x[0-9a-fA-F]{2,64}", seed) for seed in args.seeds)
    ):
        raise ValueError("Use one through twelve distinct hexadecimal seeds")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".collection.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        Collection(args).run()


if __name__ == "__main__":

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Collection failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
