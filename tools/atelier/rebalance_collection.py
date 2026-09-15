#!/usr/bin/env python3
"""Finite, opt-in CPU rebalancing for the four pinned b7 collection renders.

Default/--dry-run only validates and prints a plan. --apply runs until all four
render sets finish or the persisted deadline expires (at most 720 hours).
No app scheduler is involved. Encoders/assemblers are never stopped. The private
journal contains the original driver's environment for crash-safe exact restarts;
its directory and JSON files are owner-only, and environments are never logged.

After Aurora: Light32/Engraving32/Eclipse64. Remaining pairs receive L64/E64,
L80/G48, or G48/E80; a sole remaining renderer receives128. Before Aurora finishes,
all existing allocations are preserved. Budgets count render workers, not encoders.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import uuid

BASE = Path("/home/user/tidal-silk")
BOOT_ID = "38efcd5e-05f2-4f57-bb9b-b592804c829c"
PYTHON = Path("/usr/bin/python3.12")
ORBIT_HASH = "a9864c974e31dce21aa80ff9edd6524b9d4f9141c5cf50de26b6fdc739cc1652"
SCRIPT_HASHES = {
    "render_study.py": "54766bbd329264eb5b2ff649075a3cfbfbf2141af964c3233b1fb00ee72fe0bf",
    "finish_study.py": "b5b466b559beea010687d164494a4af70fd4d7a944e7a4dbcc4112e6a3608010",
    "verify_film.py": "6b45ede0f76eec533984d9dc4b3129209cb2b3defd93b9ef05d99bc95e99b211",
}
PYTHON_HASH = "e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f"
MAX_SECONDS = 720 * 3600


class UnsafeJob(RuntimeError):
    """An ownership, immutable-input, or process-state check failed."""


class Deferred(RuntimeError):
    """A finishing or changing process should be left alone this cycle."""


@dataclass(frozen=True)
class Spec:
    slug: str
    version: int
    initial_pid: int
    initial_start: int
    initial_workers: int
    binary_hash: str


SPECS = (
    Spec("03-aurora", 11, 109919, 5642594, 112, "2f707da3d6da01d564ab763fd17cb0cd2bfc7d4996eeee19372986290c9a847d"),
    Spec("04-light", 15, 126535, 6158625, 32, "0bb2ad19e86bb38ee01804c1d1e89bb6d9516b677f9071060eb7ffd69e119439"),
    Spec("05-engraving", 17, 164853, 7236089, 48, "34377c3aaad89823e200f9e081acfe7bf2e30c78bc73a2116a35aaf407b165d2"),
    Spec("06-eclipse", 22, 172585, 7367852, 64, "3f8756738359380d4c9467d8a083482d335488be10e72acb8e54232abf5972ed"),
)


@dataclass(frozen=True)
class Paths:
    base: Path = BASE
    python: Path = PYTHON

    @property
    def series(self):
        return self.base / "six-studies-b7"

    @property
    def scripts(self):
        return self.base / "source/tools/atelier"

    @property
    def orbit(self):
        return self.base / "results/0xb7f327f9f722.orbit"

    @property
    def private(self):
        return self.series / ".rebalance-collection"

    def output(self, spec):
        return self.series / spec.slug

    def config(self, spec):
        return self.series / "configs" / (spec.slug + ".json")

    def binary(self, spec):
        return self.base / "bin" / f"atelier-v{spec.version}"

    def log(self, spec):
        return self.series / "logs" / f"{spec.slug.split('-', 1)[1]}-final-v{spec.version}.log"


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    start: int
    uid: int
    exe: str
    device: int
    inode: int
    cwd: str
    argv: tuple[str, ...]
    state: str

    def identity(self):
        value = asdict(self)
        value.pop("state")
        value["argv"] = list(self.argv)
        return value


@dataclass
class Pin:
    process: Proc
    fd: int


class Processes:
    """Linux process access. Signals use pidfds, never numerical PID kills."""

    def __init__(self):
        if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise UnsafeJob("Linux pidfd support is required; refusing PID-only signals")
        self.boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        self.uid = os.getuid()

    @staticmethod
    def _stat(pid):
        text = (Path("/proc") / str(pid) / "stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        return fields[0], int(fields[1]), int(fields[19])

    def read(self, pid):
        root = Path("/proc") / str(pid)
        try:
            state, ppid, start = self._stat(pid)
            if state == "Z":
                return None
            uid = root.stat().st_uid
            argv = tuple(os.fsdecode(value) for value in (root / "cmdline").read_bytes().split(b"\0") if value)
            exe = os.readlink(root / "exe")
            image = (root / "exe").stat()
            cwd = os.readlink(root / "cwd")
            if self._stat(pid)[2] != start:
                raise UnsafeJob(f"PID {pid} changed while being inspected")
            return Proc(pid, ppid, start, uid, exe, image.st_dev, image.st_ino, cwd, argv, state)
        except (FileNotFoundError, ProcessLookupError):
            return None

    def all(self):
        result = []
        for directory in Path("/proc").iterdir():
            if not directory.name.isdecimal():
                continue
            try:
                # Keep a basic record for foreign-UID children so ownership
                # validation cannot silently omit an unexpected descendant.
                uid = directory.stat().st_uid
                try:
                    value = self.read(int(directory.name)) if uid == self.uid else None
                except PermissionError:
                    value = None
                if value is None:
                    state, ppid, start = self._stat(int(directory.name))
                    if state == "Z":
                        continue
                    try:
                        argv = tuple(os.fsdecode(part) for part in (directory / "cmdline").read_bytes().split(b"\0") if part)
                    except PermissionError:
                        argv = ()
                    value = Proc(int(directory.name), ppid, start, uid, "", 0, 0, "", argv, state)
                if value is not None:
                    result.append(value)
            except (FileNotFoundError, ProcessLookupError):
                continue
        return result

    def pin(self, expected):
        try:
            fd = os.pidfd_open(expected.pid, 0)
        except ProcessLookupError:
            return None
        try:
            current = self.read(expected.pid)
            if current is None:
                os.close(fd)
                return None
            same_process(current, expected.identity())
            return Pin(current, fd)
        except BaseException:
            os.close(fd)
            raise

    def signal(self, pin, signum):
        current = self.read(pin.process.pid)
        if current is None:
            return
        same_process(current, pin.process.identity(), check_parent=False)
        try:
            signal.pidfd_send_signal(pin.fd, signum)
        except ProcessLookupError:
            pass

    @staticmethod
    def close(pin):
        os.close(pin.fd)

    @staticmethod
    def exited(pin):
        poller = select.poll()
        poller.register(pin.fd, select.POLLIN)
        return bool(poller.poll(0))

    def stopped(self, pin):
        if self.exited(pin):
            return False
        tasks = Path("/proc") / str(pin.process.pid) / "task"
        try:
            return all(self._stat(int(path.name))[0] in {"T", "t", "Z"} for path in tasks.iterdir())
        except FileNotFoundError:
            return False

    def wait(self, pins, timeout, stopped=False):
        deadline = time.monotonic() + timeout
        while True:
            ready = self.stopped(pins[0]) if stopped else all(self.exited(pin) for pin in pins)
            if ready:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def environment(process):
        values = (Path("/proc") / str(process.pid) / "environ").read_bytes().split(b"\0")
        return {os.fsdecode(key): os.fsdecode(value) for key, value in (item.split(b"=", 1) for item in values if item)}

    @staticmethod
    def output_path(process, descriptor):
        return os.readlink(Path("/proc") / str(process.pid) / "fd" / str(descriptor))

    def spawn(self, command, cwd, environment, log, python):
        # Explicit executable prevents PATH substitution while preserving argv[0].
        descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "ab", buffering=0) as stream:
            child = subprocess.Popen(command, executable=str(python), cwd=cwd, env=environment,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        process = self.read(child.pid)
        if process is None:
            raise UnsafeJob("Restarted driver exited before its identity could be recorded")
        return process


def same_process(current, identity, check_parent=True):
    keys = ("pid", "start", "uid", "exe", "device", "inode", "cwd")
    if any(getattr(current, key) != identity[key] for key in keys) or list(current.argv) != identity["argv"]:
        raise UnsafeJob(f"PID reuse or changed process identity at PID {identity['pid']}")
    if check_parent and current.ppid != identity["ppid"]:
        raise UnsafeJob(f"PID {current.pid} has an unexpected parent")


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def regular(path):
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise UnsafeJob(f"Expected a fixed regular file without path redirection: {path}")


def read_json(path):
    regular(path)
    return json.loads(path.read_text())


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def options(command, offset, allowed):
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise UnsafeJob("Command must be a string argv array")
    if (len(command) - offset) % 2:
        raise UnsafeJob("Command arguments must be unambiguous option/value pairs")
    result = {}
    for index in range(offset, len(command), 2):
        key, value = command[index:index + 2]
        if key not in allowed or key in result or not value:
            raise UnsafeJob(f"Unknown, duplicate, or empty command option: {key}")
        result[key] = value
    return result


def integer(value, name):
    if not isinstance(value, str) or not value.isdecimal() or str(int(value)) != value:
        raise UnsafeJob(f"Invalid canonical integer for {name}")
    return int(value)


@dataclass
class Job:
    spec: Spec
    record: dict
    command: list[str]
    workers: int
    recipe: dict


class Collection:
    def __init__(self, paths=Paths(), specs=SPECS, process=None, known_hashes=None, expected_boot=BOOT_ID):
        self.paths = paths
        self.specs = {spec.slug: spec for spec in specs}
        self.proc = process if process is not None else Processes()
        self.expected_boot = expected_boot
        self.known_hashes = known_hashes if known_hashes is not None else {
            **{str(paths.scripts / name): value for name, value in SCRIPT_HASHES.items()},
            str(paths.python): PYTHON_HASH, str(paths.orbit): ORBIT_HASH,
            **{str(paths.binary(spec)): spec.binary_hash for spec in specs},
        }

    def immutable_pins(self):
        pins = dict(self.known_hashes)
        for spec in self.specs.values():
            for path in [self.paths.config(spec), self.paths.output(spec) / "render-request.json"]:
                regular(path)
                pins[str(path)] = digest(path)
        self.verify_pins(pins)
        return pins

    @staticmethod
    def verify_pins(pins):
        for name, expected in pins.items():
            path = Path(name)
            regular(path)
            if digest(path) != expected:
                raise UnsafeJob(f"Pinned input changed: {path}")

    def parse_driver(self, spec, command):
        if not isinstance(command, list) or len(command) < 2 or command[:2] != ["python3", str(self.paths.scripts / "render_study.py")]:
            raise UnsafeJob("Driver command does not use the fixed Python/script pair")
        allowed = {"--orbit", "--config", "--output", "--executable", "--workers", "--chunks", "--parallel-ranges", "--poster-frame", "--encoder-threads"}
        value = options(command, 2, allowed)
        expected = {"--orbit": str(self.paths.orbit), "--config": str(self.paths.config(spec)),
                    "--output": str(self.paths.output(spec)), "--executable": str(self.paths.binary(spec)),
                    "--chunks": "16", "--parallel-ranges": "4", "--poster-frame": "900"}
        if any(value.get(key) != required for key, required in expected.items()):
            raise UnsafeJob(f"Driver paths or fixed range settings changed for {spec.slug}")
        if value.get("--encoder-threads", "16") != "16":
            raise UnsafeJob("Encoder settings cannot change during rebalancing")
        workers = integer(value.get("--workers"), "workers")
        if workers < 4 or workers > 128 or workers % 4:
            raise UnsafeJob("Render worker budget must be a multiple of four, at most128")
        return workers

    def job(self, spec):
        root = self.paths.output(spec)
        if root.resolve() != root or not root.is_dir():
            raise UnsafeJob(f"Unexpected study directory: {root}")
        record = read_json(root / "final-job.json")
        if not isinstance(record, dict) or type(record.get("pid")) is not int or record["pid"] <= 1:
            raise UnsafeJob("Invalid recorded driver PID")
        command = record.get("command")
        workers = self.parse_driver(spec, command)
        recipe = read_json(self.paths.config(spec))
        if recipe != read_json(root / "render-request.json") or recipe.get("frames") != 1802 or recipe.get("kind") != spec.slug.split("-", 1)[1]:
            raise UnsafeJob("Recipe/request no longer identify the fixed full film")
        return Job(spec, record, command, workers, recipe)

    def complete(self, job):
        done = True
        for index in range(16):
            path = self.paths.output(job.spec) / f"final-chunk-{index}" / "render.json"
            if not path.exists():
                done = False
                continue
            try:
                manifest = read_json(path)
            except json.JSONDecodeError:
                done = False
                continue
            expected = list(range(1802 * index // 16, 1802 * (index + 1) // 16))
            if (manifest.get("config") != job.recipe or manifest.get("seed") != "0xb7f327f9f722"
                    or manifest.get("executable_sha256") != job.spec.binary_hash
                    or manifest.get("orbit_sha256") != self.known_hashes[str(self.paths.orbit)]
                    or manifest.get("rendered_frames") != expected
                    or type(manifest.get("complete")) is not bool):
                raise UnsafeJob(f"Chunk provenance/range mismatch: {path}")
            done &= manifest["complete"]
        return done

    def driver(self, job, identity=None):
        process = self.proc.read(job.record["pid"])
        if process is None:
            raise UnsafeJob(f"Driver disappeared before rendering completed: {job.spec.slug}")
        self.driver_process(job, process, job.command)
        if identity is not None:
            same_process(process, identity, check_parent=False)
        return process

    def driver_process(self, job, process, command):
        image = self.paths.python.stat()
        if (process.uid != self.proc.uid or process.exe != str(self.paths.python)
                or (process.device, process.inode) != (image.st_dev, image.st_ino)
                or list(process.argv) != command or process.cwd != str(self.paths.base.parent)):
            raise UnsafeJob(f"Driver identity/argv/cwd mismatch at PID {process.pid}")
        for fd in (1, 2):
            if self.proc.output_path(process, fd) != str(self.paths.log(job.spec)):
                raise UnsafeJob("Driver output no longer points to its fixed append log")

    def renderer_command(self, job, index):
        return [str(self.paths.binary(job.spec)), "--threads", str(job.workers // 4), "render",
                "--orbit", str(self.paths.orbit), "--config", str(self.paths.output(job.spec) / "render-request.json"),
                "--output", str(self.paths.output(job.spec) / f"final-chunk-{index}"),
                "--start", str(1802 * index // 16), "--end", str(1802 * (index + 1) // 16)]

    def renderer(self, job, process, parent=None):
        if parent is not None and process.ppid != parent:
            raise UnsafeJob("Renderer is not a direct child of the verified driver")
        image = self.paths.binary(job.spec).stat()
        if (process.uid != self.proc.uid or process.exe != str(self.paths.binary(job.spec))
                or (process.device, process.inode) != (image.st_dev, image.st_ino)
                or process.cwd != str(self.paths.base.parent)):
            raise UnsafeJob(f"Renderer executable/cwd mismatch at PID {process.pid}")
        matches = [index for index in range(16) if list(process.argv) == self.renderer_command(job, index)]
        if len(matches) != 1:
            raise UnsafeJob(f"Unexpected renderer arguments at PID {process.pid}")
        return matches[0]

    def finishing_child(self, job, process):
        prefix = [str(self.paths.python), str(self.paths.scripts / "finish_study.py"),
                  "--output", str(self.paths.output(job.spec)), "--executable", str(self.paths.binary(job.spec)),
                  "--poster-frame", "900", "--encoder-threads", "16"]
        expected = prefix + [part for index in range(16) for part in ("--input", str(self.paths.output(job.spec) / f"final-chunk-{index}"))]
        return (process.exe == str(self.paths.python) and bool(process.argv)
                and process.argv[0] in {str(self.paths.python), str(self.paths.python.with_name("python3"))}
                and list(process.argv[1:]) == expected[1:])

    def writers(self, job, processes):
        root = str(self.paths.output(job.spec)) + "/final-chunk-"
        result = []
        for process in processes:
            executable = str(self.paths.binary(job.spec))
            if (process.exe != executable and (not process.argv or process.argv[0] != executable)) or "render" not in process.argv:
                continue
            outputs = [process.argv[i + 1] for i, value in enumerate(process.argv[:-1]) if value == "--output"]
            if any(os.path.abspath(os.path.join(process.cwd, value)).startswith(root) for value in outputs):
                result.append(process)
        return result

    def children(self, job, driver):
        processes = self.proc.all()
        direct = [value for value in processes if value.ppid == driver.pid]
        if any(self.finishing_child(job, value) for value in direct):
            raise Deferred(f"{job.spec.slug} entered finishing; leave its driver alone")
        indices = [self.renderer(job, value, driver.pid) for value in direct]
        if len(indices) != len(set(indices)) or len(indices) > 4:
            raise UnsafeJob("Duplicate or excess renderer children")
        children = {value.pid for value in direct}
        if any(value.ppid in children for value in processes):
            raise UnsafeJob("Renderer has unexpected process descendants")
        if {value.pid for value in self.writers(job, processes)} != children:
            raise UnsafeJob("A renderer outside the owned driver is writing these chunks")
        return direct


def allocations(complete, current):
    alive = set(current) - complete
    if "03-aurora" not in complete:
        return {name: current[name] for name in alive}
    if len(alive) == 3:
        return {"04-light": 32, "05-engraving": 32, "06-eclipse": 64}
    if len(alive) == 1:
        return {next(iter(alive)): 128}
    pairs = {frozenset(("04-light", "06-eclipse")): {"04-light": 64, "06-eclipse": 64},
             frozenset(("04-light", "05-engraving")): {"04-light": 80, "05-engraving": 48},
             frozenset(("05-engraving", "06-eclipse")): {"05-engraving": 48, "06-eclipse": 80}}
    return pairs.get(frozenset(alive), {})


def changed_workers(command, workers):
    result = list(command)
    result[result.index("--workers") + 1] = str(workers)
    return result


class Controller:
    def __init__(self, collection, state, grace=20.0):
        self.collection = collection
        self.state = state
        self.grace = grace
        self.state_path = collection.paths.private / "state.json"
        remaining = state["deadline_unix"] - time.time()
        self.deadline = time.monotonic() + max(0.0, remaining)

    def remaining(self, limit=None):
        value = min(self.deadline - time.monotonic(), self.state["deadline_unix"] - time.time())
        if value <= 0:
            raise TimeoutError("Rebalancer's persisted deadline expired")
        return value if limit is None else min(value, limit)

    def save(self):
        atomic_json(self.state_path, self.state)

    def event(self, kind, **details):
        # No environment values or entire private transactions enter logs.
        value = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **details}
        data = (json.dumps(value, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(self.collection.paths.private / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        try:
            if os.write(descriptor, data) != len(data):
                raise OSError("Incomplete atomic event append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        print(json.dumps(value), flush=True)

    def observe(self):
        jobs, complete = {}, set()
        for name, spec in self.collection.specs.items():
            job = self.collection.job(spec)
            expected = self.state["jobs"][name]
            if job.record["pid"] != expected["pid"] or job.command != expected["command"]:
                raise UnsafeJob(f"Recorded job changed outside this controller: {name}")
            if self.collection.complete(job):
                complete.add(name)
            else:
                self.collection.driver(job, expected["identity"])
            jobs[name] = job
        return jobs, complete

    def plan(self):
        jobs, complete = self.observe()
        desired = allocations(complete, {name: job.workers for name, job in jobs.items()})
        return jobs, complete, desired

    def clear_transaction(self, event, **details):
        self.state["transaction"] = None
        self.save()
        self.event(event, **details)

    def begin(self, job, workers):
        if workers == job.workers:
            return
        self.remaining()
        self.collection.verify_pins(self.state["pins"])
        driver = self.collection.driver(job, self.state["jobs"][job.spec.slug]["identity"])
        children = self.collection.children(job, driver)  # Capture/validate before any signal.
        transaction = {"slug": job.spec.slug, "stage": "prepared", "old": driver.identity(),
                       "children": [value.identity() for value in children], "old_command": job.command,
                       "new_command": changed_workers(job.command, workers), "target_workers": workers,
                       "record": job.record, "environment": self.collection.proc.environment(driver)}
        self.state["transaction"] = transaction
        self.save()  # Durable recovery evidence precedes SIGSTOP.
        self.finish_transaction()

    def adopt(self, job, process, transaction):
        self.collection.driver_process(job, process, transaction["new_command"])
        updated = dict(transaction["record"])
        updated.update(pid=process.pid, command=transaction["new_command"], previous_pid=transaction["old"]["pid"],
                       process_identity={"boot_id": self.state["boot_id"], **process.identity()},
                       started_utc=datetime.now(timezone.utc).isoformat(), workers=transaction["target_workers"])
        atomic_json(self.collection.paths.output(job.spec) / "final-job.json", updated)
        self.state["jobs"][job.spec.slug] = {"pid": process.pid, "command": transaction["new_command"], "identity": process.identity()}
        self.state["transaction"] = None
        self.save()
        self.event("restarted", slug=job.spec.slug, previous_pid=transaction["old"]["pid"], pid=process.pid,
                   old_workers=job.workers, workers=transaction["target_workers"])

    def recover(self):
        transaction = self.state.get("transaction")
        if transaction is None:
            return
        self.remaining()
        self.collection.verify_pins(self.state["pins"])
        spec = self.collection.specs[transaction["slug"]]
        job = Job(spec, transaction["record"], transaction["old_command"],
                  self.collection.parse_driver(spec, transaction["old_command"]), read_json(self.collection.paths.config(spec)))
        self.collection.parse_driver(spec, transaction["new_command"])
        if (transaction["new_command"] != changed_workers(transaction["old_command"], transaction["target_workers"])
                or transaction["old"]["argv"] != transaction["old_command"]
                or transaction["record"]["command"] != transaction["old_command"]
                or transaction["record"]["pid"] != transaction["old"]["pid"]):
            raise UnsafeJob("Pending restart changes more than the worker allocation")
        current = read_json(self.collection.paths.output(spec) / "final-job.json")
        if current["command"] not in (transaction["old_command"], transaction["new_command"]):
            raise UnsafeJob("Pending restart found an unrelated job record")
        candidates = [value for value in self.collection.proc.all() if list(value.argv) == transaction["new_command"]]
        if candidates:
            if len(candidates) != 1 or self.collection.proc.read(transaction["old"]["pid"]) is not None:
                raise UnsafeJob("Ambiguous drivers during restart recovery")
            process = candidates[0]
            new_job = Job(spec, {**current, "pid": process.pid}, transaction["new_command"], transaction["target_workers"], job.recipe)
            try:
                self.collection.children(new_job, process)
            except Deferred:
                if not self.collection.complete(new_job):
                    raise
            self.adopt(job, process, transaction)
            return
        if transaction["stage"] in {"prepared", "frozen"}:
            old = self.collection.proc.read(transaction["old"]["pid"])
            if old is not None:
                same_process(old, transaction["old"], check_parent=False)
                self.collection.driver_process(job, old, transaction["old_command"])
                pin = self.collection.proc.pin(old)
                if pin is not None:
                    try:
                        if old.state in {"T", "t"}:
                            self.collection.proc.signal(pin, signal.SIGCONT)
                    finally:
                        self.collection.proc.close(pin)
                self.clear_transaction("recovered_uncommitted_pause", slug=spec.slug)
                return
        self.finish_transaction()

    def finish_transaction(self):
        transaction = self.state["transaction"]
        spec = self.collection.specs[transaction["slug"]]
        job = Job(spec, transaction["record"], transaction["old_command"],
                  self.collection.parse_driver(spec, transaction["old_command"]), read_json(self.collection.paths.config(spec)))
        pins = []
        driver_pin = None
        committed = transaction["stage"] in {"terminating", "launching", "launched"}
        try:
            old = self.collection.proc.read(transaction["old"]["pid"])
            if old is not None:
                same_process(old, transaction["old"], check_parent=False)
                self.collection.driver_process(job, old, transaction["old_command"])
                driver_pin = self.collection.proc.pin(old)
                if driver_pin is None:
                    raise UnsafeJob("Driver exited during restart preparation")
                pins.append(driver_pin)
                self.collection.proc.signal(driver_pin, signal.SIGSTOP)
                if not self.collection.proc.wait([driver_pin], self.remaining(self.grace), stopped=True):
                    raise UnsafeJob("Driver did not reach a complete thread-group stop")
                frozen = self.collection.proc.read(old.pid)
                if frozen is None:
                    raise UnsafeJob("Frozen driver disappeared")
                same_process(frozen, transaction["old"], check_parent=False)
                children = self.collection.children(job, frozen)  # Include newly queued direct children.
            else:
                children = []
                for identity in transaction["children"]:
                    child = self.collection.proc.read(identity["pid"])
                    if child is not None:
                        same_process(child, identity, check_parent=False)
                        self.collection.renderer(job, child)
                        children.append(child)
                if {value.pid for value in self.collection.writers(job, self.collection.proc.all())} != {value.pid for value in children}:
                    raise UnsafeJob("Unknown orphan writer prevents restart recovery")
            if self.collection.complete(job) and not committed:
                if driver_pin is not None:
                    self.collection.proc.signal(driver_pin, signal.SIGCONT)
                self.clear_transaction("completed_during_restart", slug=spec.slug)
                return
            self.collection.verify_pins(self.state["pins"])
            child_pins = []
            for child in children:
                pin = self.collection.proc.pin(child)
                if pin is not None:
                    pins.append(pin)
                    child_pins.append(pin)
            transaction["children"] = [pin.process.identity() for pin in child_pins]
            transaction["stage"] = "terminating"
            self.save()
            committed = True
            for pin in child_pins:
                self.collection.proc.signal(pin, signal.SIGTERM)
            if not self.collection.proc.wait(child_pins, self.remaining(self.grace)):
                for pin in child_pins:
                    if not self.collection.proc.exited(pin):
                        self.collection.proc.signal(pin, signal.SIGKILL)
                if not self.collection.proc.wait(child_pins, self.remaining(self.grace)):
                    raise UnsafeJob("Owned renderer did not exit after termination")
            if driver_pin is not None and not self.collection.proc.exited(driver_pin):
                # Its queue is frozen and every verified renderer has exited.
                # Killing this idle driver avoids ever releasing queued ranges.
                self.collection.proc.signal(driver_pin, signal.SIGKILL)
                if not self.collection.proc.wait([driver_pin], self.remaining(self.grace)):
                    raise UnsafeJob("Owned driver did not exit")
            if self.collection.writers(job, self.collection.proc.all()):
                raise UnsafeJob("A surviving writer prevents relaunch")
            transaction["stage"] = "launching"
            self.save()
            self.remaining()
            process = self.collection.proc.spawn(transaction["new_command"], transaction["old"]["cwd"],
                                                 transaction["environment"], self.collection.paths.log(spec), self.collection.paths.python)
            self.collection.driver_process(job, process, transaction["new_command"])
            transaction["stage"] = "launched"
            transaction["new"] = process.identity()
            self.save()
            self.adopt(job, process, transaction)
        except BaseException:
            if not committed and driver_pin is not None and not self.collection.proc.exited(driver_pin):
                self.collection.proc.signal(driver_pin, signal.SIGCONT)
                self.state["transaction"] = None
                self.save()
            raise
        finally:
            for pin in pins:
                self.collection.proc.close(pin)


def bootstrap(collection, timeout_hours):
    if collection.proc.boot != collection.expected_boot:
        raise UnsafeJob("Server boot identity changed; current jobs must be reviewed again")
    pins = collection.immutable_pins()
    now = time.time()
    state = {"schema": 1, "boot_id": collection.proc.boot, "started_unix": now,
             "deadline_unix": now + timeout_hours * 3600, "pins": pins, "jobs": {}, "transaction": None}
    for name, spec in collection.specs.items():
        job = collection.job(spec)
        identity = None
        if not collection.complete(job):
            process = collection.driver(job)
            if process.pid != spec.initial_pid or process.start != spec.initial_start or job.workers != spec.initial_workers:
                raise UnsafeJob(f"Initial process changed since authorization: {name}")
            try:
                collection.children(job, process)
            except Deferred:
                pass
            identity = process.identity()
        state["jobs"][name] = {"pid": job.record["pid"], "command": job.command, "identity": identity}
    return state


def validate_state(state, collection):
    if (state.get("schema") != 1 or state.get("boot_id") != collection.proc.boot
            or set(state.get("jobs", {})) != set(collection.specs)):
        raise UnsafeJob("Controller state does not match this boot/collection")
    start, end = state["started_unix"], state["deadline_unix"]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, end)) or not 0 < end - start <= MAX_SECONDS:
        raise UnsafeJob("Invalid persisted deadline")
    if time.time() < start - 1.0:
        raise UnsafeJob("Wall clock moved behind controller start")
    for name, expected in collection.known_hashes.items():
        if state["pins"].get(name) != expected:
            raise UnsafeJob("Persisted immutable-file pins changed")
    collection.verify_pins(state["pins"])


def dry_plan(controller):
    if controller.state.get("transaction"):
        transaction = controller.state["transaction"]
        return {"mode": "dry-run", "pending_transaction": {key: transaction[key] for key in ("slug", "stage", "target_workers")},
                "action": "recovery required before a new allocation plan", "signals": []}
    jobs, complete, desired = controller.plan()
    return {"mode": "dry-run", "aurora_rendering_complete": "03-aurora" in complete,
            "deadline_unix": controller.state["deadline_unix"], "signals": [],
            "jobs": [{"slug": name, "pid": job.record["pid"], "rendering_complete": name in complete,
                      "workers": job.workers, "desired_workers": desired.get(name),
                      "action": "finishing-or-complete" if name in complete else "keep" if desired[name] == job.workers else "restart",
                      "new_command": changed_workers(job.command, desired[name]) if name in desired and desired[name] != job.workers else None}
                     for name, job in jobs.items()],
            "remaining_render_worker_total": sum(desired.values())}


@contextlib.contextmanager
def controller_lock(directory):
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.resolve() != directory or directory.stat().st_uid != os.getuid():
        raise UnsafeJob("Controller directory is redirected or has another owner")
    os.chmod(directory, 0o700)
    descriptor = os.open(directory / "controller.lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise UnsafeJob("Another collection rebalancer holds the lock") from error
        yield
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true", help="explicitly authorize the already-scoped process restarts")
    modes.add_argument("--dry-run", action="store_true", help="validate and print a plan; this is the default")
    parser.add_argument("--once", action="store_true", help="with --apply, perform one allocation pass then exit")
    parser.add_argument("--timeout-hours", type=float, default=720)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--grace-seconds", type=float, default=20)
    parser.add_argument("--self-test", action="store_true", help="run local semantic tests without touching collection jobs")
    args = parser.parse_args(argv)
    if args.self_test:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_rebalance_collection.py")
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if (not all(math.isfinite(value) and value > 0 for value in (args.timeout_hours, args.poll_seconds, args.grace_seconds))
            or args.timeout_hours > 720 or args.poll_seconds > 3600 or args.grace_seconds > 60):
        parser.error("Require finite positive timeout<=720h, poll<=3600s and grace<=60s")
    collection = Collection()
    path = collection.paths.private / "state.json"
    if not args.apply:
        state = read_json(path) if path.exists() else bootstrap(collection, args.timeout_hours)
        validate_state(state, collection)
        print(json.dumps(dry_plan(Controller(collection, state, args.grace_seconds)), indent=2))
        return 0
    with controller_lock(collection.paths.private):
        state = read_json(path) if path.exists() else bootstrap(collection, args.timeout_hours)
        validate_state(state, collection)
        controller = Controller(collection, state, args.grace_seconds)
        controller.save()
        previous = None
        try:
            while True:
                controller.remaining()
                controller.recover()
                jobs, complete, desired = controller.plan()
                status = (tuple(sorted(complete)), tuple((name, job.workers) for name, job in jobs.items()))
                if status != previous:
                    controller.event("allocation_status", completed=sorted(complete), allocations=desired,
                                     deadline_unix=state["deadline_unix"])
                    previous = status
                if len(complete) == len(collection.specs):
                    controller.event("rendering_complete", note="Finishing/encoding processes remain untouched")
                    return 0
                changes = [(name, desired[name]) for name, job in jobs.items() if name in desired and desired[name] != job.workers]
                # Decreases first ensure transitions do not temporarily exceed the target budget.
                changes.sort(key=lambda item: (item[1] >= jobs[item[0]].workers, item[0]))
                for name, workers in changes:
                    try:
                        controller.begin(jobs[name], workers)
                    except Deferred as error:
                        controller.event("deferred", slug=name, reason=str(error))
                if args.once:
                    return 0
                time.sleep(controller.remaining(args.poll_seconds))
        except BaseException as error:
            controller.event("controller_failed", error=str(error), transaction_pending=state.get("transaction") is not None)
            raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (UnsafeJob, OSError, ValueError, TimeoutError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        raise SystemExit(2) from error
