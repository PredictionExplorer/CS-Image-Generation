#!/usr/bin/env python3
"""Run the Python test modules in parallel, one `unittest` process per module.

The suite's time goes into processes and waiting (fake commands, temporary git repositories,
subprocesses of the scripts under test), not the CPU, so running the modules side by side
roughly halves a local run: the pre-push hook and `just py-test` use this runner. CI keeps
`python -m unittest discover -s tests/python -v`, one process for everything, which is the
reference: a module must pass there and here alike.

Each module runs as `python -m unittest discover -s <dir> -p <file>` from the current directory
(the repository root, for the hook and `just`): the same interpreter, working directory,
`sys.path` and module name as under plain discovery, and nothing added to the environment the
tests and their subprocesses inherit. A module's output (stdout and stderr merged) is held back
until it finishes and then printed as one block, so the output of modules that run at the same
time never interleaves. A summary follows, and the exit status is 0 only when every module
passed.

Standard library only. Usage, from the repository root:

    python3 tests/python/run_parallel.py          # every tests/python/test_*.py
    python3 tests/python/run_parallel.py -v       # with unittest's per-test lines
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

TESTS_DIR = Path(__file__).resolve().parent
MODULE_PATTERN = "test_*.py"

# Exit statuses: every module passed; a module failed (or could not run); no test modules or a
# bad option; interrupted (Ctrl-C).
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


@dataclass(frozen=True)
class ModuleResult:
    """How one test module's unittest process ended."""

    name: str
    returncode: int
    seconds: float
    output: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def find_modules(start_dir: Path) -> list[Path]:
    """The test modules unittest discovery would load from `start_dir`, sorted by name."""
    return sorted(path for path in start_dir.glob(MODULE_PATTERN) if path.is_file())


def unittest_command(module: Path, *, verbose: bool) -> list[str]:
    """The command that runs one module the way `unittest discover -s <its dir>` loads it."""
    command = [sys.executable, "-m", "unittest", "discover", "-s", str(module.parent)]
    command += ["-p", module.name]
    if verbose:
        command.append("-v")
    return command


class ParallelRun:
    """Runs modules concurrently and can kill whatever still runs when interrupted."""

    def __init__(self, *, verbose: bool) -> None:
        self.verbose = verbose
        self._lock = threading.Lock()
        self._running: list[subprocess.Popen[bytes]] = []
        self._stopped = False

    def run_module(self, module: Path) -> ModuleResult:
        """Run one module to completion and collect its merged output."""
        started = time.monotonic()
        with self._lock:
            if self._stopped:
                return ModuleResult(module.stem, EXIT_INTERRUPTED, 0.0, "not started\n")
            process = subprocess.Popen(
                unittest_command(module, verbose=self.verbose),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self._running.append(process)
        try:
            output, _ = process.communicate()
        finally:
            with self._lock:
                self._running.remove(process)
        return ModuleResult(
            name=module.stem,
            returncode=process.returncode,
            seconds=time.monotonic() - started,
            output=output.decode("utf-8", errors="replace"),
        )

    def stop(self) -> None:
        """Start nothing new and kill every module still running."""
        with self._lock:
            self._stopped = True
            for process in self._running:
                process.kill()


def print_result(result: ModuleResult, out: TextIO) -> None:
    """One module's block: a header line, then its output exactly as unittest wrote it."""
    status = "ok" if result.passed else f"FAILED (exit status {result.returncode})"
    print(f"===== {result.name}: {status}, {result.seconds:.1f} s", file=out)
    out.write(result.output)
    if result.output and not result.output.endswith("\n"):
        out.write("\n")
    out.flush()


def print_summary(results: Sequence[ModuleResult], seconds: float, out: TextIO) -> None:
    """One line per module, in name order, then the verdict."""
    width = max(len(result.name) for result in results)
    print("===== summary", file=out)
    for result in sorted(results, key=lambda result: result.name):
        status = "ok" if result.passed else "FAILED"
        print(f"  {status:<7}{result.name:<{width}}  {result.seconds:6.1f} s", file=out)
    failed = [result.name for result in results if not result.passed]
    total = f"{len(results)} test module(s) in {seconds:.1f} s"
    if failed:
        print(f"FAILED: {', '.join(sorted(failed))} ({total})", file=out)
    else:
        print(f"OK: {total}", file=out)
    out.flush()


def run_all(modules: Sequence[Path], *, verbose: bool, jobs: int, out: TextIO) -> int:
    """Run `modules` with at most `jobs` at a time, printing each block as it finishes."""
    started = time.monotonic()
    runner = ParallelRun(verbose=verbose)
    results: list[ModuleResult] = []
    executor = ThreadPoolExecutor(max_workers=jobs)
    try:
        futures = [executor.submit(runner.run_module, module) for module in modules]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print_result(result, out)
    except KeyboardInterrupt:
        runner.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        print("interrupted: stopped every test module still running", file=out)
        return EXIT_INTERRUPTED
    executor.shutdown(wait=True)
    print_summary(results, time.monotonic() - started, out)
    return EXIT_OK if all(result.passed for result in results) else EXIT_FAILED


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Run every {MODULE_PATTERN} of a directory in its own unittest process, "
            "concurrently; print each module's output when it finishes, then a summary."
        )
    )
    parser.add_argument(
        "-s",
        "--start-dir",
        type=Path,
        default=TESTS_DIR,
        help="directory of the test modules (default: this script's directory)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help="modules to run at a time (default: all of them)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="pass -v to unittest")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    """Command-line entry point; returns the exit status."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out = sys.stdout if out is None else out
    modules = find_modules(args.start_dir)
    if not modules:
        print(f"error: no {MODULE_PATTERN} in {args.start_dir}", file=sys.stderr)
        return EXIT_USAGE
    if args.jobs < 0:
        print("error: --jobs must be 0 (all) or more", file=sys.stderr)
        return EXIT_USAGE
    jobs = len(modules) if args.jobs == 0 else min(args.jobs, len(modules))
    print(
        f"Running {len(modules)} test module(s), {jobs} at a time, "
        f"with {Path(sys.executable).name} {sys.version.split()[0]}...",
        file=out,
        flush=True,
    )
    return run_all(modules, verbose=args.verbose, jobs=jobs, out=out)


if __name__ == "__main__":
    sys.exit(main())
