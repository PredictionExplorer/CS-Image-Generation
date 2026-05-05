"""Fast tests for the repository's Python helper scripts."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import _utils

ROOT = Path(__file__).resolve().parents[1]


def load_script_module(module_name: str, filename: str) -> ModuleType:
    """Import a hyphenated script file as a normal module for unit tests."""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


run = load_script_module("cosmicsig_run", "run.py")
verify_reference = load_script_module("verify_reference", "ci/verify_reference.py")


def test_fmt_duration_formats_seconds_minutes_and_hours() -> None:
    assert _utils.fmt_duration(9.9) == "9s"
    assert _utils.fmt_duration(62) == "1m02s"
    assert _utils.fmt_duration(3_661) == "1h01m01s"


def test_resolve_binary_accepts_executable_and_rejects_bad_paths(tmp_path: Path) -> None:
    binary = tmp_path / "generator"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    assert _utils.resolve_binary(binary) == binary

    with pytest.raises(SystemExit) as missing:
        _utils.resolve_binary(tmp_path / "missing")
    assert missing.value.code == 1

    not_executable = tmp_path / "not-executable"
    not_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    with pytest.raises(SystemExit) as denied:
        _utils.resolve_binary(not_executable)
    assert denied.value.code == 1


def test_check_ffmpeg_exits_with_clear_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_utils.shutil, "which", lambda _name: None)

    with pytest.raises(SystemExit) as exc:
        _utils.check_ffmpeg()

    assert exc.value.code == 1


def test_load_dotenv_parses_values_without_overwriting_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "COSMICSIG_SSH_HOST=example.test",
                "export COSMICSIG_SSH_USER='frontend'",
                'COSMICSIG_API_URL="http://api.test:8353"',
                "INVALID LINE",
                "COSMICSIG_REMOTE_DIR=/srv/assets",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("COSMICSIG_SSH_HOST", raising=False)
    monkeypatch.setenv("COSMICSIG_SSH_USER", "existing")
    monkeypatch.delenv("COSMICSIG_API_URL", raising=False)
    monkeypatch.delenv("COSMICSIG_REMOTE_DIR", raising=False)

    run.load_dotenv(str(env_file))

    assert os.environ["COSMICSIG_SSH_HOST"] == "example.test"
    assert os.environ["COSMICSIG_SSH_USER"] == "existing"
    assert os.environ["COSMICSIG_API_URL"] == "http://api.test:8353"
    assert os.environ["COSMICSIG_REMOTE_DIR"] == "/srv/assets"


class FakeResponse:
    """Minimal context manager matching the part of urlopen used by run.py."""

    status = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_fetch_token_seeds_normalizes_prefixes_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "status": "1",
        "CosmicSignatureTokenList": [
            {"Seed": "0xabc"},
            {"Seed": "0Xdef"},
            {"Seed": "abc"},
            {"Seed": ""},
            {},
        ],
    }
    monkeypatch.setattr(
        run.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse(payload)
    )

    assert run.fetch_token_seeds("http://api.test", retries=1) == ["abc", "def"]


def test_fetch_token_seeds_raises_after_invalid_api_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"status": "0", "error": "nope", "CosmicSignatureTokenList": []}
    monkeypatch.setattr(
        run.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse(payload)
    )
    monkeypatch.setattr(run.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="Failed to fetch token seeds"):
        run.fetch_token_seeds("http://api.test", retries=2)


def test_find_missing_seeds_requires_png_and_mp4() -> None:
    remote = {"0xaaa.png", "0xaaa.mp4", "0xbbb.png", "0xccc.mp4"}

    assert run.find_missing_seeds(["aaa", "bbb", "ccc", "ddd"], remote) == ["bbb", "ccc", "ddd"]


def test_resolve_generator_returns_first_executable_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing"
    generator = tmp_path / "generator"
    generator.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    generator.chmod(0o755)
    monkeypatch.setattr(run, "GENERATOR_CANDIDATES", [str(missing), str(generator)])

    assert run.resolve_generator(None) == [str(generator)]
    assert run.resolve_generator(str(missing)) is None


def test_find_local_files_uses_per_seed_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "output"
    seed_dir = output_dir / "0xabc"
    seed_dir.mkdir(parents=True)
    image = seed_dir / "image.png"
    video = seed_dir / "video.mp4"
    image.write_bytes(b"png")
    video.write_bytes(b"mp4")
    monkeypatch.setattr(run, "LOCAL_OUTPUT_DIR", output_dir)

    assert run.find_local_files("abc") == (image, video)


def test_process_seed_dry_run_does_not_require_generator() -> None:
    assert run.process_seed(
        "abc",
        exec_cmd=None,
        ssh_host="host",
        ssh_user="user",
        remote_dir="/remote",
        timeout=1,
        dry_run=True,
    )


def test_verify_reference_uses_metadata_hash_when_present(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    test_image = tmp_path / "test.png"
    reference.write_bytes(b"reference")
    test_image.write_bytes(b"test")
    expected_hash = verify_reference.calculate_sha256(test_image)
    reference.with_suffix(".json").write_text(
        json.dumps({"sha256": expected_hash}),
        encoding="utf-8",
    )

    assert verify_reference.verify_image(test_image, reference)


def test_verify_reference_reports_mismatch_without_metadata(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    test_image = tmp_path / "test.png"
    reference.write_bytes(b"reference")
    test_image.write_bytes(b"test")

    assert not verify_reference.verify_image(test_image, reference)


def test_load_reference_metadata_rejects_non_object_json(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    reference.with_suffix(".json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="not a JSON object"):
        verify_reference.load_reference_metadata(reference)


def test_run_test_images_random_seed_and_worker_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_test_images = load_script_module("run_test_images", "run-test-images.py")

    monkeypatch.setattr(run_test_images.secrets, "token_hex", lambda length: "ab" * length)
    assert run_test_images.random_seed() == "0xabababababab"

    def fake_run_success(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["generator"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_test_images.subprocess, "run", fake_run_success)
    ok = run_test_images.run_one("generator", "0xabc", 1)
    assert ok.success
    assert ok.seed == "0xabc"

    def fake_run_failure(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["generator"], returncode=1, stdout="", stderr="bad"
        )

    monkeypatch.setattr(run_test_images.subprocess, "run", fake_run_failure)
    failed = run_test_images.run_one("generator", "0xdef", 2)
    assert not failed.success
    assert failed.seed == "0xdef"
