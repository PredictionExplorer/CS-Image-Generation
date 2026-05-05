# Contributing

This project treats generated-art quality and reproducibility as part of the codebase. Keep changes small, deterministic, and easy to verify.

## Before You Change Code

- Use the pinned Rust toolchain from `rust-toolchain.toml`.
- Create and activate a Python virtual environment before running Python checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quality Gates

Run the focused gate while developing:

```bash
just gate
just py-check
```

Before handing off a broad change, run:

```bash
just full-gate
```

`just full-gate` includes the slower audit and coverage checks. Coverage must stay at or above the repository's 95% line threshold.

## Testing Expectations

- Add Rust tests for boundary cases, validation behavior, deterministic contracts, and file/error handling touched by your change.
- Add Python tests under `tests_py/` for helper-script behavior that can be exercised without network, SSH, ffmpeg, or generated media.
- Avoid golden image changes unless the rendering output intentionally changes and the reference update is reviewed.

## Style

- Keep rendering and simulation algorithm changes separate from tooling, docs, or test-only improvements.
- Do not add local lint suppressions unless the reason is specific and reviewable.
- Preserve the stdlib-only runtime contract for `run.py`, `run-test-images.py`, and `ci/verify_reference.py`; developer-only Python packages belong in the `[dev]` extra.
