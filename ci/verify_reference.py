#!/usr/bin/env python3
"""CI verification script for comparing generated images against reference images."""

import hashlib
import json
import sys
from pathlib import Path
from typing import TypedDict, cast


class ReferenceMetadata(TypedDict, total=False):
    """Subset of keys stored in reference baseline JSON."""

    sha256: str


def calculate_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    with filepath.open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_reference_metadata(ref_path: Path) -> ReferenceMetadata:
    json_path = ref_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Reference metadata not found: {json_path}")

    with json_path.open(encoding="utf-8") as f:
        raw: object = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Reference metadata is not a JSON object: {json_path}")
        return cast(ReferenceMetadata, raw)


def verify_image(test_image_path: Path, reference_image_path: Path) -> bool:
    if not test_image_path.exists():
        print(f"ERROR: Test image not found: {test_image_path}")
        return False

    if not reference_image_path.exists():
        print(f"ERROR: Reference image not found: {reference_image_path}")
        return False

    test_hash = calculate_sha256(test_image_path)
    ref_hash = calculate_sha256(reference_image_path)

    try:
        metadata = load_reference_metadata(reference_image_path)
        expected_hash = metadata.get("sha256", ref_hash)
    except Exception as e:
        print(f"WARNING: Could not load reference metadata: {e}")
        expected_hash = ref_hash

    if test_hash == expected_hash:
        print("✓ Image verification PASSED")
        print(f"  Test hash:      {test_hash}")
        print(f"  Expected hash:  {expected_hash}")
        return True
    else:
        print("✗ Image verification FAILED")
        print(f"  Test hash:      {test_hash}")
        print(f"  Expected hash:  {expected_hash}")

        test_size = test_image_path.stat().st_size
        ref_size = reference_image_path.stat().st_size
        print(f"  Test file size:     {test_size} bytes")
        print(f"  Reference file size: {ref_size} bytes")

        return False


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 ci/verify_reference.py <test_image_path> [reference_image_path]")
        sys.exit(1)

    test_image_path = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        reference_image_path = Path(sys.argv[2])
    else:
        script_dir = Path(__file__).parent
        reference_image_path = script_dir / "reference" / "baseline_512x288.png"

    success = verify_image(test_image_path, reference_image_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
