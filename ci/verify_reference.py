#!/usr/bin/env python3
"""CI verification script for comparing generated images against reference images."""

import sys
import hashlib
import json
from pathlib import Path


def calculate_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_reference_metadata(ref_path: Path) -> dict:
    json_path = ref_path.with_suffix('.json')
    if not json_path.exists():
        raise FileNotFoundError(f"Reference metadata not found: {json_path}")

    with open(json_path, 'r') as f:
        return json.load(f)


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
        expected_hash = metadata.get('sha256', ref_hash)
    except Exception as e:
        print(f"WARNING: Could not load reference metadata: {e}")
        expected_hash = ref_hash

    if test_hash == expected_hash:
        print(f"✓ Image verification PASSED")
        print(f"  Test hash:      {test_hash}")
        print(f"  Expected hash:  {expected_hash}")
        return True
    else:
        print(f"✗ Image verification FAILED")
        print(f"  Test hash:      {test_hash}")
        print(f"  Expected hash:  {expected_hash}")

        test_size = test_image_path.stat().st_size
        ref_size = reference_image_path.stat().st_size
        print(f"  Test file size:     {test_size} bytes")
        print(f"  Reference file size: {ref_size} bytes")

        return False


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python verify_reference.py <test_image_path> [reference_image_path]")
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
