"""Archive integrity and material-field boundary contracts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .common import artifact, check_fields, checked, encoded, read, write


def material_fields(width=8, height=6):
    return {
        "pigment": np.full((height, width, 3), 0.1, dtype="f4"),
        "height": np.full((height, width), 0.001, dtype="f4"),
        "wetness": np.full((height, width), 0.2, dtype="f4"),
        "direction": np.zeros((height, width, 2), dtype="f4"),
        "roughness": np.full((height, width), 0.5, dtype="f4"),
        "coverage": np.ones((height, width), dtype="f4"),
    }


class FieldTests(unittest.TestCase):
    def test_numeric_fields_allow_additional_mass_accounting_arrays(self):
        fields = material_fields()
        fields["mobile"] = fields["pigment"].copy()
        statistics = check_fields(fields, (8, 6))
        self.assertEqual(set(statistics), set(material_fields()))
        self.assertAlmostEqual(statistics["pigment"]["mean"], 0.1)

    def test_invalid_shape_precision_and_physical_bounds_are_rejected(self):
        bad = [
            ("pigment", np.zeros((6, 8, 3), dtype="f8")),
            ("height", np.zeros((8, 6), dtype="f4")),
            ("wetness", np.full((6, 8), 1.1, dtype="f4")),
            ("roughness", np.full((6, 8), -0.1, dtype="f4")),
            ("height", np.full((6, 8), float("nan"), dtype="f4")),
            ("direction", np.ones((6, 8, 2), dtype="f4")),
        ]
        for key, value in bad:
            fields = material_fields()
            fields[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                check_fields(fields, (8, 6))
        fields = material_fields()
        del fields["coverage"]
        with self.assertRaises(ValueError):
            check_fields(fields, (8, 6))


class ArchiveTests(unittest.TestCase):
    def test_json_encoding_is_canonical_and_rejects_nonfinite_values(self):
        self.assertEqual(encoded({"b": 2, "a": 1}), encoded({"a": 1, "b": 2}))
        with self.assertRaises(ValueError):
            encoded({"wetness": float("nan")})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            write(path, {"value": 3})
            self.assertEqual(read(path), {"value": 3})
            self.assertFalse(path.with_suffix(".json.partial").exists())
            path.write_text('{"value": Infinity}')
            with self.assertRaises(ValueError):
                read(path)

    def test_checked_rejects_changed_content_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "archive"
            root.mkdir()
            picture = root / "poster.png"
            picture.write_bytes(b"original")
            expected = artifact(picture)
            self.assertEqual(checked(root, "poster.png", expected), picture.resolve())
            picture.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                checked(root, "poster.png", expected)
            outside = Path(tmp) / "outside.png"
            outside.write_bytes(b"outside")
            (root / "link.png").symlink_to(outside)
            for name in ("../outside.png", "link.png"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    checked(root, name, artifact(outside))


if __name__ == "__main__":
    unittest.main()
