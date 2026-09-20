"""Reproducible initialized properties, bounded transport and contact-only response."""

import os
import unittest

import numpy as np

from . import material_traits as traits
from . import test_interaction as fixtures
from .interaction import (
    DEFAULTS,
    FIELD_NAMES,
    contact_strength,
    field_names,
    initial_origins,
    maccormack_transport,
    mass_weighted_sample,
    nucleation_field,
    update_state,
    validate_config,
    velocity_derivatives,
)


class MaterialTraitTests(unittest.TestCase):
    def test_optional_configuration_canonicalizes_zero_without_changing_legacy_fields(self):
        for value in (None, {"amplitude": 0}, {"amplitude": 0, "fine_scale": 0.02}):
            self.assertEqual(validate_config({"material_variation": value}), DEFAULTS)
            self.assertEqual(field_names({"material_variation": value}), FIELD_NAMES)
        self.assertEqual(field_names(None), ())
        resolved = validate_config({"material_variation": {}})
        self.assertEqual(resolved["material_variation"], traits.DEFAULTS)
        self.assertEqual(field_names(resolved), FIELD_NAMES + traits.FIELD_NAMES)
        self.assertEqual(validate_config(resolved), resolved)
        for value in (True, [], {"version": "v0"}, {"unknown": 1}, {"fine_scale": 0.9}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                traits.validate_config(value)
        for key in traits.DEFAULTS.keys() - {"version"}:
            for value in (True, -1, "1", float("nan"), float("inf"), 10**400):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    traits.validate_config({key: value})
        with self.assertRaises(ValueError):
            traits.validate_config({"amplitude": 0.251})

    def test_seeded_traits_use_full_seed_and_independent_channel_scale_streams(self):
        world = initial_origins((57, 39), 57 / 39, 1.6)[..., :2]
        a = traits.initial_traits(world, "0X000B7", {})
        np.testing.assert_array_equal(a, traits.initial_traits(world, 0xB7, {}))
        self.assertEqual(a.dtype, np.dtype("f4"))
        self.assertTrue(np.all(np.abs(a) <= 1))
        for bit in (0, 64, 127, 255):
            other = traits.initial_traits(world, 0xB7 ^ (1 << bit), {})
            self.assertGreater(np.mean(np.abs(a - other)), 0.1)
        keys = {
            traits.seed_key(0xB7, channel, scale)
            for channel in ("aggregation", "fabric")
            for scale in ("coarse", "fine")
        }
        self.assertEqual(len(keys), 4)
        self.assertGreater(np.mean(np.abs(a[..., 0] - a[..., 1])), 0.1)
        # Amplitude controls kinetics, not the initialized material identity.
        np.testing.assert_array_equal(a, traits.initial_traits(world, 0xB7, {"amplitude": 0.25}))

    def test_traits_are_world_coordinate_properties_independent_of_sampling_grid(self):
        world = initial_origins((27, 15), 1.8, 1.6)[..., :2]
        complete = traits.initial_traits(world, 731, {})
        np.testing.assert_array_equal(
            complete[::3, ::3], traits.initial_traits(world[::3, ::3], 731, {})
        )
        for value in (np.zeros((3, 2)), np.full((2, 3, 2), np.nan), np.full((2, 3, 2), 1e12)):
            with self.assertRaises(ValueError):
                traits.initial_traits(value, 731, {})

    def test_archive_traits_are_an_atomic_bounded_float32_pair(self):
        fields = {name: np.zeros((3, 5, 2), dtype="f4") for name in traits.FIELD_NAMES}
        traits.validate_fields(fields, (3, 5), enabled=True)
        traits.validate_fields({}, (3, 5), enabled=False)
        for changed in (
            {},
            {"trait_upper": fields["trait_upper"]},
            {**fields, "trait_upper": np.ones((3, 5, 2), dtype="f8")},
            {**fields, "trait_upper": np.full((3, 5, 2), 1.01, dtype="f4")},
            {**fields, "trait_upper": np.full((3, 5, 2), np.nan, dtype="f4")},
        ):
            with self.assertRaises(ValueError):
                traits.validate_fields(changed, (3, 5), enabled=True)
        with self.assertRaises(ValueError):
            traits.validate_fields(fields, (3, 5), enabled=False)

    def test_rate_modulation_is_neutral_without_wet_contact_and_remains_bounded(self):
        rng = np.random.default_rng(6303)
        material = rng.uniform(-1, 1, (7, 9, 2))
        contact, wet = rng.uniform(0, 1, (2, 7, 9))
        contact[0] = 0
        wet[1] = 0
        factors = traits.rate_multipliers(material, contact, wet, {"amplitude": 0.25})
        for field in factors:
            np.testing.assert_array_equal(field[:2], 1)
            self.assertGreaterEqual(field.min(), 0.75)
            self.assertLessEqual(field.max(), 1.25)

    def test_actual_reaction_is_neutral_without_contact_or_when_dry(self):
        fixture = fixtures.InteractionTests("runTest")
        args = fixture.fields()
        args["state"][..., 0] = 0.7
        args["state"][..., 1] = 0.2
        args["state"][..., 3] = 0.6
        args["strain"][..., 0] = 0.4
        material = np.ones((*args["contact"].shape, 2))
        for zero in ("contact", "wetness"):
            original = args[zero].copy()
            args[zero][:] = 0
            baseline = update_state(**args)
            modified = update_state(
                **{**args, "config": {"material_variation": {"amplitude": 0.25}}}, traits=material
            )
            np.testing.assert_array_equal(baseline, modified)
            args[zero][:] = original
        modified = update_state(
            **{**args, "config": {"material_variation": {"amplitude": 0.25}}}, traits=material
        )
        baseline = update_state(**args)
        np.testing.assert_array_equal(baseline[..., 0], modified[..., 0])
        self.assertGreater(np.max(modified[..., 3] - baseline[..., 3]), 0.01)
        self.assertGreater(np.max(modified[..., 1] - baseline[..., 1]), 0.001)
        self.assertTrue(np.all(np.linalg.norm(modified[..., 1:3], axis=-1) <= modified[..., 0]))

    def test_property_transport_preserves_constants_bounds_and_excludes_empty_donors(self):
        rng = np.random.default_rng(8074)
        mass = rng.uniform(0.01, 1, (17, 31))
        mass[:, 11:18] = 0
        material = np.full((*mass.shape, 2), (0.2, -0.7))
        material[mass == 0] = (0.9, 0.8)
        velocity = np.full_like(material, (0.4, -0.2))
        yy, xx = np.mgrid[:17, :31]
        positions = np.stack((xx, yy), -1) - velocity
        support = mass_weighted_sample(np.ones((*mass.shape, 1)), mass, positions)[..., 0] > 0
        for result in (
            mass_weighted_sample(material, mass, positions),
            maccormack_transport(material, mass, velocity, pixel_size=1, dt=1),
        ):
            np.testing.assert_allclose(
                result[support], np.broadcast_to((0.2, -0.7), result[support].shape), atol=1e-15
            )
            np.testing.assert_array_equal(result[~support], 0)


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class MaterialTraitGPUTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.InteractionGPUTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_initial_traits_mask_air_and_snapshot_adds_only_atomic_optional_pair(self):
        f = self.fixture
        helper = f.helper({"material_variation": {}})
        paint = np.zeros((*f.shape, 2), dtype="f4")
        paint[2:-2, 2:-2, 0] = 0.4
        layers = f.packed(paint), f.packed(paint * 0.7)
        helper.initialize(*layers)
        expected = traits.initial_traits(
            initial_origins(f.size, f.aspect, f.domain)[..., :2], f.seed, {}
        )
        expected[paint.sum(-1) == 0] = 0
        for texture in helper.traits:
            np.testing.assert_array_equal(f.read(texture)[..., :2], expected)
            np.testing.assert_array_equal(f.read(texture)[..., 2:], 0)
        snapshot = helper.snapshot(f.read)
        self.assertEqual(tuple(snapshot), FIELD_NAMES + traits.FIELD_NAMES)
        traits.validate_fields(snapshot, f.shape, enabled=True)

    def test_zero_amplitude_is_byte_identical_to_legacy_allocation_and_evolution(self):
        f = self.fixture
        helpers = [
            f.helper({"advection": "maccormack", **extra})
            for extra in ({}, {"material_variation": {"amplitude": 0}})
        ]
        paint = np.full((*f.shape, 6), 0.2, dtype="f4")
        upper, lower = f.packed(paint), f.packed(paint * 0.8)
        carrier = f.texture(np.full((*f.shape, 4), 0.6, dtype="f4"))
        velocity = f.texture(initial_origins(f.size, f.aspect, f.domain)[..., :2] * 0.2)
        for helper in helpers:
            helper.initialize(upper, lower)
            for _ in range(3):
                helper.transport(upper, lower, velocity, 0.03)
                helper.update(upper, lower, carrier, velocity, 0.03)
        self.assertEqual(len(helpers[0]._resources), len(helpers[1]._resources))
        self.assertIsNone(helpers[1].traits)
        for left, right in zip(
            helpers[0].snapshot(f.read).values(), helpers[1].snapshot(f.read).values(), strict=True
        ):
            self.assertEqual(left.tobytes(), right.tobytes())

    def test_both_transports_match_independent_mass_weighted_references(self):
        f = self.fixture
        rng = np.random.default_rng(417)
        pixel = 2 * f.domain / f.size[1]
        velocity = np.full((*f.shape, 2), (pixel * 0.6, pixel * -0.3), dtype="f4")
        yy, xx = np.mgrid[: f.shape[0], : f.shape[1]]
        centers = np.stack((xx, yy), -1).astype("f8")
        for mode in ("linear", "maccormack"):
            for count in (2, 6):
                with self.subTest(mode=mode, count=count):
                    helper = f.helper({"advection": mode, "material_variation": {}})
                    paint = rng.uniform(0.02, 0.5, (*f.shape, count)).astype("f4")
                    paint[3:8, 4:10] = 0
                    material = rng.uniform(-1, 1, (*f.shape, 4)).astype("f4")
                    material[..., 2:] = 0
                    for texture in helper.traits:
                        texture.write(material.tobytes())
                    upper, lower = f.packed(paint), f.packed(paint * 0.8)
                    before = tuple(texture.read() for pack in (upper, lower) for texture in pack)
                    helper.transport(upper, lower, f.texture(velocity), 0.37)
                    for layer, (amount, scale) in enumerate(((paint, 1), (paint * 0.8, 0.78))):
                        expected = (
                            maccormack_transport(
                                material[..., :2],
                                amount.sum(-1),
                                velocity,
                                pixel_size=pixel,
                                dt=0.37 * scale,
                            )
                            if mode == "maccormack"
                            else mass_weighted_sample(
                                material[..., :2],
                                amount.sum(-1),
                                centers - velocity * (0.37 * scale / pixel),
                            )
                        )
                        actual = f.read(helper.traits[layer])
                        np.testing.assert_allclose(actual[..., :2], expected, atol=4e-6, rtol=4e-5)
                        np.testing.assert_array_equal(actual[..., 2:], 0)
                        self.assertTrue(np.all(np.abs(actual) <= 1))
                    self.assertEqual(
                        before, tuple(texture.read() for pack in (upper, lower) for texture in pack)
                    )

    def test_reaction_matches_reference_preserves_properties_and_clears_new_air(self):
        f = self.fixture
        config = {"material_variation": {"amplitude": 0.25}}
        helper = f.helper(config)
        rng = np.random.default_rng(26107)
        upper, lower = rng.uniform(0.01, 0.4, (2, *f.shape, 6)).astype("f4")
        upper[:2] = 0
        lower[:, :2] = 0
        origins = [f.read(texture)[..., :2] for texture in helper.origins]
        old_traits = [f.read(texture)[..., :2] for texture in helper.traits]
        state = np.full((*f.shape, 4), (0.6, 0.1, -0.2, 0.4), dtype="f4")
        for texture in helper.states:
            texture.write(state.tobytes())
        velocity = initial_origins(f.size, f.aspect, f.domain)[..., :2] * (0.7, -0.3)
        velocity = velocity.astype("f4")
        wet = rng.uniform(0, 1, (*f.shape, 4)).astype("f4")
        wet[3, :, 0] = 0
        helper.update(f.packed(upper), f.packed(lower), f.texture(wet), f.texture(velocity), 0.17)
        contact = contact_strength(upper, lower, *origins, config)
        strain, spin = velocity_derivatives(velocity, 2 * f.domain / f.size[1])
        for index, (paint, scale) in enumerate(((upper, 1), (lower, 0.78))):
            expected = update_state(
                state,
                contact,
                wet[..., 0],
                strain * scale,
                spin * scale,
                nucleation_field(origins[index], f.seed, config),
                config=config,
                dt=0.17,
                traits=old_traits[index],
            )
            empty = paint.sum(-1) <= DEFAULTS["minimum_concentration"]
            expected[empty] = 0
            old_traits[index][empty] = 0
            np.testing.assert_allclose(f.read(helper.states[index]), expected, atol=2e-5, rtol=4e-5)
            np.testing.assert_array_equal(f.read(helper.traits[index])[..., :2], old_traits[index])

    def test_constant_transport_and_snapshot_cadence_are_invariant(self):
        f = self.fixture
        config = {"advection": "maccormack", "material_variation": {}}
        helpers = [f.helper(config), f.helper(config)]
        paint = np.full((*f.shape, 2), 0.3, dtype="f4")
        paint[:, 7:12] = 0
        layers = f.packed(paint), f.packed(paint * 0.9)
        material = np.zeros((*f.shape, 4), dtype="f4")
        material[..., :2] = (0.25, -0.5)
        pixel = 2 * f.domain / f.size[1]
        velocity = f.texture(np.full((*f.shape, 2), (pixel * 0.4, 0), dtype="f4"))
        wet = f.texture(np.full((*f.shape, 4), 0.6, dtype="f4"))
        for helper in helpers:
            helper.initialize(*layers)
            for texture in helper.traits:
                texture.write(material.tobytes())
        for _ in range(4):
            for index, helper in enumerate(helpers):
                helper.transport(*layers, velocity, 0.3)
                helper.update(*layers, wet, velocity, 0.3)
                if index == 0:
                    helper.snapshot(f.read)
        for name, value in helpers[0].snapshot(f.read).items():
            self.assertEqual(value.tobytes(), helpers[1].snapshot(f.read)[name].tobytes())
        for texture in helpers[0].traits:
            value = f.read(texture)[..., :2]
            occupied = paint.sum(-1) > 0
            np.testing.assert_allclose(
                value[occupied], np.broadcast_to((0.25, -0.5), value[occupied].shape), atol=2e-7
            )
            np.testing.assert_array_equal(value[~occupied], 0)
        # Releasing an enabled helper twice must remain safe and own no live resources.
        helpers[0].close()
        helpers[0].close()
        self.assertEqual(helpers[0]._resources, [])


if __name__ == "__main__":
    unittest.main()
