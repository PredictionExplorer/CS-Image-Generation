"""Independent invariants and hardware kernel/reference checks for paint history."""

import os
import unittest

import numpy as np

from .interaction import (
    BASE_FIELDS,
    DEFAULTS,
    FIELD_NAMES,
    VERSION,
    GPUInteraction,
    contact_strength,
    initial_origins,
    maccormack_transport,
    mass_weighted_sample,
    nucleation_field,
    seed_key,
    update_state,
    validate_config,
    velocity_derivatives,
)


class InteractionTests(unittest.TestCase):
    def fields(self, shape=(7, 9)):
        return {
            "state": np.zeros((*shape, 4)),
            "contact": np.ones(shape),
            "wetness": np.ones(shape),
            "strain": np.zeros((*shape, 2)),
            "spin": np.zeros(shape),
            "nucleation": np.ones(shape),
            "config": {},
            "dt": 0.4,
        }

    def test_opt_in_config_version_and_strict_bounds(self):
        self.assertIsNone(validate_config(None))
        self.assertEqual(validate_config({}), DEFAULTS)
        self.assertEqual(validate_config({})["version"], VERSION)
        self.assertEqual(len(FIELD_NAMES), 4)
        self.assertEqual(len(BASE_FIELDS), 10)
        for value in (True, [], {"extra": 1}, {"version": "unknown"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_config(value)
        for key in DEFAULTS.keys() - {"version"}:
            for value in (True, -1, float("inf"), float("nan"), 10**400, "1"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_config({key: value})
        with self.assertRaises(ValueError):
            validate_config({"nucleation_scale": 0})
        with self.assertRaises(ValueError):
            validate_config({"nucleation_contrast": 1.01})

    def test_full_seed_stream_is_canonical_and_domain_separated(self):
        self.assertEqual(seed_key("0X00123AB"), seed_key(0x123AB))
        self.assertNotEqual(seed_key(0), seed_key(1 << 255))
        self.assertNotEqual(seed_key(1), seed_key(1 | 1 << 128))
        self.assertEqual(len(seed_key(1)), 4)
        self.assertTrue(all(0 <= n < 2**32 for n in seed_key(1)))

    def test_maccormack_is_explicit_and_preserves_old_resolved_recipe_identity(self):
        self.assertNotIn("advection", validate_config({}))
        self.assertNotIn("advection", DEFAULTS)
        for mode in ("linear", "maccormack"):
            value = validate_config({"advection": mode})
            self.assertEqual(value.pop("advection"), mode)
            self.assertEqual(value, DEFAULTS)
        for mode in (True, [], {}, None, "bfecc", ""):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_config({"advection": mode})

    def test_maccormack_retains_translating_detail_and_has_lower_analytic_error(self):
        yy, xx = np.mgrid[:64, :128]
        positions = np.stack((xx, yy), -1).astype("f8")

        def analytic(x, y):
            return (0.5 + 0.22 * np.sin(x * 2 * np.pi / 16) + 0.14 * np.cos(y * 2 * np.pi / 24))[
                ..., None
            ]

        initial = analytic(xx, yy)
        linear, corrected = initial.copy(), initial.copy()
        mass = np.ones(xx.shape)
        velocity = np.empty((*xx.shape, 2))
        velocity[:] = (0.35, -0.2)
        for _ in range(40):
            linear = mass_weighted_sample(linear, mass, positions - velocity)
            corrected = maccormack_transport(corrected, mass, velocity, pixel_size=1.0, dt=1.0)
        exact = analytic(xx - 40 * 0.35, yy + 40 * 0.2)
        interior = slice(16, -16), slice(24, -24)
        errors = [
            np.sqrt(np.mean((field[interior] - exact[interior]) ** 2))
            for field in (linear, corrected)
        ]
        self.assertLess(errors[1], errors[0] * 0.3)
        self.assertGreater(corrected[interior].std(), exact[interior].std() * 0.95)
        self.assertLess(linear[interior].std(), exact[interior].std() * 0.7)

    def test_maccormack_preserves_constants_with_variable_mass_and_excludes_air(self):
        rng = np.random.default_rng(59002)
        mass = rng.uniform(0.2, 2, (11, 23))
        mass[:, 10:15] = 0
        velocity = np.empty((*mass.shape, 2))
        velocity[:] = (0.35, -0.2)
        field = np.empty((*mass.shape, 3))
        field[:] = (0.2, -0.7, 2.9)
        field[mass == 0] = (200, -400, 999)
        yy, xx = np.mgrid[:11, :23]
        positions = np.stack((xx, yy), -1)
        support = (
            mass_weighted_sample(np.ones((*mass.shape, 1)), mass, positions - velocity)[..., 0] > 0
        )
        result = maccormack_transport(field, mass, velocity, pixel_size=1.0, dt=1.0)
        np.testing.assert_allclose(
            result[support], np.broadcast_to([0.2, -0.7, 2.9], result[support].shape), atol=9e-16
        )
        np.testing.assert_array_equal(result[~support], 0)
        # Old vacuum must not contaminate a target newly reached by paint.
        np.testing.assert_allclose(
            result[(mass == 0) & support],
            np.broadcast_to([0.2, -0.7, 2.9], result[(mass == 0) & support].shape),
            atol=9e-16,
        )

    def test_maccormack_is_bounded_and_retains_coupled_fabric_invariant(self):
        rng = np.random.default_rng(9144)
        state = np.zeros((11, 23, 4))
        state[..., 0] = rng.uniform(0, 1, state.shape[:2])
        angle = rng.uniform(-np.pi, np.pi, state.shape[:2])
        state[..., 1] = state[..., 0] * np.cos(angle)
        state[..., 2] = state[..., 0] * np.sin(angle)
        state[..., 3] = rng.uniform(0, 1, state.shape[:2])
        mass = rng.uniform(0.01, 1, state.shape[:2])
        mass[3:6, 7:9] = 0
        velocity = rng.uniform(-2, 2, (*state.shape[:2], 2))
        result = maccormack_transport(
            state, mass, velocity, pixel_size=1.0, dt=0.7, bound_state=True
        )
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all((result[..., (0, 3)] >= 0) & (result[..., (0, 3)] <= 1)))
        self.assertTrue(np.all(np.linalg.norm(result[..., 1:3], axis=-1) <= result[..., 0] + 2e-16))
        self.assertGreaterEqual(result[..., 3].min(), 0)
        self.assertLessEqual(result[..., 3].max(), state[..., 3].max())

    def test_maccormack_rejects_invalid_reference_inputs(self):
        field = np.zeros((3, 5, 4))
        mass = np.ones((3, 5))
        velocity = np.ones((3, 5, 2))
        for kwargs in (
            {"pixel_size": 0},
            {"dt": True},
            {"dt": -1},
            {"minimum_concentration": -1},
            {"bound_state": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                maccormack_transport(
                    field, mass, velocity, **{"pixel_size": 1.0, "dt": 1.0, **kwargs}
                )
        with self.assertRaises(ValueError):
            maccormack_transport(field, mass * -1, velocity, pixel_size=1.0, dt=1.0)
        with self.assertRaises(ValueError):
            maccormack_transport(
                field[..., :2], mass, velocity, pixel_size=1.0, dt=1.0, bound_state=True
            )

    def test_nucleation_is_reproducible_bounded_and_depends_on_material_origin(self):
        origins = initial_origins((37, 25), 37 / 25, 1.6)[..., :2]
        first = nucleation_field(origins, "0xb7f327f9f722", {})
        np.testing.assert_array_equal(first, nucleation_field(origins.copy(), "0xb7f327f9f722", {}))
        self.assertGreaterEqual(first.min(), 0.3 - 1e-15)
        self.assertLessEqual(first.max(), 1)
        self.assertGreater(first.std(), 0.1)
        self.assertFalse(np.array_equal(first, nucleation_field(origins, 2**255, {})))
        reordered = origins[:, ::-1]
        np.testing.assert_array_equal(
            first[:, ::-1], nucleation_field(reordered, "0xb7f327f9f722", {})
        )
        np.testing.assert_array_equal(nucleation_field(origins, 1, {"nucleation_contrast": 0}), 1)

    def test_identical_colocated_paint_has_no_contact_even_under_strain(self):
        upper = np.zeros((7, 9, 2))
        upper[..., 0] = 0.3
        lower = upper * 2
        origins = initial_origins((9, 7), 9 / 7, 1)[..., :2]
        contact = contact_strength(upper, lower, origins, origins, {})
        np.testing.assert_array_equal(contact, 0)
        fields = self.fields()
        fields["contact"] = contact
        fields["strain"][:] = (2, 1)
        np.testing.assert_array_equal(update_state(**fields), 0)

    def test_monochrome_origins_and_distinct_compositions_both_generate_contact(self):
        upper = np.zeros((7, 9, 2))
        upper[..., 0] = 0.3
        origins = initial_origins((9, 7), 9 / 7, 1)[..., :2]
        same_color = contact_strength(upper, upper, origins, origins + np.array([0.1, 0]), {})
        np.testing.assert_allclose(same_color, 1)
        other = upper[..., ::-1]
        different_colors = contact_strength(upper, other, origins, origins, {})
        np.testing.assert_allclose(different_colors, 1)
        imbalanced = contact_strength(upper, other * 0.01, origins, origins, {})
        np.testing.assert_allclose(imbalanced, 2 * 0.01 / 1.01)

    def test_vacuum_and_negligible_mass_do_not_generate_contact(self):
        upper = np.ones((7, 9, 2))
        origins = initial_origins((9, 7), 9 / 7, 1)[..., :2]
        for lower in (upper * 0, upper * 1e-7):
            np.testing.assert_array_equal(
                contact_strength(upper, lower, origins, origins + 1, {}), 0
            )

    def test_no_contact_and_dry_contact_do_not_generate_texture(self):
        for key in ("contact", "wetness"):
            fields = self.fields()
            fields[key][:] = 0
            fields["strain"][:] = (0.6, -0.8)
            np.testing.assert_array_equal(update_state(**fields), 0)

    def test_dry_state_freezes_chemistry_but_rotates_existing_fabric(self):
        fields = self.fields()
        fields["state"][:] = (0.7, 0.5, 0, 0.3)
        fields["wetness"][:] = 0
        fields["strain"][:] = (1, 1)
        fields["spin"][:] = 0.6
        result = update_state(**fields)
        np.testing.assert_array_equal(result[..., (0, 3)], fields["state"][..., (0, 3)])
        angle = 2 * fields["dt"] * 0.6
        np.testing.assert_allclose(result[..., 1], 0.5 * np.cos(angle))
        np.testing.assert_allclose(result[..., 2], 0.5 * np.sin(angle))

    def test_rigid_translation_and_rotation_have_no_interior_strain(self):
        world = initial_origins((9, 7), 9 / 7, 1)[..., :2].astype("f8")
        dx = 2 / 7
        translated = np.empty_like(world)
        translated[:] = (5.7, -3.2)
        strain, spin = velocity_derivatives(translated, dx)
        np.testing.assert_array_equal(strain, 0)
        np.testing.assert_array_equal(spin, 0)
        rotated = np.stack((-world[..., 1], world[..., 0]), -1) * 0.8
        strain, spin = velocity_derivatives(rotated, dx)
        np.testing.assert_allclose(strain[1:-1, 1:-1], 0, atol=2e-7)
        np.testing.assert_allclose(spin[1:-1, 1:-1], 0.8, atol=2e-7)
        fields = self.fields()
        fields["strain"][:] = 0
        fields["spin"][:] = 0.8
        np.testing.assert_array_equal(update_state(**fields)[..., 1:3], 0)

    def test_extensional_strain_orients_axial_fabric(self):
        fields = self.fields()
        fields["strain"][:] = (0, 2)
        result = update_state(**fields)
        np.testing.assert_array_equal(result[..., 1], 0)
        self.assertGreater(result[..., 2].min(), 0.1)
        self.assertTrue(np.all(np.linalg.norm(result[..., 1:3], axis=-1) <= result[..., 0]))
        fields["strain"][:] = (-2, 0)
        result = update_state(**fields)
        self.assertLess(result[..., 1].max(), -0.1)
        np.testing.assert_array_equal(result[..., 2], 0)

    def test_reaction_bounds_hold_at_extreme_valid_rates_and_long_exposure(self):
        rng = np.random.default_rng(87912)
        fields = self.fields((13, 17))
        fields["state"][..., 0] = rng.uniform(0, 1, (13, 17))
        fields["state"][..., 1:3] = fields["state"][..., :1] * (0.3, -0.4)
        fields["state"][..., 3] = rng.uniform(0, 1, (13, 17))
        for key in ("contact", "wetness", "nucleation"):
            fields[key] = rng.uniform(0, 1, (13, 17))
        fields["strain"] = rng.uniform(-100, 100, (13, 17, 2))
        fields["spin"] = rng.uniform(-100, 100, (13, 17))
        fields["config"] = {
            k: 100
            for k in (
                "contact_rate",
                "fabric_rate",
                "fabric_relaxation",
                "aggregation_rate",
                "breakup_rate",
            )
        }
        for dt in (1e-10, 0.07, 100):
            result = update_state(**{**fields, "dt": dt})
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(np.all((result[..., (0, 3)] >= 0) & (result[..., (0, 3)] <= 1)))
            self.assertTrue(
                np.all(np.linalg.norm(result[..., 1:3], axis=-1) <= result[..., 0] + 2e-16)
            )

    def test_scalar_kinetics_do_not_depend_on_output_or_substep_partition(self):
        fields = self.fields()
        fields["strain"][:] = (0.4, 0.3)
        whole = update_state(**{**fields, "dt": 1})
        split = fields["state"]
        for dt in (0.01, 0.1, 0.37, 0.52):
            split = update_state(**{**fields, "state": split, "dt": dt})
        np.testing.assert_allclose(whole[..., (0, 3)], split[..., (0, 3)], atol=2e-16)
        # Fabric uses operator splitting and is intentionally not claimed exact.

    def test_shear_breaks_existing_aggregate_but_cannot_create_it_without_contact(self):
        fields = self.fields()
        fields["state"][:] = (0.8, 0.3, 0.2, 0.6)
        fields["contact"][:] = 0
        fields["strain"][:] = (3, 4)
        result = update_state(**fields)
        np.testing.assert_allclose(result[..., 3], 0.6 * np.exp(-0.25 * 5 * 0.4))
        fields["state"][..., 3] = 0
        np.testing.assert_array_equal(update_state(**fields)[..., 3], 0)

    def test_aggregate_is_only_a_subpartition_of_existing_pigment(self):
        fields = self.fields()
        result = update_state(**fields)
        pigment = np.random.default_rng(188).uniform(0, 3, (7, 9, 6))
        unchanged = pigment.copy()
        aggregated = pigment * result[..., 3:4]
        dispersed = pigment - aggregated
        np.testing.assert_array_equal(pigment, unchanged)
        np.testing.assert_allclose(dispersed + aggregated, unchanged, atol=4e-16)
        self.assertGreaterEqual(dispersed.min(), 0)
        self.assertGreaterEqual(aggregated.min(), 0)

    def test_mass_weighted_transport_excludes_empty_neighbor_history(self):
        field = np.array([[[0.4, 0.2, -0.1, 0.3], [1, 0.8, 0.6, 1]]])
        mass = np.array([[0.7, 0]])
        positions = np.array([[[0.8, 0], [1, 0], [-100, 0]]])
        sampled = mass_weighted_sample(field, mass, positions)
        np.testing.assert_allclose(sampled[0, 0], field[0, 0])
        np.testing.assert_array_equal(sampled[0, 1], 0)
        np.testing.assert_allclose(sampled[0, 2], field[0, 0])

    def test_mass_weighted_transport_is_convex_in_actual_concentration(self):
        field = np.array([[[0.4, 0.2, -0.1, 0.3], [1, 0.8, 0.6, 1]]])
        mass = np.array([[3.0, 1.0]])
        sampled = mass_weighted_sample(field, mass, np.array([[[0.5, 0]]]))
        np.testing.assert_allclose(sampled[0, 0], 0.75 * field[0, 0] + 0.25 * field[0, 1])
        self.assertLessEqual(np.linalg.norm(sampled[0, 0, 1:3]), sampled[0, 0, 0])

    def test_invalid_fields_are_rejected(self):
        fields = self.fields()
        for key, value in (
            ("dt", -1),
            ("dt", True),
            ("dt", float("inf")),
            ("wetness", np.ones((7, 9)) * 1.01),
            ("contact", np.ones((7, 9)) * -1),
            ("strain", np.ones((7, 8, 2))),
            ("state", np.ones((7, 9, 4))),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                update_state(**{**fields, key: value})
        np.testing.assert_array_equal(update_state(**{**fields, "dt": 0}), fields["state"])


@unittest.skipUnless(os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires hardware OpenGL 4.3")
class InteractionGPUTests(unittest.TestCase):
    def setUp(self):
        import moderngl

        self.ctx = moderngl.create_standalone_context(require=430, backend="egl")
        self.addCleanup(self.ctx.release)
        self.size = (19, 13)
        self.shape = (13, 19)
        self.aspect, self.domain = 19 / 13, 1.3
        self.seed = hex((1 << 255) | (1 << 127) | 0x9124FB578912475B)

    def texture(self, field):
        value = np.ascontiguousarray(field, dtype="f4")
        texture = self.ctx.texture(self.size, value.shape[-1], data=value.tobytes(), dtype="f4")
        texture.repeat_x = texture.repeat_y = False
        self.addCleanup(texture.release)
        return texture

    def packed(self, field):
        textures = []
        for first in range(0, field.shape[-1], 4):
            packed = np.zeros((*self.shape, 4), dtype="f4")
            n = min(4, field.shape[-1] - first)
            packed[..., :n] = field[..., first : first + n]
            textures.append(self.texture(packed))
        return tuple(textures)

    def helper(self, config=None, lower_scale=0.78):
        helper = GPUInteraction(
            self.ctx, self.size, self.aspect, self.domain, self.seed, config or {}, lower_scale
        )
        self.addCleanup(helper.close)
        return helper

    def read(self, texture):
        return np.frombuffer(texture.read(), dtype="f4").reshape(*self.shape, 4).copy()

    def test_initialization_masks_air_and_colocated_paint_remains_silent(self):
        helper = self.helper()
        paint = np.zeros((*self.shape, 2), dtype="f4")
        paint[2:-2, 2:-2, 0] = 0.4
        upper, lower = self.packed(paint), self.packed(paint * 2)
        helper.initialize(upper, lower)
        origin = initial_origins(self.size, self.aspect, self.domain)
        origin[paint.sum(-1) == 0] = 0
        for texture in helper.origins:
            np.testing.assert_allclose(self.read(texture), origin, atol=2e-7)
        carrier = np.ones((*self.shape, 4), dtype="f4")
        world = initial_origins(self.size, self.aspect, self.domain)[..., :2]
        velocity = world.copy()
        velocity[..., 1] *= -1
        helper.update(upper, lower, self.texture(carrier), self.texture(velocity), 0.3)
        for texture in helper.states:
            np.testing.assert_array_equal(self.read(texture), 0)
        self.assertEqual(set(helper.snapshot(self.read)), set(FIELD_NAMES))
        for field in FIELD_NAMES[:2]:
            self.assertEqual(helper.snapshot(self.read)[field].shape, (*self.shape, 2))

    def test_update_matches_independent_reference_for_both_channel_pack_counts(self):
        rng = np.random.default_rng(95022)
        for count in (2, 6):
            with self.subTest(count=count):
                helper = self.helper()
                upper, lower = rng.uniform(0.01, 0.8, (2, *self.shape, count)).astype("f4")
                upper[:2] = 0
                lower[:, :2] = 0
                upper_tex, lower_tex = self.packed(upper), self.packed(lower)
                origin = initial_origins(self.size, self.aspect, self.domain)
                origins = [origin.copy(), origin.copy()]
                origins[1][..., 0] += 0.019
                old_states = []
                for index, texture in enumerate(helper.origins):
                    texture.write(origins[index].tobytes())
                    state = np.empty((*self.shape, 4), dtype="f4")
                    state[..., 0] = rng.uniform(0.1, 0.9, self.shape)
                    state[..., 1:3] = state[..., :1] * (0.3, -0.4)
                    state[..., 3] = rng.uniform(0, 1, self.shape)
                    helper.states[index].write(state.tobytes())
                    old_states.append(state)
                carrier = rng.uniform(0, 1, (*self.shape, 4)).astype("f4")
                carrier[3, :, 0] = 0
                world = origin[..., :2]
                velocity = np.stack(
                    (
                        0.7 * world[..., 0] + 0.2 * world[..., 1],
                        -0.4 * world[..., 0] - 0.1 * world[..., 1],
                    ),
                    -1,
                ).astype("f4")
                upper_before, lower_before = [
                    tuple(texture.read() for texture in textures)
                    for textures in (upper_tex, lower_tex)
                ]
                helper.update(
                    upper_tex, lower_tex, self.texture(carrier), self.texture(velocity), 0.17
                )
                contact = contact_strength(
                    upper, lower, origins[0][..., :2], origins[1][..., :2], {}
                )
                strain, spin = velocity_derivatives(velocity, 2 * self.domain / self.size[1])
                for index, scale in enumerate((1, 0.78)):
                    expected = update_state(
                        old_states[index],
                        contact,
                        carrier[..., 0],
                        strain * scale,
                        spin * scale,
                        nucleation_field(origins[index][..., :2], self.seed, {}),
                        config={},
                        dt=0.17,
                    )
                    expected[
                        (upper if index == 0 else lower).sum(-1)
                        <= DEFAULTS["minimum_concentration"]
                    ] = 0
                    np.testing.assert_allclose(
                        self.read(helper.states[index]), expected, atol=2e-5, rtol=3e-5
                    )
                self.assertEqual(upper_before, tuple(texture.read() for texture in upper_tex))
                self.assertEqual(lower_before, tuple(texture.read() for texture in lower_tex))

    def test_auxiliary_transport_matches_positive_mass_weighted_reference(self):
        rng = np.random.default_rng(86522)
        helper = self.helper(lower_scale=0.5)
        paint = rng.uniform(0, 1, (*self.shape, 6)).astype("f4")
        paint[3:7, 5:9] = 0
        state = np.empty((*self.shape, 4), dtype="f4")
        state[..., 0] = rng.uniform(0, 1, self.shape)
        state[..., 1:3] = state[..., :1] * (-0.4, 0.3)
        state[..., 3] = rng.uniform(0, 1, self.shape)
        origin = initial_origins(self.size, self.aspect, self.domain)
        for target in helper.origins:
            target.write(origin.tobytes())
        for target in helper.states:
            target.write(state.tobytes())
        pixel = 2 * self.domain / self.size[1]
        velocity = np.empty((*self.shape, 2), dtype="f4")
        velocity[:] = (pixel, -pixel * 0.5)
        upper, lower = self.packed(paint), self.packed(paint * 0.6)
        helper.transport(upper, lower, self.texture(velocity), 0.5)
        yy, xx = np.mgrid[: self.shape[0], : self.shape[1]]
        positions = np.stack((xx, yy), -1).astype("f8")
        for index, scale in enumerate((1, 0.5)):
            traced = positions - np.array([0.5, -0.25]) * scale
            mass = (paint if index == 0 else paint * 0.6).sum(-1)
            for textures, value in ((helper.origins, origin), (helper.states, state)):
                expected = mass_weighted_sample(value, mass, traced)
                np.testing.assert_allclose(
                    self.read(textures[index]), expected, atol=3e-6, rtol=1e-5
                )

    def test_maccormack_gpu_matches_reference_for_origins_and_bounded_state(self):
        rng = np.random.default_rng(241125)
        for count in (2, 6):
            with self.subTest(count=count):
                helper = self.helper({"advection": "maccormack"}, lower_scale=0.71)
                paint = rng.uniform(0.01, 1.3, (*self.shape, count)).astype("f4")
                paint[3:7, 5:9] = 0
                state = np.empty((*self.shape, 4), dtype="f4")
                state[..., 0] = rng.uniform(0, 1, self.shape)
                angle = rng.uniform(-np.pi, np.pi, self.shape)
                state[..., 1] = state[..., 0] * np.cos(angle)
                state[..., 2] = state[..., 0] * np.sin(angle)
                state[..., 3] = rng.uniform(0, 1, self.shape)
                origin = initial_origins(self.size, self.aspect, self.domain)
                for target in helper.origins:
                    target.write(origin.tobytes())
                for target in helper.states:
                    target.write(state.tobytes())
                pixel = 2 * self.domain / self.size[1]
                velocity = np.empty((*self.shape, 2), dtype="f4")
                velocity[..., 0] = pixel * (0.7 + 0.2 * origin[..., 1])
                velocity[..., 1] = pixel * (-0.3 + 0.1 * origin[..., 0])
                layers = self.packed(paint), self.packed(paint * 0.6)
                original_bytes = [tuple(texture.read() for texture in layer) for layer in layers]
                helper.transport(*layers, self.texture(velocity), 0.7)
                for index, scale in enumerate((1, 0.71)):
                    mass = (paint if index == 0 else paint * 0.6).sum(-1, dtype="f8")
                    expected_origin = maccormack_transport(
                        origin[..., :2], mass, velocity, pixel_size=pixel, dt=0.7 * scale
                    )
                    expected_state = maccormack_transport(
                        state, mass, velocity, pixel_size=pixel, dt=0.7 * scale, bound_state=True
                    )
                    actual_origin = self.read(helper.origins[index])
                    actual_state = self.read(helper.states[index])
                    np.testing.assert_allclose(
                        actual_origin[..., :2], expected_origin, atol=2e-5, rtol=4e-5
                    )
                    np.testing.assert_array_equal(actual_origin[..., 2:], 0)
                    np.testing.assert_allclose(actual_state, expected_state, atol=2e-5, rtol=4e-5)
                    self.assertEqual(
                        original_bytes[index], tuple(texture.read() for texture in layers[index])
                    )

    def test_explicit_linear_is_bit_identical_and_allocates_no_correction_scratch(self):
        rng = np.random.default_rng(76732)
        old = self.helper()
        linear = self.helper({"advection": "linear"})
        self.assertIsNone(old._forward)
        self.assertIsNone(linear._predictor)
        paint = rng.uniform(0.01, 1, (*self.shape, 2)).astype("f4")
        paint[2:6, 5:9] = 0
        layers = self.packed(paint), self.packed(paint * 0.2)
        velocity = self.texture(rng.uniform(-0.05, 0.05, (*self.shape, 2)).astype("f4"))
        carrier = self.texture(np.ones((*self.shape, 4), dtype="f4"))
        for helper in (old, linear):
            helper.initialize(*layers)
            for _ in range(3):
                helper.transport(*layers, velocity, 0.01)
                helper.update(*layers, carrier, velocity, 0.01)
        for a, b in zip(
            (*old.origins, *old.states), (*linear.origins, *linear.states), strict=True
        ):
            self.assertEqual(a.read(), b.read())

    def test_maccormack_gpu_preserves_translation_detail_better_than_linear(self):
        self.size, self.shape, self.aspect = (128, 64), (64, 128), 2.0
        yy, xx = np.mgrid[:64, :128]

        def analytic(x, y):
            return 0.5 + 0.22 * np.sin(x * 2 * np.pi / 16) + 0.14 * np.cos(y * 2 * np.pi / 24)

        state = np.zeros((*self.shape, 4), dtype="f4")
        state[..., 0] = 1
        state[..., 3] = analytic(xx, yy)
        paint = np.ones((*self.shape, 2), dtype="f4")
        layers = self.packed(paint), self.packed(paint * 0.3)
        velocity = np.empty((*self.shape, 2), dtype="f4")
        velocity[:] = np.array([0.35, -0.2]) * (2 * self.domain / self.size[1])
        velocity = self.texture(velocity)
        errors, contrasts = [], []
        interior = slice(16, -16), slice(24, -24)
        exact = analytic(xx - 40 * 0.35, yy + 40 * 0.2)[interior]
        for mode in ("linear", "maccormack"):
            helper = self.helper({"advection": mode})
            for texture in helper.states:
                texture.write(state.tobytes())
            for _ in range(40):
                helper.transport(*layers, velocity, 1.0)
            actual = self.read(helper.states[0])[..., 3][interior]
            errors.append(float(np.sqrt(np.mean((actual - exact) ** 2))))
            contrasts.append(float(actual.std()))
        self.assertLess(errors[1], errors[0] * 0.3)
        self.assertGreater(contrasts[1], float(exact.std()) * 0.95)
        self.assertLess(contrasts[0], float(exact.std()) * 0.7)

    def test_maccormack_engine_keeps_base_paint_exact_and_is_output_cadence_invariant(self):
        from tools.estuary.test_engine import SourceFixture

        from .engine import Engine
        from .palette import generate_palette

        palette = generate_palette(self.seed, chromatic_count=1)
        palette["layer_fractions"] = [0.6, 0.05]
        settings = {
            "material_model": "laminate",
            "initial_pattern": "scattered",
            "deposition": 0,
            "settling_scale": 0,
            "underpaint_strength": 0,
            "underpaint_release": 0,
            "burial_rate": 0,
            "resolution": [96, 72],
            "steps": 32,
            "carrier_velocity": [0, 0],
            "flow_strength": 0.8,
            "mass_budget_interval_steps": 7,
            "diffusion_coefficient": 0.00002,
        }
        engines = []
        for interaction in (None, {"advection": "maccormack"}, {"advection": "maccormack"}):
            engine = Engine(SourceFixture(), {**settings, "interaction": interaction}, palette, [])
            self.addCleanup(engine.close)
            engines.append(engine)
        baseline, whole, observed = engines
        baseline.advance_to(32)
        whole.advance_to(32)
        for step in (1, 9, 17, 24, 32):
            observed.advance_to(step)
            observed.snapshot()
        base_fields, whole_fields, observed_fields = [engine.snapshot() for engine in engines]
        for name in BASE_FIELDS:
            np.testing.assert_array_equal(whole_fields[name], base_fields[name], err_msg=name)
        for name in (*BASE_FIELDS, *FIELD_NAMES):
            np.testing.assert_array_equal(whole_fields[name], observed_fields[name], err_msg=name)


if __name__ == "__main__":
    unittest.main()
