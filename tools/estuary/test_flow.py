"""Independent derivative and conservation checks for the kinematic flow."""

from __future__ import annotations

import unittest

import numpy as np

from tools.estuary.flow_reference import streamfunction, velocity


class FlowReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = np.array(
            [[0.3, 0.2, -2.5, 0.4], [-0.7, -0.4, 1.2, -0.5], [0.4, -0.3, 1.3, 0.1]]
        )
        self.pairs = np.array([[-0.2, -0.1, 2.2], [0.35, -0.05, -1.7], [-0.15, -0.35, 0.3]])
        self.settings = dict(aspect=1.6, stir_radius=0.25, flow_strength=0.9, pair_swirl=0.35)

    def test_analytic_velocity_is_curl_of_complete_potential(self) -> None:
        generator = np.random.default_rng(901)
        points = generator.uniform([-1.58, -0.98], [1.58, 0.98], (1000, 2))
        h = 2e-6
        dx = np.array([h, 0])
        dy = np.array([0, h])
        psi_x = (
            streamfunction(points + dx, self.tools, self.pairs, **self.settings)
            - streamfunction(points - dx, self.tools, self.pairs, **self.settings)
        ) / (2 * h)
        psi_y = (
            streamfunction(points + dy, self.tools, self.pairs, **self.settings)
            - streamfunction(points - dy, self.tools, self.pairs, **self.settings)
        ) / (2 * h)
        np.testing.assert_allclose(
            velocity(points, self.tools, self.pairs, **self.settings),
            np.stack((psi_y, -psi_x), axis=-1),
            atol=4e-10,
            rtol=2e-7,
        )

    def test_divergence_is_zero_through_interior_and_near_boundary(self) -> None:
        generator = np.random.default_rng(902)
        points = generator.uniform([-1.59, -0.99], [1.59, 0.99], (1000, 2))
        h = 2e-5
        dx = np.array([h, 0])
        dy = np.array([0, h])
        divergence = (
            velocity(points + dx, self.tools, self.pairs, **self.settings)[..., 0]
            - velocity(points - dx, self.tools, self.pairs, **self.settings)[..., 0]
            + velocity(points + dy, self.tools, self.pairs, **self.settings)[..., 1]
            - velocity(points - dy, self.tools, self.pairs, **self.settings)[..., 1]
        ) / (2 * h)
        self.assertLess(np.max(np.abs(divergence)), 5e-8)

    def test_guard_band_domain_preserves_curl_and_divergence(self) -> None:
        settings = {**self.settings, "domain_scale": 1.6}
        generator = np.random.default_rng(903)
        points = generator.uniform([-2.54, -1.58], [2.54, 1.58], (1000, 2))
        h = 2e-6
        derivatives = []
        for offset in (np.array([h, 0]), np.array([0, h])):
            derivatives.append(
                (
                    streamfunction(points + offset, self.tools, self.pairs, **settings)
                    - streamfunction(points - offset, self.tools, self.pairs, **settings)
                )
                / (2 * h)
            )
        np.testing.assert_allclose(
            velocity(points, self.tools, self.pairs, **settings),
            np.stack((derivatives[1], -derivatives[0]), axis=-1),
            atol=4e-10,
            rtol=2e-7,
        )
        divergence = np.zeros(len(points))
        for component, offset in enumerate((np.array([h, 0]), np.array([0, h]))):
            divergence += (
                velocity(points + offset, self.tools, self.pairs, **settings)[..., component]
                - velocity(points - offset, self.tools, self.pairs, **settings)[..., component]
            ) / (2 * h)
        self.assertLess(np.max(np.abs(divergence)), 5e-8)

    def test_guard_band_stops_at_expanded_boundary_without_rescaling_source(self) -> None:
        settings = {**self.settings, "domain_scale": 1.6}
        half_width = settings["aspect"] * settings["domain_scale"]
        boundary = np.array([[half_width, 0.3], [-half_width, -0.3], [0.3, 1.6], [-0.3, -1.6]])
        np.testing.assert_array_equal(
            velocity(boundary, self.tools, self.pairs, **settings), np.zeros_like(boundary)
        )
        # The visible canvas edge is now an interior location, with live flow.
        self.assertGreater(
            np.linalg.norm(velocity([0, 1], self.tools, self.pairs, **settings)), 1e-8
        )
        tools = np.zeros((3, 4))
        tools[0, 2:] = [2, -3]
        np.testing.assert_allclose(
            velocity([0, 0], tools, np.zeros((3, 3)), **settings),
            np.array([2, -3]) * settings["flow_strength"],
        )

    def test_authored_carrier_is_included_in_complete_streamfunction(self) -> None:
        settings = {**self.settings, "domain_scale": 1.6, "carrier_velocity": (2.0, 0.2)}
        generator = np.random.default_rng(904)
        points = generator.uniform([-2.54, -1.58], [2.54, 1.58], (1000, 2))
        h = 2e-6
        derivatives = []
        divergence = np.zeros(len(points))
        for component, offset in enumerate((np.array([h, 0]), np.array([0, h]))):
            derivatives.append(
                (
                    streamfunction(points + offset, self.tools, self.pairs, **settings)
                    - streamfunction(points - offset, self.tools, self.pairs, **settings)
                )
                / (2 * h)
            )
            divergence += (
                velocity(points + offset, self.tools, self.pairs, **settings)[..., component]
                - velocity(points - offset, self.tools, self.pairs, **settings)[..., component]
            ) / (2 * h)
        np.testing.assert_allclose(
            velocity(points, self.tools, self.pairs, **settings),
            np.stack((derivatives[1], -derivatives[0]), axis=-1),
            atol=8e-10,
            rtol=2e-7,
        )
        self.assertLess(np.max(np.abs(divergence)), 5e-8)
        half_width = settings["aspect"] * settings["domain_scale"]
        boundary = np.array([[half_width, 0.3], [-half_width, -0.3], [0.3, 1.6], [-0.3, -1.6]])
        np.testing.assert_array_equal(
            velocity(boundary, self.tools, self.pairs, **settings), np.zeros_like(boundary)
        )

    def test_carrier_has_authored_velocity_at_origin(self) -> None:
        settings = {**self.settings, "domain_scale": 1.6, "carrier_velocity": (2.0, 0.2)}
        np.testing.assert_array_equal(
            velocity([0, 0], np.zeros((3, 4)), np.zeros((3, 3)), **settings), [2.0, 0.2]
        )

    def test_boundary_has_zero_flow_in_both_components(self) -> None:
        x = np.linspace(-1.6, 1.6, 101)
        y = np.linspace(-1, 1, 101)
        points = np.concatenate(
            (
                np.column_stack((x, np.ones_like(x))),
                np.column_stack((x, -np.ones_like(x))),
                np.column_stack((np.full_like(y, 1.6), y)),
                np.column_stack((np.full_like(y, -1.6), y)),
            )
        )
        np.testing.assert_array_equal(
            velocity(points, self.tools, self.pairs, **self.settings), np.zeros_like(points)
        )
        np.testing.assert_array_equal(
            streamfunction(points, self.tools, self.pairs, **self.settings), np.zeros(len(points))
        )

    def test_field_extends_as_zero_outside_painting(self) -> None:
        points = np.array([[2, 0], [-2, 0], [0, 1.1], [0, -1.1], [2, 1.1], [-2, -1.1]])
        np.testing.assert_array_equal(
            velocity(points, self.tools, self.pairs, **self.settings), np.zeros_like(points)
        )

    def test_stationary_tools_and_pairs_generate_no_flow(self) -> None:
        tools = self.tools.copy()
        tools[:, 2:] = 0
        pairs = self.pairs.copy()
        pairs[:, 2] = 0
        points = np.array([[0.1, 0.2], [-0.7, 0.3]])
        np.testing.assert_array_equal(
            velocity(points, tools, pairs, **self.settings), np.zeros_like(points)
        )

    def test_flow_is_linear_in_source_velocities_and_spins(self) -> None:
        points = np.array([[0.1, 0.2], [-0.7, 0.3]])
        tools = self.tools.copy()
        tools[:, 2:] *= 0.3
        pairs = self.pairs.copy()
        pairs[:, 2] *= 0.3
        np.testing.assert_allclose(
            velocity(points, tools, pairs, **self.settings),
            velocity(points, self.tools, self.pairs, **self.settings) * 0.3,
            atol=1e-15,
        )

    def test_single_centered_dipole_moves_with_tool_at_center(self) -> None:
        tools = np.zeros((3, 4))
        tools[0, 2:] = [2, -3]
        settings = {**self.settings, "flow_strength": 0.7}
        np.testing.assert_allclose(
            velocity([0, 0], tools, np.zeros((3, 3)), **settings), [1.4, -2.1]
        )

    def test_pair_positive_spin_moves_counterclockwise(self) -> None:
        pairs = np.zeros((3, 3))
        pairs[0, 2] = 1
        points = np.array([[0.1, 0], [-0.1, 0], [0, 0.1], [0, -0.1]])
        result = velocity(points, np.zeros((3, 4)), pairs, **self.settings)
        self.assertGreater(result[0, 1], 0)
        self.assertLess(result[1, 1], 0)
        self.assertLess(result[2, 0], 0)
        self.assertGreater(result[3, 0], 0)

    def test_invalid_input_is_rejected(self) -> None:
        invalid = (
            {"aspect": 0},
            {"domain_scale": 0},
            {"domain_scale": np.inf},
            {"carrier_velocity": (1, np.inf)},
            {"carrier_velocity": (1, 2, 3)},
            {"stir_radius": -0.1},
            {"pair_swirl": np.inf},
            {"flow_strength": np.nan},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                velocity([0, 0], self.tools, self.pairs, **{**self.settings, **changes})
        for points in ([0], [0, np.inf], 2):
            with self.subTest(points=points), self.assertRaises(ValueError):
                velocity(points, self.tools, self.pairs, **self.settings)
        with self.assertRaises(ValueError):
            velocity([0, 0], np.zeros((2, 4)), self.pairs, **self.settings)
        with self.assertRaises(ValueError):
            velocity([0, 0], self.tools, np.zeros((2, 3)), **self.settings)


if __name__ == "__main__":
    unittest.main()
