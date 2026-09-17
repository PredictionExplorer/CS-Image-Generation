"""Semantic transport tests against the actual compute shaders on a hardware GPU."""

import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from .flow_reference import velocity
from .recipe import validate_recipe


class SourceFixture:
    """Straight, unequal moving stirrers with known travel and 3D distances."""

    def frame(self, fraction):
        speeds = np.array([[0.3, 0.1], [-0.2, 0.3], [0.2, -0.2]])
        positions = np.array([[-0.5, -0.2], [0.2, 0.4], [0.3, -0.4]]) + fraction * speeds
        pairs = np.array(
            [np.linalg.norm(positions[b] - positions[a]) for a, b in ((0, 1), (1, 2), (2, 0))]
        )
        return SimpleNamespace(
            positions=positions,
            velocities=speeds,
            pair_distances=pairs,
            arc_lengths=np.linalg.norm(speeds, axis=1) * fraction,
        )


@unittest.skipUnless(
    os.environ.get("ESTUARY_TEST_GPU") == "1", "Requires a hardware OpenGL 4.3 context"
)
class EngineTests(unittest.TestCase):
    def setUp(self):
        from .engine import Engine

        self.recipe = validate_recipe(
            {
                "simulation": {
                    "resolution": [128, 96],
                    "steps": 360,
                    "deposition": 0,
                    "domain_scale": 1.6,
                    "carrier_velocity": [0.25, -0.1],
                },
                "render": {"resolution": [128, 96], "frames": 31},
            }
        )
        self.engine = Engine(SourceFixture(), self.recipe)
        self.addCleanup(self.engine.close)

    def test_shader_flow_matches_float64_stream_function_reference(self):
        from .engine import tool_uniforms

        engine = self.engine
        maximum = engine._flow(0.37)
        actual = np.frombuffer(engine.velocity.read(), dtype="f4").reshape(96, 128, 2)
        x = ((np.arange(128) + 0.5) / 128 * 2 - 1) * (128 / 96) * engine.domain
        y = ((np.arange(96) + 0.5) / 96 * 2 - 1) * engine.domain
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx, yy], axis=-1)
        tools, pairs = tool_uniforms(engine.source.frame(0.37), 0.2)
        expected = velocity(
            points,
            tools,
            pairs,
            aspect=128 / 96,
            stir_radius=0.2,
            flow_strength=0.7,
            pair_swirl=0.2,
            domain_scale=1.6,
            carrier_velocity=(0.25, -0.1),
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)
        self.assertAlmostEqual(maximum, float(np.linalg.norm(actual, axis=-1).max()), places=6)

    def test_nonzero_flow_preserves_a_constant_material_without_sources(self):
        state = np.zeros((96, 128, 4), dtype="f4")
        state[:, :, :3] = [0.2, 0.5, 0.9]
        self.engine.restore(state, 0)
        self.engine.advance_to(36)
        np.testing.assert_allclose(self.engine.read_state(), state, atol=3e-6, rtol=0)
        self.assertLessEqual(self.engine.maximum_courant, 1.500001)

    def test_output_requests_do_not_change_the_source_evolution(self):
        from .engine import Engine

        recipe = validate_recipe(
            {**self.recipe, "simulation": {**self.recipe["simulation"], "deposition": 1.0}}
        )
        self.engine.close()
        a = Engine(SourceFixture(), recipe)
        a.advance_to(24)
        a.render(128, 96)
        a.advance_to(48)
        expected = a.read_state()
        a.close()
        b = Engine(SourceFixture(), recipe)
        self.addCleanup(b.close)
        b.advance_to(48)
        np.testing.assert_array_equal(expected, b.read_state())

    def test_checkpoint_restore_reproduces_continuation_and_material_bounds(self):
        from .engine import Engine

        recipe = validate_recipe(
            {**self.recipe, "simulation": {**self.recipe["simulation"], "deposition": 1.0}}
        )
        self.engine.close()
        a = Engine(SourceFixture(), recipe)
        a.advance_to(24)
        state = a.read_state()
        counters = {"internal_steps": a.internal_steps, "maximum_courant": a.maximum_courant}
        a.advance_to(48)
        result = a.read_state()
        a.close()
        b = Engine(SourceFixture(), recipe)
        self.addCleanup(b.close)
        b.restore(state, 24, **counters)
        b.advance_to(48)
        np.testing.assert_array_equal(result, b.read_state())
        self.assertTrue(np.all(result >= 0))
        self.assertGreater(float(result.sum()), 0)
        reflected = b.render(128, 96)
        self.assertTrue(np.isfinite(reflected).all())
        self.assertTrue(np.all((reflected >= 0) & (reflected <= 1)))

    def test_invalid_checkpoint_and_backward_time_fail(self):
        with self.assertRaises(ValueError):
            self.engine.restore(np.zeros((96, 128, 4), dtype="f8"), 0)
        state = self.engine.read_state()
        state[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.engine.restore(state, 0)
        self.engine.advance_to(1)
        with self.assertRaises(ValueError):
            self.engine.advance_to(0)

    def test_interleaved_contexts_do_not_modify_each_others_paint(self):
        from .engine import Engine

        other = Engine(SourceFixture(), self.recipe)
        self.addCleanup(other.close)
        zero = self.engine.read_state()
        colored = zero.copy()
        colored[:, :, :3] = [0.2, 0.3, 0.4]
        other.restore(colored, 0)
        self.engine.advance_to(5)
        other.advance_to(5)
        np.testing.assert_array_equal(self.engine.read_state(), zero)
        np.testing.assert_allclose(other.read_state(), colored, atol=1e-6)


class AdaptiveClockTests(unittest.TestCase):
    def test_concentrated_motion_is_bounded_per_transport_not_only_on_average(self):
        from .engine import Engine

        engine = Engine.__new__(Engine)
        engine.recipe = validate_recipe({"simulation": {"steps": 360}, "render": {"frames": 31}})
        engine.step, engine.steps, engine.height = 0, 360, 768
        engine.domain = 1.0
        engine.ctx = nullcontext()
        engine.maximum_courant = 0.0

        def sample(time):
            arc = 0.1 * np.clip((time - 0.0006) / 0.00001, 0, 1)
            return SimpleNamespace(arc_lengths=np.full(3, arc))

        engine.source = SimpleNamespace(frame=sample)
        engine._flow = lambda _: 0
        distances = []
        engine._transport = lambda a, b: distances.append(
            float(sample(b).arc_lengths[0] - sample(a).arc_lengths[0])
        )
        engine.advance_to(1)
        self.assertLessEqual(max(distances), 0.035 * 0.4 + 1e-10)
        self.assertAlmostEqual(sum(distances), 0.1)


if __name__ == "__main__":
    unittest.main()
