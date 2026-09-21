"""Deterministic cache eviction races without serializing expensive planning."""

from __future__ import annotations

import copy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, get_ident
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

from . import choreography as planner
from . import test_choreography as fixtures
from .test_choreography import config, prepared, source

_SOURCE_DATA = planner._source_data


class ChoreographyCacheTests(TestCase):
    setUp = fixtures.ChoreographyContracts.setUp
    plan = fixtures.ChoreographyContracts.plan

    def racing_hit(self, name, key, reader, *, copy_outside_lock=False):
        """Pause a cache lookup while another thread attempts eviction.

        The unlocked old contains/get path deterministically loses its key.
        With locking, eviction waits for the lookup/touch transaction and the
        reader retains a valid reference even after that entry is removed.
        """
        ready, attempted, evicted = Event(), Event(), Event()
        lock = planner._CACHE_LOCK
        reader_ident = None

        class PausedCache(OrderedDict):
            def pause(self, candidate, found):
                if candidate == key and found and get_ident() == reader_ident:
                    locked = lock.locked()
                    ready.set()
                    if not attempted.wait(3):
                        raise AssertionError("Eviction worker did not attempt its transaction")
                    if not locked and not evicted.wait(3):
                        raise AssertionError("Unlocked eviction did not finish")

            def __contains__(self, candidate):
                found = super().__contains__(candidate)
                self.pause(candidate, found)
                return found

            def get(self, candidate, default=None):
                value = super().get(candidate, default)
                self.pause(candidate, value is not None)
                return value

        cache = PausedCache(getattr(planner, name))
        stored = OrderedDict.__getitem__(cache, key)
        real_copy = copy.deepcopy

        def checked_copy(value, *args, **kwargs):
            if copy_outside_lock and value is stored and not evicted.wait(3):
                raise AssertionError("Copying held the cache lock and blocked eviction")
            return real_copy(value, *args, **kwargs)

        def evict():
            attempted.set()
            with lock:
                cache.pop(key)
            evicted.set()

        def read_cached():
            nonlocal reader_ident
            reader_ident = get_ident()
            return reader()

        with (
            patch.object(planner, name, cache),
            patch.object(planner.copy, "deepcopy", side_effect=checked_copy),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            read = pool.submit(read_cached)
            self.assertTrue(ready.wait(3), "Reader did not reach its cached lookup")
            write = pool.submit(evict)
            result = read.result(timeout=5)
            write.result(timeout=5)
            self.assertNotIn(key, cache)
        return result, stored

    def test_layout_hit_survives_eviction_and_copies_outside_lock(self):
        expected = self.plan()
        key = next(iter(planner._LAYOUTS))
        actual, stored = self.racing_hit("_LAYOUTS", key, self.plan, copy_outside_lock=True)
        self.assertEqual(actual, expected)
        self.assertIsNot(actual, stored)

    def source_fixture(self, sample_barrier=None):
        value = source()

        def sample(fractions):
            if sample_barrier is not None:
                sample_barrier.wait(timeout=3)
            from types import SimpleNamespace

            return SimpleNamespace(
                positions=np.tile([[0.2, 0], [-0.2, 0], [0, 0.2]], (len(fractions), 1, 1)),
                velocities=np.tile([[0.1, 0], [0, 0.1], [-0.1, -0.1]], (len(fractions), 1, 1)),
                pair_distances=np.ones((len(fractions), 3)),
            )

        value.sample = sample
        return value

    def safe_anchors(self):
        return patch.object(
            planner.engaged,
            "plan_engaged_layout",
            return_value={"pools": [{"position": point.tolist()} for point in prepared()[4]]},
        )

    def test_prepared_hit_survives_eviction_after_lookup(self):
        origin = self.source_fixture()
        with self.safe_anchors():
            expected = _SOURCE_DATA(origin, config())
            actual, stored = self.racing_hit(
                "_PREPARED", expected[-1], lambda: _SOURCE_DATA(origin, config())
            )
        self.assertIs(actual, expected)
        self.assertIs(actual, stored)

    def test_duplicate_preparation_does_not_evict_an_unrelated_entry(self):
        original_keys = [f"prepared-{index}" for index in range(11)]
        cache = dict.fromkeys(original_keys, ())
        origin = self.source_fixture(Barrier(2))
        with (
            patch.object(planner, "_PREPARED", cache),
            self.safe_anchors(),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            futures = [pool.submit(_SOURCE_DATA, origin, config()) for _ in range(2)]
            results = [future.result(timeout=5) for future in futures]
        self.assertEqual(results[0][-1], results[1][-1])
        self.assertEqual(set(cache), {*original_keys, results[0][-1]})
        self.assertEqual(len(cache), 12)
        for left, right in zip(results[0][1:3], results[1][1:3], strict=True):
            np.testing.assert_array_equal(left, right)

    def test_parallel_pilots_publish_bounded_cache_without_changing_layouts(self):
        setups = ("active-pools", "compact-pools")
        expected = [self.plan(setup) for setup in setups]
        barrier = Barrier(2)
        real_pilot = planner._pilot

        def simultaneous_pilot(*args, **kwargs):
            barrier.wait(timeout=3)
            return real_pilot(*args, **kwargs)

        cache = OrderedDict((f"layout-{i}", {}) for i in range(127))
        with (
            patch.object(planner, "_LAYOUTS", cache),
            patch.object(planner, "_pilot", side_effect=simultaneous_pilot),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            futures = [pool.submit(self.plan, setup) for setup in setups]
            actual = [future.result(timeout=5) for future in futures]
        self.assertEqual(actual, expected)
        self.assertEqual(len(cache), 128)
        self.assertEqual(sum(key.startswith("layout-") for key in cache), 126)


if __name__ == "__main__":
    main()
