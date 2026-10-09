"""Order statistics, permutation schedule and replication counts (verbatim legacy tests)."""
import unittest

import numpy as np

from omniaware import order


class OrderTests(unittest.TestCase):
    def test_order_std_bootstrap_resamples_whole_questions_and_recomputes_dataset_means(self):
        baseline = np.array([[0, 1]*6, [1, 0]*6], dtype=float)
        candidate = np.full_like(baseline, .5)
        result = order.order_std_statistics(baseline, candidate, resamples=1000, seed=17)
        self.assertEqual(result["baseline_std"], 0.)
        self.assertEqual(result["candidate_std"], 0.)
        self.assertEqual(result["difference"], 0.)
        # A bootstrap cluster sample is either both distinct questions or one
        # question twice: baseline's order std is exactly 0 or .5, not 24 IID cells.
        self.assertEqual(result["ci95"], [-.5, 0.])
        constant_rows = np.array([[0, 1]*6]*2, dtype=float)
        exact = order.order_std_statistics(constant_rows, candidate, resamples=100, seed=23)
        self.assertEqual(exact["difference"], -.5)
        self.assertEqual(exact["ci95"], [-.5, -.5])

    def test_permutation_schedule_has_twelve_unique_random_orders_and_two_anchors(self):
        schedule = order.permutation_schedule()
        self.assertEqual(schedule, order.permutation_schedule())
        self.assertEqual(len(schedule), 14)
        self.assertEqual(len({tuple(row["permutation"]) for row in schedule}), 14)
        self.assertEqual(schedule[0]["permutation"], list(range(5)))
        self.assertEqual(schedule[1]["permutation"], list(reversed(range(5))))
        for seed in (17, 23, 29):
            self.assertEqual(sum(row["seed"] == seed for row in schedule), 4)
        self.assertTrue(all(sorted(row["permutation"]) == list(range(5)) for row in schedule))

    def test_replication_ceil_gate_boundaries_and_adjacent_full_copies(self):
        self.assertEqual(order.replication_counts([0., .5, 1.], 1., True), [1, 2, 4])
        self.assertEqual(order.replication_counts([0., .5, 1.], 0., True), [1, 1, 1])
        self.assertEqual(order.replication_counts([0., .5, 1.], 1., False), [1, 1, 1])
        self.assertEqual(order.replication_counts([0., 1e-6], 1., True), [1, 2])
