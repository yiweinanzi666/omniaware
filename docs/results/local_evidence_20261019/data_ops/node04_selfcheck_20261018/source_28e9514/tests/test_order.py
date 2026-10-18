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

    def test_replication_rounds_gamma_u_r_gate_boundaries(self):
        # round(exp(1.2 * 1 * r)): r=0 -> 1; r=0.5 -> round(1.82) = 2; r=1 -> round(3.32) = 3
        self.assertEqual(order.replication_counts([0., .5, 1.], 1., True), [1, 2, 3])
        self.assertEqual(order.replication_counts([0., .5, 1.], 0., True), [1, 1, 1])
        self.assertEqual(order.replication_counts([0., .5, 1.], 1., False), [1, 1, 1])
        # A tiny positive bias no longer doubles the chunk (the old ceiling gave 2 copies here).
        self.assertEqual(order.replication_counts([0., 1e-6], 1., True), [1, 1])

    def test_replication_log_copies_track_the_bias(self):
        import math
        relevance = np.linspace(0.0, 4.0, 401)              # gamma * U * r = 1.2 * r with U = 1
        counts = np.array(order.replication_counts(list(relevance), 1., True, gamma=1.2))
        self.assertTrue(np.all(counts >= 1))
        self.assertLessEqual(np.max(np.abs(np.log(counts) - 1.2 * relevance)), math.log(1.5) + 1e-9)
