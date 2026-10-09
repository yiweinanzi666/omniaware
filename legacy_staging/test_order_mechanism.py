"""T17 clustered statistics and tokenizer-only prompt checks; no model loads."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoTokenizer

from . import run_order_mechanism as runner
from . import score_order_mechanism as scoring


def synthetic_records():
    orders = runner.permutation_schedule()
    config = dict(samples=2, methods=list(runner.METHODS), orders=orders, max_new_tokens=128,
                  cost_boundary="synthetic separated observation and generation costs")
    observations = [dict(qid=f"q{q}", observation_seconds=.5, input_tokens=7) for q in range(2)]
    rows = []
    random_index = {order["order_id"]: i for i, order in enumerate(orders[2:])}
    for q in range(2):
        for order_index, order in enumerate(orders):
            for method_index, method in enumerate(runner.METHODS):
                if order["kind"] == "random":
                    i = random_index[order["order_id"]]
                    value = ([.2, .6][(i+q) % 2] if method == "static" else
                             .3+.2*q+.01*i if method == "replication" else .5+.2*q)
                else:
                    value = (.8-.2*q if order["order_id"] == "sorted" else .4-.2*q)
                inputs = [1, q, order_index] + ([8, 8] if q and method == "replication" else [])
                tokens = [11, method_index if q else 0, 99]
                rows.append(dict(qid=f"q{q}", method=method, order_id=order["order_id"],
                    variant=method+"__"+order["order_id"], references=[f"reference{q}"], category="text",
                    prediction=f"answer-{q}-{method}-{order_index}",
                    input_token_ids=inputs, answer_token_ids=tokens, gate_enabled=bool(q),
                    input_tokens=len(inputs), output_tokens=len(tokens), ended_with_eos=True,
                    max_new_tokens=128, inference_seconds=1., reranker_seconds=.2, reorder_seconds=.1,
                    observation_seconds=.5, measured_method_seconds=1.3+(.5 if method != "static" else 0),
                    generation_calls=1, acc_approx_x_normalized_bart=value, acc_approx=value+.1,
                    normalized_bart_score=value/(value+.1)))
    return rows, config, observations


class OrderStatisticsTests(unittest.TestCase):
    def test_random_means_dataset_std_anchors_seed_groups_and_costs(self):
        rows, config, observations = synthetic_records()
        result = scoring.summarize(rows, config, observations, resamples=200)
        primary = scoring.METRICS[0]
        static = result["methods"]["static"]
        self.assertAlmostEqual(static["random"][primary]["mean"], .4)
        self.assertAlmostEqual(static["random"][primary]["dataset_order_std"], 0.)
        self.assertAlmostEqual(static["anchors"][primary]["signed_gap"], .4)
        self.assertAlmostEqual(static["anchors"][primary]["absolute_gap"], .4)
        self.assertEqual(set(static["per_seed"]), {"17", "23", "29"})
        self.assertTrue(all(len(row["order_ids"]) == 4 for row in static["per_seed"].values()))
        comparison = result["paired_comparisons"]["paper-eab_minus_static"][primary]["mean"]
        self.assertAlmostEqual(comparison["mean_difference"], .2)
        self.assertEqual(set(comparison["question_differences"]), {"q0", "q1"})
        self.assertEqual(comparison["wins"], 2)
        self.assertAlmostEqual(static["all_orders"]["mean_measured_method_seconds"], 1.3)
        self.assertAlmostEqual(result["methods"]["replication"]["all_orders"]["mean_measured_method_seconds"], 1.8)
        self.assertEqual(result["checks"]["method_cost_mismatches"], [])
        self.assertEqual(result["actual_workload"]["generation_calls"], 84)
        self.assertEqual(result["actual_workload"]["observation_calls"], 2)
        self.assertEqual(result["actual_workload"]["generation_output_tokens"], 252)
        self.assertEqual(result["actual_workload"]["observation_input_tokens"], 14)
        self.assertTrue(result["checks"]["static_eab_input_and_first_token_match"])
        self.assertTrue(result["checks"]["gate_off_all_tokens_match"])

    def test_order_std_bootstrap_resamples_whole_questions_and_recomputes_dataset_means(self):
        baseline = np.array([[0, 1]*6, [1, 0]*6], dtype=float)
        candidate = np.full_like(baseline, .5)
        result = scoring.order_std_statistics(baseline, candidate, resamples=1000, seed=17)
        self.assertEqual(result["baseline_std"], 0.)
        self.assertEqual(result["candidate_std"], 0.)
        self.assertEqual(result["difference"], 0.)
        # A bootstrap cluster sample is either both distinct questions or one
        # question twice: baseline's order std is exactly 0 or .5, not 24 IID cells.
        self.assertEqual(result["ci95"], [-.5, 0.])
        constant_rows = np.array([[0, 1]*6]*2, dtype=float)
        exact = scoring.order_std_statistics(constant_rows, candidate, resamples=100, seed=23)
        self.assertEqual(exact["difference"], -.5)
        self.assertEqual(exact["ci95"], [-.5, -.5])

    def test_failures_tokens_and_same_answer_numerical_differences_are_retained(self):
        rows, config, observations = synthetic_records()
        target = rows[0]
        target.update(prediction="", ended_with_eos=False, output_tokens=128,
                      answer_token_ids=[11]*128)
        # Same answer text with different automatic scores must remain visible.
        rows[3]["prediction"] = rows[6]["prediction"] = "identical answer"
        rows[3]["acc_approx_x_normalized_bart"] = .21
        rows[6]["acc_approx_x_normalized_bart"] = .22
        rows[2]["input_token_ids"] = [999]
        rows[2]["answer_token_ids"] = [999]
        result = scoring.summarize(rows, config, observations, resamples=100)
        self.assertEqual(result["prediction_count"], 84)
        self.assertEqual(result["methods"]["static"]["all_orders"]["empty_prediction_count"], 1)
        self.assertEqual(result["methods"]["static"]["all_orders"]["token_limit_without_eos_count"], 1)
        checks = result["checks"]
        self.assertFalse(checks["static_eab_input_and_first_token_match"])
        self.assertFalse(checks["gate_off_all_tokens_match"])
        self.assertEqual(checks["static_eab_input_mismatches"], [dict(qid="q0", order_id="sorted")])
        differing = next(row for row in checks["same_answer_score_differences"] if row["prediction"] == "identical answer")
        self.assertAlmostEqual(differing["metrics"][scoring.METRICS[0]]["range"], .01)
        self.assertEqual(len(differing["records"]), 2)

    def test_unbalanced_or_duplicate_rows_cannot_change_denominator_silently(self):
        rows, config, observations = synthetic_records()
        for invalid in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError):
                scoring.summarize(invalid, config, observations, resamples=10)
        with self.assertRaises(ValueError):
            scoring.summarize(rows, config, observations+observations[:1], resamples=10)

    def test_score_entrypoint_reuses_score_records_and_writes_all_rows(self):
        rows, config, observations = synthetic_records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, values in (("predictions.jsonl", rows), ("observations.jsonl", observations)):
                (root/name).write_text("".join(json.dumps(row)+"\n" for row in values))
            with patch.object(scoring, "load_official_metric", return_value=("lexical", {})), \
                 patch.object(scoring, "load_parabank", return_value=("bart", {})), \
                 patch.object(scoring, "score_records", side_effect=lambda records, *args: records) as call, \
                 patch.object(scoring.torch, "set_num_threads"):
                result = scoring.score(root, config)
            self.assertEqual(len(call.call_args.args[0]), 84)
            self.assertEqual(call.call_args.args[1:], ("lexical", "bart"))
            self.assertEqual(result["prediction_count"], 84)
            self.assertEqual(len((root/"scored/scored_predictions.jsonl").read_text().splitlines()), 84)
            self.assertTrue((root/"scored/summary.json").exists())


class OrderPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(runner.MODEL, local_files_only=True)

    def original(self):
        blocks = []
        for index in range(5):
            text = ("retained "*300 if index == 0 else "Source text. ") + f"TAIL-{index}"
            blocks.append(dict(fact_id=f"fact-{index}", text=text, original_rank=index+1,
                block=f"[Evidence {index+1}]\n{text}\n[/Evidence {index+1}]", truncated=False))
        return dict(qid="synthetic", question="Which fact answers this?", evidence_blocks=blocks,
                    evidence_sha256="stored-source", references=["GOLD_SENTINEL"])

    def test_permutation_schedule_has_twelve_unique_random_orders_and_two_anchors(self):
        schedule = runner.permutation_schedule()
        self.assertEqual(schedule, runner.permutation_schedule())
        self.assertEqual(len(schedule), 14)
        self.assertEqual(len({tuple(row["permutation"]) for row in schedule}), 14)
        self.assertEqual(schedule[0]["permutation"], list(range(5)))
        self.assertEqual(schedule[1]["permutation"], list(reversed(range(5))))
        for seed in (17, 23, 29):
            self.assertEqual(sum(row["seed"] == seed for row in schedule), 4)
        self.assertTrue(all(sorted(row["permutation"]) == list(range(5)) for row in schedule))

    def test_whole_blocks_labels_tail_and_map_follow_permutation_without_retruncation(self):
        original = self.original()
        before = copy.deepcopy(original)
        order = runner.permutation_schedule()[2]
        relevance = [0., .2, .5, .8, 1.]
        observation = dict(u_initial=1., gate_enabled=True)
        _, static_map, static = runner.prepare_input(self.tokenizer, original, order, "static", relevance, observation)
        _, eab_map, candidate = runner.prepare_input(self.tokenizer, original, order, "paper-eab", relevance, observation)
        self.assertEqual(static["input_token_ids"], candidate["input_token_ids"])
        self.assertTrue(torch.equal(static_map, eab_map))
        self.assertEqual(static["evidence_ids"], [f"fact-{i}" for i in order["permutation"]])
        expected = "\n\n".join([original["evidence_blocks"][i]["block"] for i in order["permutation"]]
                                 +["Question: "+original["question"]])
        self.assertEqual(static["user_content"], expected)
        self.assertNotIn("GOLD_SENTINEL", static["user_content"])
        self.assertGreater(len(self.tokenizer.encode(original["evidence_blocks"][0]["text"])), 256)
        self.assertIn("TAIL-0", static["user_content"])
        encoded = self.tokenizer(static["rendered_prompt"], add_special_tokens=False,
                                 return_offsets_mapping=True, return_tensors="pt")
        self.assertEqual(encoded["input_ids"][0].tolist(), static["input_token_ids"])
        offsets = encoded["offset_mapping"][0].tolist()
        expected_map = torch.zeros_like(static_map)
        cursor = 0
        for rank in order["permutation"]:
            block = original["evidence_blocks"][rank]["block"]
            left = static["rendered_prompt"].index(block, cursor)
            right = left+len(block)
            for token, (start, end) in enumerate(offsets):
                if start < right and end > left:
                    expected_map[0, token] = relevance[rank]
            cursor = right
        self.assertTrue(torch.equal(static_map, expected_map))
        self.assertEqual(original, before)

    def test_replication_ceil_gate_boundaries_and_adjacent_full_copies(self):
        self.assertEqual(runner.replication_counts([0., .5, 1.], 1., True), [1, 2, 4])
        self.assertEqual(runner.replication_counts([0., .5, 1.], 0., True), [1, 1, 1])
        self.assertEqual(runner.replication_counts([0., .5, 1.], 1., False), [1, 1, 1])
        self.assertEqual(runner.replication_counts([0., 1e-6], 1., True), [1, 2])
        original, order = self.original(), runner.permutation_schedule()[1]
        relevance = [0., .2, .5, .8, 1.]
        observation = dict(u_initial=1., gate_enabled=True)
        _, weights, trace = runner.prepare_input(self.tokenizer, original, order, "replication", relevance, observation)
        counts = runner.replication_counts(relevance, 1., True)
        expanded = [rank for rank in order["permutation"] for _ in range(counts[rank])]
        self.assertEqual(trace["evidence_ids"], [f"fact-{rank}" for rank in expanded])
        self.assertEqual(trace["per_original_block_copies"], counts)
        self.assertEqual(len(trace["evidence_token_counts"]), len(expanded))
        self.assertEqual(weights.shape[-1], trace["input_tokens"])
        for rank in range(5):
            self.assertEqual(trace["user_content"].count(original["evidence_blocks"][rank]["block"]), counts[rank])
            positions = [row for row in trace["block_positions"] if row["original_rank"] == rank+1]
            self.assertEqual([row["copy_index"] for row in positions], list(range(1, counts[rank]+1)))
        off = dict(u_initial=1., gate_enabled=False)
        traces = [runner.prepare_input(self.tokenizer, original, order, method, relevance, off)[2]
                  for method in runner.METHODS]
        self.assertTrue(all(t["input_token_ids"] == traces[0]["input_token_ids"] for t in traces))


if __name__ == "__main__":
    unittest.main()
