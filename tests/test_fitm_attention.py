"""Numerical tests for document means, small temperature and mass conservation."""

import unittest

import torch

from omniaware.baselines.fitm_attention import (
    TEMPERATURE, calibrated_alpha, document_means, redistribute_document_attention,
)


class FitmAttentionTests(unittest.TestCase):
    def test_unequal_lengths_preserve_each_non_document_token_and_total_mass(self):
        log_attention = torch.tensor([.1, .1, .1, .1, .05, .15, .1, .3], dtype=torch.float64).log()
        documents = [[0, 1], [2, 3, 4, 5]]
        calibrated, statistics = redistribute_document_attention(log_attention, torch.tensor([.5, .5]).double(), documents)
        self.assertAlmostEqual(statistics["C"].item(), .2)
        torch.testing.assert_close(statistics["document_mass_after"], torch.tensor([.2, .4]).double())
        torch.testing.assert_close(statistics["document_mean_after"], torch.tensor([.1, .1]).double())
        self.assertTrue(torch.equal(calibrated[[6, 7]], log_attention.exp()[[6, 7]]))
        self.assertAlmostEqual(calibrated.sum().item(), 1.)
        self.assertAlmostEqual(calibrated[4].item() / calibrated[5].item(), 1 / 3)

    def test_temperature_is_stable_and_rows_layers_heads_stay_independent(self):
        original = torch.tensor([[[.9, .1, .3], [.1, .9, .3]],
                                 [[.3, .1, .9], [.4, .4, .4]]], dtype=torch.float64)
        reference = torch.zeros_like(original)
        actual = calibrated_alpha(original, reference)
        self.assertEqual(TEMPERATURE, .00005)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.sum(-1), torch.ones(2, 2).double())
        expected = torch.tensor([[[1., 0, 0], [0, 1., 0]], [[0, 0, 1.], [1 / 3, 1 / 3, 1 / 3]]]).double()
        torch.testing.assert_close(actual, expected)
        log_attention = torch.randn(2, 2, 9, generator=torch.Generator().manual_seed(3), dtype=torch.float64).log_softmax(-1)
        documents = [[0], [1, 2], [3, 4, 5]]
        calibrated, stats = redistribute_document_attention(log_attention, actual, documents)
        for layer in range(2):
            for head in range(2):
                single, _ = redistribute_document_attention(log_attention[layer, head], actual[layer, head], documents)
                torch.testing.assert_close(single, calibrated[layer, head], rtol=0, atol=0)
        torch.testing.assert_close(stats["document_mass_after"].sum(-1), stats["document_mass_before"].sum(-1))
        self.assertTrue(torch.equal(calibrated[..., 6:], log_attention.exp()[..., 6:]))

    def test_identity_target_and_underflowed_original_document(self):
        documents = [[0, 1], [2, 3, 4]]
        log_attention = torch.randn(2, 4, 1, 7, generator=torch.Generator().manual_seed(5), dtype=torch.float64).log_softmax(-1)
        means = document_means(log_attention, documents)
        identity, _ = redistribute_document_attention(log_attention, means / means.sum(-1, keepdim=True), documents)
        torch.testing.assert_close(identity, log_attention.exp(), rtol=1e-14, atol=1e-16)
        extreme = torch.tensor([-1000., -1001., 1., 2., 3., 0., -torch.inf]).log_softmax(-1)
        calibrated, stats = redistribute_document_attention(extreme, torch.tensor([.5, .5]), documents)
        self.assertTrue(torch.isfinite(calibrated).all())
        self.assertGreater(calibrated[:2].sum().item(), 0.)
        self.assertEqual(calibrated[-1].item(), 0.)
        self.assertEqual(calibrated[5].item(), extreme.exp()[5].item())
        torch.testing.assert_close(stats["document_mass_before"].sum(), stats["document_mass_after"].sum())


if __name__ == "__main__":
    unittest.main()
