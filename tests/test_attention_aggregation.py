"""Smoke test Bab IV.5.2.2-4.5.2.5: learned attention, propagasi variance,
dan reparameterization di level parent."""

import torch

from hiprobcbm.models.attention_aggregation import (
    LearnedAttentionAggregator,
    UniformAggregator,
    reparameterize_parent,
)


def test_attention_weights_sum_to_one():
    B, K, d_c = 4, 5, 16
    agg = LearnedAttentionAggregator(concept_dim=d_c)
    mu_sub = torch.randn(B, K, d_c)

    alpha = agg.attention_scores(mu_sub)
    assert alpha.shape == (B, K)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_attention_respects_mask():
    B, K, d_c = 2, 4, 8
    agg = LearnedAttentionAggregator(concept_dim=d_c)
    mu_sub = torch.randn(B, K, d_c)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    alpha = agg.attention_scores(mu_sub, mask=mask)
    # Bobot pada posisi ter-mask harus (mendekati) nol.
    assert torch.allclose(alpha[~mask], torch.zeros_like(alpha[~mask]), atol=1e-6)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_variance_propagation_uses_squared_weights():
    """sigma_i^{2,p} = sum_k alpha_ik^2 * sigma_ik^2 (Eq. 4.5.2.4) - bukan
    weighted average biasa (sum alpha_ik * sigma_ik^2)."""
    B, K, d_c = 1, 2, 1
    agg = LearnedAttentionAggregator(concept_dim=d_c)

    mu_sub = torch.zeros(B, K, d_c)
    sigma2_sub = torch.tensor([[[1.0], [1.0]]])  # kedua subkonsep sigma^2=1

    # Paksa alpha = [0.5, 0.5] secara manual untuk verifikasi rumus persis.
    agg.attention_scores = lambda *a, **k: torch.tensor([[0.5, 0.5]])
    out = agg.aggregate(mu_sub, sigma2_sub)

    expected_variance_propagation = 0.5**2 * 1.0 + 0.5**2 * 1.0  # = 0.5
    expected_naive_average = 0.5 * 1.0 + 0.5 * 1.0  # = 1.0 (SALAH jika dipakai)

    assert torch.isclose(out["sigma2_parent"][0, 0], torch.tensor(expected_variance_propagation), atol=1e-6)
    assert not torch.isclose(out["sigma2_parent"][0, 0], torch.tensor(expected_naive_average), atol=1e-6)


def test_uniform_aggregator_equal_weights():
    B, K, d_c = 2, 4, 8
    agg = UniformAggregator(concept_dim=d_c)
    mu_sub = torch.randn(B, K, d_c)

    alpha = agg.attention_scores(mu_sub)
    assert torch.allclose(alpha, torch.full((B, K), 1.0 / K), atol=1e-6)


def test_reparameterize_parent_shape():
    B, d_c, S = 3, 16, 7
    mu = torch.randn(B, d_c)
    sigma2 = torch.rand(B, d_c) + 0.01

    z = reparameterize_parent(mu, sigma2, n_samples=S)
    assert z.shape == (B, S, d_c)


if __name__ == "__main__":
    test_attention_weights_sum_to_one()
    test_attention_respects_mask()
    test_variance_propagation_uses_squared_weights()
    test_uniform_aggregator_equal_weights()
    test_reparameterize_parent_shape()
    print("OK: test_attention_aggregation")
