"""Smoke test Bab IV.5.1: bentuk keluaran (mu, sigma^2) harus (B, C, d_c) -
memverifikasi perbaikan Temuan #1 (bukan (B, C))."""

import torch

from hiprobcbm.models.probabilistic_concepts import (
    ConceptExistenceScorer,
    ProbabilisticConceptPredictor,
    kl_to_standard_normal,
    reparameterize,
)


def test_predictor_output_shape():
    B, d_h, C, d_c = 4, 32, 5, 16
    predictor = ProbabilisticConceptPredictor(feature_dim=d_h, num_concepts=C, concept_dim=d_c)
    h = torch.randn(B, d_h)

    mu, sigma2 = predictor(h)

    assert mu.shape == (B, C, d_c)
    assert sigma2.shape == (B, C, d_c)
    assert (sigma2 > 0).all(), "sigma^2 harus positif (softplus)"


def test_reparameterize_shape_and_gradient():
    B, C, d_c, S = 2, 3, 8, 5
    mu = torch.randn(B, C, d_c, requires_grad=True)
    # `+ 0.1` sebelum requires_grad_() dilakukan agar sigma2 tetap leaf tensor
    # (kalau requires_grad diset dulu baru ditambah, hasilnya non-leaf dan
    # .grad tidak akan terisi walau backward berjalan benar).
    sigma2 = (torch.rand(B, C, d_c) + 0.1).requires_grad_()

    z = reparameterize(mu, sigma2, n_samples=S)
    assert z.shape == (B, C, S, d_c)

    z.sum().backward()
    assert mu.grad is not None and sigma2.grad is not None


def test_kl_zero_at_standard_normal():
    mu = torch.zeros(2, 3, 4)
    sigma2 = torch.ones(2, 3, 4)
    kl = kl_to_standard_normal(mu, sigma2)
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-5)


def test_kl_positive_when_shifted():
    mu = torch.ones(2, 3, 4) * 2.0
    sigma2 = torch.ones(2, 3, 4)
    kl = kl_to_standard_normal(mu, sigma2)
    assert kl.item() > 0


def test_existence_scorer_probability_range():
    B, C, d_c, S = 4, 3, 16, 8
    mu = torch.randn(B, C, d_c)
    sigma2 = torch.rand(B, C, d_c) + 0.05
    scorer = ConceptExistenceScorer(concept_dim=d_c)

    p = scorer.probability_from_distribution(mu, sigma2, n_samples=S)
    assert p.shape == (B, C)
    assert (p >= 0).all() and (p <= 1).all()


if __name__ == "__main__":
    test_predictor_output_shape()
    test_reparameterize_shape_and_gradient()
    test_kl_zero_at_standard_normal()
    test_kl_positive_when_shifted()
    test_existence_scorer_probability_range()
    print("OK: test_probabilistic_concepts")
