"""Smoke test Bab IV.5.1.3-4.5.1.4: kedua varian SAE serta pipeline
discovery penuh (dari mean konsep -> pseudo subconcept label)."""

import torch

from hiprobcbm.models.sae import BatchTopKSAE, KLSparseAutoencoder, build_sae, train_sae
from hiprobcbm.models.subconcept_discovery import build_pseudo_hierarchy, discover_subconcepts_for_concept


def test_kl_sae_forward_shapes():
    N, d_c, latent = 20, 16, 64
    sae = KLSparseAutoencoder(input_dim=d_c, latent_dim=latent)
    x = torch.randn(N, d_c)

    out = sae(x)
    assert out.reconstruction.shape == (N, d_c)
    assert out.activations.shape == (N, latent)
    assert out.loss.item() >= 0


def test_batchtopk_sae_respects_sparsity():
    N, d_c, latent, k = 10, 16, 32, 4
    sae = BatchTopKSAE(input_dim=d_c, latent_dim=latent, top_k=k)
    x = torch.randn(N, d_c)

    out = sae(x)
    n_active = (out.activations > 0).sum().item()
    assert n_active <= k * N, "Jumlah aktivasi tidak boleh melebihi top_k * batch_size"


def test_build_sae_latent_dim_ratio():
    d_c = 16
    sae = build_sae("kl", input_dim=d_c, latent_dim_ratio=8)
    assert sae.encoder.out_features == d_c * 8

    sae2 = build_sae("batchtopk", input_dim=d_c)  # default ratio = 8
    assert sae2.W_enc.shape == (d_c, d_c * 8)


def test_train_sae_runs_and_returns_correct_shape():
    N, d_c = 32, 8
    features = torch.randn(N, d_c)
    sae = build_sae("kl", input_dim=d_c, latent_dim=16)

    activations = train_sae(sae, features, n_epochs=2, batch_size=8)
    assert activations.shape == (N, 16)


def test_discover_subconcepts_for_concept_end_to_end():
    torch.manual_seed(0)
    N, d_c = 50, 8
    mu_concept = torch.randn(N, d_c)
    presence_mask = torch.rand(N) > 0.4  # sebagian sampel "mengandung" konsep ini

    result = discover_subconcepts_for_concept(
        concept_index=0,
        concept_name="dummy_concept",
        mu_concept=mu_concept,
        presence_mask=presence_mask,
        sae_variant="kl",
        threshold=0.5,
        sae_kwargs={"latent_dim": 16},
        train_kwargs={"n_epochs": 3, "batch_size": 16},
    )

    assert result.pseudo_labels.shape[0] == N
    # Sampel di luar presence_mask harus tetap 0 (Bab IV.5.1.4: subkonsep
    # hanya berarti untuk sampel yang memang mengandung konsep induknya).
    assert (result.pseudo_labels[~presence_mask] == 0).all()


def test_build_pseudo_hierarchy_multi_concept():
    torch.manual_seed(0)
    N, C, d_c = 40, 3, 8
    mu_all = torch.randn(N, C, d_c)
    concept_probs = torch.rand(N, C)

    hierarchy = build_pseudo_hierarchy(
        mu_all=mu_all,
        concept_probs=concept_probs,
        concept_names=[f"c{i}" for i in range(C)],
        sae_variant="kl",
        sae_kwargs={"latent_dim": 16},
        train_kwargs={"n_epochs": 2, "batch_size": 16},
    )

    assert len(hierarchy.concept_results) == C
    assert all(r.pseudo_labels.shape[0] == N for r in hierarchy.concept_results)


if __name__ == "__main__":
    test_kl_sae_forward_shapes()
    test_batchtopk_sae_respects_sparsity()
    test_build_sae_latent_dim_ratio()
    test_train_sae_runs_and_returns_correct_shape()
    test_discover_subconcepts_for_concept_end_to_end()
    test_build_pseudo_hierarchy_multi_concept()
    print("OK: test_sae")
