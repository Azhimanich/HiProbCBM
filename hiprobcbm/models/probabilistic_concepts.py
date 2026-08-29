"""Probabilistic Concept Predictor (Bab IV.5.1.1-4.5.1.2 & Bab III.3.6).

Formulasi (mengikuti notasi baku di Bagian 3/4 lembar rujukan formulasi):

    h            = f_enc(x)                                  # (B, d_h)
    mu_i         = W_mu^(i) h + b_mu^(i)                      # (B, d_c)
    sigma_i^2    = softplus(W_sigma^(i) h + b_sigma^(i))      # (B, d_c)
    c_i ~ N(mu_i, diag(sigma_i^2))

Setiap konsep memiliki kepala (head) linear terpisah, sejalan dengan
batasan masalah proposal (representasi mean/variance per-konsep, D_c=16 -
lihat Bab IV.5.1.1) - desain ini SENGAJA lebih sederhana daripada modul
PIENet+attention milik ProbCBM asli (Kim dkk., 2023), karena proposal
memformulasikan concept predictor sebagai lapisan fully-connected
sederhana (bukan berbasis attention), sebagaimana dicatat di kalimat
"Probabilistic concept predictor diimplementasikan sebagai lapisan
fully-connected terpisah untuk setiap konsep" (Bab IV.6).

Modul ini juga menyediakan `ConceptExistenceScorer`, yang mengisi celah
notasi yang diidentifikasi sebagai Temuan #2 (lembar rujukan, Bagian 2):
Bab IV.5 tidak pernah memformulasikan bagaimana p_i (probabilitas
keberadaan konsep) dihitung dari representasi probabilistik, padahal
dipakai di Bab IV.7.1.3 (Concept Accuracy). Di sini p_i dihitung via
fungsi skor sigmoid BERBAGI BOBOT lintas-konsep (mengikuti pola
`scoring_function` HiCEM, subbab 4.2 arsitektur HiCEM), diterapkan pada
setiap sampel Monte Carlo dari distribusi konsep tersebut.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbabilisticConceptPredictor(nn.Module):
    """Tahap 1 (Bab IV.5.1): menghasilkan (mu, sigma^2) untuk C konsep top-level.

    Keluaran mu, sigma2 berbentuk (B, C, d_c) - BUKAN (B, C) seperti yang
    tertulis literal di draf Bab IV.5.1.2 (`mu, sigma^2 in R^C`). Ini adalah
    perbaikan atas Temuan #1 (Kritis) pada lembar rujukan: setiap konsep
    tetap direpresentasikan sebagai vektor d_c-dimensi (sejalan dengan
    d_c=16 di Bab IV.5.1.1 dan konvensi ProbCBM Eq. 1), bukan skalar.
    """

    def __init__(self, feature_dim: int, num_concepts: int, concept_dim: int = 16, hidden_dim: int | None = None):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        hidden_dim = hidden_dim or feature_dim

        self.mean_heads = nn.ModuleList(
            [nn.Linear(feature_dim, concept_dim) for _ in range(num_concepts)]
        )
        self.logvar_heads = nn.ModuleList(
            [nn.Linear(feature_dim, concept_dim) for _ in range(num_concepts)]
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (B, d_h) -> mu, sigma2: (B, C, d_c)."""
        mu = torch.stack([head(h) for head in self.mean_heads], dim=1)
        raw = torch.stack([head(h) for head in self.logvar_heads], dim=1)
        sigma2 = F.softplus(raw) + 1e-6  # Bab IV.5.1.2: sigma_i^2 = softplus(s_i)
        return mu, sigma2


def reparameterize(mu: torch.Tensor, sigma2: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
    """Reparameterization trick (Bab III.3.6, Bab IV.5.2.5; Kingma & Welling, 2014).

    mu, sigma2: (..., d) -> z: (..., S, d), dengan S = n_samples.
    """
    sigma = torch.sqrt(sigma2.clamp_min(1e-12))
    mu_e = mu.unsqueeze(-2).expand(*mu.shape[:-1], n_samples, mu.shape[-1])
    sigma_e = sigma.unsqueeze(-2).expand(*sigma.shape[:-1], n_samples, sigma.shape[-1])
    eps = torch.randn_like(mu_e)
    return mu_e + sigma_e * eps


def kl_to_standard_normal(mu: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """L_KL = D_KL(N(mu, diag(sigma2)) || N(0, I)), dirata-rata per elemen batch.

    Bab III.3.6 & Eq. (6) ProbCBM (Kim dkk., 2023):
        KL = 1/2 * sum(sigma2 + mu^2 - 1 - log(sigma2))
    """
    sigma2 = sigma2.clamp_min(1e-12)
    kl = 0.5 * (sigma2 + mu.pow(2) - 1.0 - torch.log(sigma2))
    return kl.sum(dim=-1).mean()


class ConceptExistenceScorer(nn.Module):
    """Fungsi skor bersama s(.) yang memetakan representasi konsep (mean atau
    sampel) menjadi probabilitas keberadaan p_i (mengisi Temuan #2).

    Dipakai baik untuk konsep top-level (Tahap 1) maupun subkonsep
    (Tahap 2, subbab 4.5.2.1), sehingga satu fungsi skor konsisten dipakai
    di seluruh tingkat hierarki - sejalan dengan `scoring_function` HiCEM
    yang juga dibagi lintas subkonsep (subbab 4.2 arsitektur HiCEM).
    """

    def __init__(self, concept_dim: int):
        super().__init__()
        self.score = nn.Linear(concept_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., d_c) -> p: (...) dalam [0, 1]."""
        return torch.sigmoid(self.score(z)).squeeze(-1)

    def probability_from_distribution(self, mu: torch.Tensor, sigma2: torch.Tensor, n_samples: int = 8) -> torch.Tensor:
        """p_i^(bar) = (1/S) sum_s sigmoid(score(z^(s))), z^(s) ~ N(mu, sigma2).

        Ini formula konkret p_i^(s) yang dipakai Bab IV.7.1.3 (Concept
        Accuracy) dan Bab IV.7.1.4 (ROC-AUC), sekarang didefinisikan
        eksplisit di lapisan metode (bukan hanya muncul di evaluasi).
        """
        samples = reparameterize(mu, sigma2, n_samples=n_samples)  # (..., S, d_c)
        probs = self.forward(samples)  # (..., S)
        return probs.mean(dim=-1)
