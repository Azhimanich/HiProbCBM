"""Hierarchical Concept Aggregation - Learned Attention (Bab IV.5.2.1-4.5.2.5).

Ini modul yang menjadi novelty utama HiProbCBM dibandingkan HiCEM (lihat
Bagian 1, aspek 03 & 04 pada lembar rujukan formulasi): bobot agregasi
DIPELAJARI lewat parameter (W, b, v) terpisah dari probabilitas prediksi
subkonsep - berbeda dari HiCEM yang memakai probabilitas subkonsep itu
sendiri sebagai bobot mixture.

Formulasi:
    e_ik      = v^T tanh(W mu_ik + b)                    (4.5.2.2)
    alpha_ik  = softmax_k(e_ik)
    mu_i^p    = sum_k alpha_ik * mu_ik                   (4.5.2.3, agregasi mean)
    sigma_i^{2,p} = sum_k alpha_ik^2 * sigma_ik^2         (4.5.2.4, propagasi variance)
    z_i^p     = mu_i^p + sigma_i^p * eps                 (4.5.2.5, reparameterization)

dengan sigma_i^p = sqrt(sigma_i^{2,p}) (akar kuadrat yang secara eksplisit
disisipkan di sini - lihat Temuan #4 pada lembar rujukan: draf Bab IV
memakai sigma_i^p pada reparameterization tanpa menuliskan hubungan akar
kuadrat ini).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedAttentionAggregator(nn.Module):
    """Satu instance dipakai bersama (weight-shared) untuk seluruh konsep
    induk, mengikuti pola `scoring_function`/`label_predictor` yang
    dibagi lintas konsep pada HiCEM (subbab 4.2 arsitektur HiCEM) - supaya
    jumlah parameter tidak bertumbuh linear terhadap jumlah konsep induk C.
    """

    def __init__(self, concept_dim: int, attention_hidden_dim: int | None = None):
        super().__init__()
        attention_hidden_dim = attention_hidden_dim or concept_dim
        self.W = nn.Linear(concept_dim, attention_hidden_dim, bias=True)  # W, b pada e_ik
        self.v = nn.Linear(attention_hidden_dim, 1, bias=False)
        self.tanh = nn.Tanh()

    def attention_scores(self, mu_subconcepts: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """mu_subconcepts: (B, K_i, d_c) -> alpha: (B, K_i), sum_k alpha_k = 1.

        `mask` (B, K_i) bool: True untuk subkonsep valid (K_i bisa berbeda
        antar-konsep induk; padding dipakai saat dibatch, mask menyaring
        posisi padding dari softmax).
        """
        e = self.v(self.tanh(self.W(mu_subconcepts))).squeeze(-1)  # (B, K_i)
        if mask is not None:
            e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=-1)
        if mask is not None:
            # Jika seluruh subkonsep di suatu baris ter-mask (K_i=0 utk sampel itu),
            # softmax(-inf,...) menghasilkan NaN; ganti dengan nol.
            alpha = torch.nan_to_num(alpha, nan=0.0)
        return alpha

    def aggregate(
        self,
        mu_subconcepts: torch.Tensor,
        sigma2_subconcepts: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """mu_subconcepts, sigma2_subconcepts: (B, K_i, d_c).

        Mengembalikan mu_parent, sigma2_parent: (B, d_c) dan alpha: (B, K_i)
        (dipakai untuk analisis kualitatif Bab IV.7.2 - bobot kontribusi
        subkonsep terhadap konsep induk).
        """
        alpha = self.attention_scores(mu_subconcepts, mask=mask)  # (B, K_i)
        alpha_exp = alpha.unsqueeze(-1)  # (B, K_i, 1)

        mu_parent = (alpha_exp * mu_subconcepts).sum(dim=1)  # (B, d_c) - Eq 4.5.2.3
        sigma2_parent = (alpha_exp.pow(2) * sigma2_subconcepts).sum(dim=1)  # (B, d_c) - Eq 4.5.2.4

        return {"mu_parent": mu_parent, "sigma2_parent": sigma2_parent, "alpha": alpha}


def reparameterize_parent(mu_parent: torch.Tensor, sigma2_parent: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
    """z_i^(s) = mu_i^p + sigma_i^p * eps^(s), sigma_i^p = sqrt(sigma_i^{2,p}).

    Bab IV.5.2.5 - menyisipkan langkah akar kuadrat yang implisit di draf
    proposal (Temuan #4).
    """
    sigma_parent = torch.sqrt(sigma2_parent.clamp_min(1e-12))
    mu_e = mu_parent.unsqueeze(-2).expand(*mu_parent.shape[:-1], n_samples, mu_parent.shape[-1])
    sigma_e = sigma_parent.unsqueeze(-2).expand(*sigma_parent.shape[:-1], n_samples, sigma_parent.shape[-1])
    eps = torch.randn_like(mu_e)
    return mu_e + sigma_e * eps


class UniformAggregator(LearnedAttentionAggregator):
    """HiProbCBM-A1 (studi ablasi, Tabel 4.8): tanpa learned attention -
    seluruh subkonsep diberi bobot seragam alpha_ik = 1/K_i.

    Mewarisi `LearnedAttentionAggregator` supaya antarmuka `aggregate`
    identik (drop-in replacement saat ablasi), tapi `attention_scores`
    di-override agar tidak memakai (W, b, v) sama sekali.
    """

    def attention_scores(self, mu_subconcepts: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, k, _ = mu_subconcepts.shape
        if mask is None:
            return torch.full((batch, k), 1.0 / max(k, 1), device=mu_subconcepts.device)
        counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        alpha = mask.float() / counts
        return alpha
