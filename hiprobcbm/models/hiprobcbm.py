"""Perakitan penuh HiProbCBM - Tahap 1 & Tahap 2 (Bab IV.5).

- ``HiProbCBMStage1``: backbone + Probabilistic Concept Predictor +
  ConceptExistenceScorer (subbab 4.5.1.1-4.5.1.2). Dipakai untuk melatih
  representasi konsep top-level DAN untuk mengekstrak mean yang menjadi
  masukan automatic subconcept discovery (subbab 4.5.1.3-4.5.1.4,
  lihat ``subconcept_discovery.py``).

- ``HiProbCBMStage2``: model penuh setelah pseudo hierarchical subconcept
  dataset terbentuk - memprediksi distribusi tiap subkonsep, mengagregasi-
  kannya lewat learned attention (subbab 4.5.2), lalu melakukan
  hierarchical probabilistic reasoning (subbab 4.5.2.5) untuk prediksi
  kelas via Monte Carlo sampling + classifier terstandardisasi.

Karena K_i (jumlah subkonsep per konsep induk) bisa berbeda-beda, tensor
subkonsep di-pad ke K_max dengan mask boolean (C, K_max); konsep tanpa
subkonsep (K_i=0) jatuh kembali ke representasi konsep-nya sendiri dari
Tahap 1, sejalan dengan aturan HiCEM ("jika c_i tidak memiliki
subkonsep, ĉ_i^+ = ĉ_i^{+'}").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hiprobcbm.models.attention_aggregation import (
    LearnedAttentionAggregator,
    UniformAggregator,
    reparameterize_parent,
)
from hiprobcbm.models.backbones import Backbone, build_backbone
from hiprobcbm.models.classifier import LinearSoftmaxClassifier
from hiprobcbm.models.probabilistic_concepts import (
    ConceptExistenceScorer,
    ProbabilisticConceptPredictor,
    kl_to_standard_normal,
    reparameterize,
)


@dataclass
class Stage1Output:
    mu: torch.Tensor  # (B, C, d_c)
    sigma2: torch.Tensor  # (B, C, d_c)
    concept_probs: torch.Tensor  # (B, C)
    kl_loss: torch.Tensor


class HiProbCBMStage1(nn.Module):
    """Tahap 1 (Bab IV.5.1): representasi konsep probabilistik flat, identik
    strukturnya dengan ProbCBM (dipakai juga sebagai dasar mean untuk SAE)."""

    def __init__(self, backbone_name: str, num_concepts: int, concept_dim: int = 16, pretrained: bool = True):
        super().__init__()
        self.backbone: Backbone = build_backbone(backbone_name, pretrained=pretrained)
        self.concept_predictor = ProbabilisticConceptPredictor(
            feature_dim=self.backbone.output_dim, num_concepts=num_concepts, concept_dim=concept_dim
        )
        self.scorer = ConceptExistenceScorer(concept_dim=concept_dim)
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim

    def forward(self, x: torch.Tensor, n_samples: int = 8) -> Stage1Output:
        h = self.backbone(x)
        mu, sigma2 = self.concept_predictor(h)
        concept_probs = self.scorer.probability_from_distribution(mu, sigma2, n_samples=n_samples)
        kl = kl_to_standard_normal(mu, sigma2)
        return Stage1Output(mu=mu, sigma2=sigma2, concept_probs=concept_probs, kl_loss=kl)

    @torch.no_grad()
    def extract_means_for_discovery(self, dataloader, device: torch.device, n_samples: int = 8):
        """Mengumpulkan (mu, concept_probs) untuk seluruh data train - masukan
        `subconcept_discovery.build_pseudo_hierarchy` (Bab IV.5.1.3)."""
        self.eval()
        all_mu, all_probs = [], []
        for batch in dataloader:
            x = batch["image"].to(device)
            out = self.forward(x, n_samples=n_samples)
            all_mu.append(out.mu.cpu())
            all_probs.append(out.concept_probs.cpu())
        return torch.cat(all_mu, dim=0), torch.cat(all_probs, dim=0)


class SubconceptPredictor(nn.Module):
    """Memprediksi (mu_ik, sigma2_ik) untuk seluruh subkonsep hasil
    discovery (Bab IV.5.2.1). Satu kepala linear per subkonsep, dikelompokkan
    per konsep induk lalu di-pad ke K_max."""

    def __init__(self, feature_dim: int, subconcepts_per_concept: list[int], concept_dim: int = 16):
        super().__init__()
        self.subconcepts_per_concept = subconcepts_per_concept
        self.num_concepts = len(subconcepts_per_concept)
        self.concept_dim = concept_dim
        self.k_max = max(subconcepts_per_concept) if subconcepts_per_concept else 0

        self.mean_heads = nn.ModuleList()
        self.logvar_heads = nn.ModuleList()
        for k_i in subconcepts_per_concept:
            self.mean_heads.append(nn.ModuleList([nn.Linear(feature_dim, concept_dim) for _ in range(k_i)]))
            self.logvar_heads.append(nn.ModuleList([nn.Linear(feature_dim, concept_dim) for _ in range(k_i)]))

        mask = torch.zeros(self.num_concepts, max(self.k_max, 1), dtype=torch.bool)
        for i, k_i in enumerate(subconcepts_per_concept):
            mask[i, :k_i] = True
        self.register_buffer("mask", mask)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (B, d_h) -> mu, sigma2: (B, C, K_max, d_c), padding = 0."""
        import torch.nn.functional as F

        batch = h.shape[0]
        k_max = max(self.k_max, 1)
        mu = torch.zeros(batch, self.num_concepts, k_max, self.concept_dim, device=h.device)
        sigma2 = torch.zeros_like(mu)
        for i, k_i in enumerate(self.subconcepts_per_concept):
            for k in range(k_i):
                mu[:, i, k, :] = self.mean_heads[i][k](h)
                sigma2[:, i, k, :] = F.softplus(self.logvar_heads[i][k](h)) + 1e-6
        return mu, sigma2


@dataclass
class Stage2Output:
    mu_sub: torch.Tensor  # (B, C, K_max, d_c)
    sigma2_sub: torch.Tensor
    mu_parent: torch.Tensor  # (B, C, d_c)
    sigma2_parent: torch.Tensor
    attention: torch.Tensor  # (B, C, K_max)
    sub_concept_probs: torch.Tensor  # (B, C, K_max)
    parent_concept_probs: torch.Tensor  # (B, C)
    task_logits_per_sample: torch.Tensor  # (B, S, num_classes)
    task_probs: torch.Tensor  # (B, num_classes) - rata-rata S sampel (p_bar)
    kl_loss: torch.Tensor


class HiProbCBMStage2(nn.Module):
    """Tahap 2 (Bab IV.5.2-4.5.2.5): hierarchical concept aggregation +
    hierarchical probabilistic reasoning, dilatih end-to-end."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        subconcepts_per_concept: list[int],
        concept_dim: int = 16,
        pretrained: bool = True,
        n_mc_samples_train: int = 8,
        n_mc_samples_eval: int = 32,
        use_attention: bool = True,
        classifier_hidden_dims: tuple[int, ...] = (),
    ):
        super().__init__()
        self.backbone: Backbone = build_backbone(backbone_name, pretrained=pretrained)
        self.subconcept_predictor = SubconceptPredictor(
            feature_dim=self.backbone.output_dim,
            subconcepts_per_concept=subconcepts_per_concept,
            concept_dim=concept_dim,
        )
        self.subconcepts_per_concept = subconcepts_per_concept
        self.num_concepts = len(subconcepts_per_concept)
        self.concept_dim = concept_dim
        self.n_mc_samples_train = n_mc_samples_train
        self.n_mc_samples_eval = n_mc_samples_eval

        # HiProbCBM-A1 (Tabel 4.8): use_attention=False -> UniformAggregator.
        self.aggregator = (LearnedAttentionAggregator if use_attention else UniformAggregator)(
            concept_dim=concept_dim
        )
        self.scorer = ConceptExistenceScorer(concept_dim=concept_dim)
        self.classifier = LinearSoftmaxClassifier(
            input_dim=self.num_concepts * concept_dim,
            num_classes=num_classes,
            hidden_dims=classifier_hidden_dims,
        )

    def _aggregate_all_concepts(
        self, mu_sub: torch.Tensor, sigma2_sub: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Terapkan aggregator per konsep induk i; konsep tanpa subkonsep
        (K_i=0) jatuh kembali ke rata-rata trivial (mask kosong -> alpha nol,
        mu/sigma2 diisi dari sub tensor yang memang sudah nol) - dalam
        praktiknya K_i=0 sebaiknya dihindari di Bab IV.5.1.4 dengan
        menjamin minimal 1 subkonsep per konsep induk (mis. subkonsep
        tunggal = konsep itu sendiri saat SAE tidak menemukan struktur)."""
        batch = mu_sub.shape[0]
        mask = self.subconcept_predictor.mask.unsqueeze(0).expand(batch, -1, -1)  # (B, C, K_max)

        mu_parent = torch.zeros(batch, self.num_concepts, self.concept_dim, device=mu_sub.device)
        sigma2_parent = torch.zeros_like(mu_parent)
        alpha_all = torch.zeros(batch, self.num_concepts, mask.shape[-1], device=mu_sub.device)

        for i in range(self.num_concepts):
            agg = self.aggregator.aggregate(mu_sub[:, i], sigma2_sub[:, i], mask=mask[:, i])
            mu_parent[:, i] = agg["mu_parent"]
            sigma2_parent[:, i] = agg["sigma2_parent"]
            alpha_all[:, i] = agg["alpha"]

        return mu_parent, sigma2_parent, alpha_all

    def forward(
        self,
        x: torch.Tensor,
        subconcept_intervention: torch.Tensor | None = None,
        subconcept_intervention_mask: torch.Tensor | None = None,
    ) -> Stage2Output:
        """subconcept_intervention / _mask: (B, C, K_max, d_c) & (B, C, K_max)
        bool - dipakai `intervention.py` untuk Concept Intervention subbab
        4.5.4 (mengganti mu_ik dengan nilai target sebelum agregasi)."""
        h = self.backbone(x)
        mu_sub, sigma2_sub = self.subconcept_predictor(h)

        if subconcept_intervention is not None and subconcept_intervention_mask is not None:
            mask = subconcept_intervention_mask.unsqueeze(-1)
            mu_sub = torch.where(mask, subconcept_intervention, mu_sub)
            sigma2_sub = torch.where(mask, torch.zeros_like(sigma2_sub), sigma2_sub)

        mu_parent, sigma2_parent, alpha = self._aggregate_all_concepts(mu_sub, sigma2_sub)

        n_samples = self.n_mc_samples_train if self.training else self.n_mc_samples_eval
        z = reparameterize_parent(mu_parent, sigma2_parent, n_samples=n_samples)  # (B, C, S, d_c)

        bottleneck = z.permute(0, 2, 1, 3).reshape(x.shape[0], n_samples, -1)  # (B, S, C*d_c)
        task_logits_per_sample = self.classifier(bottleneck)  # (B, S, num_classes)
        task_probs = torch.softmax(task_logits_per_sample, dim=-1).mean(dim=1)  # p_bar, Bab IV.5.2.5

        parent_probs = self.scorer.probability_from_distribution(mu_parent, sigma2_parent, n_samples=n_samples)
        valid_mask = self.subconcept_predictor.mask
        sub_probs = self.scorer(mu_sub) * valid_mask.unsqueeze(0)

        kl_loss = kl_to_standard_normal(mu_parent, sigma2_parent)

        return Stage2Output(
            mu_sub=mu_sub,
            sigma2_sub=sigma2_sub,
            mu_parent=mu_parent,
            sigma2_parent=sigma2_parent,
            attention=alpha,
            sub_concept_probs=sub_probs,
            parent_concept_probs=parent_probs,
            task_logits_per_sample=task_logits_per_sample,
            task_probs=task_probs,
            kl_loss=kl_loss,
        )
