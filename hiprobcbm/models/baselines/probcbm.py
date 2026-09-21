"""Probabilistic Concept Bottleneck Model - adaptasi (Kim dkk., 2023),
Bab III.3.3 & Tabel 2.1 no. 3.

Representasi konsep memakai modul yang SAMA dengan Tahap 1 HiProbCBM
(`ProbabilisticConceptPredictor` + `ConceptExistenceScorer`) - ini memang
disengaja: HiProbCBM adalah pengembangan langsung dari ProbCBM (Bab I.1.3
Batasan Masalah, poin 1), sehingga satu-satunya perbedaan arsitektur pada
level konsep top-level seharusnya adalah keberadaan struktur hierarkis di
Tahap 2 milik HiProbCBM, bukan cara memprediksi (mu, sigma^2) itu sendiri.

Menyediakan dua mode classifier:
- ``classifier_head="linear"`` (default): head terstandardisasi
  (Keputusan Desain Eksperimen #1), dipakai pada perbandingan utama.
- ``classifier_head="anchor"``: mekanisme asli ProbCBM (jarak ke anchor
  kelas, Eq. 4-5 Kim dkk. 2023) - dipakai HANYA untuk sanity-check
  replikasi terhadap Tabel 1 paper asli sebelum classifier distandardisasi.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hiprobcbm.models.backbones import Backbone, build_backbone
from hiprobcbm.models.classifier import AnchorClassifier, LinearSoftmaxClassifier
from hiprobcbm.models.probabilistic_concepts import (
    ConceptExistenceScorer,
    ProbabilisticConceptPredictor,
    kl_to_standard_normal,
    reparameterize,
)


class _PEM(nn.Module):
    """Attention-based probabilistic embedding module used by reference ProbCBM."""
    def __init__(self, feature_dim: int, concept_dim: int):
        super().__init__()
        hidden = max(1, concept_dim // 2)
        self.attention_1 = nn.Linear(feature_dim, hidden, bias=False)
        self.attention_2 = nn.Linear(hidden, 1, bias=False)
        self.mean = nn.Linear(feature_dim, concept_dim)
        self.residual = nn.Linear(feature_dim, concept_dim)
        self.logvar = nn.Linear(feature_dim, concept_dim)

    def forward(self, pooled: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.attention_2(torch.tanh(self.attention_1(tokens))).squeeze(-1), dim=-1)
        attended = (weights.unsqueeze(-1) * tokens).sum(dim=1)
        mu = F.normalize(self.mean(pooled) + torch.sigmoid(self.residual(attended)), p=2, dim=-1)
        return mu, self.logvar(attended).clamp(max=10.)


@dataclass
class ProbCBMOutput:
    mu: torch.Tensor
    sigma2: torch.Tensor
    concept_probs: torch.Tensor
    task_logits_per_sample: torch.Tensor
    task_probs: torch.Tensor
    kl_loss: torch.Tensor


class ProbabilisticConceptBottleneckModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_concepts: int,
        num_classes: int,
        concept_dim: int = 16,
        pretrained: bool = True,
        classifier_head: str = "linear",
        class_embedding_dim: int = 128,
        n_mc_samples_train: int = 8,
        n_mc_samples_eval: int = 32,
        implementation: str = "controlled",
    ):
        super().__init__()
        self.backbone: Backbone = build_backbone(backbone_name, pretrained=pretrained)
        self.concept_predictor = ProbabilisticConceptPredictor(
            feature_dim=self.backbone.output_dim, num_concepts=num_concepts, concept_dim=concept_dim
        )
        self.scorer = ConceptExistenceScorer(concept_dim=concept_dim)
        self.n_mc_samples_train = n_mc_samples_train
        self.n_mc_samples_eval = n_mc_samples_eval
        self.implementation = implementation
        if implementation not in {"controlled", "reference"}:
            raise ValueError("implementation harus 'controlled' atau 'reference'.")
        if implementation == "reference":
            if backbone_name not in {"resnet18", "inception_v3"}:
                raise ValueError("ProbCBM reference membutuhkan backbone dengan feature map spasial.")
            self.reference_heads = nn.ModuleList([
                _PEM(self.backbone.output_dim, concept_dim) for _ in range(num_concepts)
            ])
            self.concept_anchors = nn.Parameter(torch.randn(num_concepts, concept_dim))
            self.concept_scale = nn.Parameter(torch.ones(1))

        bottleneck_dim = num_concepts * concept_dim
        if classifier_head == "linear":
            self.classifier: nn.Module = LinearSoftmaxClassifier(bottleneck_dim, num_classes)
        elif classifier_head == "anchor":
            self.classifier = AnchorClassifier(bottleneck_dim, num_classes, class_embedding_dim)
        else:
            raise ValueError("classifier_head harus 'linear' atau 'anchor'.")

    def forward(self, x: torch.Tensor) -> ProbCBMOutput:
        h = self.backbone(x)
        if self.implementation == "reference":
            spatial = self.backbone.spatial_features(x)
            tokens = spatial.flatten(2).transpose(1, 2)
            outputs = [head(h, tokens) for head in self.reference_heads]
            mu = torch.stack([item[0] for item in outputs], dim=1)
            # Reference models parameterize log sigma. Preserve project-wide
            # sigma² output contract for sampling/KL.
            sigma2 = torch.stack([item[1].exp().clamp(max=1e6) for item in outputs], dim=1)
        else:
            mu, sigma2 = self.concept_predictor(h)

        n_samples = self.n_mc_samples_train if self.training else self.n_mc_samples_eval
        z = reparameterize(mu, sigma2, n_samples=n_samples)  # (B, C, S, d_c)

        bottleneck = z.permute(0, 2, 1, 3).reshape(x.shape[0], n_samples, -1)  # (B, S, C*d_c)
        task_logits_per_sample = self.classifier(bottleneck)
        task_probs = torch.softmax(task_logits_per_sample, dim=-1).mean(dim=1)

        if self.implementation == "reference":
            anchors = F.normalize(self.concept_anchors, p=2, dim=-1)
            distances = ((z - anchors.unsqueeze(0).unsqueeze(2)).square().sum(dim=-1) + 1e-10).sqrt()
            concept_probs = torch.sigmoid(-self.concept_scale.square() * distances).mean(dim=-1)
        else:
            concept_probs = self.scorer.probability_from_distribution(mu, sigma2, n_samples=n_samples)
        kl_loss = kl_to_standard_normal(mu, sigma2)

        return ProbCBMOutput(
            mu=mu,
            sigma2=sigma2,
            concept_probs=concept_probs,
            task_logits_per_sample=task_logits_per_sample,
            task_probs=task_probs,
            kl_loss=kl_loss,
        )
