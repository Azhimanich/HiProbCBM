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

from hiprobcbm.models.backbones import Backbone, build_backbone
from hiprobcbm.models.classifier import AnchorClassifier, LinearSoftmaxClassifier
from hiprobcbm.models.probabilistic_concepts import (
    ConceptExistenceScorer,
    ProbabilisticConceptPredictor,
    kl_to_standard_normal,
    reparameterize,
)


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
    ):
        super().__init__()
        self.backbone: Backbone = build_backbone(backbone_name, pretrained=pretrained)
        self.concept_predictor = ProbabilisticConceptPredictor(
            feature_dim=self.backbone.output_dim, num_concepts=num_concepts, concept_dim=concept_dim
        )
        self.scorer = ConceptExistenceScorer(concept_dim=concept_dim)
        self.n_mc_samples_train = n_mc_samples_train
        self.n_mc_samples_eval = n_mc_samples_eval

        bottleneck_dim = num_concepts * concept_dim
        if classifier_head == "linear":
            self.classifier: nn.Module = LinearSoftmaxClassifier(bottleneck_dim, num_classes)
        elif classifier_head == "anchor":
            self.classifier = AnchorClassifier(bottleneck_dim, num_classes, class_embedding_dim)
        else:
            raise ValueError("classifier_head harus 'linear' atau 'anchor'.")

    def forward(self, x: torch.Tensor) -> ProbCBMOutput:
        h = self.backbone(x)
        mu, sigma2 = self.concept_predictor(h)

        n_samples = self.n_mc_samples_train if self.training else self.n_mc_samples_eval
        z = reparameterize(mu, sigma2, n_samples=n_samples)  # (B, C, S, d_c)

        bottleneck = z.permute(0, 2, 1, 3).reshape(x.shape[0], n_samples, -1)  # (B, S, C*d_c)
        task_logits_per_sample = self.classifier(bottleneck)
        task_probs = torch.softmax(task_logits_per_sample, dim=-1).mean(dim=1)

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
