"""Concept Bottleneck Model (Koh dkk., 2020) - Bab III.3.2 & Tabel 2.1 no. 1.

Representasi konsep paling sederhana: satu nilai skalar (probabilitas
biner) per konsep, tanpa embedding maupun distribusi. Formulasi:

    c_hat = g(x)   # (B, C) via sigmoid
    y_hat = f(c_hat)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hiprobcbm.models.backbones import Backbone, build_backbone
from hiprobcbm.models.classifier import LinearSoftmaxClassifier


@dataclass
class CBMOutput:
    concept_probs: torch.Tensor  # (B, C)
    task_logits: torch.Tensor  # (B, num_classes)


class ConceptBottleneckModel(nn.Module):
    def __init__(self, backbone_name: str, num_concepts: int, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone: Backbone = build_backbone(backbone_name, pretrained=pretrained)
        self.concept_head = nn.Linear(self.backbone.output_dim, num_concepts)
        self.classifier = LinearSoftmaxClassifier(input_dim=num_concepts, num_classes=num_classes)

    def forward(self, x: torch.Tensor, concept_labels: torch.Tensor | None = None) -> CBMOutput:
        h = self.backbone(x)
        concept_logits = self.concept_head(h)
        concept_probs = torch.sigmoid(concept_logits)

        # Mode independen (Koh dkk., 2020): saat training, classifier menerima
        # label konsep ground-truth, bukan prediksi - menghindari kebocoran
        # kesalahan tahap-1 ke tahap-2. Saat eval, prediksi sendiri dipakai.
        bottleneck = concept_labels if (self.training and concept_labels is not None) else concept_probs
        task_logits = self.classifier(bottleneck)
        return CBMOutput(concept_probs=concept_probs, task_logits=task_logits)
