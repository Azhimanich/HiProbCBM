"""Concept Embedding Model (Espinosa Zarlenga dkk., 2022) - Bab III.2.1 &
Tabel 2.1 no. 2.

Setiap konsep direpresentasikan lewat dua embedding (aktif/tidak-aktif):

    c_hat_i^+, c_hat_i^-  = phi_i^+(h), phi_i^-(h)
    p_hat_i               = s(c_hat_i^+)              # scoring function bersama
    c_hat_i               = p_hat_i * c_hat_i^+ + (1-p_hat_i) * c_hat_i^-

Bottleneck = konkatenasi seluruh c_hat_i, diteruskan ke classifier
terstandardisasi (Keputusan Desain Eksperimen #1).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hiprobcbm.models.backbones import Backbone, build_backbone
from hiprobcbm.models.classifier import LinearSoftmaxClassifier


@dataclass
class CEMOutput:
    concept_probs: torch.Tensor  # (B, C)
    concept_embeddings: torch.Tensor  # (B, C, embedding_dim) - mixed
    task_logits: torch.Tensor  # (B, num_classes)
    positive_embeddings: torch.Tensor
    negative_embeddings: torch.Tensor


class ConceptEmbeddingModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_concepts: int,
        num_classes: int,
        embedding_dim: int = 16,
        pretrained: bool = True,
        intervention_probability: float = .25,
    ):
        super().__init__()
        self.backbone: Backbone = build_backbone(backbone_name, pretrained=pretrained)
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.intervention_probability = intervention_probability

        self.positive_generators = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.backbone.output_dim, embedding_dim), nn.LeakyReLU()) for _ in range(num_concepts)]
        )
        self.negative_generators = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.backbone.output_dim, embedding_dim), nn.LeakyReLU()) for _ in range(num_concepts)]
        )
        self.scoring_function = nn.Linear(2 * embedding_dim, 1)
        self.classifier = LinearSoftmaxClassifier(input_dim=num_concepts * embedding_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor, concept_labels=None, intervention_mask=None) -> CEMOutput:
        h = self.backbone(x)
        pos = torch.stack([gen(h) for gen in self.positive_generators], dim=1)  # (B, C, d)
        neg = torch.stack([gen(h) for gen in self.negative_generators], dim=1)  # (B, C, d)

        p = torch.sigmoid(self.scoring_function(torch.cat([pos, neg], dim=-1))).squeeze(-1)
        if self.training and concept_labels is not None and intervention_mask is None:
            intervention_mask = torch.rand(self.num_concepts, device=x.device) < self.intervention_probability
        effective = p if intervention_mask is None else torch.where(intervention_mask, concept_labels, p)
        mixed = effective.unsqueeze(-1) * pos + (1 - effective).unsqueeze(-1) * neg

        bottleneck = mixed.reshape(x.shape[0], -1)
        task_logits = self.classifier(bottleneck)
        return CEMOutput(concept_probs=p, concept_embeddings=mixed, task_logits=task_logits,
                         positive_embeddings=pos, negative_embeddings=neg)

    @torch.no_grad()
    def extract_for_discovery(self, loader, device):
        """Deterministic train views and paths for HiCEM's CEM→SAE phase."""
        self.eval()
        embeddings, probabilities, paths = [], [], []
        for batch in loader:
            out = self(batch["image"].to(device))
            embeddings.append(out.positive_embeddings.cpu())
            probabilities.append(out.concept_probs.cpu())
            paths.extend(batch["path"])
        return torch.cat(embeddings), torch.cat(probabilities), paths
