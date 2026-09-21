"""Hierarchical Concept Embedding Model - adaptasi (Hill dkk., 2026),
Bab III.3.4 & Tabel 2.1 no. 4.

Berbeda dari HiProbCBM, bobot agregasi di sini BUKAN learned attention,
melainkan probabilitas prediksi subkonsep itu sendiri (lihat aspek 03 pada
lembar rujukan formulasi):

    p_hat_k^+  = s(c_hat_k^+)                                  (scoring bersama)
    c_hat_k^+  = sum_j p_hat_kj^+ * c_hat_kj^+                  (weighted mixture)
    p_hat_k^+  (estimasi top-level) = sum_j softmax(200 p_hat_kj^+ - 100)_j * p_hat_kj^+

Representasi tetap deterministik (tanpa sigma) - inilah yang membedakannya
dari HiProbCBM (lihat Tabel 2.1: "Belum mengintegrasikan representasi
probabilistik").

Untuk kesetaraan head-to-head, label predictor memakai
`LinearSoftmaxClassifier` yang sama dengan model lain (HiCEM asli memang
sudah memakai linear label predictor, jadi ini BUKAN penyimpangan dari
desain aslinya, berbeda dengan kasus ProbCBM).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hiprobcbm.models.backbones import Backbone, build_backbone
from hiprobcbm.models.classifier import LinearSoftmaxClassifier


def _soft_maximum(probs: torch.Tensor, alpha: float = 200.0, beta: float = 100.0) -> torch.Tensor:
    """Estimasi differentiable dari max(probs) sepanjang dim terakhir,
    Bab III.3.4 & subbab 4.2 arsitektur HiCEM (Hill dkk., 2026)."""
    weights = torch.softmax(alpha * probs - beta, dim=-1)
    return (weights * probs).sum(dim=-1)


@dataclass
class HiCEMOutput:
    top_concept_probs: torch.Tensor  # (B, C)
    sub_concept_probs: torch.Tensor  # positive children only: (B, C, K_max)
    negative_sub_concept_probs: torch.Tensor
    positive_mask: torch.Tensor  # (C, K_max)
    negative_mask: torch.Tensor
    task_logits: torch.Tensor


class _SubconceptBranch(nn.Module):
    """Satu cabang (positif ATAU negatif) untuk satu konsep induk - Bab III.3.4:
    'sejumlah subkonsep (child concepts) yang memberikan informasi lebih rinci'."""

    def __init__(self, embedding_dim: int, n_subconcepts: int, scoring_function: nn.Linear):
        super().__init__()
        self.n_subconcepts = n_subconcepts
        self.generators = nn.ModuleList(
            [nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU()) for _ in range(n_subconcepts)]
        )
        self.scoring_function = scoring_function  # dibagi lintas seluruh subkonsep (weight sharing)

    def forward(self, top_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """top_embedding: (B, d) -> (mixed_embedding, sub_probs, top_prob_estimate)."""
        if self.n_subconcepts == 0:
            prob = torch.sigmoid(self.scoring_function(top_embedding)).squeeze(-1)
            empty = torch.zeros(top_embedding.shape[0], 0, device=top_embedding.device)
            return top_embedding, empty, prob

        embeddings = torch.stack([gen(top_embedding) for gen in self.generators], dim=1)  # (B, K, d)
        probs = torch.sigmoid(self.scoring_function(embeddings)).squeeze(-1)  # (B, K)
        mixed = (probs.unsqueeze(-1) * embeddings).sum(dim=1)  # (B, d) - weighted mixture
        top_prob = _soft_maximum(probs)
        return mixed, probs, top_prob


class HierarchicalConceptEmbeddingModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        subconcepts_per_concept: list[tuple[int, int]],
        embedding_dim: int = 16,
        pretrained: bool = True,
    ):
        """subconcepts_per_concept: list of (n_positive_sub, n_negative_sub)
        per konsep induk - mengikuti `sub_concepts` pada HiCEM asli."""
        super().__init__()
        self.backbone: Backbone = build_backbone(backbone_name, pretrained=pretrained)
        self.num_concepts = len(subconcepts_per_concept)
        self.embedding_dim = embedding_dim
        self.subconcepts_per_concept = subconcepts_per_concept
        self.k_max = max((max(n_pos, n_neg) for n_pos, n_neg in subconcepts_per_concept), default=0)

        self.top_generators = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.backbone.output_dim, embedding_dim * 2), nn.LeakyReLU())
             for _ in subconcepts_per_concept]
        )
        self.scoring_function = nn.Linear(embedding_dim, 1)

        self.positive_branches = nn.ModuleList(
            [_SubconceptBranch(embedding_dim, n_pos, self.scoring_function) for n_pos, _ in subconcepts_per_concept]
        )
        self.negative_branches = nn.ModuleList(
            [_SubconceptBranch(embedding_dim, n_neg, self.scoring_function) for _, n_neg in subconcepts_per_concept]
        )

        self.classifier = LinearSoftmaxClassifier(
            input_dim=self.num_concepts * embedding_dim, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> HiCEMOutput:
        h = self.backbone(x)
        batch = x.shape[0]

        bottleneck, top_probs = [], []
        sub_probs_padded = torch.zeros(batch, self.num_concepts, max(self.k_max, 1), device=x.device)
        neg_probs_padded = torch.zeros_like(sub_probs_padded)
        pos_mask = torch.zeros(self.num_concepts, max(self.k_max, 1), dtype=torch.bool, device=x.device)
        neg_mask = torch.zeros_like(pos_mask)

        for i, top_gen in enumerate(self.top_generators):
            top_embedding = top_gen(h)
            pos_embed, neg_embed = top_embedding[:, : self.embedding_dim], top_embedding[:, self.embedding_dim :]

            pos_mixed, pos_probs, pos_top_prob = self.positive_branches[i](pos_embed)
            neg_mixed, neg_probs, neg_top_prob = self.negative_branches[i](neg_embed)

            # A discovery-only hierarchy has no labelled negative children.
            # Do not invent a negative target in that case: the parent is
            # represented by its positive children, exactly the information
            # available to this controlled HiCEM run.  When both branches
            # exist, retain the original HiCEM combination.
            n_pos, n_neg = self.subconcepts_per_concept[i]
            if n_neg == 0:
                p_i, mixed = pos_top_prob, pos_mixed
            elif n_pos == 0:
                p_i, mixed = 1 - neg_top_prob, neg_mixed
            else:
                p_i = (pos_top_prob + (1 - neg_top_prob)) / 2
                mixed = p_i.unsqueeze(-1) * pos_mixed + (1 - p_i).unsqueeze(-1) * neg_mixed
            top_probs.append(p_i)
            bottleneck.append(mixed)

            if pos_probs.shape[1] > 0:
                sub_probs_padded[:, i, : pos_probs.shape[1]] = pos_probs
                pos_mask[i, :pos_probs.shape[1]] = True
            if neg_probs.shape[1] > 0:
                neg_probs_padded[:, i, : neg_probs.shape[1]] = neg_probs
                neg_mask[i, :neg_probs.shape[1]] = True

        task_logits = self.classifier(torch.cat(bottleneck, dim=1))
        return HiCEMOutput(top_concept_probs=torch.stack(top_probs, dim=1), sub_concept_probs=sub_probs_padded,
                           negative_sub_concept_probs=neg_probs_padded, positive_mask=pos_mask,
                           negative_mask=neg_mask, task_logits=task_logits)
