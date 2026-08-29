"""Concept Intervention (Bab IV.5.4 & Bab IV.7.1.5).

HiProbCBM mendukung intervensi pada tingkat subkonsep MAUPUN konsep induk
(Bab IV.5.4):

    (mu_ik, sigma_ik^2) -> (mu_ik*, sigma_ik*^2)
        =(agregasi)=> (mu_i^p, sigma_i^{2,p}) -> (mu_i^{p*}, sigma_i^{2,p*})

Baseline (CBM/CEM/ProbCBM/HiCEM) diintervensi per-grup atribut (mengikuti
Koh dkk. 2020 & ProbCBM Sec. 5.2.4: 28 grup pada CUB), opsional diurutkan
berdasarkan ketidakpastian tertinggi terlebih dulu (hanya berlaku untuk
model probabilistik).
"""

from __future__ import annotations

import torch

from hiprobcbm.data.base import ConceptGroup
from hiprobcbm.models.hiprobcbm import HiProbCBMStage2, Stage2Output


def build_subconcept_intervention_tensors(
    model: HiProbCBMStage2,
    ground_truth_subconcept_labels: torch.Tensor,  # (B, C, K_max) {0,1}
    concept_anchors_positive: torch.Tensor,  # (d_c,) atau (C, K_max, d_c)
    concept_anchors_negative: torch.Tensor,
    intervene_mask: torch.Tensor,  # (C, K_max) bool - subkonsep mana yang dikoreksi
) -> tuple[torch.Tensor, torch.Tensor]:
    """Membentuk (mu_ik*, mask) untuk diteruskan ke `HiProbCBMStage2.forward`.

    Anchor "positif"/"negatif" di sini adalah rata-rata mu_ik pada sampel
    training yang berlabel pseudo-subconcept 1 / 0 (dihitung sekali dari
    hasil Tahap 1, dianalogikan dengan anchor z+/z- pada ProbCBM)."""
    batch = ground_truth_subconcept_labels.shape[0]
    device = ground_truth_subconcept_labels.device
    num_concepts, k_max = intervene_mask.shape

    target = torch.where(
        ground_truth_subconcept_labels.unsqueeze(-1).bool(),
        concept_anchors_positive.expand(batch, num_concepts, k_max, -1) if concept_anchors_positive.dim() == 1
        else concept_anchors_positive.unsqueeze(0).expand(batch, -1, -1, -1),
        concept_anchors_negative.expand(batch, num_concepts, k_max, -1) if concept_anchors_negative.dim() == 1
        else concept_anchors_negative.unsqueeze(0).expand(batch, -1, -1, -1),
    ).to(device)

    mask = intervene_mask.unsqueeze(0).expand(batch, -1, -1).to(device)
    return target, mask


def intervene_hiprobcbm(
    model: HiProbCBMStage2,
    x: torch.Tensor,
    ground_truth_subconcept_labels: torch.Tensor,
    concept_anchors_positive: torch.Tensor,
    concept_anchors_negative: torch.Tensor,
    intervene_mask: torch.Tensor,
) -> Stage2Output:
    target, mask = build_subconcept_intervention_tensors(
        model, ground_truth_subconcept_labels, concept_anchors_positive, concept_anchors_negative, intervene_mask
    )
    return model(x, subconcept_intervention=target, subconcept_intervention_mask=mask)


def progressive_group_intervention_order(
    concept_groups: list[ConceptGroup],
    concept_uncertainty: torch.Tensor | None = None,
    mode: str = "random",
    generator: torch.Generator | None = None,
) -> list[ConceptGroup]:
    """Urutan intervensi grup konsep (Bab IV.7.1.5 & ProbCBM Sec. 5.2.4).

    mode="random": urutan acak (dirata-rata atas beberapa run pada
    evaluasi, Bab IV.7.1.5).
    mode="uncertainty": grup dengan ketidakpastian rata-rata TERTINGGI
    diintervensi lebih dulu (hanya valid untuk model probabilistik).
    """
    groups = list(concept_groups)
    if mode == "random":
        idx = torch.randperm(len(groups), generator=generator).tolist()
        return [groups[i] for i in idx]
    if mode == "uncertainty":
        if concept_uncertainty is None:
            raise ValueError("concept_uncertainty wajib diisi untuk mode='uncertainty'.")
        scores = [concept_uncertainty[g.concept_indices].mean().item() for g in groups]
        order = sorted(range(len(groups)), key=lambda i: scores[i], reverse=True)
        return [groups[i] for i in order]
    raise ValueError("mode harus 'random' atau 'uncertainty'.")


def apply_group_intervention(
    concept_probs: torch.Tensor,
    concept_labels: torch.Tensor,
    groups_to_intervene: list[ConceptGroup],
) -> torch.Tensor:
    """Mengganti probabilitas konsep pada grup yang diintervensi dengan
    label ground-truth (do(c_J = c_J*), Bab IV.7.1.5)."""
    intervened = concept_probs.clone()
    for group in groups_to_intervene:
        idx = group.concept_indices
        intervened[:, idx] = concept_labels[:, idx]
    return intervened
