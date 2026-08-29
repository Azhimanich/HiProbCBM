"""Fungsi objektif setiap model (Bab III.3.2 Eq. CBM; Bab IV.5.2.5 Eq. HiProbCBM).

Setiap fungsi mengembalikan (total_loss, komponen: dict[str, Tensor]) agar
mudah dicatat ke W&B/TensorBoard sekaligus diperiksa manual saat debugging.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def concept_bce_loss(concept_probs: torch.Tensor, concept_labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(concept_probs.clamp(1e-6, 1 - 1e-6), concept_labels)


def task_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def cbm_loss(concept_probs, concept_labels, task_logits, task_labels, concept_weight: float = 1.0):
    l_concept = concept_bce_loss(concept_probs, concept_labels)
    l_class = task_ce_loss(task_logits, task_labels)
    total = l_class + concept_weight * l_concept
    return total, {"concept": l_concept.detach(), "class": l_class.detach()}


def cem_loss(concept_probs, concept_labels, task_logits, task_labels, concept_weight: float = 1.0):
    return cbm_loss(concept_probs, concept_labels, task_logits, task_labels, concept_weight)


def hicem_loss(top_probs, top_labels, sub_probs, sub_mask, sub_labels, task_logits, task_labels,
               concept_weight: float = 1.0):
    """L = E[L_task + lambda * L_CrossEntr(c, p_hat)] (Bab III.3.4), dengan c
    mencakup konsep top-level DAN subkonsep."""
    l_top = concept_bce_loss(top_probs, top_labels)
    if sub_mask.sum() > 0:
        l_sub = F.binary_cross_entropy(
            sub_probs[sub_mask].clamp(1e-6, 1 - 1e-6), sub_labels[sub_mask]
        )
    else:
        l_sub = torch.zeros((), device=top_probs.device)
    l_class = task_ce_loss(task_logits, task_labels)
    total = l_class + concept_weight * (l_top + l_sub)
    return total, {"top_concept": l_top.detach(), "sub_concept": l_sub.detach(), "class": l_class.detach()}


def probcbm_loss(concept_probs, concept_labels, task_logits_per_sample, task_labels, kl_loss,
                  lambda_kl: float = 5e-5, concept_weight: float = 1.0):
    """L_concept = L_BCE + lambda_KL * L_KL (Eq. 6-7, Kim dkk. 2023);
    L_class = cross-entropy rata-rata atas S sampel Monte Carlo."""
    l_concept_bce = concept_bce_loss(concept_probs, concept_labels)
    l_kl = kl_loss

    n_samples = task_logits_per_sample.shape[1]
    labels_expanded = task_labels.unsqueeze(1).expand(-1, n_samples).reshape(-1)
    logits_flat = task_logits_per_sample.reshape(-1, task_logits_per_sample.shape[-1])
    l_class = task_ce_loss(logits_flat, labels_expanded)

    total = l_class + concept_weight * l_concept_bce + lambda_kl * l_kl
    return total, {"concept_bce": l_concept_bce.detach(), "kl": l_kl.detach(), "class": l_class.detach()}


def hiprobcbm_stage1_loss(concept_probs, concept_labels, kl_loss, lambda_kl: float = 5e-5):
    """Tahap 1 (Bab IV.5.1): identik dengan L_concept ProbCBM, karena
    struktur konsep top-level Tahap 1 memang belum hierarkis."""
    l_bce = concept_bce_loss(concept_probs, concept_labels)
    total = l_bce + lambda_kl * kl_loss
    return total, {"concept_bce": l_bce.detach(), "kl": kl_loss.detach()}


def hiprobcbm_stage2_loss(
    task_logits_per_sample: torch.Tensor,
    task_labels: torch.Tensor,
    parent_concept_probs: torch.Tensor,
    parent_concept_labels: torch.Tensor,
    sub_concept_probs: torch.Tensor,
    sub_concept_mask: torch.Tensor,
    sub_concept_labels: torch.Tensor,
    kl_loss: torch.Tensor,
    lambda_concept: float = 1.0,
    lambda_sub: float = 1.0,
    lambda_kl: float = 5e-5,
):
    """L = L_cls + lambda1 L_concept + lambda2 L_sub + lambda3 L_KL (Bab IV.5.2.5).

    `sub_concept_labels` adalah pseudo-label hasil `subconcept_discovery`
    (Bab IV.5.1.4), bukan anotasi manusia - konsisten dengan batasan
    masalah "label-free" (Bab I.1.3 poin 2).
    """
    n_samples = task_logits_per_sample.shape[1]
    labels_expanded = task_labels.unsqueeze(1).expand(-1, n_samples).reshape(-1)
    logits_flat = task_logits_per_sample.reshape(-1, task_logits_per_sample.shape[-1])
    l_cls = task_ce_loss(logits_flat, labels_expanded)

    l_concept = concept_bce_loss(parent_concept_probs, parent_concept_labels)

    if sub_concept_mask.sum() > 0:
        l_sub = F.binary_cross_entropy(
            sub_concept_probs[sub_concept_mask].clamp(1e-6, 1 - 1e-6),
            sub_concept_labels[sub_concept_mask],
        )
    else:
        l_sub = torch.zeros((), device=task_logits_per_sample.device)

    total = l_cls + lambda_concept * l_concept + lambda_sub * l_sub + lambda_kl * kl_loss
    return total, {
        "class": l_cls.detach(),
        "concept": l_concept.detach(),
        "sub_concept": l_sub.detach(),
        "kl": kl_loss.detach(),
    }
