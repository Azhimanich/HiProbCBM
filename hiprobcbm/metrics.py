"""Metrik evaluasi kuantitatif (Bab IV.7.1) - dipakai seragam untuk seluruh
model pembanding maupun HiProbCBM, supaya Tabel 4.7 bisa dihitung dengan
satu fungsi yang sama untuk setiap baris.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def task_accuracy(pred_labels: torch.Tensor, true_labels: torch.Tensor) -> float:
    """Bab IV.7.1.1."""
    return (pred_labels == true_labels).float().mean().item()


def task_f1(pred_labels: torch.Tensor, true_labels: torch.Tensor, average: str = "macro") -> float:
    """Bab IV.7.1.2."""
    return float(f1_score(true_labels.cpu().numpy(), pred_labels.cpu().numpy(), average=average, zero_division=0))


def concept_accuracy(concept_preds: torch.Tensor, concept_labels: torch.Tensor, threshold: float = 0.5) -> float:
    """Bab IV.7.1.3: Concept Accuracy = (1/NC) sum_n sum_i 1(c_hat_ni = c*_ni).

    `concept_preds` boleh berupa probabilitas (di-threshold di sini) atau
    sudah biner 0/1.
    """
    binary_pred = (concept_preds >= threshold).float()
    return (binary_pred == concept_labels).float().mean().item()


def concept_roc_auc(concept_probs: torch.Tensor, concept_labels: torch.Tensor) -> float:
    """Bab IV.7.1.4 - dirata-rata lintas konsep, mengabaikan konsep yang
    seluruh labelnya satu kelas (ROC-AUC tak terdefinisi)."""
    probs = concept_probs.cpu().numpy()
    labels = concept_labels.cpu().numpy()
    scores = []
    for i in range(labels.shape[-1]):
        col_labels = labels[..., i].reshape(-1)
        col_probs = probs[..., i].reshape(-1)
        if len(np.unique(col_labels)) < 2:
            continue
        scores.append(roc_auc_score(col_labels, col_probs))
    return float(np.mean(scores)) if scores else float("nan")


def intervention_performance(acc_after_intervention: float, acc_before: float) -> float:
    """Bab IV.7.1.5: Delta Acc_int = Acc(y | do(c_J=c_J*)) - Acc(y_hat)."""
    return acc_after_intervention - acc_before


def expected_calibration_error(confidences: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    """Bab IV.7.1.6 - ECE = sum_m (|B_m|/N) * |acc(B_m) - conf(B_m)|."""
    confidences = confidences.cpu().numpy()
    correct = correct.cpu().numpy().astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def subconcept_discovery_roc_auc(discovered_labels: torch.Tensor, ground_truth_labels: torch.Tensor) -> float:
    """Bab IV.4 / Tabel 4.3 (validasi terhadap PseudoKitchens): mencocokkan
    subkonsep yang ditemukan dengan ground-truth ingredient, mengikuti
    protokol Tabel 1 HiCEM (ROC-AUC per pasangan, ambil skor tertinggi)."""
    discovered = discovered_labels.cpu().numpy()
    truth = ground_truth_labels.cpu().numpy()
    if len(np.unique(truth)) < 2:
        return float("nan")
    best = 0.0
    for k in range(discovered.shape[1]):
        if len(np.unique(discovered[:, k])) < 2:
            continue
        score = roc_auc_score(truth, discovered[:, k])
        best = max(best, score)
    return best


@dataclass
class MetricReport:
    task_accuracy: float
    task_f1: float
    concept_accuracy: float
    concept_roc_auc: float
    ece: float | None = None
    intervention_delta: float | None = None

    def as_dict(self) -> dict[str, float]:
        return {
            "task_accuracy": self.task_accuracy,
            "task_f1": self.task_f1,
            "concept_accuracy": self.concept_accuracy,
            "concept_roc_auc": self.concept_roc_auc,
            "ece": self.ece,
            "intervention_delta": self.intervention_delta,
        }


def compute_standard_metrics(
    task_probs: torch.Tensor,
    task_labels: torch.Tensor,
    concept_probs: torch.Tensor,
    concept_labels: torch.Tensor,
    compute_ece: bool = True,
) -> MetricReport:
    pred_labels = task_probs.argmax(dim=-1)
    confidences, _ = task_probs.max(dim=-1)
    correct = (pred_labels == task_labels)

    return MetricReport(
        task_accuracy=task_accuracy(pred_labels, task_labels),
        task_f1=task_f1(pred_labels, task_labels),
        concept_accuracy=concept_accuracy(concept_probs, concept_labels),
        concept_roc_auc=concept_roc_auc(concept_probs, concept_labels),
        ece=expected_calibration_error(confidences, correct) if compute_ece else None,
    )
