"""Smoke test Bab IV.7.1: metrik evaluasi kuantitatif."""

import torch

from hiprobcbm.metrics import (
    compute_standard_metrics,
    concept_accuracy,
    concept_roc_auc,
    expected_calibration_error,
    intervention_performance,
    task_accuracy,
    task_f1,
)


def test_task_accuracy_perfect():
    pred = torch.tensor([0, 1, 2, 1])
    true = torch.tensor([0, 1, 2, 1])
    assert task_accuracy(pred, true) == 1.0


def test_task_accuracy_half():
    pred = torch.tensor([0, 1, 2, 0])
    true = torch.tensor([0, 1, 2, 1])
    assert task_accuracy(pred, true) == 0.75


def test_task_f1_range():
    pred = torch.tensor([0, 1, 1, 0])
    true = torch.tensor([0, 1, 0, 0])
    f1 = task_f1(pred, true)
    assert 0.0 <= f1 <= 1.0


def test_concept_accuracy_with_probabilities():
    # threshold 0.5 -> binary_pred = [[1,0],[0,1]]; label sampel ke-2 sengaja
    # dibalik pada 1 dari 2 konsep, jadi 3 dari 4 elemen cocok = 0.75.
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    acc = concept_accuracy(probs, labels)
    assert abs(acc - 0.75) < 1e-6


def test_concept_roc_auc_perfect_separation():
    probs = torch.tensor([[0.1], [0.2], [0.8], [0.9]])
    labels = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    auc = concept_roc_auc(probs, labels)
    assert auc == 1.0


def test_intervention_performance_delta():
    assert abs(intervention_performance(0.9, 0.7) - 0.2) < 1e-6


def test_expected_calibration_error_zero_when_perfectly_calibrated():
    # confidence = accuracy pada setiap bin -> ECE seharusnya 0.
    confidences = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    correct = torch.tensor([1.0] * 9 + [0.0])  # 90% benar, confidence 0.9 -> selaras
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece < 1e-6


def test_compute_standard_metrics_end_to_end():
    task_probs = torch.softmax(torch.randn(10, 4), dim=-1)
    task_labels = torch.randint(0, 4, (10,))
    concept_probs = torch.rand(10, 6)
    concept_labels = torch.randint(0, 2, (10, 6)).float()

    report = compute_standard_metrics(task_probs, task_labels, concept_probs, concept_labels)
    d = report.as_dict()
    assert set(d.keys()) == {"task_accuracy", "task_f1", "concept_accuracy", "concept_roc_auc", "ece", "intervention_delta"}
    assert 0.0 <= d["task_accuracy"] <= 1.0
    assert 0.0 <= d["concept_accuracy"] <= 1.0


if __name__ == "__main__":
    test_task_accuracy_perfect()
    test_task_accuracy_half()
    test_task_f1_range()
    test_concept_accuracy_with_probabilities()
    test_concept_roc_auc_perfect_separation()
    test_intervention_performance_delta()
    test_expected_calibration_error_zero_when_perfectly_calibrated()
    test_compute_standard_metrics_end_to_end()
    print("OK: test_metrics")
