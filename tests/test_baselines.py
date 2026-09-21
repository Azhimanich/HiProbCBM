"""Smoke test Bab III.3.2-3.4: forward pass keempat baseline dengan
backbone kecil (ResNet18, pretrained=False) supaya cepat dijalankan di CPU."""

import torch

from hiprobcbm.models.baselines.cbm import ConceptBottleneckModel
from hiprobcbm.models.baselines.cem import ConceptEmbeddingModel
from hiprobcbm.models.baselines.hicem import HierarchicalConceptEmbeddingModel
from hiprobcbm.models.baselines.probcbm import ProbabilisticConceptBottleneckModel

BATCH, IMG = 2, 224
NUM_CONCEPTS, NUM_CLASSES = 5, 3


def _dummy_batch():
    x = torch.randn(BATCH, 3, IMG, IMG)
    c = torch.randint(0, 2, (BATCH, NUM_CONCEPTS)).float()
    y = torch.randint(0, NUM_CLASSES, (BATCH,))
    return x, c, y


def test_cbm_forward():
    model = ConceptBottleneckModel("resnet18", NUM_CONCEPTS, NUM_CLASSES, pretrained=False)
    x, c, y = _dummy_batch()
    out = model(x, concept_labels=c)
    assert out.concept_probs.shape == (BATCH, NUM_CONCEPTS)
    assert out.task_logits.shape == (BATCH, NUM_CLASSES)


def test_cem_forward():
    model = ConceptEmbeddingModel("resnet18", NUM_CONCEPTS, NUM_CLASSES, embedding_dim=8, pretrained=False)
    x, c, y = _dummy_batch()
    out = model(x)
    assert out.concept_probs.shape == (BATCH, NUM_CONCEPTS)
    assert out.concept_embeddings.shape == (BATCH, NUM_CONCEPTS, 8)
    assert out.task_logits.shape == (BATCH, NUM_CLASSES)


def test_probcbm_forward_train_and_eval():
    model = ProbabilisticConceptBottleneckModel(
        "resnet18", NUM_CONCEPTS, NUM_CLASSES, concept_dim=8, pretrained=False,
        classifier_head="linear", n_mc_samples_train=3, n_mc_samples_eval=5,
    )
    x, c, y = _dummy_batch()

    model.train()
    out_train = model(x)
    assert out_train.task_logits_per_sample.shape == (BATCH, 3, NUM_CLASSES)

    model.eval()
    out_eval = model(x)
    assert out_eval.task_logits_per_sample.shape == (BATCH, 5, NUM_CLASSES)
    assert out_eval.task_probs.shape == (BATCH, NUM_CLASSES)
    assert torch.allclose(out_eval.task_probs.sum(dim=-1), torch.ones(BATCH), atol=1e-4)


def test_probcbm_anchor_classifier_variant():
    """Sanity-check replikasi (Keputusan Desain Eksperimen #1)."""
    model = ProbabilisticConceptBottleneckModel(
        "resnet18", NUM_CONCEPTS, NUM_CLASSES, concept_dim=8, pretrained=False,
        classifier_head="anchor", class_embedding_dim=12,
    )
    x, _, _ = _dummy_batch()
    out = model(x)
    assert out.task_probs.shape == (BATCH, NUM_CLASSES)


def test_probcbm_pem_anchor_variant():
    model = ProbabilisticConceptBottleneckModel(
        "resnet18", NUM_CONCEPTS, NUM_CLASSES, concept_dim=8, pretrained=False,
        classifier_head="anchor", implementation="reference", n_mc_samples_train=2,
    )
    x, _, _ = _dummy_batch()
    out = model(x)
    assert out.concept_probs.shape == (BATCH, NUM_CONCEPTS)
    assert torch.isfinite(out.concept_probs).all()


def test_hicem_forward():
    subconcepts_per_concept = [(2, 2) for _ in range(NUM_CONCEPTS)]
    model = HierarchicalConceptEmbeddingModel(
        "resnet18", NUM_CLASSES, subconcepts_per_concept, embedding_dim=8, pretrained=False
    )
    x, c, y = _dummy_batch()
    out = model(x)
    assert out.top_concept_probs.shape == (BATCH, NUM_CONCEPTS)
    assert out.sub_concept_probs.shape == (BATCH, NUM_CONCEPTS, 2)
    assert out.task_logits.shape == (BATCH, NUM_CLASSES)


def test_hicem_concept_without_subconcepts_falls_back():
    """Konsep dengan (0, 0) subkonsep tetap harus menghasilkan bottleneck
    valid (jatuh kembali ke embedding top-level sendiri)."""
    subconcepts_per_concept = [(0, 0)] * NUM_CONCEPTS
    model = HierarchicalConceptEmbeddingModel(
        "resnet18", NUM_CLASSES, subconcepts_per_concept, embedding_dim=8, pretrained=False
    )
    x, _, _ = _dummy_batch()
    out = model(x)
    assert out.task_logits.shape == (BATCH, NUM_CLASSES)
    assert not torch.isnan(out.task_logits).any()


def test_hicem_discovery_only_children_do_not_create_negative_targets():
    model = HierarchicalConceptEmbeddingModel(
        "resnet18", NUM_CLASSES, [(2, 0) for _ in range(NUM_CONCEPTS)], embedding_dim=8, pretrained=False
    )
    x, _, _ = _dummy_batch()
    out = model(x)
    assert out.positive_mask.all()
    assert not out.negative_mask.any()


if __name__ == "__main__":
    test_cbm_forward()
    test_cem_forward()
    test_probcbm_forward_train_and_eval()
    test_probcbm_anchor_classifier_variant()
    test_hicem_forward()
    test_hicem_concept_without_subconcepts_falls_back()
    print("OK: test_baselines")
