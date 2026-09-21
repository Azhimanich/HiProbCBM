import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hiprobcbm.config import Config


@pytest.mark.parametrize("activations, expected", [
    ([[.1, .2], [.1, .3]], [[1.], [1.]]),  # fallback; no threshold-positive features
    ([[.7, .2], [.1, .3]], [[1.], [0.]]),  # only first feature is alive after threshold
])
def test_discovery_drops_sigmoid_features_without_positive_labels(monkeypatch, activations, expected):
    from hiprobcbm.models import subconcept_discovery as discovery
    monkeypatch.setattr(discovery, "build_sae", lambda *a, **k: None)
    monkeypatch.setattr(discovery, "train_sae", lambda *a, **k: torch.tensor(activations))
    result = discovery.discover_subconcepts_for_concept(0, "parent", torch.ones(2, 3), torch.ones(2, dtype=torch.bool))
    assert result.num_subconcepts == 1
    assert torch.equal(result.pseudo_labels, torch.tensor(expected))


def test_evaluation_restores_a1_and_inception_behavior(tmp_path, monkeypatch):
    from hiprobcbm.engine import evaluate
    cfg = {"dataset": "tiny", "data": {}, "model": {"backbone": "inception_v3", "image_size": 299,
           "concept_dim": 3, "pretrained": True, "use_attention": False}, "train": {"batch_size": 2}}
    (tmp_path / "run_manifest.json").write_text(json.dumps({"config": cfg}))
    path = tmp_path / "stage2_best.pth"
    torch.save({}, path)
    seen = {}

    class Model(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            seen.update(kwargs)
            self.backbone = SimpleNamespace(module=SimpleNamespace(transform_input=False))

    monkeypatch.setattr(evaluate, "HiProbCBMStage2", Model)
    monkeypatch.setattr(evaluate, "build_dataset", lambda *a, **k: SimpleNamespace(
        num_classes=2, get_dataloader=lambda *a, **k: []))

    def assess(model, loader, device):
        assert model.backbone.module.transform_input is True
        return SimpleNamespace(as_dict=lambda: {"task_accuracy": .5})

    monkeypatch.setattr(evaluate, "evaluate_stage2", assess)
    # Deliberately wrong fallback simulates loading the base YAML for an A1 run.
    result = evaluate.evaluate_hiprobcbm_checkpoint(Config({}), path, [1, 1], torch.device("cpu"))
    assert seen["use_attention"] is False
    assert seen["pretrained"] is False  # no download, flag restored separately
    assert result["task_accuracy"] == .5


def test_hicem_uses_dedicated_discovery_pipeline(tmp_path, monkeypatch):
    """HiCEM must not fall back to the former all-zero child-label loop."""
    from hiprobcbm.engine import train_baseline, train_hicem
    seen = {}

    def run(cfg, device, log_dir, resume):
        seen.update(cfg=cfg, device=device, log_dir=log_dir, resume=resume)

    monkeypatch.setattr(train_hicem, "run", run)
    cfg = Config({"baseline": "hicem"})
    train_baseline.run(cfg, torch.device("cpu"), tmp_path, resume="auto")
    assert seen == {"cfg": cfg, "device": torch.device("cpu"), "log_dir": tmp_path, "resume": "auto"}


def test_hicem_evaluator_uses_saved_discovery_hierarchy(tmp_path, monkeypatch):
    from hiprobcbm.engine import evaluate
    from hiprobcbm.utils.checkpoint import save_artifact

    cfg = {"dataset": "tiny", "baseline": "hicem", "data": {},
           "model": {"backbone": "resnet18", "image_size": 2, "embedding_dim": 3, "pretrained": False},
           "train": {"batch_size": 2}}
    (tmp_path / "run_manifest.json").write_text(json.dumps({"config": cfg}))
    save_artifact({"subconcepts_per_concept": [1, 1]}, tmp_path / "hicem_pseudo_hierarchy.pt")
    torch.save({}, tmp_path / "hicem_best.pth")

    class Model(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return SimpleNamespace(task_logits=torch.tensor([[2., 0.], [0., 2.]]),
                                   top_concept_probs=torch.tensor([[.8, .2], [.2, .8]]))

    batch = {"image": torch.zeros(2, 3, 2, 2), "label": torch.tensor([0, 1]),
             "concepts": torch.tensor([[1., 0.], [0., 1.]])}
    monkeypatch.setattr(evaluate, "HierarchicalConceptEmbeddingModel", Model)
    monkeypatch.setattr(evaluate, "build_dataset", lambda *a, **k: SimpleNamespace(
        num_concepts=2, num_classes=2, get_dataloader=lambda *a, **k: [batch]))
    report = evaluate.evaluate_baseline_checkpoint(Config({}), tmp_path / "hicem_best.pth", torch.device("cpu"))
    assert report["task_accuracy"] == 1.0


def test_stage2_rejects_dead_child_legacy_artifact_but_accepts_single_fallback(tmp_path):
    from hiprobcbm.engine.train_stage2 import load_pseudo_hierarchy
    path = tmp_path / "pseudo_hierarchy.pt"
    torch.save({"paths": ["a", "b"], "subconcepts_per_concept": [2],
                "pseudo_labels": torch.tensor([[[1., 0.]], [[0., 0.]]])}, path)
    with pytest.raises(ValueError, match="tanpa contoh positif"):
        load_pseudo_hierarchy(path)
    torch.save({"paths": ["a", "b"], "subconcepts_per_concept": [1],
                "pseudo_labels": torch.zeros(2, 1, 1)}, path)
    assert load_pseudo_hierarchy(path)[0] == [1]
