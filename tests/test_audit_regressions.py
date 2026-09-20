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


def test_hicem_placeholder_cannot_start_expensive_training(tmp_path):
    from hiprobcbm.engine import train_baseline
    with pytest.raises(ValueError, match="placeholder"):
        train_baseline.run(Config({"baseline": "hicem"}), torch.device("cpu"), tmp_path)


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
