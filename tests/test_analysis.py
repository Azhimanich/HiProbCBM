"""Analisis uncertainty (ProbCBM) dan subkonsep hasil discovery (HiCEM)."""

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from hiprobcbm.config import Config
from hiprobcbm.data import DATASET_REGISTRY
from hiprobcbm.data.base import ConceptDataset, ConceptGroup, DatasetSplits
from hiprobcbm.engine import analysis, train_stage1, train_stage2
from tests.test_stage_orchestration import _build_config

N_TRAIN, N_TEST, IMG = 24, 12, 64
GROUPS = {"fruit": ["Apple", "Banana"], "veg": ["Carrot", "Onion"]}
NAMES = ["fruit", "veg", "Salt", "Sugar"]


def test_concept_uncertainty_matches_probcbm_definition():
    sigma2 = torch.tensor([[[0.25, 1.0, 4.0]]])
    logsigma = 0.5 * torch.log(sigma2)
    assert torch.allclose(analysis.concept_uncertainty(sigma2), logsigma.mean(-1).exp())


def test_class_uncertainty_is_normalized_entropy():
    uniform = torch.full((1, 4), .25)
    onehot = torch.tensor([[1., 0., 0., 0.]])
    assert analysis.class_uncertainty(uniform).item() == pytest.approx(1.0)
    assert analysis.class_uncertainty(onehot).item() == pytest.approx(0.0, abs=1e-6)


def test_cutout_zeroes_center_patch_like_probcbm():
    images = torch.ones(2, 3, 224, 224)
    cut = analysis.cutout(images, 64)
    start = 224 // 2 - 32
    assert cut[:, :, start:start + 64, start:start + 64].abs().sum() == 0
    assert cut.sum() == images.sum() - 2 * 3 * 64 * 64 and images.min() == 1  # input tidak diubah


def test_uncertainty_groups_equal_size_and_correlation():
    unc = np.linspace(0, 1, 100)
    correct = (unc < 0.5).astype(float)
    groups = analysis.uncertainty_groups(unc, correct, 10)
    assert len(groups["accuracy"]) == 10
    assert groups["accuracy_lowest_uncertainty_group"] == 1.0 and groups["accuracy_highest_uncertainty_group"] == 0.0
    assert groups["spearman_uncertainty_accuracy"] < -0.8


def test_summarize_intervention_order():
    curves = {"uncertainty": [.5, .7, .8, .85, .9, .92, 1.0], "random_mean": [.5, .55, .6, .65, .7, .75, 1.0]}
    summary = analysis.summarize_intervention_order(curves)
    assert summary["uncertainty_order_better"] and summary["accuracy_after_3_groups_uncertainty"] == .85


def test_matching_follows_hicem_threshold_and_parent_scope():
    rng = np.random.default_rng(0)
    apple = rng.integers(0, 2, 200)
    pseudo = np.zeros((200, 4, 2))
    pseudo[:, 0, 1] = apple            # child 1 parent fruit = Apple
    pseudo[:, 1, 0] = rng.integers(0, 2, 200)  # noise
    pseudo[:, 2, 0] = apple            # parent lain: tidak boleh dipakai untuk Apple
    bank = {"Apple": apple, "Banana": rng.integers(0, 2, 200), "Carrot": rng.integers(0, 2, 200),
            "Onion": rng.integers(0, 2, 200)}
    matches = analysis.match_discovered_subconcepts(pseudo, [2, 1, 1, 1], NAMES, GROUPS, lambda n: bank[n])
    assert [(m["ingredient"], m["parent"], m["subconcept"]) for m in matches] == [("Apple", 0, 1)]
    assert matches[0]["train_matching_roc_auc"] == pytest.approx(1.0)


class _KitchenSplit(Dataset):
    def __init__(self, frame, seed):
        g = torch.Generator().manual_seed(seed)
        self.frame = frame.reset_index(drop=True)
        self.images = torch.randn(len(frame), 3, IMG, IMG, generator=g)
        self.concepts = torch.tensor(frame[NAMES].to_numpy(dtype="float32"))
        self.labels = torch.randint(0, 3, (len(frame),), generator=g)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        return {"image": self.images[i], "concepts": self.concepts[i], "label": self.labels[i],
                "path": self.frame.loc[i, "image"]}


def _manifest():
    rng = np.random.default_rng(1)
    rows = []
    for split, n in (("train", N_TRAIN), ("val", 8), ("test", N_TEST)):
        for i in range(n):
            row = {"image": f"{split}/{i:04d}.png", "split": split}
            for ing in ["Apple", "Banana", "Carrot", "Onion", "Salt", "Sugar"]:
                row[ing] = int(rng.random() < .5)
            rows.append(row)
    frame = pd.DataFrame(rows)
    for group, members in GROUPS.items():
        frame[group] = (frame[members].sum(axis=1) > 0).astype(int)
    return frame


class _FakeKitchens(ConceptDataset):
    def __init__(self, **kwargs):
        super().__init__(name="fake_kitchens", num_concepts=4, num_classes=3, concept_names=NAMES,
                         concept_groups=[ConceptGroup(n, [i]) for i, n in enumerate(NAMES)])
        self.ingredient_groups = GROUPS
        self._manifest = _manifest()

    def get_splits(self, image_size, backbone):
        m = self._manifest
        return DatasetSplits(train=_KitchenSplit(m[m.split == "train"], 1), val=_KitchenSplit(m[m.split == "val"], 2),
                             test=_KitchenSplit(m[m.split == "test"], 3))


def _cfg(protocol):
    values = _build_config().to_dict()
    values["dataset"] = "fake_kitchens"
    values["model"]["parent_probability"] = "hicem_mix"
    values["train"]["intervention_training"] = {"mode": protocol, "prob": 0.5}
    values["sae"]["presence_threshold"] = 0.0
    return Config(values)


@pytest.mark.parametrize("protocol", ["probcbm", "hicem"])
def test_analysis_pipeline_end_to_end(tmp_path, protocol):
    DATASET_REGISTRY["fake_kitchens"] = _FakeKitchens
    try:
        cfg = _cfg(protocol)
        device = torch.device("cpu")
        s1, s2 = tmp_path / "s1", tmp_path / "s2"
        s1.mkdir()
        s2.mkdir()
        train_stage1.run(cfg, device, s1)
        train_stage2.run(cfg, device, s2, s1)
        hierarchy = torch.load(s1 / "pseudo_hierarchy.pt", map_location="cpu", weights_only=True)

        unc = analysis.evaluate_uncertainty(cfg, s2 / "stage2_best.pth", hierarchy["subconcepts_per_concept"],
                                            device, patch_size=16, n_groups=4)
        assert set(unc["cutout"]) == {"patch_size", "concept_increase_own", "concept_increase_propagated",
                                      "class_increase"}
        assert len(unc["class_groups"]["accuracy"]) == 4
        json.dumps(unc)

        disc = analysis.evaluate_discovered_subconcepts(cfg, s2 / "stage2_best.pth", s1 / "pseudo_hierarchy.pt",
                                                        device, threshold=0.0)
        json.dumps(disc)
        assert disc["n_ground_truth_subconcepts"] == 4
        if disc["n_matched"]:
            assert len(disc["curves"]["cumulative"]) == disc["n_matched"] + 1
            assert len(disc["curves"]["one_at_a_time"]) == disc["n_matched"] + 1
            assert math.isnan(disc["mean_discovered_concept_roc_auc"]) or 0 <= disc["mean_discovered_concept_roc_auc"] <= 1
    finally:
        DATASET_REGISTRY.pop("fake_kitchens", None)
