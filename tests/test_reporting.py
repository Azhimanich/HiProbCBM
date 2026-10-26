"""Penjelasan multilevel dan grafik naskah."""

import json

import numpy as np
import torch

from hiprobcbm import reporting
from hiprobcbm.data import DATASET_REGISTRY
from hiprobcbm.engine import analysis, train_stage1, train_stage2
from tests.test_analysis import _FakeKitchens, _cfg


def test_select_samples_is_deterministic():
    correct = np.array([True, True, False, True, False, True])
    confidence = np.array([.9, .2, .95, .6, .3, .99])
    assert reporting.select_samples(correct, confidence, 2, 1, 2) == [5, 0, 1, 2, 4]


def test_cub_class_names_from_folder():
    class Dummy:
        num_classes = 3
    names = reporting.class_names_for(Dummy(), ["002.Laysan_Albatross/a.jpg"], [1])
    assert names == ["class_0", "Laysan Albatross", "class_2"]


def test_sub_names_join_multiple_ingredients(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"matches": [{"parent": 0, "subconcept": 1, "ingredient": "Apple"},
                                            {"parent": 0, "subconcept": 1, "ingredient": "Pear"}]}))
    assert reporting.sub_names_from_discovered(path) == {(0, 1): "Apple/Pear"}


def test_explanation_and_figures_end_to_end(tmp_path):
    DATASET_REGISTRY["fake_kitchens"] = _FakeKitchens
    try:
        cfg = _cfg("hicem")
        device = torch.device("cpu")
        s1, s2 = tmp_path / "s1", tmp_path / "s2"
        s1.mkdir()
        s2.mkdir()
        train_stage1.run(cfg, device, s1)
        train_stage2.run(cfg, device, s2, s1)

        result = reporting.explain_test_samples(cfg, s2 / "stage2_best.pth", s1 / "pseudo_hierarchy.pt", device,
                                                sub_names={(0, 0): "Apple"}, top_parents=2, top_subconcepts=2,
                                                n_confident=1, n_uncertain_correct=1, n_wrong=1)
        json.dumps(result)
        sample = result["samples"][0]
        assert len(sample["parents"]) == 2 and sample["parents"][0]["subconcepts"]
        counts = torch.load(s1 / "pseudo_hierarchy.pt", weights_only=True)["subconcepts_per_concept"]
        if any(k > 1 for k in counts):  # hanya induk bertingkat yang ditampilkan
            assert all(p["n_subconcepts"] > 1 for p in sample["parents"])
        contributions = [p["contribution_to_predicted_logit"] for p in sample["parents"]]
        assert contributions == sorted(contributions, reverse=True)
        attentions = [s["attention"] for s in sample["parents"][0]["subconcepts"]]
        assert attentions == sorted(attentions, reverse=True)
        assert reporting.plot_explanation(sample, tmp_path / "fig" / "x.png").exists()

        unc = analysis.evaluate_uncertainty(cfg, s2 / "stage2_best.pth",
                                            torch.load(s1 / "pseudo_hierarchy.pt", weights_only=True)["subconcepts_per_concept"],
                                            device, patch_size=16, n_groups=4)
        assert reporting.plot_uncertainty_groups({"utama": [unc, unc]}, tmp_path / "u.png", "t").exists()
        curves = {"protocol": "probcbm", "curves": {"random_mean": [.5, .7, .9], "uncertainty": [.5, .8, .9]}}
        assert reporting.plot_intervention_curves({"utama": [curves, curves]}, tmp_path / "i.png", "t").exists()
        hicem_curves = {"protocol": "hicem", "curves": {"cumulative": [.5, .6, .7, .8]}}
        assert reporting.plot_intervention_curves({"utama": [hicem_curves]}, tmp_path / "h.png", "t").exists()
        disc = {"curves": {"cumulative": [.5, .55, .6]}}
        assert reporting.plot_discovered_interventions({"utama": [(42, disc)]}, tmp_path / "d.png", "t").exists()
    finally:
        DATASET_REGISTRY.pop("fake_kitchens", None)
