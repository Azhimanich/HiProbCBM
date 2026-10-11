"""Regresi Tahap 2 v2 (11 Okt 2026): probabilitas induk ala HiCEM, jalur
konsep Tahap 1 yang dibekukan ala ProbCBM sequential, loss konsep berbobot
HiCEM, dan metrik ROC-AUC protokol HiCEM. Rumus dibandingkan langsung
dengan source baseline bila source itu dapat diimpor tanpa dependensi berat.
"""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hiprobcbm.config import Config
from hiprobcbm.data import DATASET_REGISTRY
from hiprobcbm.engine import evaluate, train_stage1, train_stage2
from hiprobcbm.engine.protocol import optimizer_for
from hiprobcbm.losses import hicem_concept_loss_weights, hiprobcbm_stage2_loss
from hiprobcbm.metrics import concept_roc_auc_hicem_protocol
from hiprobcbm.models.hiprobcbm import HiProbCBMStage1, HiProbCBMStage2, hicem_soft_max
from tests.test_stage_orchestration import _FakeDataset, _build_config

HICEM_METRICS = Path(__file__).resolve().parents[2] / "HiCEM" / "cemcd" / "metrics.py"


def test_hicem_soft_max_matches_source_formula_and_ignores_padding():
    probs = torch.tensor([[[0.9, 0.1, 0.0], [0.2, 0.3, 0.0]]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    out = hicem_soft_max(probs, mask)
    for i in range(2):
        p = probs[0, i, :2]
        expected = (torch.softmax(200 * p - 100, dim=-1) * p).sum()
        assert torch.allclose(out[0, i], expected)
    assert out[0, 0] > 0.89  # ~max: satu child aktif cukup


def test_hicem_mix_parent_probability_formula():
    torch.manual_seed(0)
    model = HiProbCBMStage2("resnet18", num_classes=3, subconcepts_per_concept=[3, 1], concept_dim=4,
                            pretrained=False, n_mc_samples_train=2, n_mc_samples_eval=4,
                            parent_probability="hicem_mix").eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        torch.manual_seed(5)
        out = model(x)
        h = model.backbone(x)
        mu_sub, sigma2_sub = model.subconcept_predictor(h)
        mu_parent, sigma2_parent, _ = model._aggregate_all_concepts(mu_sub, sigma2_sub)
        torch.manual_seed(5)
        from hiprobcbm.models.attention_aggregation import reparameterize_parent
        reparameterize_parent(mu_parent, sigma2_parent, n_samples=4)  # samakan konsumsi RNG forward
        own = model.scorer.probability_from_distribution(*model.parent_predictor(h), n_samples=4)
        mask = model.subconcept_predictor.mask
        expected = 0.5 * (own + hicem_soft_max(model.scorer(mu_sub) * mask, mask))
    assert torch.allclose(out.parent_concept_probs, expected, atol=1e-6)
    with pytest.raises(ValueError):
        HiProbCBMStage2("resnet18", num_classes=3, subconcepts_per_concept=[1], parent_probability="salah")


def test_legacy_mode_has_no_parent_predictor():
    model = HiProbCBMStage2("resnet18", num_classes=3, subconcepts_per_concept=[2], concept_dim=4, pretrained=False)
    assert model.parent_predictor is None
    with pytest.raises(ValueError):
        model.freeze_concept_path()


def test_initialize_from_stage1_copies_and_freezes(tmp_path):
    torch.manual_seed(0)
    stage1 = HiProbCBMStage1("resnet18", num_concepts=2, concept_dim=4, pretrained=False)
    torch.save(stage1.state_dict(), tmp_path / "stage1_best.pth")
    stage2 = HiProbCBMStage2("resnet18", num_classes=3, subconcepts_per_concept=[2, 1], concept_dim=4,
                             pretrained=False, n_mc_samples_train=2, parent_probability="hicem_mix")
    train_stage2.initialize_from_stage1(stage2, tmp_path / "stage1_best.pth")

    for name, value in stage1.backbone.state_dict().items():
        assert torch.equal(stage2.backbone.state_dict()[name], value)
    for name, value in stage1.concept_predictor.state_dict().items():
        assert torch.equal(stage2.parent_predictor.state_dict()[name], value)
    assert torch.equal(stage2.scorer.score.weight, stage1.scorer.score.weight)

    stage2.train()
    assert not stage2.backbone.training and stage2.subconcept_predictor.training
    frozen = {id(p) for m in (stage2.backbone, stage2.parent_predictor, stage2.scorer) for p in m.parameters()}
    optimizer = optimizer_for(stage2, Config({"optimizer": "adam"}), 1e-2)
    assert not frozen & {id(p) for g in optimizer.param_groups for p in g["params"]}

    before = copy.deepcopy(stage2.backbone.state_dict())
    out = stage2(torch.randn(3, 3, 64, 64))
    loss = out.task_logits_per_sample.mean() + out.parent_concept_probs.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for name, value in before.items():  # termasuk running stats BatchNorm
        assert torch.equal(stage2.backbone.state_dict()[name], value)


def test_hicem_concept_loss_weights_match_source_formula():
    labels = torch.tensor([[1., 0., 0.], [1., 1., 0.], [0., 0., 0.], [1., 0., 0.]])
    counts = np.array([3., 1., 1.])  # kolom tanpa positif dihitung satu, sesuai HiCEM
    assert torch.allclose(hicem_concept_loss_weights(labels), torch.tensor(4 / counts - 1, dtype=torch.float32))


def test_joint_reduction_equals_hicem_weighted_bce():
    torch.manual_seed(0)
    b, c, k = 5, 2, 3
    mask = torch.tensor([[True, True, False], [True, False, False]])
    parent_p, parent_y = torch.rand(b, c), (torch.rand(b, c) > .5).float()
    sub_p, sub_y = torch.rand(b, c, k) * mask, ((torch.rand(b, c, k) > .5) & mask).float()
    pw, sw = torch.rand(c) + .5, torch.rand(c, k) + .5
    logits = torch.randn(b, 2, 4)
    y = torch.randint(0, 4, (b,))
    total, _ = hiprobcbm_stage2_loss(logits, y, parent_p, parent_y, sub_p, mask.expand(b, -1, -1), sub_y,
                                     torch.tensor(0.), lambda_concept=10., parent_weights=pw, sub_weights=sw,
                                     reduction="joint")
    probs = torch.cat([parent_p, sub_p[:, mask]], 1).clamp(1e-6, 1 - 1e-6)
    expected_concept = torch.nn.BCELoss(weight=torch.cat([pw, sw[mask]]))(probs, torch.cat([parent_y, sub_y[:, mask]], 1))
    expected_task = F.cross_entropy(logits.reshape(-1, 4), y.repeat_interleave(2))
    assert torch.allclose(total, expected_task + 10 * expected_concept)


def test_separate_reduction_unchanged_for_legacy_runs():
    torch.manual_seed(1)
    b, c, k = 4, 2, 2
    mask = torch.tensor([[True, True], [True, False]]).expand(b, -1, -1)
    parent_p, parent_y = torch.rand(b, c), (torch.rand(b, c) > .5).float()
    sub_p, sub_y = torch.rand(b, c, k), (torch.rand(b, c, k) > .5).float()
    logits, y = torch.randn(b, 3, 4), torch.randint(0, 4, (b,))
    total, _ = hiprobcbm_stage2_loss(logits, y, parent_p, parent_y, sub_p, mask, sub_y, torch.tensor(2.), lambda_kl=.1)
    old = (F.cross_entropy(logits.reshape(-1, 4), y.repeat_interleave(3))
           + F.binary_cross_entropy(parent_p.clamp(1e-6, 1 - 1e-6), parent_y)
           + F.binary_cross_entropy(sub_p[mask].clamp(1e-6, 1 - 1e-6), sub_y[mask]) + .1 * 2.)
    assert torch.allclose(total, old)


@pytest.mark.skipif(not HICEM_METRICS.exists(), reason="clone HiCEM tidak tersedia")
def test_concept_auc_matches_hicem_source():
    spec = importlib.util.spec_from_file_location("hicem_metrics", HICEM_METRICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    probs = torch.rand(40, 5)
    labels = (torch.rand(40, 5) > .5).float()
    labels[:, 2] = 1.0  # kolom satu kelas: HiCEM memakai accuracy
    _, _, expected, _ = module.calculate_concept_accuracies(probs, labels)
    assert concept_roc_auc_hicem_protocol(probs, labels) == pytest.approx(expected)


def _v2_config(**train_overrides):
    values = _build_config().to_dict()
    values["model"]["parent_probability"] = "hicem_mix"
    values["train"].update(train_overrides)
    return Config(values)


@pytest.mark.parametrize("train_overrides", [
    {"stage2_init": "stage1_frozen_concepts", "lr_stage2": 1e-2, "scheduler": "cosine"},
    {"lambda_concept": 10.0, "concept_loss_reduction": "joint", "concept_loss_weighting": "hicem_imbalance"},
], ids=["probcbm_protocol", "hicem_protocol"])
def test_v2_stage2_trains_and_evaluates(tmp_path, train_overrides):
    DATASET_REGISTRY["dummy_e2e"] = _FakeDataset
    try:
        cfg = _v2_config(**train_overrides)
        device = torch.device("cpu")
        stage1_dir, stage2_dir = tmp_path / "stage1", tmp_path / "stage2"
        stage1_dir.mkdir()
        stage2_dir.mkdir()
        train_stage1.run(cfg, device, stage1_dir)
        train_stage2.run(cfg, device, stage2_dir, stage1_dir)

        counts = torch.load(stage1_dir / "pseudo_hierarchy.pt", map_location="cpu",
                            weights_only=True)["subconcepts_per_concept"]
        metrics = evaluate.evaluate_hiprobcbm_checkpoint(cfg, stage2_dir / "stage2_best.pth", counts, device)
        assert 0.0 <= metrics["task_accuracy"] <= 1.0
        assert 0.0 <= metrics["concept_roc_auc_hicem_protocol"] <= 1.0

        if train_overrides.get("stage2_init") == "stage1_frozen_concepts":
            stage1 = torch.load(stage1_dir / "stage1_best.pth", map_location="cpu", weights_only=True)
            stage2 = torch.load(stage2_dir / "stage2_best.pth", map_location="cpu", weights_only=True)
            for name, value in stage1.items():
                if name.startswith("backbone."):
                    assert torch.equal(stage2[name], value)
            # Resume harus menerima run yang sama (identitas memuat stage1_best).
            train_stage2.run(cfg, device, stage2_dir, stage1_dir)
    finally:
        DATASET_REGISTRY.pop("dummy_e2e", None)


# --- Revisi Tahap 1 + intervensi (11 Okt 2026) -------------------------------------------

def test_cub_concept_names_give_probcbm_28_groups():
    from hiprobcbm.data.base import derive_attribute_groups
    from hiprobcbm.data.cub_concept_names import CUB_CONCEPT_NAMES
    assert len(CUB_CONCEPT_NAMES) == 112 and len(set(CUB_CONCEPT_NAMES)) == 112
    assert len(derive_attribute_groups(CUB_CONCEPT_NAMES)) == 28  # ProbCBM Sec. 5.2.4


def test_probcbm_transform_matches_source_order():
    from torchvision import transforms as T
    from hiprobcbm.data.transforms import build_transform
    train = build_transform("resnet18", 224, train=True, protocol="probcbm").transforms
    assert [type(t) for t in train] == [T.ColorJitter, T.RandomHorizontalFlip, T.Resize, T.RandomResizedCrop,
                                       T.ToTensor, T.Normalize]
    assert train[0].brightness == pytest.approx((1 - 32 / 255, 1 + 32 / 255)) and train[0].contrast is None
    assert list(train[-1].mean) == [0.5, 0.5, 0.5]
    with pytest.raises(ValueError):
        build_transform("clip_vit_l14", 224, train=True, protocol="probcbm")


def test_optimizer_lr_ratio_groups_like_probcbm():
    model = HiProbCBMStage1("resnet18", num_concepts=2, concept_dim=4, pretrained=False)
    opt = optimizer_for(model, Config({"optimizer": "adamp", "lr_ratio": 10.0}), 1e-3)
    backbone = {id(p) for p in model.backbone.parameters()}
    assert type(opt).__name__ == "AdamP"
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([1e-3, 1e-2])
    assert {id(p) for p in opt.param_groups[0]["params"]} == backbone


def test_training_interventions_follow_protocols():
    model = HiProbCBMStage2("resnet18", num_classes=3, subconcepts_per_concept=[2, 1], concept_dim=4,
                            pretrained=False, n_mc_samples_train=6, parent_probability="hicem_mix")
    parents = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    subs = torch.zeros(3, 2, 2)
    torch.manual_seed(0)
    p = train_stage2.training_interventions(model, Config({"intervention_training": {"mode": "probcbm", "prob": .5}}),
                                            parents, subs)
    assert p["sample_intervention_mask"].shape == (3, 6) and p["parent_labels"] is parents
    h = train_stage2.training_interventions(model, Config({"intervention_training": {"mode": "hicem", "prob": 1.0}}),
                                            parents, subs)
    assert torch.equal(h["parent_intervention"], parents)
    assert torch.isnan(h["sub_intervention"][:, 1, 1]).all()  # padding tidak pernah diintervensi
    assert train_stage2.training_interventions(model, Config({}), parents, subs) == {}


def test_anchor_intervention_changes_only_classifier_path():
    torch.manual_seed(0)
    model = HiProbCBMStage2("resnet18", num_classes=3, subconcepts_per_concept=[2, 1], concept_dim=4,
                            pretrained=False, n_mc_samples_eval=3, parent_probability="hicem_mix").eval()
    with torch.no_grad():
        h = model.encode(torch.randn(2, 3, 64, 64))
        torch.manual_seed(1)
        base = model.forward_features(h)
        torch.manual_seed(1)
        values = torch.tensor([[1., float("nan")], [0., float("nan")]])
        intervened = model.forward_features(h, parent_intervention=values)
    assert torch.allclose(base.parent_concept_probs, intervened.parent_concept_probs)
    assert not torch.allclose(base.task_probs, intervened.task_probs)
    legacy = HiProbCBMStage2("resnet18", num_classes=3, subconcepts_per_concept=[1], pretrained=False)
    with pytest.raises(ValueError):
        legacy.forward_features(legacy.encode(torch.randn(1, 3, 64, 64)), parent_intervention=torch.ones(1, 1))


def test_stage1_complete_requires_same_config_and_seed(tmp_path):
    import json
    from hiprobcbm.config import apply_seed_override, load_config
    from hiprobcbm.engine.train_stage1 import stage1_complete
    config = Path(__file__).resolve().parents[1] / "configs" / "cub_stage1_probcbm_reported.yaml"
    for name in ("pseudo_hierarchy.pt", "stage1_best.pth"):
        (tmp_path / name).write_bytes(b"x")
    values = apply_seed_override(load_config(config), 43).to_dict()
    for key in ("config_path", "log_dir"):  # sama seperti run_identity menulis manifest
        values.pop(key)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"config": values}))
    assert stage1_complete(config, 43, tmp_path)
    assert not stage1_complete(config, 44, tmp_path)
    values["train"]["lr_stage1"] = 1e-4  # konfigurasi lama
    (tmp_path / "run_manifest.json").write_text(json.dumps({"config": values}))
    assert not stage1_complete(config, 43, tmp_path)


def test_reset_script_dry_run_then_apply(tmp_path):
    import subprocess
    import sys
    root = tmp_path / "train_log"
    (root / "probcbm" / "utama").mkdir(parents=True)
    (root / "probcbm" / "utama" / "a.pth").write_bytes(b"1234")
    (root / "hicem").mkdir()
    script = Path(__file__).resolve().parents[1] / "scripts" / "reset_protocol_results.py"

    def run(*extra):
        return subprocess.run([sys.executable, str(script), "--protocol", "probcbm", "--train-log", str(root), *extra],
                              capture_output=True, text=True, check=True)

    assert "DRY-RUN" in run().stdout and (root / "probcbm").exists()
    run("--apply")
    assert not (root / "probcbm").exists() and (root / "hicem").exists()


def test_stage1_warm_epoch_freezes_only_backbone():
    torch.manual_seed(0)
    model = HiProbCBMStage1("resnet18", num_concepts=2, concept_dim=4, pretrained=False)
    opt = optimizer_for(model, Config({"optimizer": "adamp", "lr_ratio": 10.0}), 1e-3)
    loader = [{"image": torch.randn(4, 3, 64, 64), "concepts": (torch.rand(4, 2) > .5).float()}]
    conv = model.backbone.module.conv1.weight.detach().clone()
    head = model.concept_predictor.mean_heads[0].weight.detach().clone()
    train_stage1.train_one_epoch(model, loader, opt, torch.device("cpu"), 5e-5, 2,
                                 Config({"clip_grad_norm": 2.0}), warm=True)
    assert torch.equal(model.backbone.module.conv1.weight, conv)
    assert not torch.equal(model.concept_predictor.mean_heads[0].weight, head)
    assert model.backbone.module.conv1.weight.requires_grad  # dipulihkan setelah warm
    train_stage1.train_one_epoch(model, loader, opt, torch.device("cpu"), 5e-5, 2, Config({}), warm=False)
    assert not torch.equal(model.backbone.module.conv1.weight, conv)


class _GroupedFakeDataset(_FakeDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from hiprobcbm.data.base import ConceptGroup
        self.concept_groups = [ConceptGroup("g0", [0, 1]), ConceptGroup("g1", [2, 3])]


@pytest.mark.parametrize("protocol", ["probcbm", "hicem"])
def test_full_revision_pipeline_with_interventions(tmp_path, protocol):
    DATASET_REGISTRY["dummy_e2e"] = _GroupedFakeDataset
    try:
        values = _build_config().to_dict()
        values["model"]["parent_probability"] = "hicem_mix"
        if protocol == "probcbm":
            values["train"].update({"optimizer": "adamp", "lr_stage1": 1e-3, "lr_stage2": 1e-3, "lr_ratio": 10.0,
                                    "warm_epochs": 1, "clip_grad_norm": 2.0, "scheduler": "cosine",
                                    "stage2_init": "stage1_frozen_concepts",
                                    "intervention_training": {"mode": "probcbm", "prob": 0.5}})
        else:
            values["train"].update({"lambda_concept": 10.0, "concept_loss_weighting": "hicem_imbalance",
                                    "concept_loss_reduction": "joint",
                                    "intervention_training": {"mode": "hicem", "prob": 0.25}})
        cfg = Config(values)
        device = torch.device("cpu")
        stage1_dir, stage2_dir = tmp_path / "stage1", tmp_path / "stage2"
        stage1_dir.mkdir()
        stage2_dir.mkdir()
        train_stage1.run(cfg, device, stage1_dir)
        train_stage2.run(cfg, device, stage2_dir, stage1_dir)
        counts = torch.load(stage1_dir / "pseudo_hierarchy.pt", map_location="cpu",
                            weights_only=True)["subconcepts_per_concept"]
        result = evaluate.evaluate_hiprobcbm_interventions(cfg, stage2_dir / "stage2_best.pth", counts, device,
                                                           n_random_orders=2)
        assert result["protocol"] == protocol
        if protocol == "probcbm":
            assert set(result["curves"]) == {"random_0", "random_1", "uncertainty", "random_mean"}
            assert all(len(v) == 3 for v in result["curves"].values())  # 0, 1, 2 grup
        else:
            assert len(result["curves"]["cumulative"]) == len(result["curves"]["one_at_a_time"]) == 5
        assert -1.0 <= result["intervention_delta"] <= 1.0
    finally:
        DATASET_REGISTRY.pop("dummy_e2e", None)
