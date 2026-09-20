"""Fault injection: compare uninterrupted training with an actual restart."""
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hiprobcbm.config import Config, apply_seed_override
from hiprobcbm.utils.checkpoint import RunCheckpoint, file_hash, save_artifact
from hiprobcbm.utils.seed import set_random_seed


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def new_run(path, identity=None, resume="auto"):
    model = nn.Sequential(nn.Linear(3, 5), nn.Dropout(.3), nn.Linear(5, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    return RunCheckpoint(path, "toy", model, optimizer, identity or {"config": {"seed": 7}}, 4, resume)


def epoch(run, index):
    for _ in range(3):
        x = torch.randn(4, 3) + random.random() + np.random.rand()
        loss = run.model(x).square().mean()
        run.optimizer.zero_grad()
        loss.backward()
        run.optimizer.step()
    run.save_epoch(index, [.2, .7, .4, .5][index])


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_exact_resume_optimizer_rng_best_and_next_epoch(tmp_path):
    set_random_seed(7)
    full = new_run(tmp_path / "full")
    for i in range(4):
        epoch(full, i)
    set_random_seed(7)
    split = new_run(tmp_path / "split")
    epoch(split, 0)
    epoch(split, 1)
    # Simulate partial next epoch: these unsaved updates must be discarded.
    split.optimizer.zero_grad()
    split.model(torch.randn(5, 3)).sum().backward()
    split.optimizer.step()
    set_random_seed(999)
    split = new_run(tmp_path / "split")
    assert split.start_epoch == 2
    for i in range(split.start_epoch, 4):
        epoch(split, i)
    assert_nested_equal(load(full.path), load(split.path))
    assert split.best_epoch == 1 and split.best_val == .7
    split.export_weights()
    assert_nested_equal(load(tmp_path / "split" / "toy_best.pth"), split.best_model)


def test_corruption_fallback_and_no_overwrite_on_mismatch(tmp_path):
    run = new_run(tmp_path)
    epoch(run, 0)
    epoch(run, 1)
    run.path.write_bytes(b"interrupted upload")
    restored = new_run(tmp_path)
    assert restored.start_epoch == 1
    epoch(restored, 1)
    assert new_run(tmp_path).start_epoch == 2
    before = file_hash(run.path)
    with pytest.raises(ValueError, match="config berbeda"):
        new_run(tmp_path, {"config": {"seed": 9}})
    assert file_hash(run.path) == before


def test_legacy_checkpoint_and_missing_optimizer_rejected(tmp_path):
    torch.save(nn.Linear(3, 1).state_dict(), tmp_path / "toy_best.pth")
    with pytest.raises(ValueError, match="artifact lama"):
        new_run(tmp_path)
    run = new_run(tmp_path / "valid")
    epoch(run, 0)
    payload = load(run.path)
    payload["optimizer"]["state"] = {}
    save_artifact(payload, run.path)
    with pytest.raises(ValueError, match="Optimizer state kosong"):
        new_run(tmp_path / "valid")


def test_seed_zero_and_new_run_refuses_overwrite(tmp_path):
    assert apply_seed_override(Config({"seed": 42}), 0).seed == 0
    new_run(tmp_path)
    with pytest.raises(ValueError, match="Direktori berisi"):
        new_run(tmp_path, resume="never")


def test_nonfinite_does_not_replace_checkpoint(tmp_path):
    run = new_run(tmp_path)
    before = file_hash(run.path)
    with pytest.raises(ValueError, match="NaN"):
        run.save_epoch(0, float("nan"))
    assert file_hash(run.path) == before


def test_interruption_during_atomic_write_preserves_last_good_epoch(tmp_path, monkeypatch):
    run = new_run(tmp_path)
    epoch(run, 0)
    before = file_hash(run.path)
    original = torch.save

    def interrupted(value, stream):
        stream.write(b"incomplete")
        raise OSError("disk disconnected")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="disconnected"):
        epoch(run, 1)
    assert file_hash(run.path) == before
    monkeypatch.setattr(torch, "save", original)
    assert new_run(tmp_path).start_epoch == 1


@pytest.mark.parametrize("field", ["config", "code", "data", "environment", "artifacts"])
def test_identity_change_rejected_before_loading_weights(tmp_path, field):
    identity = {k: "original" for k in ["config", "code", "data", "environment", "artifacts"]}
    run = new_run(tmp_path, identity)
    epoch(run, 0)
    changed = dict(identity)
    changed[field] = "changed"
    with pytest.raises(ValueError, match=field + " berbeda"):
        new_run(tmp_path, changed)


class TinySplit(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        return {"image": torch.full((3, 2, 2), i / 8), "label": torch.tensor(i % 2),
                "concepts": torch.tensor([i % 2, (i // 2) % 2], dtype=torch.float), "path": f"{i}.jpg"}


class TinyDataset:
    num_concepts, num_classes = 2, 2
    concept_names = ["a", "b"]

    def get_dataloader(self, split, image_size, backbone, batch_size, **kwargs):
        return DataLoader(TinySplit(), batch_size=batch_size, shuffle=kwargs.get("shuffle", split == "train"),
                          num_workers=0, drop_last=kwargs.get("drop_last", split == "train"))


class TinyBackbone(nn.Module):
    output_dim = 3

    def forward(self, x):
        return x.mean(dim=(2, 3))


def tiny_config():
    return Config({"dataset": "tiny", "seed": 7, "data": {}, "baseline": "cbm",
                   "model": {"backbone": "resnet18", "image_size": 2, "concept_dim": 3, "pretrained": False,
                             "n_mc_samples_train": 2, "n_mc_samples_eval": 3},
                   "train": {"batch_size": 4, "num_workers": 0, "lr": .01, "lr_stage1": .01, "lr_stage2": .01,
                             "epochs": 3, "epochs_stage1": 3, "epochs_stage2": 3},
                   "sae": {"presence_threshold": .01, "kwargs": {"latent_dim": 4},
                           "train_kwargs": {"n_epochs": 3, "batch_size": 4, "checkpoint_interval": 1}}})


@pytest.mark.parametrize("stage", ["stage1", "sae", "stage2", "cbm"])
def test_real_trainers_resume_after_injected_disconnect(tmp_path, monkeypatch, stage):
    from hiprobcbm.engine import train_stage1, train_stage2, train_baseline
    import hiprobcbm.models.hiprobcbm as models
    import hiprobcbm.models.baselines.cbm as cbm
    for trainer in (train_stage1, train_stage2, train_baseline):
        monkeypatch.setattr(trainer, "build_dataset", lambda *a, **k: TinyDataset())
    monkeypatch.setattr(models, "build_backbone", lambda *a, **k: TinyBackbone())
    monkeypatch.setattr(cbm, "build_backbone", lambda *a, **k: TinyBackbone())
    cfg, device = tiny_config(), torch.device("cpu")
    parent = tmp_path / "parent"
    if stage == "stage2":
        train_stage1.run(cfg, device, parent)

    def train(path):
        if stage in ("stage1", "sae"):
            train_stage1.run(cfg, device, path)
        elif stage == "stage2":
            train_stage2.run(cfg, device, path, parent)
        else:
            train_baseline.run(cfg, device, path)

    set_random_seed(7)
    train(tmp_path / "full")
    original = RunCheckpoint.save_epoch
    interrupted = False

    def disconnect(self, epoch, val):
        nonlocal interrupted
        original(self, epoch, val)
        if self.name == stage and epoch == 0 and not interrupted:
            interrupted = True
            raise ConnectionError("injected disconnect")

    monkeypatch.setattr(RunCheckpoint, "save_epoch", disconnect)
    set_random_seed(7)
    with pytest.raises(ConnectionError, match="injected"):
        train(tmp_path / "split")
    set_random_seed(999)
    train(tmp_path / "split")
    name = "stage1" if stage == "sae" else stage
    for suffix in ("resume.pth", "best.pth", "last.pth"):
        assert_nested_equal(load(tmp_path / "full" / f"{name}_{suffix}"), load(tmp_path / "split" / f"{name}_{suffix}"))
    if stage in ("stage1", "sae"):
        assert_nested_equal(load(tmp_path / "full" / "pseudo_hierarchy.pt"), load(tmp_path / "split" / "pseudo_hierarchy.pt"))
        before = file_hash(tmp_path / "split" / "pseudo_hierarchy.pt")
        train(tmp_path / "split")  # completed runs must preserve the downstream artifact hash
        assert file_hash(tmp_path / "split" / "pseudo_hierarchy.pt") == before
