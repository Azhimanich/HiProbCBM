from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hiprobcbm.data.feature_cache import cached_feature_loader, prepare_clip_feature_cache


class _RawSplit(Dataset):
    def __init__(self, split: str) -> None:
        self.split = split

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.full((3, 4, 4), float(index)),
            "label": torch.tensor(index % 2),
            "concepts": torch.tensor([index % 2, 1.0]),
            "path": f"{self.split}/{index}.png",
        }


class _Dataset:
    name = "dummy_kitchens"

    def __init__(self, root: Path) -> None:
        self.data_root = root
        self.calls = 0

    def get_dataloader(self, split, image_size, backbone, batch_size, **kwargs):
        assert backbone == "clip_vit_l14"
        self.calls += 1
        return DataLoader(_RawSplit(split), batch_size=batch_size, shuffle=False)


class _FakeCLIP(nn.Module):
    # The test raw dataset deliberately has no transform attribute.
    preprocess = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1).repeat(1, 768)


def test_clip_feature_cache_is_reused_without_reopening_source(monkeypatch, tmp_path):
    (tmp_path / "manifest.csv").write_text("id\n0\n", encoding="utf-8")
    (tmp_path / "info.json").write_text("{}", encoding="utf-8")
    dataset = _Dataset(tmp_path)
    monkeypatch.setattr("hiprobcbm.data.feature_cache.build_backbone", lambda *args, **kwargs: _FakeCLIP())

    payloads, artifacts = prepare_clip_feature_cache(
        dataset, image_size=224, batch_size=2, num_workers=0, device=torch.device("cpu")
    )
    assert dataset.calls == 3
    assert all(path.exists() and path.with_suffix(".pt.sha256").exists() for path in artifacts)
    assert payloads["train"]["features"].shape == (3, 768)

    loader = cached_feature_loader(payloads["train"], batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 768)
    assert batch["path"] == ["train/0.png", "train/1.png"]

    def should_not_build(*args, **kwargs):
        raise AssertionError("CLIP tidak boleh dibangun ulang ketika cache valid")

    monkeypatch.setattr("hiprobcbm.data.feature_cache.build_backbone", should_not_build)
    reused, _ = prepare_clip_feature_cache(
        dataset, image_size=224, batch_size=2, num_workers=0, device=torch.device("cpu")
    )
    assert dataset.calls == 3
    assert torch.equal(reused["val"]["features"], payloads["val"]["features"])


def test_incomplete_feature_cache_is_rebuilt(monkeypatch, tmp_path):
    (tmp_path / "manifest.csv").write_text("id\n0\n", encoding="utf-8")
    (tmp_path / "info.json").write_text("{}", encoding="utf-8")
    partial = tmp_path / "representation_cache" / "clip_vitl14"
    partial.mkdir(parents=True)
    torch.save({"unfinished": True}, partial / "train.pt")
    dataset = _Dataset(tmp_path)
    monkeypatch.setattr("hiprobcbm.data.feature_cache.build_backbone", lambda *args, **kwargs: _FakeCLIP())

    payloads, artifacts = prepare_clip_feature_cache(
        dataset, image_size=224, batch_size=2, num_workers=0, device=torch.device("cpu"), splits=("train",)
    )
    assert dataset.calls == 1
    assert payloads["train"]["features"].shape == (3, 768)
    assert artifacts[0].with_suffix(".pt.sha256").exists()
