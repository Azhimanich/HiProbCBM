"""Persistent frozen-backbone representations for large image encoders.

HiCEM's reference implementation first encodes each dataset split with the
frozen foundation model and then trains its concept heads on a TensorDataset.
Doing the same here is essential for CLIP ViT-L/14: repeatedly encoding images
from Google Drive at every epoch is needlessly slow and makes a Colab run
impractical.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from hiprobcbm.models.backbones import build_backbone
from hiprobcbm.utils.checkpoint import file_hash, load_artifact, save_artifact


LOGGER = logging.getLogger(__name__)
_SCHEMA_VERSION = 1
_SPLITS = ("train", "val", "test")


class FeatureCacheIdentityError(ValueError):
    """A complete cache belongs to a different dataset/protocol."""


class CachedFeatureDataset(Dataset):
    """A dataset of CLIP features and labels with the normal project contract."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.features = payload["features"]
        self.labels = payload["labels"]
        self.concepts = payload["concepts"]
        self.paths = payload["paths"]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.features[index],
            "label": self.labels[index],
            "concepts": self.concepts[index],
            "path": self.paths[index],
        }


def _cache_root(dataset: Any) -> Path:
    data_root = Path(dataset.data_root)
    return data_root / "representation_cache" / "clip_vitl14"


def _identity(dataset: Any, image_size: int) -> dict[str, Any]:
    data_root = Path(dataset.data_root)
    source_files = [data_root / name for name in ("manifest.csv", "info.json")]
    return {
        "schema": _SCHEMA_VERSION,
        "dataset": getattr(dataset, "name", type(dataset).__name__),
        "backbone": "clip_vit_l14",
        "image_size": int(image_size),
        "preprocess": "openai_clip_vitl14",
        "source": {
            path.name: file_hash(path) if path.exists() else None
            for path in source_files
        },
    }


def _validate_payload(payload: Any, expected_identity: dict[str, Any], path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("identity") != expected_identity:
        raise FeatureCacheIdentityError(
            f"Cache fitur {path} tidak cocok dengan dataset/transform saat ini. "
            "Hapus hanya folder representation_cache/clip_vitl14 lalu jalankan ulang."
        )
    required = ("features", "labels", "concepts", "paths")
    if any(key not in payload for key in required):
        raise RuntimeError(f"Cache fitur {path} tidak lengkap.")
    count = int(payload["features"].shape[0])
    if (
        payload["features"].ndim != 2
        or payload["features"].shape[1] != 768
        or payload["labels"].shape[0] != count
        or payload["concepts"].shape[0] != count
        or len(payload["paths"]) != count
    ):
        raise RuntimeError(f"Cache fitur {path} memiliki bentuk data tidak valid.")
    return payload


def _cache_one_split(
    dataset: Any,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    backbone: torch.nn.Module,
    identity: dict[str, Any],
    cache_file: Path,
) -> dict[str, Any]:
    source_loader = dataset.get_dataloader(
        split,
        image_size,
        "clip_vit_l14",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        augment=False,
    )
    # `clip.load` owns the exact resize/crop/interpolation contract used by
    # HiCEM.  The project-level transform is deliberately replaced here so
    # cached representations use the same foundation-model preprocessing.
    if hasattr(source_loader.dataset, "transform"):
        source_loader.dataset.transform = backbone.preprocess
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_concepts: list[torch.Tensor] = []
    all_paths: list[str] = []
    with torch.no_grad():
        for batch in tqdm(source_loader, desc=f"cache CLIP/{split}"):
            all_features.append(backbone(batch["image"].to(device, non_blocking=True)).cpu())
            all_labels.append(batch["label"].cpu())
            all_concepts.append(batch["concepts"].cpu())
            all_paths.extend(str(item) for item in batch["path"])
    payload = {
        "identity": identity,
        "features": torch.cat(all_features),
        "labels": torch.cat(all_labels),
        "concepts": torch.cat(all_concepts),
        "paths": all_paths,
    }
    save_artifact(payload, cache_file)
    LOGGER.info("CACHE CLIP SAVED: %s (%d sampel)", cache_file, len(all_paths))
    return payload


def prepare_clip_feature_cache(
    dataset: Any,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    splits: Iterable[str] = _SPLITS,
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    """Load valid split caches, creating only missing caches with frozen CLIP."""
    split_names = tuple(splits)
    if any(split not in _SPLITS for split in split_names):
        raise ValueError(f"Split cache tidak didukung: {split_names}")
    identity = _identity(dataset, image_size)
    root = _cache_root(dataset)
    root.mkdir(parents=True, exist_ok=True)
    cache_files = {split: root / f"{split}.pt" for split in split_names}
    payloads: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for split, cache_file in cache_files.items():
        if cache_file.exists():
            try:
                payloads[split] = _validate_payload(load_artifact(cache_file), identity, cache_file)
            except FeatureCacheIdentityError:
                raise
            except (FileNotFoundError, ValueError) as exc:
                # A cache is a reproducible derivative, unlike a training
                # checkpoint.  If Colab stops between the atomic tensor and
                # checksum writes, rebuild only this split on the next run.
                checksum = cache_file.with_suffix(cache_file.suffix + ".sha256")
                cache_file.unlink(missing_ok=True)
                checksum.unlink(missing_ok=True)
                LOGGER.warning("CACHE CLIP TIDAK LENGKAP (%s); membangun ulang %s", exc, split)
                missing.append(split)
            else:
                LOGGER.info("CACHE CLIP VALID: %s", cache_file)
        else:
            missing.append(split)

    if missing:
        LOGGER.info("Membuat cache CLIP sekali untuk split: %s", ", ".join(missing))
        backbone = build_backbone("clip_vit_l14", pretrained=True).to(device).eval()
        try:
            for split in missing:
                payloads[split] = _cache_one_split(
                    dataset, split, image_size, batch_size, num_workers, device,
                    backbone, identity, cache_files[split],
                )
        finally:
            del backbone
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return payloads, [cache_files[split] for split in split_names]


def cached_feature_loader(
    payload: dict[str, Any],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
) -> DataLoader:
    """Return a loader that never reopens the original images."""
    return DataLoader(
        CachedFeatureDataset(payload),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
