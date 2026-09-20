"""Antarmuka dataset bersama untuk seluruh model (baseline & HiProbCBM).

Menyediakan satu abstraksi (`ConceptDataset`) yang dipakai baik oleh
baseline (CBM/CEM/ProbCBM/HiCEM) maupun HiProbCBM, supaya pipeline data -
augmentasi, split train/val/test, definisi grup konsep - identik di
seluruh model yang dibandingkan (syarat head-to-head yang adil, lihat
Bab IV.4 & catatan "Keputusan Desain Eksperimen").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class ConceptGroup:
    """Satu grup konsep top-level, dipakai untuk concept intervention
    berkelompok (ProbCBM Sec. 5.2.4 / Koh dkk. 2020) maupun untuk hierarki
    ground-truth eksplisit (PseudoKitchens: ingredient_group -> ingredient).
    """

    name: str
    concept_indices: list[int]


@dataclass
class DatasetSplits:
    train: Dataset
    val: Dataset
    test: Dataset


@dataclass
class ConceptDataset:
    """Kontrak minimal yang harus dipenuhi setiap dataset konsep.

    Atribut:
        name: identitas dataset ("cub" / "kitchens").
        num_concepts: jumlah konsep top-level (C pada notasi Bab IV).
        num_classes: jumlah kelas tugas akhir.
        concept_names: nama tiap konsep, urutan sejajar dengan label vektor.
        concept_groups: pengelompokan konsep untuk intervention & (opsional)
            ground-truth hierarki eksplisit.
        has_ground_truth_hierarchy: True untuk PseudoKitchens (Tabel 4.3:
            "ground-truth struktur konsep hierarkis eksplisit"), False untuk
            CUB (struktur harus diasumsikan dari nama atribut).
    """

    name: str
    num_concepts: int
    num_classes: int
    concept_names: list[str]
    concept_groups: list[ConceptGroup] = field(default_factory=list)
    has_ground_truth_hierarchy: bool = False

    def get_splits(self, image_size: int, backbone: str) -> DatasetSplits:  # pragma: no cover - abstrak
        raise NotImplementedError

    def get_dataloader(
        self,
        split: str,
        image_size: int,
        backbone: str,
        batch_size: int,
        num_workers: int = 4,
        shuffle: bool | None = None,
        drop_last: bool | None = None,
        augment: bool | None = None,
    ) -> DataLoader:
        splits = self.get_splits(image_size=image_size, backbone=backbone)
        dataset = getattr(splits, split)
        if augment is not None and hasattr(dataset, "transform"):
            from hiprobcbm.data.transforms import build_transform
            dataset.transform = build_transform(backbone, image_size, train=augment)
        if shuffle is None:
            shuffle = split == "train"
        if drop_last is None:
            drop_last = split == "train"
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
        )


def collate_concept_batch(batch: Iterable[dict]) -> dict[str, torch.Tensor]:
    """Collate default: menumpuk tensor per-kunci, aman dipakai lintas dataset
    selama setiap item punya kunci {'image', 'concepts', 'label'}."""
    out: dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out


def derive_attribute_groups(attribute_names: list[str], separator: str = "::") -> list[ConceptGroup]:
    """Mengelompokkan atribut lewat prefiks bersama sebelum `separator`,
    mengikuti prosedur pengelompokan atribut CUB pada Koh dkk. (2020) dan
    dipakai ulang oleh ProbCBM (Lampiran C.2 proposal, `attr_group_dict`).

    Contoh: "has_bill_shape::hooked" dan "has_bill_shape::pointed" masuk ke
    grup "has_bill_shape" yang sama.
    """
    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(attribute_names):
        prefix = name.split(separator)[0] if separator in name else name
        groups.setdefault(prefix, []).append(idx)
    return [ConceptGroup(name=prefix, concept_indices=idxs) for prefix, idxs in groups.items()]
