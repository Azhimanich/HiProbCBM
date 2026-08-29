"""Caltech-UCSD Birds-200-2011 (CUB) - dataset pengujian utama (Bab IV.4,
Tabel 4.3 & Lampiran proposal).

Format berkas mengikuti konvensi standar yang dipakai lintas literatur CBM
(Koh dkk. 2020; diwarisi oleh CEM, ProbCBM, HiCEM): tiga berkas pickle
`train.pkl` / `val.pkl` / `test.pkl` di `<data_root>/metadata/`, masing-
masing berisi list of dict dengan kunci minimal:

    {
        "img_path": "<kelas>/<nama_file>.jpg",   # relatif terhadap <data_root>/images
        "class_label": int,                       # 0..199
        "attribute_label": list[int] (112,),       # 0/1 per konsep bersih
        "attribute_certainty": list[int] (112,),   # opsional, dipakai untuk analisis
    }

Nama 112 konsep bersih dan pengelompokannya (mis. "has_bill_shape",
"has_wing_color", dst.) dibaca dari `<data_root>/metadata/attributes.txt`
bila tersedia; jika tidak, dipakai nama generik `concept_{i}` supaya kode
tetap berjalan tanpa berkas tambahan (mis. untuk smoke test).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from hiprobcbm.data.base import ConceptDataset, DatasetSplits, derive_attribute_groups
from hiprobcbm.data.transforms import build_transform

DEFAULT_NUM_CONCEPTS = 112
DEFAULT_NUM_CLASSES = 200


def _load_attribute_names(data_root: Path, num_concepts: int) -> list[str]:
    attr_file = data_root / "metadata" / "attributes.txt"
    if attr_file.exists():
        names = []
        with open(attr_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # format umum: "<idx> <nama_atribut>"
                parts = line.split(maxsplit=1)
                names.append(parts[-1] if len(parts) > 1 else parts[0])
        if len(names) == num_concepts:
            return names
    return [f"concept_{i}" for i in range(num_concepts)]


class _CUBSplitDataset(Dataset):
    def __init__(self, records: list[dict], image_root: Path, transform):
        self.records = records
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        img_path = self.image_root / record["img_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        concepts = torch.tensor(record["attribute_label"], dtype=torch.float32)
        label = torch.tensor(record["class_label"], dtype=torch.long)

        return {
            "image": image,
            "concepts": concepts,
            "label": label,
            "path": record["img_path"],
        }


class CUBConceptDataset(ConceptDataset):
    def __init__(self, data_root: str | Path, num_concepts: int = DEFAULT_NUM_CONCEPTS):
        self.data_root = Path(data_root)
        concept_names = _load_attribute_names(self.data_root, num_concepts)
        concept_groups = derive_attribute_groups(concept_names)
        super().__init__(
            name="cub",
            num_concepts=num_concepts,
            num_classes=DEFAULT_NUM_CLASSES,
            concept_names=concept_names,
            concept_groups=concept_groups,
            has_ground_truth_hierarchy=False,
        )

    def _load_split(self, split: str) -> list[dict]:
        pkl_path = self.data_root / "metadata" / f"{split}.pkl"
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    def get_splits(self, image_size: int, backbone: str) -> DatasetSplits:
        image_root = self.data_root / "images"
        train_tf = build_transform(backbone, image_size, train=True)
        eval_tf = build_transform(backbone, image_size, train=False)
        return DatasetSplits(
            train=_CUBSplitDataset(self._load_split("train"), image_root, train_tf),
            val=_CUBSplitDataset(self._load_split("val"), image_root, eval_tf),
            test=_CUBSplitDataset(self._load_split("test"), image_root, eval_tf),
        )
