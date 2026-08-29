"""PseudoKitchens - dataset validasi hierarchical subconcept discovery
(Bab IV.4, Tabel 4.3 & Lampiran proposal; diperkenalkan oleh Hill dkk., 2026).

Dataset asli HiCEM disebar sebagai render 3D mentah (PNG + metadata EXR
cryptomatte per-instance) yang hanya bisa dibaca dengan pustaka `OpenEXR`.
Supaya loader ini tetap ringan dan tidak mewajibkan dependensi tersebut
untuk training/evaluasi sehari-hari, HiProbCBM membaca satu **manifest
rata** (`manifest.csv` + `info.json`) yang sudah diekstrak sekali di awal:

    info.json:
        {
          "ingredient_groups": {"mengandung buah": ["Apple", "Banana", ...], ...},
          "recipes": ["Fruit Salad", "Vegetable Pasta", ...]
        }

    manifest.csv (satu baris per citra, kolom biner per ingredient):
        image,split,recipe,Banana,Onion,Garlic,...

Ini persis format yang dicontohkan pada Lampiran proposal ("Contoh Isi
Dataset PseudoKitchens"). Skrip `scripts/prepare_pseudokitchens_manifest.py`
menyediakan konversi dari format render mentah HiCEM ke manifest ini
(memakai OpenEXR sebagai dependensi opsional), sehingga kedua model tetap
dilatih di atas citra yang identik demi menjaga head-to-head fairness.

Berbeda dari CUB, hierarki di sini bersifat **ground-truth eksplisit**:
`ingredient_group` adalah konsep induk, `ingredient` adalah subkonsep -
dipakai pada Bab IV.4 untuk mengukur validitas automatic subconcept
discovery terhadap struktur yang sudah diketahui (`has_ground_truth_hierarchy=True`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from hiprobcbm.data.base import ConceptDataset, ConceptGroup, DatasetSplits
from hiprobcbm.data.transforms import build_transform


class _KitchensSplitDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: Path, concept_columns: list[str],
                 recipe_to_idx: dict[str, int], transform):
        self.frame = frame.reset_index(drop=True)
        self.image_root = image_root
        self.concept_columns = concept_columns
        self.recipe_to_idx = recipe_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        image = Image.open(self.image_root / row["image"]).convert("RGB")
        image = self.transform(image)

        concepts = torch.tensor(row[self.concept_columns].to_numpy(dtype="float32"))
        label = torch.tensor(self.recipe_to_idx[row["recipe"]], dtype=torch.long)

        return {"image": image, "concepts": concepts, "label": label, "path": row["image"]}


class PseudoKitchensDataset(ConceptDataset):
    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        with open(self.data_root / "info.json", "r", encoding="utf-8") as f:
            info = json.load(f)

        self.ingredient_groups: dict[str, list[str]] = info["ingredient_groups"]
        self.recipes: list[str] = info["recipes"]
        self.recipe_to_idx = {name: i for i, name in enumerate(self.recipes)}

        manifest = pd.read_csv(self.data_root / "manifest.csv")
        self._manifest = manifest

        # Konsep top-level = ingredient_group (subkonsep) + ingredient tunggal
        # yang tidak tergabung dalam grup manapun (mengikuti KitchensDatasets
        # milik HiCEM: top_level = ingredient_groups + ingredient lepas).
        grouped_ingredients = {ing for group in self.ingredient_groups.values() for ing in group}
        all_ingredient_columns = [c for c in manifest.columns if c not in ("image", "split", "recipe")]
        standalone = [c for c in all_ingredient_columns if c not in grouped_ingredients]
        concept_names = sorted(self.ingredient_groups.keys()) + sorted(standalone)

        concept_groups = [
            ConceptGroup(name=group, concept_indices=[concept_names.index(group)])
            for group in self.ingredient_groups
        ]

        self._concept_columns_for_group: dict[str, list[str]] = dict(self.ingredient_groups)
        self._standalone_columns = standalone

        super().__init__(
            name="kitchens",
            num_concepts=len(concept_names),
            num_classes=len(self.recipes),
            concept_names=concept_names,
            concept_groups=concept_groups,
            has_ground_truth_hierarchy=True,
        )

    def _materialize_top_level_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Membentuk kolom konsep induk = OR dari subkonsepnya (ingredient
        group aktif jika salah satu ingredient anggotanya muncul di citra),
        sejalan dengan `KitchensDataset.__getitem__` pada repo HiCEM."""
        frame = frame.copy()
        for group, members in self._concept_columns_for_group.items():
            frame[group] = (frame[members].sum(axis=1) > 0).astype("float32")
        return frame

    def get_ground_truth_hierarchy(self) -> dict[str, list[str]]:
        """Dipakai Bab IV.7.1.1 (ROC-AUC subkonsep) untuk mencocokkan hasil
        automatic subconcept discovery terhadap struktur asli."""
        return dict(self.ingredient_groups)

    def get_splits(self, image_size: int, backbone: str) -> DatasetSplits:
        image_root = self.data_root / "images"
        manifest = self._materialize_top_level_columns(self._manifest)
        train_tf = build_transform(backbone, image_size, train=True)
        eval_tf = build_transform(backbone, image_size, train=False)

        splits = {}
        for split_name, transform in (("train", train_tf), ("val", eval_tf), ("test", eval_tf)):
            frame = manifest[manifest["split"] == split_name]
            splits[split_name] = _KitchensSplitDataset(
                frame, image_root, self.concept_names, self.recipe_to_idx, transform
            )
        return DatasetSplits(train=splits["train"], val=splits["val"], test=splits["test"])
