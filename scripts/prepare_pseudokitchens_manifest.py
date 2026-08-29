#!/usr/bin/env python
"""Konversi PseudoKitchens format render mentah (HiCEM) -> manifest rata
yang dipakai `hiprobcbm.data.kitchens.PseudoKitchensDataset`.

Dataset asli (lihat repo `OscarPi/cem-concept-discovery`, `cemcd/data/kitchens.py`)
disebar sebagai citra PNG + metadata EXR cryptomatte per-split, dibaca
lewat pustaka `OpenEXR` (dependensi opsional, HANYA dibutuhkan skrip ini,
bukan oleh training/evaluasi sehari-hari).

Penggunaan:
    python scripts/prepare_pseudokitchens_manifest.py \\
        --raw-dir /path/ke/pseudokitchens_V2 \\
        --out-dir datasets/PseudoKitchens

Keluaran (di --out-dir):
    images/<split>/<idx>.png   - citra disalin apa adanya
    info.json                  - {"ingredient_groups": ..., "recipes": ...}
    manifest.csv                - image, split, recipe, <kolom per ingredient>

Menjaga head-to-head fairness dengan HiCEM: citra & label yang dipakai
identik dengan dataset asli, hanya formatnya yang diratakan.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
from pathlib import Path

import numpy as np


def hex_str_to_id(id_hex_string: str) -> float:
    packed = struct.Struct("=I").pack(int(id_hex_string, 16))
    return struct.Struct("=f").unpack(packed)[0]


def image_contains_ingredient(channel1, channel2, manifest: dict, dataset_info: dict, ingredient: str) -> bool:
    count = dataset_info["object_counts"]["ingredient_counts"].get(ingredient, 0)
    for i in range(1, count + 1):
        key = f"{ingredient} {i}"
        if key in manifest:
            obj_id = hex_str_to_id(manifest[key])
            if np.any(channel1 == obj_id) or np.any(channel2 == obj_id):
                return True
    return False


def convert_split(raw_dir: Path, out_dir: Path, split: str, dataset_info: dict, ingredients: list[str]) -> list[dict]:
    import OpenEXR  # dependensi opsional - hanya dibutuhkan skrip konversi ini

    split_dir = raw_dir / split
    n = dataset_info[f"{split}_size"]
    width = len(str(n))

    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(1, n + 1):
        stem = str(i).zfill(width)
        src_png = split_dir / f"{stem}.png"
        dst_png = out_dir / "images" / split / f"{stem}.png"
        shutil.copyfile(src_png, dst_png)

        with (split_dir / f"{stem}.json").open() as f:
            recipe_idx = json.load(f)["recipe_idx"]
        recipe_name = dataset_info["recipes"][recipe_idx]

        with OpenEXR.File(str(split_dir / f"{stem}.exr")) as exrfile:
            manifest = json.loads(exrfile.header()["cryptomatte/f42029d/manifest"])
            channel1 = exrfile.channels()["CryptoAsset00.r"].pixels
            channel2 = exrfile.channels()["CryptoAsset00.b"].pixels

        row = {"image": f"{split}/{stem}.png", "split": split, "recipe": recipe_name}
        for ingredient in ingredients:
            row[ingredient] = int(image_contains_ingredient(channel1, channel2, manifest, dataset_info, ingredient))
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=str, required=True, help="Direktori pseudokitchens_V2 hasil unduhan HiCEM.")
    parser.add_argument("--out-dir", type=str, required=True, help="Direktori keluaran manifest (dipakai --data-root).")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (raw_dir / "info.json").open() as f:
        dataset_info = json.load(f)

    # Ingredient "lepas" (bukan anggota ingredient_group manapun) tetap
    # dicatat di manifest supaya bisa dipakai HiCEM baseline (Tabel 4.7).
    grouped = {ing for group in dataset_info["ingredient_groups"].values() for ing in group}
    all_ingredients = sorted(dataset_info["object_counts"]["ingredient_counts"].keys())

    rows: list[dict] = []
    for split in ("train", "val", "test"):
        print(f"Memproses split '{split}'...")
        rows.extend(convert_split(raw_dir, out_dir, split, dataset_info, all_ingredients))

    with (out_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"ingredient_groups": dataset_info["ingredient_groups"], "recipes": dataset_info["recipes"]},
            f,
            indent=2,
        )

    fieldnames = ["image", "split", "recipe", *all_ingredients]
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Selesai. {len(rows)} baris ditulis ke {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
