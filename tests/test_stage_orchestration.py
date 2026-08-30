"""Integrasi orkestrasi penuh: `scripts/run_stage1.py` -> `pseudo_hierarchy.pt`
-> `scripts/run_stage2.py`, lewat I/O disk sungguhan (bukan cuma memanggil
modul model secara langsung seperti `test_hiprobcbm_end_to_end.py`).

Ditambahkan setelah ditemukan bug penyelarasan: sebelum perbaikan,
`train_stage1.run()` mengekstrak mean konsep lewat `train_loader` yang
SAMA dengan yang dipakai training (shuffle=True), lalu menyimpannya tanpa
identitas apa pun - sementara `train_stage2.run()` mengonsumsi pseudo-label
itu lewat pengindeksan posisi yang mengasumsikan urutan loader shuffle=False.
Akibatnya pseudo-label subkonsep bisa tertaut ke citra yang salah tanpa
ada error apa pun (silent bug). Perbaikannya: Tahap 1 kini mengekstrak
lewat loader khusus (shuffle=False, drop_last=False) dan menyimpan `path`
per-baris; Tahap 2 mencocokkan lewat `path`, bukan indeks posisi.

Test ini memakai dataset sintetis kecil (bukan CUB/PseudoKitchens asli)
supaya bisa dijalankan cepat di CPU tanpa dataset eksternal.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from hiprobcbm.config import Config
from hiprobcbm.data import DATASET_REGISTRY
from hiprobcbm.data.base import ConceptDataset, DatasetSplits
from hiprobcbm.engine import train_stage1, train_stage2

N_TRAIN, N_VAL, NUM_CONCEPTS, NUM_CLASSES, IMG_SIZE = 18, 8, 4, 3, 64


class _FakeSplit(Dataset):
    """Setiap sampel punya `path` unik ("fake_0000.png", dst.) yang dipakai
    untuk memverifikasi penautan pseudo-label -> citra."""

    def __init__(self, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, IMG_SIZE, IMG_SIZE, generator=g)
        self.concepts = (torch.rand(n, NUM_CONCEPTS, generator=g) > 0.5).float()
        self.labels = torch.randint(0, NUM_CLASSES, (n,), generator=g)
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],
            "concepts": self.concepts[idx],
            "label": self.labels[idx],
            "path": f"fake_{idx:04d}.png",
        }


class _FakeDataset(ConceptDataset):
    """Dataset dummy yang taat kontrak `ConceptDataset`, tanpa perlu citra
    asli di disk - dipakai HANYA oleh test ini lewat pendaftaran sementara
    ke `DATASET_REGISTRY`."""

    def __init__(self, **kwargs):
        super().__init__(
            name="dummy_e2e",
            num_concepts=NUM_CONCEPTS,
            num_classes=NUM_CLASSES,
            concept_names=[f"concept_{i}" for i in range(NUM_CONCEPTS)],
        )

    def get_splits(self, image_size: int, backbone: str) -> DatasetSplits:
        return DatasetSplits(
            train=_FakeSplit(N_TRAIN, seed=1),
            val=_FakeSplit(N_VAL, seed=2),
            test=_FakeSplit(N_VAL, seed=3),
        )


def _build_config() -> Config:
    return Config(
        {
            "dataset": "dummy_e2e",
            "seed": 0,
            "data": {},
            "model": {
                "backbone": "resnet18",
                "image_size": IMG_SIZE,
                "concept_dim": 8,
                "pretrained": False,
                "n_mc_samples_train": 2,
                "n_mc_samples_eval": 3,
                "use_attention": True,
            },
            "train": {
                "batch_size": 4,  # tidak habis membagi N_TRAIN=18 -> drop_last benar-benar diuji
                "lr_stage1": 1e-3,
                "lr_stage2": 1e-3,
                "epochs_stage1": 1,
                "epochs_stage2": 1,
                "lambda_kl": 1e-5,
                "lambda_concept": 1.0,
                "lambda_sub": 1.0,
            },
            "sae": {
                "variant": "kl",
                "activation_threshold": 0.5,
                "presence_threshold": 0.3,  # dilonggarkan supaya sample_filter tidak kosong pada data acak
                "kwargs": {"latent_dim": 16},
                "train_kwargs": {"n_epochs": 2, "batch_size": 8},
            },
        }
    )


def test_stage1_then_stage2_orchestration(tmp_path):
    DATASET_REGISTRY["dummy_e2e"] = _FakeDataset
    try:
        cfg = _build_config()
        device = torch.device("cpu")

        stage1_dir = tmp_path / "stage1"
        stage1_dir.mkdir()
        train_stage1.run(cfg, device, stage1_dir)

        pseudo_path = stage1_dir / "pseudo_hierarchy.pt"
        assert pseudo_path.exists(), "pseudo_hierarchy.pt harus tersimpan setelah Tahap 1"

        payload = torch.load(pseudo_path, map_location="cpu", weights_only=True)
        # Ekstraksi memakai drop_last=False -> HARUS mencakup seluruh N_TRAIN
        # sampel, membuktikan tidak ada sampel yang diam-diam hilang.
        assert len(payload["paths"]) == N_TRAIN
        assert set(payload["paths"]) == {f"fake_{i:04d}.png" for i in range(N_TRAIN)}
        assert payload["pseudo_labels"].shape[0] == N_TRAIN

        stage2_dir = tmp_path / "stage2"
        stage2_dir.mkdir()
        train_stage2.run(cfg, device, stage2_dir, stage1_dir)

        assert (stage2_dir / "stage2_last.pth").exists()
        assert (stage2_dir / "stage2_best.pth").exists()
    finally:
        DATASET_REGISTRY.pop("dummy_e2e", None)


def test_evaluate_baseline_checkpoint(tmp_path):
    """Bab IV.7: `evaluate.py` harus bisa memuat checkpoint baseline yang
    baru dilatih dan menghasilkan metrik yang valid - jalur ini belum
    pernah benar-benar dieksekusi sebelumnya (hanya diperiksa statis)."""
    from hiprobcbm.config import Config
    from hiprobcbm.engine import evaluate, train_baseline

    DATASET_REGISTRY["dummy_e2e"] = _FakeDataset
    try:
        cfg = Config(
            {
                "dataset": "dummy_e2e",
                "baseline": "cbm",
                "data": {},
                "model": {"backbone": "resnet18", "image_size": IMG_SIZE, "pretrained": False},
                "train": {"batch_size": 4, "lr": 1e-3, "epochs": 1, "concept_weight": 1.0},
            }
        )
        device = torch.device("cpu")
        log_dir = tmp_path / "cbm"
        log_dir.mkdir()

        train_baseline.run(cfg, device, log_dir)
        assert (log_dir / "cbm_best.pth").exists()

        metrics = evaluate.evaluate_baseline_checkpoint(cfg, log_dir / "cbm_best.pth", device)
        assert 0.0 <= metrics["task_accuracy"] <= 1.0
        assert 0.0 <= metrics["concept_accuracy"] <= 1.0
    finally:
        DATASET_REGISTRY.pop("dummy_e2e", None)


def test_evaluate_hiprobcbm_checkpoint(tmp_path):
    from hiprobcbm.engine import evaluate

    DATASET_REGISTRY["dummy_e2e"] = _FakeDataset
    try:
        cfg = _build_config()
        device = torch.device("cpu")

        stage1_dir = tmp_path / "stage1"
        stage1_dir.mkdir()
        train_stage1.run(cfg, device, stage1_dir)

        stage2_dir = tmp_path / "stage2"
        stage2_dir.mkdir()
        train_stage2.run(cfg, device, stage2_dir, stage1_dir)

        payload = torch.load(stage1_dir / "pseudo_hierarchy.pt", map_location="cpu", weights_only=True)
        subconcepts_per_concept = payload["subconcepts_per_concept"]

        metrics = evaluate.evaluate_hiprobcbm_checkpoint(
            cfg, stage2_dir / "stage2_best.pth", subconcepts_per_concept, device
        )
        assert 0.0 <= metrics["task_accuracy"] <= 1.0
        assert 0.0 <= metrics["concept_accuracy"] <= 1.0
    finally:
        DATASET_REGISTRY.pop("dummy_e2e", None)


def test_path_to_row_lookup_is_bijective():
    from hiprobcbm.engine.train_stage2 import build_path_to_row

    paths = [f"fake_{i:04d}.png" for i in range(N_TRAIN)]
    mapping = build_path_to_row(paths)
    assert len(mapping) == N_TRAIN
    assert all(mapping[p] == i for i, p in enumerate(paths))


def test_path_to_row_rejects_duplicates():
    from hiprobcbm.engine.train_stage2 import build_path_to_row

    with __import__("pytest").raises(ValueError):
        build_path_to_row(["a.png", "b.png", "a.png"])


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_stage1_then_stage2_orchestration(Path(d))
    test_path_to_row_lookup_is_bijective()
    test_path_to_row_rejects_duplicates()
    print("OK: test_stage_orchestration")
