"""Tahap 2: Hierarchical Concept Aggregation + Hierarchical Probabilistic
Reasoning (Bab IV.5.2), dilatih end-to-end memakai pseudo hierarchical
subconcept dataset dari Tahap 1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from tqdm import tqdm

from hiprobcbm.config import Config
from hiprobcbm.data import build_dataset
from hiprobcbm.losses import hiprobcbm_stage2_loss
from hiprobcbm.metrics import compute_standard_metrics
from hiprobcbm.models.hiprobcbm import HiProbCBMStage2
from hiprobcbm.utils.checkpoint import RunCheckpoint, run_identity, require_finite, load_artifact

logger = logging.getLogger(__name__)


def load_pseudo_hierarchy(path: Path) -> tuple[list[int], torch.Tensor, list[str]]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "identity" in payload:
        payload = load_artifact(path)
    paths = payload.get("paths")
    if paths is None:
        raise ValueError(
            f"'{path}' tidak memuat kunci 'paths' - kemungkinan dihasilkan sebelum perbaikan "
            "bug penyelarasan pseudo-label (lihat train_stage1.run). Jalankan ulang run_stage1.py."
        )
    counts, labels = payload["subconcepts_per_concept"], payload["pseudo_labels"]
    if (not counts or any(type(k) is not int or k < 1 for k in counts)
            or labels.ndim != 3 or labels.shape[:2] != (len(paths), len(counts))
            or labels.shape[2] != max(counts) or not paths):
        raise ValueError("Shape/jumlah child/path pseudo hierarchy tidak valid.")
    require_finite(labels)
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Pseudo-label harus biner.")
    build_path_to_row(paths)
    for index, count in enumerate(counts):
        if labels[:, index, count:].any():
            raise ValueError("Padding pseudo-label harus nol.")
        support = labels[:, index, :count].sum(dim=0)
        if count > 1 and (support == 0).any():
            raise ValueError(f"Parent {index}: child tanpa contoh positif (artifact SAE lama/rusak). "
                             "Periksa discovery sebelum Stage 2; jangan lanjutkan training panjang.")
        if count == 1 and support.item() == 0:
            logger.warning("Parent %d: fallback tunggal tanpa contoh positif; belum ada bukti discovery.", index)
    return counts, labels, paths


def build_path_to_row(paths: list[str]) -> dict[str, int]:
    """Peta identitas citra -> baris pada `pseudo_labels_all`, dipakai untuk
    menautkan pseudo-label Tahap 1 ke batch Tahap 2 TANPA bergantung pada
    kesamaan urutan iterasi dua DataLoader yang berbeda (lihat catatan di
    `HiProbCBMStage1.extract_means_for_discovery`)."""
    mapping = {path: i for i, path in enumerate(paths)}
    if len(mapping) != len(paths):
        raise ValueError("Ditemukan path citra duplikat pada pseudo_hierarchy.pt - identitas tidak unik.")
    return mapping


def train_one_epoch(model, loader, pseudo_labels_all, path_to_row, optimizer, device, cfg_train):
    model.train()
    running = {"total": 0.0, "class": 0.0, "concept": 0.0, "sub_concept": 0.0, "kl": 0.0}
    n_batches = 0

    for batch in tqdm(loader, desc="stage2/train", leave=False):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        c_parent = batch["concepts"].to(device)
        batch_size = x.shape[0]

        row_idx = torch.tensor([path_to_row[p] for p in batch["path"]], dtype=torch.long)
        sub_labels = pseudo_labels_all[row_idx].to(device)

        out = model(x)
        sub_mask = model.subconcept_predictor.mask.unsqueeze(0).expand(batch_size, -1, -1)

        loss, components = hiprobcbm_stage2_loss(
            task_logits_per_sample=out.task_logits_per_sample,
            task_labels=y,
            parent_concept_probs=out.parent_concept_probs,
            parent_concept_labels=c_parent,
            sub_concept_probs=out.sub_concept_probs,
            sub_concept_mask=sub_mask,
            sub_concept_labels=sub_labels,
            kl_loss=out.kl_loss,
            lambda_concept=cfg_train.get("lambda_concept", 1.0),
            lambda_sub=cfg_train.get("lambda_sub", 1.0),
            lambda_kl=cfg_train.get("lambda_kl", 5e-5),
        )

        require_finite(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running["total"] += loss.item()
        for k, v in components.items():
            running[k] += v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate_stage2(model: HiProbCBMStage2, loader, device):
    model.eval()
    all_task_probs, all_labels, all_concept_probs, all_concept_labels = [], [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        c = batch["concepts"].to(device)
        out = model(x)
        all_task_probs.append(out.task_probs.cpu())
        all_labels.append(y.cpu())
        all_concept_probs.append(out.parent_concept_probs.cpu())
        all_concept_labels.append(c.cpu())

    return compute_standard_metrics(
        torch.cat(all_task_probs), torch.cat(all_labels), torch.cat(all_concept_probs), torch.cat(all_concept_labels)
    )


def run(cfg: Config, device: torch.device, log_dir: Path, stage1_log_dir: Path, resume="auto") -> None:
    dataset = build_dataset(cfg.dataset, **cfg.data.to_dict())
    # shuffle=True aman dipakai di sini karena penautan sub_labels sekarang
    # berbasis identitas `path` per-batch (build_path_to_row), bukan lagi
    # asumsi urutan posisi yang identik dengan loader ekstraksi Tahap 1.
    train_loader = dataset.get_dataloader(
        "train", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size, num_workers=cfg.train.get("num_workers", 2), shuffle=True
    )
    val_loader = dataset.get_dataloader(
        "val", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size, num_workers=cfg.train.get("num_workers", 2), shuffle=False
    )

    subconcepts_per_concept, pseudo_labels_all, pseudo_paths = load_pseudo_hierarchy(
        stage1_log_dir / "pseudo_hierarchy.pt"
    )
    path_to_row = build_path_to_row(pseudo_paths)
    if len(subconcepts_per_concept) != dataset.num_concepts:
        raise ValueError("Jumlah parent hierarchy berbeda dengan dataset Stage 2.")

    model = HiProbCBMStage2(
        backbone_name=cfg.model.backbone,
        num_classes=dataset.num_classes,
        subconcepts_per_concept=subconcepts_per_concept,
        concept_dim=cfg.model.concept_dim,
        pretrained=cfg.model.get("pretrained", True),
        n_mc_samples_train=cfg.model.get("n_mc_samples_train", 8),
        n_mc_samples_eval=cfg.model.get("n_mc_samples_eval", 32),
        use_attention=cfg.model.get("use_attention", True),  # False -> HiProbCBM-A1 (Tabel 4.8)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr_stage2)

    checkpoint = RunCheckpoint(log_dir, "stage2", model, optimizer,
                               run_identity(cfg, device, [stage1_log_dir / "pseudo_hierarchy.pt"]),
                               cfg.train.epochs_stage2, resume)
    for epoch in range(checkpoint.start_epoch, cfg.train.epochs_stage2):
        train_stats = train_one_epoch(model, train_loader, pseudo_labels_all, path_to_row, optimizer, device, cfg.train)
        val_report = evaluate_stage2(model, val_loader, device)
        logger.info("epoch=%d train=%s val=%s", epoch, train_stats, val_report.as_dict())

        checkpoint.save_epoch(epoch, val_report.task_accuracy)
    checkpoint.export_weights()
