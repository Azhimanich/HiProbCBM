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
from hiprobcbm.data.feature_cache import cached_feature_loader, prepare_clip_feature_cache
from hiprobcbm.losses import hicem_concept_loss_weights, hiprobcbm_stage2_loss
from hiprobcbm.metrics import compute_standard_metrics
from hiprobcbm.models.hiprobcbm import HiProbCBMStage2
from hiprobcbm.utils.checkpoint import run_identity, require_finite, load_artifact
from hiprobcbm.engine.protocol import checkpoint_for, clip_gradients, optimizer_for, train_concept_labels

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


STAGE2_INIT_MODES = ("scratch", "stage1_frozen_concepts")


def build_stage2_model(cfg: Config, num_classes: int, subconcepts_per_concept: list[int],
                       pretrained: bool) -> HiProbCBMStage2:
    """Satu-satunya pembuat model Tahap 2, dipakai trainer dan evaluator agar
    konfigurasi yang dievaluasi selalu sama dengan yang dilatih."""
    return HiProbCBMStage2(
        backbone_name=cfg.model.backbone,
        num_classes=num_classes,
        subconcepts_per_concept=subconcepts_per_concept,
        concept_dim=cfg.model.concept_dim,
        pretrained=pretrained,
        n_mc_samples_train=cfg.model.get("n_mc_samples_train", 8),
        n_mc_samples_eval=cfg.model.get("n_mc_samples_eval", 32),
        use_attention=cfg.model.get("use_attention", True),  # False -> HiProbCBM-A1 (Tabel 4.8)
        use_cached_features=cfg.model.get("use_cached_features", False),
        parent_probability=cfg.model.get("parent_probability", "aggregate"),
    )


def initialize_from_stage1(model: HiProbCBMStage2, stage1_best: Path) -> None:
    """Muat backbone, prediktor konsep, dan scorer dari bobot terbaik Tahap 1,
    lalu bekukan jalur konsep (padanan ProbCBM sequential concept -> class)."""
    state = torch.load(stage1_best, map_location="cpu", weights_only=True)
    parts = {"backbone.": model.backbone, "concept_predictor.": model.parent_predictor, "scorer.": model.scorer}
    for prefix, module in parts.items():
        sub_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not sub_state:
            raise ValueError(f"{stage1_best} tidak memuat parameter '{prefix}'.")
        module.load_state_dict(sub_state, strict=True)
    model.freeze_concept_path()


def concept_loss_weights(cfg_train, train_loader, pseudo_labels_all, device):
    mode = cfg_train.get("concept_loss_weighting", "none")
    if mode == "none":
        return None, None
    if mode != "hicem_imbalance":
        raise ValueError(f"concept_loss_weighting tidak dikenal: {mode}")
    parent_labels = train_concept_labels(train_loader)
    if parent_labels.shape[0] != pseudo_labels_all.shape[0]:
        raise ValueError("Jumlah sampel train berbeda dengan pseudo hierarchy; bobot imbalance tidak sah.")
    return (hicem_concept_loss_weights(parent_labels).to(device),
            hicem_concept_loss_weights(pseudo_labels_all).to(device))


def training_interventions(model, cfg_train, parent_labels, sub_labels) -> dict:
    """Intervensi acak saat training, mengikuti baseline protokolnya:

    - ``probcbm``: tiap (sampel, sampel MC) dengan peluang ``prob`` memakai
      anchor ground truth untuk SELURUH konsep (build_model_resnset.py:
      ``torch.rand_like(gt[..., :1, :1]) < intervention_prob``; config_exp.yaml 0.5).
    - ``hicem``: RandInt, satu mask Bernoulli(``prob``) per langkah atas seluruh
      kolom top concept dan subkonsep, dibagi semua sampel batch
      (hicem.py: ``torch.bernoulli(torch.full((n_concepts,), 0.25))``).
    """
    spec = cfg_train.get("intervention_training") or {}
    mode = spec.get("mode", "none")
    if mode == "none":
        return {}
    prob = float(spec["prob"])
    device = parent_labels.device
    if mode == "probcbm":
        mask = torch.rand(parent_labels.shape[0], model.n_mc_samples_train, device=device) < prob
        return {"sample_intervention_mask": mask, "parent_labels": parent_labels}
    if mode == "hicem":
        nan = torch.tensor(float("nan"), device=device)
        parent_mask = torch.bernoulli(torch.full(parent_labels.shape[1:], prob, device=device)).bool()
        valid = model.subconcept_predictor.mask
        sub_mask = torch.bernoulli(torch.full(valid.shape, prob, device=device)).bool() & valid
        return {"parent_intervention": torch.where(parent_mask, parent_labels, nan),
                "sub_intervention": torch.where(sub_mask, sub_labels, nan)}
    raise ValueError(f"intervention_training.mode tidak dikenal: {mode}")


def train_one_epoch(model, loader, pseudo_labels_all, path_to_row, optimizer, device, cfg_train,
                    parent_weights=None, sub_weights=None):
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

        out = model(x, **training_interventions(model, cfg_train, c_parent, sub_labels))
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
            parent_weights=parent_weights,
            sub_weights=sub_weights,
            reduction=cfg_train.get("concept_loss_reduction", "separate"),
        )

        require_finite(loss)
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model, cfg_train)
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
    use_cached_features = cfg.model.get("use_cached_features", False)
    # shuffle=True aman dipakai di sini karena penautan sub_labels sekarang
    # berbasis identitas `path` per-batch (build_path_to_row), bukan lagi
    # asumsi urutan posisi yang identik dengan loader ekstraksi Tahap 1.
    feature_artifacts: list[Path] = []
    if use_cached_features:
        payloads, feature_artifacts = prepare_clip_feature_cache(
            dataset,
            image_size=cfg.model.image_size,
            batch_size=cfg.model.get("feature_cache_batch_size", cfg.train.batch_size),
            num_workers=cfg.train.get("num_workers", 2),
            device=device,
            splits=("train", "val"),
        )
        train_loader = cached_feature_loader(
            payloads["train"], batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=cfg.train.get("num_workers", 2), drop_last=True,
        )
        val_loader = cached_feature_loader(
            payloads["val"], batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=cfg.train.get("num_workers", 2), drop_last=False,
        )
    else:
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

    model = build_stage2_model(cfg, dataset.num_classes, subconcepts_per_concept,
                               pretrained=cfg.model.get("pretrained", True))
    init_mode = cfg.train.get("stage2_init", "scratch")
    if init_mode not in STAGE2_INIT_MODES:
        raise ValueError(f"stage2_init harus salah satu dari {STAGE2_INIT_MODES}.")
    identity_artifacts = feature_artifacts + [stage1_log_dir / "pseudo_hierarchy.pt"]
    if init_mode == "stage1_frozen_concepts":
        stage1_best = stage1_log_dir / "stage1_best.pth"
        initialize_from_stage1(model, stage1_best)
        identity_artifacts.append(stage1_best)
        logger.info("Tahap 2 memakai bobot konsep Tahap 1 yang dibekukan: %s", stage1_best)
    model = model.to(device)
    parent_weights, sub_weights = concept_loss_weights(cfg.train, train_loader, pseudo_labels_all, device)

    optimizer = optimizer_for(model, cfg.train, cfg.train.lr_stage2)

    checkpoint = checkpoint_for(log_dir, "stage2", model, optimizer,
                                run_identity(cfg, device, identity_artifacts),
                                cfg.train.epochs_stage2, cfg.train, resume)
    for epoch in range(checkpoint.start_epoch, cfg.train.epochs_stage2):
        train_stats = train_one_epoch(model, train_loader, pseudo_labels_all, path_to_row, optimizer, device,
                                      cfg.train, parent_weights, sub_weights)
        val_report = evaluate_stage2(model, val_loader, device)
        logger.info("epoch=%d train=%s val=%s", epoch, train_stats, val_report.as_dict())

        checkpoint.save_epoch(epoch, val_report.task_accuracy)
        if checkpoint.completed:
            break
    checkpoint.export_weights()
