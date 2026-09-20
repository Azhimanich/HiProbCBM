"""Trainer generik untuk baseline (CBM/CEM/ProbCBM/HiCEM) - Tabel 4.7.

Satu fungsi `run()` melayani keempat model lewat `cfg.baseline`, memakai
dataset & metrik yang SAMA dengan HiProbCBM demi head-to-head fairness.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from tqdm import tqdm

from hiprobcbm.config import Config
from hiprobcbm.data import build_dataset
from hiprobcbm.losses import cbm_loss, cem_loss, hicem_loss, probcbm_loss
from hiprobcbm.metrics import compute_standard_metrics
from hiprobcbm.models.baselines import BASELINE_REGISTRY
from hiprobcbm.utils.checkpoint import RunCheckpoint, run_identity, require_finite

logger = logging.getLogger(__name__)


def build_model(name: str, cfg: Config, dataset):
    model_cfg = cfg.model.to_dict()
    if name == "cbm":
        return BASELINE_REGISTRY["cbm"](
            backbone_name=model_cfg["backbone"], num_concepts=dataset.num_concepts,
            num_classes=dataset.num_classes, pretrained=model_cfg.get("pretrained", True),
        )
    if name == "cem":
        return BASELINE_REGISTRY["cem"](
            backbone_name=model_cfg["backbone"], num_concepts=dataset.num_concepts,
            num_classes=dataset.num_classes, embedding_dim=model_cfg.get("embedding_dim", 16),
            pretrained=model_cfg.get("pretrained", True),
        )
    if name == "probcbm":
        return BASELINE_REGISTRY["probcbm"](
            backbone_name=model_cfg["backbone"], num_concepts=dataset.num_concepts,
            num_classes=dataset.num_classes, concept_dim=model_cfg.get("concept_dim", 16),
            pretrained=model_cfg.get("pretrained", True),
            classifier_head=model_cfg.get("classifier_head", "linear"),
            n_mc_samples_train=model_cfg.get("n_mc_samples_train", 8),
            n_mc_samples_eval=model_cfg.get("n_mc_samples_eval", 32),
        )
    if name == "hicem":
        n_top = dataset.num_concepts
        n_sub = model_cfg.get("n_subconcepts_per_side", 4)
        subconcepts_per_concept = [(n_sub, n_sub) for _ in range(n_top)]
        return BASELINE_REGISTRY["hicem"](
            backbone_name=model_cfg["backbone"], num_classes=dataset.num_classes,
            subconcepts_per_concept=subconcepts_per_concept,
            embedding_dim=model_cfg.get("embedding_dim", 16), pretrained=model_cfg.get("pretrained", True),
        )
    raise ValueError(f"Baseline tidak dikenal: {name!r}")


def training_step(name: str, model, batch, device, concept_weight: float):
    x = batch["image"].to(device)
    y = batch["label"].to(device)
    c = batch["concepts"].to(device)

    if name == "cbm":
        out = model(x, concept_labels=c)
        loss, components = cbm_loss(out.concept_probs, c, out.task_logits, y, concept_weight)
        probs = torch.softmax(out.task_logits, dim=-1)
        concept_probs = out.concept_probs
    elif name == "cem":
        out = model(x)
        loss, components = cem_loss(out.concept_probs, c, out.task_logits, y, concept_weight)
        probs = torch.softmax(out.task_logits, dim=-1)
        concept_probs = out.concept_probs
    elif name == "probcbm":
        out = model(x)
        loss, components = probcbm_loss(out.concept_probs, c, out.task_logits_per_sample, y, out.kl_loss,
                                         concept_weight=concept_weight)
        probs = out.task_probs
        concept_probs = out.concept_probs
    elif name == "hicem":
        out = model(x)
        sub_mask = out.sub_concept_probs > 0  # padding bernilai persis 0 -> aman dipakai sebagai mask kasar
        sub_labels = torch.zeros_like(out.sub_concept_probs)  # tanpa anotasi subkonsep manual -> self-supervised (bab III.3.4)
        loss, components = hicem_loss(out.top_concept_probs, c, out.sub_concept_probs, sub_mask, sub_labels,
                                       out.task_logits, y, concept_weight)
        probs = torch.softmax(out.task_logits, dim=-1)
        concept_probs = out.top_concept_probs
    else:
        raise ValueError(name)

    return loss, components, probs, concept_probs, y, c


def run(cfg: Config, device: torch.device, log_dir: Path, resume="auto") -> None:
    name = cfg.baseline
    if name == "hicem":
        raise ValueError("Baseline HiCEM internal belum layak eksperimen: target subkonsep masih placeholder nol. "
                         "Gunakan implementasi referensi dengan discovery valid; lihat docs/head_to_head_audit_2026-09-20.md.")
    dataset = build_dataset(cfg.dataset, **cfg.data.to_dict())
    train_loader = dataset.get_dataloader("train", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size, num_workers=cfg.train.get("num_workers", 2))
    val_loader = dataset.get_dataloader("val", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size, num_workers=cfg.train.get("num_workers", 2), shuffle=False)

    model = build_model(name, cfg, dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    concept_weight = cfg.train.get("concept_weight", 1.0)

    checkpoint = RunCheckpoint(log_dir, name, model, optimizer, run_identity(cfg, device), cfg.train.epochs, resume)
    for epoch in range(checkpoint.start_epoch, cfg.train.epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"{name}/train", leave=False):
            loss, _, _, _, _, _ = training_step(name, model, batch, device, concept_weight)
            require_finite(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        model.eval()
        all_probs, all_labels, all_cprobs, all_clabels = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                _, _, probs, cprobs, y, c = training_step(name, model, batch, device, concept_weight)
                all_probs.append(probs.cpu())
                all_labels.append(y.cpu())
                all_cprobs.append(cprobs.cpu())
                all_clabels.append(c.cpu())

        report = compute_standard_metrics(
            torch.cat(all_probs), torch.cat(all_labels), torch.cat(all_cprobs), torch.cat(all_clabels)
        )
        logger.info(
            "[%s] epoch=%d train_loss=%.4f val=%s", name, epoch, running_loss / max(n_batches, 1), report.as_dict()
        )

        checkpoint.save_epoch(epoch, report.task_accuracy)
    checkpoint.export_weights()
