"""Evaluasi terpadu (Bab IV.7) - dipakai untuk mengisi Tabel 4.7 (Skenario
Pengujian) dan Tabel 4.8 (Studi Ablasi) dari checkpoint yang sudah dilatih.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import torch

from hiprobcbm.config import Config
from hiprobcbm.data import build_dataset
from hiprobcbm.engine.train_baseline import build_model, training_step
from hiprobcbm.engine.train_stage2 import evaluate_stage2
from hiprobcbm.metrics import expected_calibration_error
from hiprobcbm.models.hiprobcbm import HiProbCBMStage2
from hiprobcbm.models.baselines.hicem import HierarchicalConceptEmbeddingModel
from hiprobcbm.utils.checkpoint import load_artifact

logger = logging.getLogger(__name__)


def saved_config(checkpoint_path: Path, fallback: Config) -> Config:
    """Use the configuration actually trained, including A1/A2 and seed overrides."""
    manifest = Path(checkpoint_path).parent / "run_manifest.json"
    if manifest.exists():
        return Config(json.loads(manifest.read_text(encoding="utf-8"))["config"])
    logger.warning("Checkpoint legacy tanpa manifest; konfigurasi evaluator harus ditetapkan manual.")
    return fallback


def evaluate_baseline_checkpoint(cfg: Config, checkpoint_path: Path, device: torch.device) -> dict[str, float]:
    cfg = saved_config(checkpoint_path, cfg)
    dataset = build_dataset(cfg.dataset, **cfg.data.to_dict())
    test_loader = dataset.get_dataloader("test", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size, shuffle=False)

    if cfg.baseline == "hicem":
        hierarchy_path = Path(checkpoint_path).parent / "hicem_pseudo_hierarchy.pt"
        if not hierarchy_path.exists():
            raise FileNotFoundError(f"Artifact discovery HiCEM tidak ditemukan: {hierarchy_path}")
        hierarchy = load_artifact(hierarchy_path)
        counts = hierarchy["subconcepts_per_concept"]
        if len(counts) != dataset.num_concepts or any(type(k) is not int or k < 1 for k in counts):
            raise ValueError("Artifact hierarchy HiCEM tidak sesuai dengan dataset checkpoint.")
        model = HierarchicalConceptEmbeddingModel(
            cfg.model.backbone, dataset.num_classes, [(count, 0) for count in counts],
            embedding_dim=cfg.model.get("embedding_dim", 16), pretrained=cfg.model.get("pretrained", True),
        ).to(device)
    else:
        model = build_model(cfg.baseline, cfg, dataset).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    from hiprobcbm.metrics import compute_standard_metrics

    all_probs, all_labels, all_cprobs, all_clabels = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if cfg.baseline == "hicem":
                out = model(batch["image"].to(device))
                probs, cprobs = torch.softmax(out.task_logits, dim=-1), out.top_concept_probs
                y, c = batch["label"].to(device), batch["concepts"].to(device)
            else:
                _, _, probs, cprobs, y, c = training_step(
                    cfg.baseline, model, batch, device, cfg.train.get("concept_weight", 1.0)
                )
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
            all_cprobs.append(cprobs.cpu())
            all_clabels.append(c.cpu())

    report = compute_standard_metrics(
        torch.cat(all_probs), torch.cat(all_labels), torch.cat(all_cprobs), torch.cat(all_clabels)
    )
    return report.as_dict()


def evaluate_hiprobcbm_checkpoint(
    cfg: Config, stage2_checkpoint: Path, subconcepts_per_concept: list[int], device: torch.device
) -> dict[str, float]:
    cfg = saved_config(stage2_checkpoint, cfg)
    dataset = build_dataset(cfg.dataset, **cfg.data.to_dict())
    test_loader = dataset.get_dataloader("test", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size, shuffle=False)

    model = HiProbCBMStage2(
        backbone_name=cfg.model.backbone,
        num_classes=dataset.num_classes,
        subconcepts_per_concept=subconcepts_per_concept,
        concept_dim=cfg.model.concept_dim,
        pretrained=False,
        n_mc_samples_train=cfg.model.get("n_mc_samples_train", 8),
        n_mc_samples_eval=cfg.model.get("n_mc_samples_eval", 32),
        use_attention=cfg.model.get("use_attention", True),
    ).to(device)
    if cfg.model.backbone == "inception_v3":
        # torchvision's flag is not a tensor in state_dict. Match the training
        # factory behavior without downloading pretrained weights at evaluation.
        model.backbone.module.transform_input = cfg.model.get("pretrained", True)
    model.load_state_dict(torch.load(stage2_checkpoint, map_location=device, weights_only=True))

    report = evaluate_stage2(model, test_loader, device)
    return report.as_dict()


def write_results_table(rows: list[dict], out_path: Path) -> None:
    """Menulis hasil ke CSV supaya mudah ditempel jadi Tabel 4.7/4.8 di
    naskah tesis (kolom disesuaikan otomatis dari baris pertama)."""
    if not rows:
        logger.warning("Tidak ada baris hasil untuk ditulis.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Tabel hasil tersimpan di %s", out_path)
