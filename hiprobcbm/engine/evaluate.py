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
from hiprobcbm.data.feature_cache import cached_feature_loader, prepare_clip_feature_cache
from hiprobcbm.engine.train_baseline import build_model, training_step
from hiprobcbm.engine.train_stage2 import build_stage2_model, evaluate_stage2
from hiprobcbm.metrics import expected_calibration_error

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
    if cfg.baseline == "hicem":
        raise ValueError("HiCEM placeholder tidak boleh dimasukkan tabel hasil; lihat audit A01/B05.")
    dataset = build_dataset(cfg.dataset, **cfg.data.to_dict())
    test_loader = dataset.get_dataloader("test", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size, shuffle=False)

    model = build_model(cfg.baseline, cfg, dataset).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    from hiprobcbm.metrics import compute_standard_metrics

    all_probs, all_labels, all_cprobs, all_clabels = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            _, _, probs, cprobs, y, c = training_step(cfg.baseline, model, batch, device, cfg.train.get("concept_weight", 1.0))
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
            all_cprobs.append(cprobs.cpu())
            all_clabels.append(c.cpu())

    report = compute_standard_metrics(
        torch.cat(all_probs), torch.cat(all_labels), torch.cat(all_cprobs), torch.cat(all_clabels)
    )
    return report.as_dict()


def _hiprobcbm_model_and_test_loader(cfg: Config, stage2_checkpoint: Path, subconcepts_per_concept: list[int],
                                     device: torch.device):
    dataset = build_dataset(cfg.dataset, **cfg.data.to_dict())
    use_cached_features = cfg.model.get("use_cached_features", False)
    if use_cached_features:
        payloads, _ = prepare_clip_feature_cache(
            dataset,
            image_size=cfg.model.image_size,
            batch_size=cfg.model.get("feature_cache_batch_size", cfg.train.batch_size),
            num_workers=cfg.train.get("num_workers", 2),
            device=device,
            splits=("test",),
        )
        test_loader = cached_feature_loader(
            payloads["test"], batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=cfg.train.get("num_workers", 2), drop_last=False,
        )
    else:
        test_loader = dataset.get_dataloader("test", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size, shuffle=False)

    model = build_stage2_model(cfg, dataset.num_classes, subconcepts_per_concept, pretrained=False).to(device)
    if cfg.model.backbone == "inception_v3":
        # torchvision's flag is not a tensor in state_dict. Match the training
        # factory behavior without downloading pretrained weights at evaluation.
        model.backbone.module.transform_input = cfg.model.get("pretrained", True)
    model.load_state_dict(torch.load(stage2_checkpoint, map_location=device, weights_only=True))
    return dataset, model, test_loader


def evaluate_hiprobcbm_checkpoint(
    cfg: Config, stage2_checkpoint: Path, subconcepts_per_concept: list[int], device: torch.device
) -> dict[str, float]:
    cfg = saved_config(stage2_checkpoint, cfg)
    _, model, test_loader = _hiprobcbm_model_and_test_loader(cfg, stage2_checkpoint, subconcepts_per_concept, device)
    report = evaluate_stage2(model, test_loader, device)
    return report.as_dict()


def _probcbm_uncertainty(sigma2: torch.Tensor) -> torch.Tensor:
    """``get_uncertainty`` ProbCBM: exp(mean(log sigma)) per konsep."""
    return (0.5 * torch.log(sigma2.clamp_min(1e-12))).mean(dim=-1).exp()


@torch.no_grad()
def evaluate_hiprobcbm_interventions(
    cfg: Config, stage2_checkpoint: Path, subconcepts_per_concept: list[int], device: torch.device,
    n_random_orders: int = 5, seed: int = 0,
) -> dict:
    """Kurva task accuracy terhadap jumlah intervensi konsep induk pada test.

    - Protokol ``probcbm`` (ProbCBM Sec. 5.2.4, Fig. 11;
      ``predict_class_with_gt_concepts``): 28 grup atribut CUB diintervensi
      kumulatif; urutan acak (``n_random_orders`` kali, urutan sama untuk semua
      sampel) dan urutan uncertainty per sampel (``uncertainty_avg``).
    - Protokol ``hicem`` (experiments/run.py,
      ``get_intervention_accuracies``): konsep yang disediakan diintervensi
      menurut urutan indeks, kumulatif dan satu per satu. Intervensi subkonsep
      hasil discovery membutuhkan matching concept bank dan belum dihitung.

    ``intervention_delta`` = akurasi semua konsep diintervensi - tanpa
    intervensi (Bab IV.7.1.5), dari kurva kumulatif (rata-rata urutan acak
    untuk probcbm).
    """
    cfg = saved_config(stage2_checkpoint, cfg)
    protocol = (cfg.train.get("intervention_training") or {}).get("mode", "none")
    if protocol not in ("probcbm", "hicem"):
        raise ValueError("Evaluasi intervensi membutuhkan train.intervention_training.mode probcbm/hicem.")
    dataset, model, test_loader = _hiprobcbm_model_and_test_loader(cfg, stage2_checkpoint, subconcepts_per_concept, device)
    model.eval()
    torch.manual_seed(seed)
    n_concepts = model.num_concepts

    if protocol == "probcbm":
        groups = [g.concept_indices for g in dataset.concept_groups]
        if len(groups) < 2:
            raise ValueError("Grup atribut CUB tidak tersedia; intervensi grup ProbCBM tidak dapat dihitung.")
        group2concept = torch.zeros(len(groups), n_concepts, dtype=torch.bool, device=device)
        for g, members in enumerate(groups):
            group2concept[g, members] = True
        generator = torch.Generator().manual_seed(seed)
        orders = {f"random_{r}": torch.randperm(len(groups), generator=generator).to(device) for r in range(n_random_orders)}
        orders["uncertainty"] = None
        steps = len(groups) + 1
    else:
        orders = {"cumulative": torch.arange(n_concepts, device=device), "one_at_a_time": torch.arange(n_concepts, device=device)}
        steps = n_concepts + 1

    correct = {name: torch.zeros(steps) for name in orders}
    total = 0
    for batch in test_loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        c = batch["concepts"].to(device).float()
        h = model.encode(x)
        total += y.shape[0]
        if protocol == "probcbm":
            _, sigma2_own = model.parent_predictor(h)
            unc = _probcbm_uncertainty(sigma2_own)  # (B, C)
            group_unc = (unc @ group2concept.float().t()) / group2concept.sum(dim=1)
            per_sample_order = group_unc.argsort(dim=1, descending=True)  # (B, G)
        for name, order in orders.items():
            for k in range(steps):
                if protocol == "probcbm":
                    chosen = per_sample_order[:, :k] if order is None else order[:k].expand(y.shape[0], -1)
                    mask = group2concept[chosen].any(dim=1) if k else torch.zeros_like(c, dtype=torch.bool)
                elif name == "cumulative":
                    mask = torch.zeros_like(c, dtype=torch.bool)
                    mask[:, order[:k]] = True
                else:  # one_at_a_time: langkah k>0 hanya konsep ke-(k-1)
                    mask = torch.zeros_like(c, dtype=torch.bool)
                    if k:
                        mask[:, order[k - 1]] = True
                values = torch.where(mask, c, torch.full_like(c, float("nan")))
                out = model.forward_features(h, parent_intervention=values)
                correct[name][k] += (out.task_probs.argmax(dim=-1) == y).sum().item()

    curves = {name: (counts / max(total, 1)).tolist() for name, counts in correct.items()}
    if protocol == "probcbm":
        random_names = [n for n in curves if n.startswith("random_")]
        mean = torch.tensor([curves[n] for n in random_names]).mean(dim=0)
        curves["random_mean"] = mean.tolist()
        delta = float(mean[-1] - mean[0])
    else:
        delta = curves["cumulative"][-1] - curves["cumulative"][0]
    return {"protocol": protocol, "curves": curves, "intervention_delta": delta}


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
