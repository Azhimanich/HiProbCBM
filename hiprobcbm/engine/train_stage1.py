"""Tahap 1: Probabilistic Concept Predictor + Automatic Subconcept Discovery
(Bab IV.5.1). Alur:

1. Latih `HiProbCBMStage1` (mirip ProbCBM) sampai konvergen pada L_concept.
2. Ekstrak mean konsep untuk seluruh data train.
3. Jalankan `subconcept_discovery.build_pseudo_hierarchy` untuk membentuk
   pseudo hierarchical subconcept dataset (disimpan ke disk untuk dipakai
   Tahap 2, lihat `run_stage2.py`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from tqdm import tqdm

from hiprobcbm.config import Config
from hiprobcbm.data import build_dataset
from hiprobcbm.losses import hiprobcbm_stage1_loss
from hiprobcbm.models.hiprobcbm import HiProbCBMStage1
from hiprobcbm.models.subconcept_discovery import PseudoHierarchy, build_pseudo_hierarchy

logger = logging.getLogger(__name__)


def train_one_epoch(model: HiProbCBMStage1, loader, optimizer, device, lambda_kl: float) -> dict[str, float]:
    model.train()
    running = {"total": 0.0, "concept_bce": 0.0, "kl": 0.0}
    n_batches = 0
    for batch in tqdm(loader, desc="stage1/train", leave=False):
        x = batch["image"].to(device)
        c = batch["concepts"].to(device)

        out = model(x)
        loss, components = hiprobcbm_stage1_loss(out.concept_probs, c, out.kl_loss, lambda_kl=lambda_kl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running["total"] += loss.item()
        running["concept_bce"] += components["concept_bce"].item()
        running["kl"] += components["kl"].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate_stage1(model: HiProbCBMStage1, loader, device) -> dict[str, float]:
    from hiprobcbm.metrics import concept_accuracy, concept_roc_auc

    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        x = batch["image"].to(device)
        c = batch["concepts"].to(device)
        out = model(x)
        all_probs.append(out.concept_probs.cpu())
        all_labels.append(c.cpu())
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    return {
        "concept_accuracy": concept_accuracy(probs, labels),
        "concept_roc_auc": concept_roc_auc(probs, labels),
    }


def _stack_padded_pseudo_labels(hierarchy: PseudoHierarchy) -> torch.Tensor:
    """Menumpuk `pseudo_labels` (N, K_i) tiap konsep menjadi satu tensor
    (N, C, K_max), mem-pad dengan nol subkonsep yang jumlahnya < K_max."""
    if not hierarchy.concept_results:
        return torch.zeros(0, 0, 0)

    n = hierarchy.concept_results[0].pseudo_labels.shape[0]
    k_max = max(hierarchy.subconcepts_per_concept, default=0)
    stacked = torch.zeros(n, len(hierarchy.concept_results), max(k_max, 1))
    for i, result in enumerate(hierarchy.concept_results):
        k_i = result.pseudo_labels.shape[1]
        if k_i > 0:
            stacked[:, i, :k_i] = result.pseudo_labels
    return stacked


def run(cfg: Config, device: torch.device, log_dir: Path) -> PseudoHierarchy:
    dataset = build_dataset(cfg.dataset, **cfg.data.to_dict())
    train_loader = dataset.get_dataloader("train", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size)
    val_loader = dataset.get_dataloader("val", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size, shuffle=False)
    # Loader KHUSUS untuk automatic subconcept discovery: shuffle=False dan
    # drop_last=False supaya SETIAP sampel train terekstrak tepat sekali,
    # dan `path` per-batch bisa dipakai Tahap 2 untuk mencocokkan
    # pseudo-label ke citra yang benar (lihat `extract_means_for_discovery`
    # - ini memperbaiki bug kritis: `train_loader` di atas memakai
    # shuffle=True, jadi TIDAK BOLEH dipakai langsung untuk ekstraksi mean).
    discovery_loader = dataset.get_dataloader(
        "train", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size,
        shuffle=False, drop_last=False,
    )

    model = HiProbCBMStage1(
        backbone_name=cfg.model.backbone,
        num_concepts=dataset.num_concepts,
        concept_dim=cfg.model.concept_dim,
        pretrained=cfg.model.get("pretrained", True),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr_stage1)

    best_val_acc = -1.0
    for epoch in range(cfg.train.epochs_stage1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, cfg.train.get("lambda_kl", 5e-5))
        val_stats = evaluate_stage1(model, val_loader, device)
        logger.info("epoch=%d train=%s val=%s", epoch, train_stats, val_stats)

        if val_stats["concept_accuracy"] > best_val_acc:
            best_val_acc = val_stats["concept_accuracy"]
            torch.save(model.state_dict(), log_dir / "stage1_best.pth")

    torch.save(model.state_dict(), log_dir / "stage1_last.pth")

    # Bab IV.5.1.3-4.5.1.4: automatic subconcept discovery pada data train.
    model.load_state_dict(torch.load(log_dir / "stage1_best.pth", map_location=device, weights_only=True))
    mu_all, concept_probs_all, paths = model.extract_means_for_discovery(
        discovery_loader, device, n_samples=cfg.model.get("n_mc_samples_eval", 32)
    )

    hierarchy = build_pseudo_hierarchy(
        mu_all=mu_all,
        concept_probs=concept_probs_all,
        concept_names=dataset.concept_names,
        sae_variant=cfg.sae.get("variant", "kl"),
        threshold=cfg.sae.get("activation_threshold", 0.5),
        presence_threshold=cfg.sae.get("presence_threshold", 0.5),
        sae_kwargs=cfg.sae.get("kwargs", {}),
        train_kwargs=cfg.sae.get("train_kwargs", {}),
        device=device,
    )

    torch.save(
        {
            "subconcepts_per_concept": hierarchy.subconcepts_per_concept,
            "pseudo_labels": _stack_padded_pseudo_labels(hierarchy),
            "paths": paths,  # penaut posisi -> identitas citra (Tahap 2 mencocokkan lewat ini)
            "concept_names": dataset.concept_names,
        },
        log_dir / "pseudo_hierarchy.pt",
    )
    logger.info("Pseudo hierarchical subconcept dataset tersimpan di %s", log_dir / "pseudo_hierarchy.pt")
    return hierarchy
