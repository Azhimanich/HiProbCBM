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
from hiprobcbm.data.feature_cache import cached_feature_loader, prepare_clip_feature_cache
from hiprobcbm.losses import hiprobcbm_stage1_loss
from hiprobcbm.models.hiprobcbm import HiProbCBMStage1
from hiprobcbm.models.subconcept_discovery import PseudoHierarchy, build_pseudo_hierarchy
from hiprobcbm.utils.checkpoint import run_identity, save_artifact, load_artifact, capture_rng, restore_rng, require_finite, file_hash
from hiprobcbm.engine.protocol import checkpoint_for, optimizer_for

logger = logging.getLogger(__name__)


def train_one_epoch(model: HiProbCBMStage1, loader, optimizer, device, lambda_kl: float, n_samples=8) -> dict[str, float]:
    model.train()
    running = {"total": 0.0, "concept_bce": 0.0, "kl": 0.0}
    n_batches = 0
    for batch in tqdm(loader, desc="stage1/train", leave=False):
        x = batch["image"].to(device)
        c = batch["concepts"].to(device)

        out = model(x, n_samples=n_samples)
        loss, components = hiprobcbm_stage1_loss(out.concept_probs, c, out.kl_loss, lambda_kl=lambda_kl)

        require_finite(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running["total"] += loss.item()
        running["concept_bce"] += components["concept_bce"].item()
        running["kl"] += components["kl"].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate_stage1(model: HiProbCBMStage1, loader, device, n_samples=32) -> dict[str, float]:
    from hiprobcbm.metrics import concept_accuracy, concept_roc_auc

    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        x = batch["image"].to(device)
        c = batch["concepts"].to(device)
        out = model(x, n_samples=n_samples)
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


def run(cfg: Config, device: torch.device, log_dir: Path, resume="auto") -> PseudoHierarchy:
    dataset = build_dataset(cfg.dataset, **cfg.data.to_dict())
    use_cached_features = cfg.model.get("use_cached_features", False)
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
        discovery_loader = cached_feature_loader(
            payloads["train"], batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=cfg.train.get("num_workers", 2), drop_last=False,
        )
    else:
        train_loader = dataset.get_dataloader("train", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size, num_workers=cfg.train.get("num_workers", 2))
        val_loader = dataset.get_dataloader("val", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size, num_workers=cfg.train.get("num_workers", 2), shuffle=False)
    # Loader KHUSUS untuk automatic subconcept discovery: shuffle=False dan
    # drop_last=False supaya SETIAP sampel train terekstrak tepat sekali,
    # dan `path` per-batch bisa dipakai Tahap 2 untuk mencocokkan
    # pseudo-label ke citra yang benar (lihat `extract_means_for_discovery`
    # - ini memperbaiki bug kritis: `train_loader` di atas memakai
    # shuffle=True, jadi TIDAK BOLEH dipakai langsung untuk ekstraksi mean).
        discovery_loader = dataset.get_dataloader(
            "train", cfg.model.image_size, cfg.model.backbone, cfg.train.batch_size,
            shuffle=False, drop_last=False, augment=False, num_workers=cfg.train.get("num_workers", 2),
        )

    model = HiProbCBMStage1(
        backbone_name=cfg.model.backbone,
        num_concepts=dataset.num_concepts,
        concept_dim=cfg.model.concept_dim,
        pretrained=cfg.model.get("pretrained", True),
        use_cached_features=use_cached_features,
    ).to(device)

    optimizer = optimizer_for(model, cfg.train, cfg.train.lr_stage1)

    identity = run_identity(cfg, device, feature_artifacts)
    checkpoint = checkpoint_for(log_dir, "stage1", model, optimizer, identity, cfg.train.epochs_stage1, cfg.train, resume)
    if checkpoint.completed:
        logger.info("Stage 1 checkpoint sudah selesai; ekspor bobot terbaik.")
    else:
        for epoch in range(checkpoint.start_epoch, cfg.train.epochs_stage1):
            train_stats = train_one_epoch(model, train_loader, optimizer, device, cfg.train.get("lambda_kl", 5e-5),
                                          cfg.model.get("n_mc_samples_train", 8))
            val_stats = evaluate_stage1(model, val_loader, device, cfg.model.get("n_mc_samples_eval", 32))
            logger.info("epoch=%d train=%s val=%s", epoch, train_stats, val_stats)

            checkpoint.save_epoch(epoch, val_stats["concept_accuracy"])
            if checkpoint.completed:
                break
    checkpoint.export_weights()

    # Bab IV.5.1.3-4.5.1.4: automatic subconcept discovery pada data train.
    model.load_state_dict(torch.load(log_dir / "stage1_best.pth", map_location=device, weights_only=True))
    identity = {**identity, "parent_weights_sha256": file_hash(log_dir / "stage1_best.pth")}
    feature_path = log_dir / "discovery_features.pt"
    if feature_path.exists():
        cached = load_artifact(feature_path)
        if cached["identity"] != identity:
            raise ValueError("Cache fitur discovery berasal dari run berbeda.")
        mu_all, concept_probs_all, paths = cached["mu"], cached["probs"], cached["paths"]
        restore_rng(cached["rng"])
    else:
        mu_all, concept_probs_all, paths = model.extract_means_for_discovery(
            discovery_loader, device, n_samples=cfg.model.get("n_mc_samples_eval", 32)
        )
        save_artifact({"identity": identity, "mu": mu_all, "probs": concept_probs_all, "paths": paths,
                     "rng": capture_rng()}, feature_path)

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
        checkpoint_dir=log_dir / "discovery",
        checkpoint_identity=identity,
    )

    payload = {
        "subconcepts_per_concept": hierarchy.subconcepts_per_concept,
        "pseudo_labels": _stack_padded_pseudo_labels(hierarchy),
        "paths": paths,  # penaut posisi -> identitas citra (Tahap 2 mencocokkan lewat ini)
        "concept_names": dataset.concept_names,
        "identity": identity,
    }
    hierarchy_path = log_dir / "pseudo_hierarchy.pt"
    if hierarchy_path.exists():
        existing = load_artifact(hierarchy_path)
        if (existing.get("identity") != identity or existing["paths"] != paths
                or existing["subconcepts_per_concept"] != payload["subconcepts_per_concept"]
                or not torch.equal(existing["pseudo_labels"], payload["pseudo_labels"])):
            raise ValueError("Artifact hierarchy yang ada tidak cocok; tidak ditimpa.")
    else:
        save_artifact(payload, hierarchy_path)
    logger.info("Pseudo hierarchical subconcept dataset tersimpan di %s", log_dir / "pseudo_hierarchy.pt")
    return hierarchy
