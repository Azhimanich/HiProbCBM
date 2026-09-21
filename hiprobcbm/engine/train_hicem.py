"""HiCEM pipeline: train CEM → discover positive children → train HiCEM.

The discovery phase uses only the training split.  It does not use the
PseudoKitchens ground-truth child labels, which remain reserved for a later
held-out discovery analysis.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from hiprobcbm.config import Config
from hiprobcbm.data import build_dataset
from hiprobcbm.engine.protocol import checkpoint_for, optimizer_for
from hiprobcbm.losses import cem_loss, hicem_loss
from hiprobcbm.metrics import compute_standard_metrics
from hiprobcbm.models.baselines.cem import ConceptEmbeddingModel
from hiprobcbm.models.baselines.hicem import HierarchicalConceptEmbeddingModel
from hiprobcbm.models.subconcept_discovery import build_pseudo_hierarchy
from hiprobcbm.engine.train_stage1 import _stack_padded_pseudo_labels
from hiprobcbm.engine.train_stage2 import build_path_to_row
from hiprobcbm.utils.checkpoint import (
    capture_rng, file_hash, load_artifact, require_finite,
    restore_rng, run_identity, save_artifact,
)

logger = logging.getLogger(__name__)


def _phase_config(cfg: Config, key: str) -> Config:
    values = cfg.train.to_dict()
    values.update(cfg.get(key, {}))
    return Config(values)


def _loaders(dataset, cfg):
    workers = cfg.train.get("num_workers", 2)
    train = dataset.get_dataloader("train", cfg.model.image_size, cfg.model.backbone,
                                   cfg.train.batch_size, num_workers=workers, shuffle=True)
    val = dataset.get_dataloader("val", cfg.model.image_size, cfg.model.backbone,
                                 cfg.train.batch_size, num_workers=workers, shuffle=False)
    discovery = dataset.get_dataloader("train", cfg.model.image_size, cfg.model.backbone,
                                       cfg.train.batch_size, num_workers=workers, shuffle=False,
                                       drop_last=False, augment=False)
    return train, val, discovery


def _evaluate_cem(model, loader, device):
    model.eval()
    probs, labels, cprobs, concepts = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            probs.append(torch.softmax(out.task_logits, -1).cpu())
            labels.append(batch["label"].cpu())
            cprobs.append(out.concept_probs.cpu())
            concepts.append(batch["concepts"].cpu())
    return compute_standard_metrics(torch.cat(probs), torch.cat(labels), torch.cat(cprobs), torch.cat(concepts))


def _train_initial_cem(cfg, dataset, train_loader, val_loader, device, log_dir, resume):
    phase = _phase_config(cfg, "hicem_initial")
    model = ConceptEmbeddingModel(
        cfg.model.backbone, dataset.num_concepts, dataset.num_classes,
        embedding_dim=cfg.model.get("embedding_dim", 16), pretrained=cfg.model.get("pretrained", True),
        intervention_probability=phase.get("intervention_probability", .25),
    ).to(device)
    optimizer = optimizer_for(model, phase, phase.get("lr", cfg.train.lr))
    checkpoint = checkpoint_for(log_dir, "hicem_initial_cem", model, optimizer, run_identity(cfg, device),
                                phase.get("epochs", cfg.train.epochs), phase, resume)
    if checkpoint.completed:
        logger.info("[hicem initial CEM] checkpoint sudah selesai; ekspor bobot terbaik.")
    else:
        for epoch in range(checkpoint.start_epoch, checkpoint.total_epochs):
            model.train()
            for batch in train_loader:
                out = model(batch["image"].to(device), concept_labels=batch["concepts"].to(device))
                loss, _ = cem_loss(out.concept_probs, batch["concepts"].to(device), out.task_logits,
                                   batch["label"].to(device), phase.get("concept_weight", 10.0))
                require_finite(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            report = _evaluate_cem(model, val_loader, device)
            logger.info("[hicem initial CEM] epoch=%d val=%s", epoch, report.as_dict())
            checkpoint.save_epoch(epoch, report.task_accuracy)
            if checkpoint.completed:
                break
    checkpoint.export_weights()
    model.load_state_dict(torch.load(log_dir / "hicem_initial_cem_best.pth", map_location=device, weights_only=True))
    return model


def _discover(cfg, dataset, initial_cem, loader, device, log_dir, initial_checkpoint):
    identity = {**run_identity(cfg, device), "initial_cem_sha256": file_hash(initial_checkpoint)}
    features_path = log_dir / "hicem_discovery_features.pt"
    if features_path.exists():
        cached = load_artifact(features_path)
        if cached["identity"] != identity:
            raise ValueError("Cache discovery HiCEM bukan milik initial CEM ini.")
        embeddings, probabilities, paths = cached["embeddings"], cached["probabilities"], cached["paths"]
        restore_rng(cached["rng"])
    else:
        embeddings, probabilities, paths = initial_cem.extract_for_discovery(loader, device)
        save_artifact({"identity": identity, "embeddings": embeddings, "probabilities": probabilities,
                       "paths": paths, "rng": capture_rng()}, features_path)
    discovery_cfg = cfg.get("hicem_discovery", {})
    hierarchy = build_pseudo_hierarchy(
        embeddings, probabilities, dataset.concept_names,
        sae_variant=discovery_cfg.get("variant", "batchtopk"),
        threshold=discovery_cfg.get("activation_threshold", 0.0),
        presence_threshold=discovery_cfg.get("presence_threshold", .5),
        sae_kwargs=discovery_cfg.get("kwargs", {}), train_kwargs=discovery_cfg.get("train_kwargs", {}),
        device=device, checkpoint_dir=log_dir / "hicem_discovery", checkpoint_identity=identity,
    )
    payload = {"identity": identity, "subconcepts_per_concept": hierarchy.subconcepts_per_concept,
               "pseudo_labels": _stack_padded_pseudo_labels(hierarchy), "paths": paths,
               "concept_names": dataset.concept_names}
    path = log_dir / "hicem_pseudo_hierarchy.pt"
    if path.exists():
        existing = load_artifact(path)
        if (existing["identity"] != identity or existing["paths"] != paths
                or existing["subconcepts_per_concept"] != payload["subconcepts_per_concept"]
                or not torch.equal(existing["pseudo_labels"], payload["pseudo_labels"])):
            raise ValueError("Hierarchy HiCEM yang ada tidak sesuai; tidak ditimpa.")
    else:
        save_artifact(payload, path)
    return payload, path


def _evaluate_hicem(model, loader, device):
    model.eval()
    probs, labels, cprobs, concepts = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            probs.append(torch.softmax(out.task_logits, -1).cpu())
            labels.append(batch["label"].cpu())
            cprobs.append(out.top_concept_probs.cpu())
            concepts.append(batch["concepts"].cpu())
    return compute_standard_metrics(torch.cat(probs), torch.cat(labels), torch.cat(cprobs), torch.cat(concepts))


def run(cfg: Config, device: torch.device, log_dir: Path, resume="auto"):
    dataset = build_dataset(cfg.dataset, **cfg.data.to_dict())
    train_loader, val_loader, discovery_loader = _loaders(dataset, cfg)
    # RunCheckpoint deliberately rejects unrelated *.pth files in a run
    # directory. Keep the preliminary CEM in its own child directory, while
    # the final HiCEM remains the authority for this experiment directory.
    initial_dir = log_dir / "initial_cem"
    initial = _train_initial_cem(cfg, dataset, train_loader, val_loader, device, initial_dir, resume)
    hierarchy, hierarchy_path = _discover(
        cfg, dataset, initial, discovery_loader, device, log_dir,
        initial_dir / "hicem_initial_cem_best.pth",
    )
    counts, labels, paths = hierarchy["subconcepts_per_concept"], hierarchy["pseudo_labels"], hierarchy["paths"]
    if len(counts) != dataset.num_concepts or any(k < 1 for k in counts):
        raise ValueError("Hierarchy HiCEM tidak mempunyai child valid untuk semua parent.")
    mapping = build_path_to_row(paths)
    model = HierarchicalConceptEmbeddingModel(
        cfg.model.backbone, dataset.num_classes, [(k, 0) for k in counts],
        embedding_dim=cfg.model.get("embedding_dim", 16), pretrained=cfg.model.get("pretrained", True),
    ).to(device)
    phase = _phase_config(cfg, "hicem_train")
    optimizer = optimizer_for(model, phase, phase.get("lr", cfg.train.lr))
    checkpoint = checkpoint_for(log_dir, "hicem", model, optimizer,
                                run_identity(cfg, device, [hierarchy_path]), phase.get("epochs", cfg.train.epochs),
                                phase, resume)
    if checkpoint.completed:
        logger.info("[hicem] checkpoint sudah selesai; ekspor bobot terbaik.")
    else:
        for epoch in range(checkpoint.start_epoch, checkpoint.total_epochs):
            model.train()
            for batch in train_loader:
                rows = torch.tensor([mapping[p] for p in batch["path"]], dtype=torch.long)
                out = model(batch["image"].to(device))
                mask = out.positive_mask.unsqueeze(0).expand(out.sub_concept_probs.shape[0], -1, -1)
                loss, _ = hicem_loss(out.top_concept_probs, batch["concepts"].to(device), out.sub_concept_probs, mask,
                                     labels[rows].to(device), out.task_logits, batch["label"].to(device),
                                     phase.get("concept_weight", 10.0))
                require_finite(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            report = _evaluate_hicem(model, val_loader, device)
            logger.info("[hicem] epoch=%d val=%s", epoch, report.as_dict())
            checkpoint.save_epoch(epoch, report.task_accuracy)
            if checkpoint.completed:
                break
    checkpoint.export_weights()
