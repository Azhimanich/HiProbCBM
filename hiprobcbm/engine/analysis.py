"""Analisis tambahan setelah training: uncertainty dan subkonsep hasil discovery.

Semua fungsi di sini hanya inferensi pada ``stage2_best.pth``; tidak ada
training. Setiap prosedur mengikuti sumber baseline:

- Uji cutout: ``run_cutout_test`` di ../ProbCBM/utils/evaluate.py (patch 64
  di tengah citra yang sudah dinormalisasi dinolkan; dilaporkan proporsi
  sampel yang uncertainty-nya naik).
- Uncertainty vs performa: ProbCBM Sec. 5.2.3 / Fig. 10 ("10 groups according
  to the degree of uncertainty, where each group has the same number of
  samples"). Kode figur ini tidak ada di repo ProbCBM, sehingga prosedurnya
  direkonstruksi dari teks paper.
- Subkonsep hasil discovery: ``match_discovered_concepts_to_concept_bank``
  dan ``get_accuracies`` di ../HiCEM/experiments/run.py (pencocokan di train
  dengan ROC-AUC > 0.7; Table 1 = rata-rata ROC-AUC test prediksi model untuk
  subkonsep yang cocok) serta ``get_intervention_accuracies`` di
  ../HiCEM/experiments/experiment_utils.py.

Definisi uncertainty konsep mengikuti ``get_uncertainty`` ProbCBM:
``exp(mean(log sigma))``. Uncertainty kelas memakai entropi prediktif
ternormalisasi dari rata-rata sampel Monte Carlo; ProbCBM memakai
``get_class_uncertainty`` yang khusus untuk head berbasis anchor, sehingga ini
adalah adaptasi dan harus dilaporkan demikian.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from hiprobcbm.config import Config
from hiprobcbm.engine.evaluate import _hiprobcbm_model_and_test_loader, saved_config
from hiprobcbm.utils.checkpoint import load_artifact


def concept_uncertainty(sigma2: torch.Tensor) -> torch.Tensor:
    """``get_uncertainty`` ProbCBM: exp(mean(log sigma)) per konsep."""
    return (0.5 * torch.log(sigma2.clamp_min(1e-12))).mean(dim=-1).exp()


def class_uncertainty(task_probs: torch.Tensor) -> torch.Tensor:
    """Entropi prediktif ternormalisasi ke [0, 1]."""
    entropy = -(task_probs.clamp_min(1e-12) * task_probs.clamp_min(1e-12).log()).sum(dim=-1)
    return entropy / math.log(task_probs.shape[-1])


def cutout(images: torch.Tensor, patch_size: int = 64) -> torch.Tensor:
    """Salinan ``run_cutout_test`` ProbCBM: patch tengah citra ternormalisasi = 0."""
    cut = images.clone()
    x = y = cut.shape[-1] // 2 - patch_size // 2
    cut[:, :, x:x + patch_size, y:y + patch_size] = 0
    return cut


def uncertainty_groups(uncertainty: np.ndarray, correct: np.ndarray, n_groups: int = 10) -> dict:
    """Bagi menjadi ``n_groups`` kelompok berukuran sama menurut uncertainty."""
    order = np.argsort(uncertainty, kind="stable")
    groups = np.array_split(order, n_groups)
    medians = [float(np.median(uncertainty[g])) for g in groups]
    accuracies = [float(correct[g].mean()) for g in groups]
    rho = spearmanr(medians, accuracies).correlation if len(set(medians)) > 1 else float("nan")
    return {"median_uncertainty": medians, "accuracy": accuracies,
            "spearman_uncertainty_accuracy": float(rho),
            "accuracy_lowest_uncertainty_group": accuracies[0],
            "accuracy_highest_uncertainty_group": accuracies[-1]}


@torch.no_grad()
def evaluate_uncertainty(cfg: Config, stage2_checkpoint: Path, subconcepts_per_concept: list[int],
                         device: torch.device, patch_size: int = 64, n_groups: int = 10, seed: int = 0) -> dict:
    """Uji uncertainty ala ProbCBM untuk satu checkpoint Tahap 2.

    ``own``: uncertainty prediktor konsep induk langsung (padanan ProbCBM).
    ``propagated``: uncertainty induk hasil propagasi variance subkonsep
    (``sigma2_parent = sum(alpha^2 sigma2_sub)``), klaim khas HiProbCBM.
    Uji cutout hanya untuk input citra; jalur cache fitur CLIP tidak memiliki
    citra mentah sehingga dilewati dan dicatat.
    """
    cfg = saved_config(stage2_checkpoint, cfg)
    _, model, loader = _hiprobcbm_model_and_test_loader(cfg, stage2_checkpoint, subconcepts_per_concept, device)
    if model.parent_predictor is None:
        raise ValueError("Analisis uncertainty membutuhkan model parent_probability='hicem_mix'.")
    model.eval()
    torch.manual_seed(seed)
    use_cutout = not cfg.model.get("use_cached_features", False)

    keys = ("own", "propagated", "class", "concept_correct", "class_correct")
    store = {k: [] for k in keys}
    cut_store = {k: [] for k in ("own", "propagated", "class")}

    def measure(h):
        out = model.forward_features(h)
        _, sigma2_own = model.parent_predictor(h)
        return out, concept_uncertainty(sigma2_own), concept_uncertainty(out.sigma2_parent), class_uncertainty(out.task_probs)

    for batch in loader:
        x = batch["image"].to(device)
        c = batch["concepts"].to(device).float()
        y = batch["label"].to(device)
        out, u_own, u_prop, u_cls = measure(model.encode(x))
        store["own"].append(u_own.cpu())
        store["propagated"].append(u_prop.cpu())
        store["class"].append(u_cls.cpu())
        store["concept_correct"].append(((out.parent_concept_probs >= 0.5).float() == c).float().cpu())
        store["class_correct"].append((out.task_probs.argmax(-1) == y).float().cpu())
        if use_cutout:
            _, cu_own, cu_prop, cu_cls = measure(model.encode(cutout(x, patch_size)))
            cut_store["own"].append(cu_own.cpu())
            cut_store["propagated"].append(cu_prop.cpu())
            cut_store["class"].append(cu_cls.cpu())

    data = {k: torch.cat(v).numpy() for k, v in store.items()}
    result = {
        "class_uncertainty_definition": "normalized predictive entropy (adaptasi; ProbCBM memakai get_class_uncertainty)",
        "concept_groups_own": uncertainty_groups(data["own"].reshape(-1), data["concept_correct"].reshape(-1), n_groups),
        "concept_groups_propagated": uncertainty_groups(data["propagated"].reshape(-1), data["concept_correct"].reshape(-1), n_groups),
        "class_groups": uncertainty_groups(data["class"], data["class_correct"], n_groups),
    }
    if use_cutout:
        cut = {k: torch.cat(v).numpy() for k, v in cut_store.items()}
        result["cutout"] = {
            "patch_size": patch_size,
            # ProbCBM: rata-rata uncertainty seluruh konsep per sampel, lalu proporsi yang naik.
            "concept_increase_own": float((cut["own"].mean(1) > data["own"].mean(1)).mean()),
            "concept_increase_propagated": float((cut["propagated"].mean(1) > data["propagated"].mean(1)).mean()),
            "class_increase": float((cut["class"] > data["class"]).mean()),
        }
    else:
        result["cutout"] = "dilewati: input berupa cache fitur CLIP, citra mentah tidak tersedia"
    return result


def summarize_intervention_order(curves: dict) -> dict:
    """Bandingkan urutan uncertainty vs acak dari ``intervention_curves.json``
    protokol probcbm (ProbCBM Fig. 11): rata-rata akurasi sepanjang kurva
    (area di bawah kurva per langkah) dan akurasi setelah 1, 3, 5 grup."""
    unc, rnd = curves["uncertainty"], curves["random_mean"]
    summary = {"area_uncertainty": float(np.mean(unc[1:])), "area_random": float(np.mean(rnd[1:]))}
    for k in (1, 3, 5):
        summary[f"accuracy_after_{k}_groups_uncertainty"] = unc[k]
        summary[f"accuracy_after_{k}_groups_random"] = rnd[k]
    summary["uncertainty_order_better"] = summary["area_uncertainty"] > summary["area_random"]
    return summary


def match_discovered_subconcepts(pseudo_labels: np.ndarray, counts: list[int], concept_names: list[str],
                                 ingredient_groups: dict[str, list[str]], train_bank, threshold: float = 0.7) -> list[dict]:
    """``match_discovered_concepts_to_concept_bank`` HiCEM: untuk tiap
    subkonsep ground truth, pilih subkonsep hasil discovery parent yang sama
    dengan ROC-AUC train tertinggi; dianggap cocok bila > ``threshold``.
    ``train_bank(name)`` mengembalikan label train ingredient dengan urutan
    baris yang sama seperti ``pseudo_labels``."""
    matches = []
    for parent, name in enumerate(concept_names):
        for ingredient in ingredient_groups.get(name, []):
            truth = train_bank(ingredient)
            if len(np.unique(truth)) < 2:
                continue
            best_auc, best_k = 0.0, None
            for k in range(counts[parent]):
                score = roc_auc_score(truth, pseudo_labels[:, parent, k])
                if score > best_auc:
                    best_auc, best_k = score, k
            if best_auc > threshold:
                matches.append({"parent": parent, "parent_name": name, "subconcept": best_k,
                                "ingredient": ingredient, "train_matching_roc_auc": float(best_auc)})
    return matches


@torch.no_grad()
def evaluate_discovered_subconcepts(cfg: Config, stage2_checkpoint: Path, hierarchy_path: Path,
                                    device: torch.device, threshold: float = 0.7, seed: int = 0) -> dict:
    """Table 1 dan Fig. 4 HiCEM untuk satu checkpoint PseudoKitchens."""
    cfg = saved_config(stage2_checkpoint, cfg)
    hierarchy_path = Path(hierarchy_path)
    sha = hierarchy_path.with_suffix(hierarchy_path.suffix + ".sha256")
    hierarchy = load_artifact(hierarchy_path) if sha.exists() else torch.load(hierarchy_path, map_location="cpu", weights_only=True)
    counts = hierarchy["subconcepts_per_concept"]
    dataset, model, loader = _hiprobcbm_model_and_test_loader(cfg, stage2_checkpoint, counts, device)
    if not hasattr(dataset, "ingredient_groups"):
        raise ValueError("Evaluasi subkonsep hasil discovery hanya untuk PseudoKitchens.")
    manifest = dataset._manifest.set_index("image")

    train_paths = list(hierarchy["paths"])
    matches = match_discovered_subconcepts(
        hierarchy["pseudo_labels"].numpy(), counts, dataset.concept_names, dataset.ingredient_groups,
        lambda ingredient: manifest.loc[train_paths, ingredient].to_numpy(dtype=float), threshold,
    )
    result = {"threshold": threshold, "matches": matches, "n_matched": len(matches),
              "n_ground_truth_subconcepts": sum(len(v) for v in dataset.ingredient_groups.values())}
    if not matches:
        result.update(mean_discovered_concept_roc_auc=float("nan"), curves={}, intervention_delta=float("nan"))
        return result

    model.eval()
    torch.manual_seed(seed)
    batches, probs, truths = [], [], []
    for batch in loader:
        h = model.encode(batch["image"].to(device))
        gt = torch.tensor(manifest.loc[list(batch["path"]), [m["ingredient"] for m in matches]].to_numpy(dtype="float32"))
        out = model.forward_features(h)
        probs.append(torch.stack([out.sub_concept_probs[:, m["parent"], m["subconcept"]] for m in matches], 1).cpu())
        truths.append(gt)
        batches.append((h, batch["label"].to(device), gt.to(device)))
    probs, truths = torch.cat(probs).numpy(), torch.cat(truths).numpy()
    for j, match in enumerate(matches):
        both = len(np.unique(truths[:, j])) > 1
        match["test_roc_auc"] = float(roc_auc_score(truths[:, j], probs[:, j])) if both else float("nan")
    result["mean_discovered_concept_roc_auc"] = float(np.nanmean([m["test_roc_auc"] for m in matches]))

    k_max = model.subconcept_predictor.mask.shape[1]
    curves = {"cumulative": [], "one_at_a_time": []}
    for name in curves:
        for step in range(len(matches) + 1):
            chosen = range(step) if name == "cumulative" else ([step - 1] if step else [])
            correct = total = 0
            for h, y, gt in batches:
                values = torch.full((h.shape[0], model.num_concepts, k_max), float("nan"), device=device)
                for j in chosen:
                    values[:, matches[j]["parent"], matches[j]["subconcept"]] = gt[:, j]
                out = model.forward_features(h, sub_intervention=values)
                correct += (out.task_probs.argmax(-1) == y).sum().item()
                total += y.shape[0]
            curves[name].append(correct / max(total, 1))
    result["curves"] = curves
    result["intervention_delta"] = curves["cumulative"][-1] - curves["cumulative"][0]
    return result
