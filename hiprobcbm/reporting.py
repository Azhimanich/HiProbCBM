"""Penjelasan multilevel per sampel dan grafik untuk naskah (Bab IV.7.2).

Penjelasan mengikuti alur penalaran HiProbCBM: subkonsep -> konsep induk ->
kelas. Karena classifier adalah satu lapisan linear di atas konkatenasi
representasi induk, kontribusi konsep induk i ke logit kelas c dihitung
persis sebagai ``W[c, i*d:(i+1)*d] . mu_parent_i`` (pada mean, tanpa bias dan
tanpa noise Monte Carlo). Di dalam tiap induk, kontribusi subkonsep adalah
bobot learned attention ``alpha``. Uncertainty memakai definisi ProbCBM
``exp(mean(log sigma))`` di setiap tingkat.

Semua fungsi hanya inferensi.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hiprobcbm.config import Config
from hiprobcbm.engine.analysis import class_uncertainty, concept_uncertainty
from hiprobcbm.engine.evaluate import _hiprobcbm_model_and_test_loader, saved_config
from hiprobcbm.utils.checkpoint import load_artifact


def _load_hierarchy(path: Path) -> dict:
    path = Path(path)
    sha = path.with_suffix(path.suffix + ".sha256")
    return load_artifact(path) if sha.exists() else torch.load(path, map_location="cpu", weights_only=True)


def class_names_for(dataset, paths: list[str], labels: list[int]) -> list[str]:
    """Nama kelas: recipe PseudoKitchens, atau folder kelas CUB (``068.Ruby...``)."""
    if hasattr(dataset, "recipes"):
        return list(dataset.recipes)
    names = [f"class_{i}" for i in range(dataset.num_classes)]
    for path, label in zip(paths, labels):
        folder = str(path).replace("\\", "/").split("/")[0]
        if "." in folder:
            names[label] = folder.split(".", 1)[1].replace("_", " ")
    return names


def select_samples(correct: np.ndarray, confidence: np.ndarray, n_confident: int = 2,
                   n_uncertain_correct: int = 1, n_wrong: int = 2) -> list[int]:
    """Pilihan sampel deterministik: benar paling yakin, benar paling ragu,
    dan salah paling yakin (untuk menganalisis kesalahan)."""
    right = np.where(correct)[0]
    wrong = np.where(~correct)[0]
    chosen = list(right[np.argsort(-confidence[right], kind="stable")][:n_confident])
    chosen += [i for i in right[np.argsort(confidence[right], kind="stable")] if i not in chosen][:n_uncertain_correct]
    chosen += list(wrong[np.argsort(-confidence[wrong], kind="stable")][:n_wrong])
    return [int(i) for i in chosen]


@torch.no_grad()
def explain_test_samples(cfg: Config, stage2_checkpoint: Path, hierarchy_path: Path, device: torch.device,
                         sub_names: dict | None = None, top_parents: int = 3, top_subconcepts: int = 4,
                         n_prototypes: int = 3, seed: int = 0, hierarchical_only: bool = True, **selection) -> dict:
    """Penjelasan multilevel untuk sampel test terpilih.

    ``hierarchical_only``: hanya tampilkan konsep induk dengan lebih dari satu
    subkonsep (bila ada), agar tingkat subkonsep benar-benar terlihat; induk
    dengan satu subkonsep memiliki attention 1 secara trivial.
    ``sub_names``: {(parent, subkonsep): nama}, mis. ingredient hasil matching
    ``discovered_subconcepts.json``. Subkonsep tanpa nama diberi label
    ``sub-k`` dan diberi contoh citra train yang pseudo-labelnya positif
    (prototipe) agar maknanya dapat ditafsirkan.
    """
    cfg = saved_config(stage2_checkpoint, cfg)
    hierarchy = _load_hierarchy(hierarchy_path)
    counts = hierarchy["subconcepts_per_concept"]
    dataset, model, loader = _hiprobcbm_model_and_test_loader(cfg, stage2_checkpoint, counts, device)
    if model.parent_predictor is None:
        raise ValueError("Penjelasan multilevel membutuhkan model parent_probability='hicem_mix'.")
    model.eval()
    torch.manual_seed(seed)
    sub_names = sub_names or {}

    feats, labels, concepts, paths, confidence, preds = [], [], [], [], [], []
    for batch in loader:
        h = model.encode(batch["image"].to(device))
        out = model.forward_features(h)
        conf, pred = out.task_probs.max(-1)
        feats.append(h.cpu())
        labels.append(batch["label"])
        concepts.append(batch["concepts"].float())
        paths.extend(batch["path"])
        confidence.append(conf.cpu())
        preds.append(pred.cpu())
    feats, labels, concepts = torch.cat(feats), torch.cat(labels), torch.cat(concepts)
    confidence, preds = torch.cat(confidence), torch.cat(preds)
    correct = (preds == labels).numpy()
    chosen = select_samples(correct, confidence.numpy(), **selection)
    class_names = class_names_for(dataset, paths, labels.tolist())

    h = feats[chosen].to(device)
    out = model.forward_features(h)
    _, sigma2_own = model.parent_predictor(h)
    unc_own = concept_uncertainty(sigma2_own).cpu()
    unc_prop = concept_uncertainty(out.sigma2_parent).cpu()
    unc_sub = concept_uncertainty(out.sigma2_sub).cpu()
    d = model.concept_dim
    weight = model.classifier.net[0].weight.detach().cpu()  # (kelas, C*d)
    mu_parent = out.mu_parent.cpu()
    pseudo = hierarchy["pseudo_labels"]
    train_paths = list(hierarchy["paths"])
    image_root = Path(getattr(dataset, "data_root", ".")) / "images"  # CUB & PseudoKitchens: <data_root>/images

    samples = []
    for row, index in enumerate(chosen):
        pred = int(preds[index])
        contribution = (weight[pred].view(model.num_concepts, d) * mu_parent[row]).sum(-1)
        parents = []
        ranked = torch.argsort(contribution, descending=True).tolist()
        if hierarchical_only and any(counts[i] > 1 for i in ranked):
            ranked = [i for i in ranked if counts[i] > 1]
        for i in ranked[:top_parents]:
            subs = []
            for k in torch.argsort(out.attention[row, i].cpu(), descending=True)[:min(top_subconcepts, counts[i])].tolist():
                positives = torch.nonzero(pseudo[:, i, k]).flatten()[:n_prototypes].tolist()
                subs.append({"index": k, "name": sub_names.get((i, k), f"sub-{k}"),
                             "attention": float(out.attention[row, i, k]),
                             "probability": float(out.sub_concept_probs[row, i, k]),
                             "uncertainty": float(unc_sub[row, i, k]),
                             "prototype_images": [str(image_root / train_paths[p]) for p in positives]})
            parents.append({"index": i, "name": dataset.concept_names[i],
                            "contribution_to_predicted_logit": float(contribution[i]),
                            "probability": float(out.parent_concept_probs[row, i]),
                            "ground_truth": float(concepts[index, i]),
                            "uncertainty_own": float(unc_own[row, i]),
                            "uncertainty_propagated": float(unc_prop[row, i]),
                            "n_subconcepts": counts[i], "subconcepts": subs})
        samples.append({"test_index": index, "image": str(image_root / paths[index]),
                        "true_class": class_names[int(labels[index])], "predicted_class": class_names[pred],
                        "correct": bool(correct[index]), "confidence": float(confidence[index]),
                        "class_uncertainty": float(class_uncertainty(out.task_probs[row:row + 1]).item()),
                        "parents": parents})
    return {"checkpoint": str(stage2_checkpoint), "samples": samples}


# --- Grafik ---------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _open_image(path: str):
    from PIL import Image
    try:
        return Image.open(path).convert("RGB")
    except OSError:
        return None


def plot_explanation(sample: dict, out_path: Path, title: str = "") -> Path:
    """Satu gambar: citra uji di kiri; untuk tiap konsep induk teratas, bar
    bobot attention subkonsepnya beserta probabilitas dan uncertainty, serta
    satu citra prototipe subkonsep dengan attention terbesar."""
    plt = _plt()
    parents = sample["parents"]
    fig = plt.figure(figsize=(15, 3.2 * max(len(parents), 1) + 0.6))
    grid = fig.add_gridspec(max(len(parents), 1), 3, width_ratios=[1.2, 2.2, 0.9])
    ax_img = fig.add_subplot(grid[:, 0])
    image = _open_image(sample["image"])
    if image is not None:
        ax_img.imshow(image)
    ax_img.axis("off")
    status = "BENAR" if sample["correct"] else "SALAH"
    ax_img.set_title(f"{status}\nlabel: {sample['true_class']}\nprediksi: {sample['predicted_class']}\n"
                     f"confidence {sample['confidence']:.2f} | uncertainty kelas {sample['class_uncertainty']:.2f}",
                     fontsize=9)
    for row, parent in enumerate(parents):
        ax = fig.add_subplot(grid[row, 1])
        subs = parent["subconcepts"]
        names = [s["name"] for s in subs][::-1]
        ax.barh(names, [s["attention"] for s in subs][::-1], color="#4C72B0")
        for y, s in enumerate(subs[::-1]):
            ax.text(s["attention"], y, f"  p={s['probability']:.2f}  u={s['uncertainty']:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.35)
        ax.set_xlabel("bobot attention", fontsize=8)
        ax.set_title(f"Konsep induk: {parent['name']} (kontribusi {parent['contribution_to_predicted_logit']:+.2f}) | "
                     f"p={parent['probability']:.2f}, label={int(parent['ground_truth'])}, "
                     f"u_langsung={parent['uncertainty_own']:.3f}, u_propagasi={parent['uncertainty_propagated']:.3f} "
                     f"| {parent['n_subconcepts']} subkonsep", fontsize=8)
        ax.tick_params(labelsize=8)
        ax_proto = fig.add_subplot(grid[row, 2])
        ax_proto.axis("off")
        proto = subs[0]["prototype_images"][0] if subs and subs[0]["prototype_images"] else None
        proto_image = _open_image(proto) if proto else None
        if proto_image is not None:
            ax_proto.imshow(proto_image)
            ax_proto.set_title(f"prototipe {subs[0]['name']}", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _mean_std(series: list[list[float]]):
    array = np.asarray(series, dtype=float)
    return array.mean(0), (array.std(0, ddof=1) if len(array) > 1 else np.zeros(array.shape[1]))


def plot_intervention_curves(runs: dict[str, list[dict]], out_path: Path, title: str) -> Path:
    """``runs``: {varian: [isi intervention_curves.json per seed]}. Mean +/- std antar-seed."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, seeds in runs.items():
        protocol = seeds[0]["protocol"]
        key = "random_mean" if protocol == "probcbm" else "cumulative"
        mean, std = _mean_std([s["curves"][key] for s in seeds])
        x = np.arange(len(mean))
        line, = ax.plot(x, mean, label=f"{variant} ({'acak' if protocol == 'probcbm' else 'kumulatif'})")
        ax.fill_between(x, mean - std, mean + std, alpha=.15, color=line.get_color())
        if protocol == "probcbm":
            u_mean, _ = _mean_std([s["curves"]["uncertainty"] for s in seeds])
            ax.plot(x, u_mean, linestyle="--", color=line.get_color(), label=f"{variant} (uncertainty)")
    ax.set_xlabel("jumlah grup konsep diintervensi" if protocol == "probcbm" else "jumlah konsep diintervensi")
    ax.set_ylabel("task accuracy (test)")
    ax.set_title(title)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def plot_uncertainty_groups(runs: dict[str, list[dict]], out_path: Path, title: str) -> Path:
    """Tiga panel (kelas, konsep langsung, konsep propagasi): akurasi per
    kelompok uncertainty (10 kelompok), mean antar-seed per varian."""
    plt = _plt()
    panels = [("class_groups", "kelas"), ("concept_groups_own", "konsep (langsung)"),
              ("concept_groups_propagated", "konsep (propagasi)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, label) in zip(axes, panels):
        for variant, seeds in runs.items():
            mean, std = _mean_std([s[key]["accuracy"] for s in seeds])
            x = np.arange(1, len(mean) + 1)
            line, = ax.plot(x, mean, marker="o", label=variant)
            ax.fill_between(x, mean - std, mean + std, alpha=.15, color=line.get_color())
        ax.set_xlabel("kelompok uncertainty (1 = paling yakin)")
        ax.set_ylabel("akurasi")
        ax.set_title(f"Uncertainty {label}")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def plot_discovered_interventions(runs: dict[str, list[dict]], out_path: Path, title: str) -> Path:
    """Kurva intervensi subkonsep hasil discovery (HiCEM Fig. 4), per seed
    karena jumlah subkonsep yang cocok berbeda antar-seed."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, seeds in runs.items():
        for seed, result in seeds:
            curve = result.get("curves", {}).get("cumulative")
            if curve:
                ax.plot(np.arange(len(curve)), curve, marker=".", label=f"{variant} seed {seed} ({len(curve) - 1} cocok)")
    ax.set_xlabel("jumlah subkonsep hasil discovery diintervensi (kumulatif)")
    ax.set_ylabel("task accuracy (test)")
    ax.set_title(title)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def sub_names_from_discovered(path: Path) -> dict:
    """{(parent, subkonsep): ingredient} dari ``discovered_subconcepts.json``."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    names: dict = {}
    for m in data.get("matches", []):  # beberapa ingredient bisa cocok ke subkonsep yang sama
        names.setdefault((m["parent"], m["subconcept"]), []).append(m["ingredient"])
    return {key: "/".join(value) for key, value in names.items()}
