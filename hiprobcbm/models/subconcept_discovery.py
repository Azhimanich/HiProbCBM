"""Automatic Subconcept Discovery (Bab IV.5.1.3-4.5.1.4).

Alur per konsep induk i (mengikuti pola `split_with_sae` pada HiCEM, tetapi
beroperasi pada MEAN dari representasi PROBABILISTIK, bukan embedding
deterministik - lihat Distingsi HiProbCBM vs HiCEM, Bab III.3.8):

1. Kumpulkan mu_i untuk seluruh sampel training di mana konsep i diprediksi
   aktif (p_i > 0.5), analog `sample_filter` pada Concept Splitting HiCEM.
2. Latih SAE (KL-sparsity atau BatchTopK, lihat sae.py) pada kumpulan
   mu_i tersebut, mu_i in R^{d_c} per sampel.
3. Neuron laten yang aktif konsisten pada klaster sampel menjadi kandidat
   subkonsep; threshold tau mengubah aktivasi menjadi pseudo-label biner
   (Eq. Bab IV.5.1.4: y_ik^sub = 1(a_ik > tau)).
4. Buang fitur "mati" (tidak pernah aktif) - mengikuti
   `non_dead_feature_acts` pada implementasi HiCEM.

Hasil akhirnya adalah *pseudo hierarchical subconcept dataset*: untuk tiap
konsep induk i, sekumpulan label subkonsep biner K_i yang menjadi target
supervisi Tahap 2 (subbab 4.5.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

from hiprobcbm.models.sae import build_sae, train_sae
from hiprobcbm.utils.checkpoint import save_artifact, load_artifact, capture_rng, restore_rng


@dataclass
class ConceptSubconceptResult:
    concept_index: int
    concept_name: str
    sample_filter: torch.Tensor  # (N,) bool - sampel yang dipakai melatih SAE konsep ini
    pseudo_labels: torch.Tensor  # (N, K_i) float32 {0,1}, nol di luar sample_filter
    num_subconcepts: int


@dataclass
class PseudoHierarchy:
    concept_results: list[ConceptSubconceptResult] = field(default_factory=list)

    @property
    def subconcepts_per_concept(self) -> list[int]:
        return [r.num_subconcepts for r in self.concept_results]

    def total_subconcepts(self) -> int:
        return sum(self.subconcepts_per_concept)


def discover_subconcepts_for_concept(
    concept_index: int,
    concept_name: str,
    mu_concept: torch.Tensor,
    presence_mask: torch.Tensor,
    sae_variant: str = "kl",
    threshold: float = 0.5,
    sae_kwargs: dict | None = None,
    train_kwargs: dict | None = None,
    device: str | torch.device = "cpu",
) -> ConceptSubconceptResult:
    """mu_concept: (N, d_c) mean konsep i untuk seluruh dataset training.
    presence_mask: (N,) bool, True jika konsep i diprediksi aktif pada
    sampel tsb (Bab IV.5.1.3: "hanya mean yang digunakan sebagai masukan
    SAE", dibatasi pada sampel yang relevan dengan konsep tsb, mengikuti
    `sample_filter` HiCEM).
    """
    sae_kwargs = sae_kwargs or {}
    train_kwargs = train_kwargs or {}

    d_c = mu_concept.shape[-1]
    selected = mu_concept[presence_mask]
    n = mu_concept.shape[0]

    def _fallback_single_subconcept() -> ConceptSubconceptResult:
        """K_i=0 TIDAK PERNAH dikembalikan oleh fungsi ini secara sengaja:
        `HiProbCBMStage2._aggregate_all_concepts` mengagregasi lewat softmax
        attention atas subkonsep - kalau K_i=0 (mask seluruhnya False),
        softmax menghasilkan NaN yang ditangani jadi alpha=0, dan mu_parent
        jatuh menjadi VEKTOR NOL, bukan representasi konsep yang berarti.
        Sebagai gantinya, konsep tanpa subkonsep hasil discovery diberi SATU
        subkonsep trivial yang labelnya = presence_mask konsep itu sendiri -
        persis padanan aturan HiCEM "jika c_i tidak memiliki subkonsep, maka
        c_i^+ = c_i^{+'}" (Bab III.3.4): subkonsep tunggal ini berperan
        sebagai representasi konsep induk itu sendiri.
        """
        return ConceptSubconceptResult(
            concept_index=concept_index,
            concept_name=concept_name,
            sample_filter=presence_mask,
            pseudo_labels=presence_mask.float().unsqueeze(1),
            num_subconcepts=1,
        )

    if selected.shape[0] < 2:
        # Tidak cukup sampel untuk melatih SAE.
        return _fallback_single_subconcept()

    sae = build_sae(sae_variant, input_dim=d_c, **sae_kwargs)
    activations = train_sae(sae, selected, device=device, **train_kwargs)

    alive = (activations > threshold).any(dim=0)
    activations = activations[:, alive]

    if activations.shape[1] == 0:
        # SAE konvergen tapi tidak ada fitur laten yang pernah aktif sama
        # sekali ("seluruh fitur mati") - kembali ke fallback yang sama.
        return _fallback_single_subconcept()

    pseudo_binary = (activations > threshold).float()

    k = pseudo_binary.shape[1]
    full_labels = torch.zeros(n, k)
    full_labels[presence_mask] = pseudo_binary

    return ConceptSubconceptResult(
        concept_index=concept_index,
        concept_name=concept_name,
        sample_filter=presence_mask,
        pseudo_labels=full_labels,
        num_subconcepts=k,
    )


def build_pseudo_hierarchy(
    mu_all: torch.Tensor,
    concept_probs: torch.Tensor,
    concept_names: list[str],
    sae_variant: str = "kl",
    threshold: float = 0.5,
    presence_threshold: float = 0.5,
    sae_kwargs: dict | None = None,
    train_kwargs: dict | None = None,
    device: str | torch.device = "cpu",
    checkpoint_dir=None,
    checkpoint_identity=None,
) -> PseudoHierarchy:
    """Jalankan discovery untuk seluruh C konsep top-level sekaligus.

    mu_all: (N, C, d_c) - mean konsep Tahap 1 untuk seluruh dataset training.
    concept_probs: (N, C) - probabilitas keberadaan konsep (dari
    `ConceptExistenceScorer`), dipakai untuk menentukan sample_filter per
    konsep, sejalan dengan Bab IV.5.1.3 & mekanisme HiCEM.
    """
    n, num_concepts, _ = mu_all.shape
    results: list[ConceptSubconceptResult] = []
    for c in range(num_concepts):
        result_path = Path(checkpoint_dir) / f"concept_{c}" / "result.pt" if checkpoint_dir is not None else None
        identity = {"run": checkpoint_identity, "concept_index": c, "name": concept_names[c]}
        if result_path is not None and result_path.exists():
            cached = load_artifact(result_path)
            if cached["identity"] != identity:
                raise ValueError("Cache discovery tidak cocok dengan run/konsep.")
            results.append(ConceptSubconceptResult(**cached["result"]))
            restore_rng(cached["rng"])
            continue
        kwargs = dict(train_kwargs or {})
        if result_path is not None:
            kwargs.update(checkpoint_dir=result_path.parent, checkpoint_identity=identity)
        presence_mask = concept_probs[:, c] > presence_threshold
        result = discover_subconcepts_for_concept(
            concept_index=c,
            concept_name=concept_names[c],
            mu_concept=mu_all[:, c, :],
            presence_mask=presence_mask,
            sae_variant=sae_variant,
            threshold=threshold,
            sae_kwargs=sae_kwargs,
            train_kwargs=kwargs,
            device=device,
        )
        results.append(result)
        if result_path is not None:
            save_artifact({"identity": identity, "result": asdict(result), "rng": capture_rng()}, result_path)
    return PseudoHierarchy(concept_results=results)
