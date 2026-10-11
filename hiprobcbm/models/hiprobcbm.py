"""Perakitan penuh HiProbCBM - Tahap 1 & Tahap 2 (Bab IV.5).

- ``HiProbCBMStage1``: backbone + Probabilistic Concept Predictor +
  ConceptExistenceScorer (subbab 4.5.1.1-4.5.1.2). Dipakai untuk melatih
  representasi konsep top-level DAN untuk mengekstrak mean yang menjadi
  masukan automatic subconcept discovery (subbab 4.5.1.3-4.5.1.4,
  lihat ``subconcept_discovery.py``).

- ``HiProbCBMStage2``: model penuh setelah pseudo hierarchical subconcept
  dataset terbentuk - memprediksi distribusi tiap subkonsep, mengagregasi-
  kannya lewat learned attention (subbab 4.5.2), lalu melakukan
  hierarchical probabilistic reasoning (subbab 4.5.2.5) untuk prediksi
  kelas via Monte Carlo sampling + classifier terstandardisasi.

Karena K_i (jumlah subkonsep per konsep induk) bisa berbeda-beda, tensor
subkonsep di-pad ke K_max dengan mask boolean (C, K_max); konsep tanpa
subkonsep (K_i=0) jatuh kembali ke representasi konsep-nya sendiri dari
Tahap 1, sejalan dengan aturan HiCEM ("jika c_i tidak memiliki
subkonsep, ĉ_i^+ = ĉ_i^{+'}").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hiprobcbm.models.attention_aggregation import (
    LearnedAttentionAggregator,
    UniformAggregator,
    reparameterize_parent,
)
from hiprobcbm.models.backbones import Backbone, build_backbone
from hiprobcbm.models.classifier import LinearSoftmaxClassifier
from hiprobcbm.models.probabilistic_concepts import (
    ConceptExistenceScorer,
    ProbabilisticConceptPredictor,
    kl_to_standard_normal,
    reparameterize,
)


@dataclass
class Stage1Output:
    mu: torch.Tensor  # (B, C, d_c)
    sigma2: torch.Tensor  # (B, C, d_c)
    concept_probs: torch.Tensor  # (B, C)
    kl_loss: torch.Tensor


class HiProbCBMStage1(nn.Module):
    """Tahap 1 (Bab IV.5.1): representasi konsep probabilistik flat, identik
    strukturnya dengan ProbCBM (dipakai juga sebagai dasar mean untuk SAE)."""

    def __init__(
        self,
        backbone_name: str,
        num_concepts: int,
        concept_dim: int = 16,
        pretrained: bool = True,
        use_cached_features: bool = False,
    ):
        super().__init__()
        self.backbone: Backbone = build_backbone(
            backbone_name, pretrained=pretrained, use_cached_features=use_cached_features
        )
        self.concept_predictor = ProbabilisticConceptPredictor(
            feature_dim=self.backbone.output_dim, num_concepts=num_concepts, concept_dim=concept_dim
        )
        self.scorer = ConceptExistenceScorer(concept_dim=concept_dim)
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim

    def forward(self, x: torch.Tensor, n_samples: int = 8) -> Stage1Output:
        h = self.backbone(x)
        mu, sigma2 = self.concept_predictor(h)
        concept_probs = self.scorer.probability_from_distribution(mu, sigma2, n_samples=n_samples)
        kl = kl_to_standard_normal(mu, sigma2)
        return Stage1Output(mu=mu, sigma2=sigma2, concept_probs=concept_probs, kl_loss=kl)

    @torch.no_grad()
    def extract_means_for_discovery(self, dataloader, device: torch.device, n_samples: int = 8):
        """Mengumpulkan (mu, concept_probs, path) untuk seluruh data train -
        masukan `subconcept_discovery.build_pseudo_hierarchy` (Bab IV.5.1.3).

        PENTING: `dataloader` di sini WAJIB dibuat dengan `shuffle=False,
        drop_last=False` (lihat `engine/train_stage1.py`), supaya setiap
        sampel ikut terekstrak tepat sekali. `path` dikembalikan agar Tahap 2
        bisa mencocokkan pseudo-label ke citra lewat identitas file, bukan
        lewat asumsi urutan posisi - dua DataLoader terpisah (satu untuk
        training Tahap 1, satu untuk training Tahap 2) tidak dijamin
        menghasilkan urutan iterasi yang sama walau sama-sama shuffle=False
        jika parameter lain (mis. num_workers) berbeda, jadi identitas path
        adalah satu-satunya penaut yang aman.
        """
        self.eval()
        all_mu, all_probs, all_paths = [], [], []
        for batch in dataloader:
            x = batch["image"].to(device)
            out = self.forward(x, n_samples=n_samples)
            all_mu.append(out.mu.cpu())
            all_probs.append(out.concept_probs.cpu())
            all_paths.extend(batch["path"])
        return torch.cat(all_mu, dim=0), torch.cat(all_probs, dim=0), all_paths


class SubconceptPredictor(nn.Module):
    """Memprediksi (mu_ik, sigma2_ik) untuk seluruh subkonsep hasil
    discovery (Bab IV.5.2.1). Satu kepala linear per subkonsep, dikelompokkan
    per konsep induk lalu di-pad ke K_max.

    Konsep dengan K_i=0 (tidak ditemukan subkonsep) DIPAKSA punya tepat 1
    kepala - bukan 0 - supaya baris masknya tidak seluruhnya False. Tanpa
    ini, `LearnedAttentionAggregator.aggregate` akan menghasilkan softmax
    atas skor kosong (-inf semua -> NaN -> alpha dipaksa 0), sehingga
    mu_parent untuk konsep tsb menjadi VEKTOR NOL, bukan representasi
    konsep yang berarti. Kepala tunggal ini berperan sebagai representasi
    konsep induk itu sendiri - padanan aturan HiCEM "jika c_i tidak
    memiliki subkonsep, maka c_i^+ = c_i^{+'}" (Bab III.3.4). Lihat juga
    fallback yang sepadan di `subconcept_discovery.discover_subconcepts_for_concept`.
    """

    def __init__(self, feature_dim: int, subconcepts_per_concept: list[int], concept_dim: int = 16):
        super().__init__()
        self.subconcepts_per_concept = subconcepts_per_concept
        # Jumlah kepala AKTUAL per konsep (>=1) - dipakai konsisten di
        # __init__ maupun forward, terpisah dari `subconcepts_per_concept`
        # asli (yang boleh berisi 0) supaya niat "K_i=0 -> 1 kepala fallback"
        # tetap terbaca jelas dari kode, bukan cuma dari max(...,1) tersebar.
        self._effective_k = [max(k_i, 1) for k_i in subconcepts_per_concept]
        self.num_concepts = len(subconcepts_per_concept)
        self.concept_dim = concept_dim
        self.k_max = max(self._effective_k) if self._effective_k else 0

        self.mean_heads = nn.ModuleList()
        self.logvar_heads = nn.ModuleList()
        for k_i in self._effective_k:
            self.mean_heads.append(nn.ModuleList([nn.Linear(feature_dim, concept_dim) for _ in range(k_i)]))
            self.logvar_heads.append(nn.ModuleList([nn.Linear(feature_dim, concept_dim) for _ in range(k_i)]))

        mask = torch.zeros(self.num_concepts, max(self.k_max, 1), dtype=torch.bool)
        for i, k_i in enumerate(self._effective_k):
            mask[i, :k_i] = True
        self.register_buffer("mask", mask)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (B, d_h) -> mu, sigma2: (B, C, K_max, d_c), padding = 0."""
        import torch.nn.functional as F

        batch = h.shape[0]
        k_max = max(self.k_max, 1)
        mu = torch.zeros(batch, self.num_concepts, k_max, self.concept_dim, device=h.device)
        sigma2 = torch.zeros_like(mu)
        for i, k_i in enumerate(self._effective_k):
            for k in range(k_i):
                mu[:, i, k, :] = self.mean_heads[i][k](h)
                sigma2[:, i, k, :] = F.softplus(self.logvar_heads[i][k](h)) + 1e-6
        return mu, sigma2


PARENT_PROBABILITY_MODES = ("aggregate", "hicem_mix")


def hicem_soft_max(sub_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Soft-OR subkonsep -> konsep induk, salinan rumus HiCEM:
    ``sum(softmax(200 p - 100) * p)`` (../HiCEM/cemcd/models/hicem.py,
    ``predicted_top_concept_probs_1``). Hasilnya ~max_k p_k: satu subkonsep
    aktif cukup untuk mengaktifkan induk. Posisi padding dikeluarkan dari
    softmax lewat ``mask`` (C, K_max) bool."""
    logits = (200.0 * sub_probs - 100.0).masked_fill(~mask, float("-inf"))
    weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
    return (weights * sub_probs).sum(dim=-1)


@dataclass
class Stage2Output:
    mu_sub: torch.Tensor  # (B, C, K_max, d_c)
    sigma2_sub: torch.Tensor
    mu_parent: torch.Tensor  # (B, C, d_c)
    sigma2_parent: torch.Tensor
    attention: torch.Tensor  # (B, C, K_max)
    sub_concept_probs: torch.Tensor  # (B, C, K_max)
    parent_concept_probs: torch.Tensor  # (B, C)
    task_logits_per_sample: torch.Tensor  # (B, S, num_classes)
    task_probs: torch.Tensor  # (B, num_classes) - rata-rata S sampel (p_bar)
    kl_loss: torch.Tensor


class HiProbCBMStage2(nn.Module):
    """Tahap 2 (Bab IV.5.2-4.5.2.5): hierarchical concept aggregation +
    hierarchical probabilistic reasoning, dilatih end-to-end."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        subconcepts_per_concept: list[int],
        concept_dim: int = 16,
        pretrained: bool = True,
        n_mc_samples_train: int = 8,
        n_mc_samples_eval: int = 32,
        use_attention: bool = True,
        classifier_hidden_dims: tuple[int, ...] = (),
        use_cached_features: bool = False,
        parent_probability: str = "aggregate",
    ):
        super().__init__()
        if parent_probability not in PARENT_PROBABILITY_MODES:
            raise ValueError(f"parent_probability harus salah satu dari {PARENT_PROBABILITY_MODES}.")
        self.parent_probability = parent_probability
        self._concept_path_frozen = False
        self.backbone: Backbone = build_backbone(
            backbone_name, pretrained=pretrained, use_cached_features=use_cached_features
        )
        # "hicem_mix": konsep induk juga punya prediktor probabilistiknya sendiri
        # (struktur Tahap 1/ProbCBM). Probabilitas induk = rata-rata prediksi
        # langsung ini dan soft-OR subkonsep, mengikuti HiCEM yang merata-rata
        # dua estimasi top concept: (p_1 + p_2) / 2.
        self.parent_predictor = (
            ProbabilisticConceptPredictor(
                feature_dim=self.backbone.output_dim,
                num_concepts=len(subconcepts_per_concept),
                concept_dim=concept_dim,
            )
            if parent_probability == "hicem_mix"
            else None
        )
        self.subconcept_predictor = SubconceptPredictor(
            feature_dim=self.backbone.output_dim,
            subconcepts_per_concept=subconcepts_per_concept,
            concept_dim=concept_dim,
        )
        self.subconcepts_per_concept = subconcepts_per_concept
        self.num_concepts = len(subconcepts_per_concept)
        self.concept_dim = concept_dim
        self.n_mc_samples_train = n_mc_samples_train
        self.n_mc_samples_eval = n_mc_samples_eval

        # HiProbCBM-A1 (Tabel 4.8): use_attention=False -> UniformAggregator.
        self.aggregator = (LearnedAttentionAggregator if use_attention else UniformAggregator)(
            concept_dim=concept_dim
        )
        self.scorer = ConceptExistenceScorer(concept_dim=concept_dim)
        self.classifier = LinearSoftmaxClassifier(
            input_dim=self.num_concepts * concept_dim,
            num_classes=num_classes,
            hidden_dims=classifier_hidden_dims,
        )
        if parent_probability == "hicem_mix":
            # Anchor intervensi [negatif, positif] per konsep induk dan subkonsep.
            # Padanan ``concept_vectors`` ProbCBM: intervensi mengganti embedding
            # prediksi dengan anchor ground truth dan uncertainty menjadi nol
            # (build_model_resnset.py, predict_class_with_gt_concepts).
            k_max = self.subconcept_predictor.mask.shape[1]
            self.parent_anchors = nn.Parameter(0.1 * torch.randn(self.num_concepts, 2, concept_dim))
            self.sub_anchors = nn.Parameter(0.1 * torch.randn(self.num_concepts, k_max, 2, concept_dim))
        else:
            self.parent_anchors = None
            self.sub_anchors = None

    def freeze_concept_path(self) -> None:
        """Bekukan backbone, prediktor konsep induk, dan scorer setelah bobotnya
        diambil dari Tahap 1. Padanan tahap ``class`` ProbCBM sequential
        (../ProbCBM/main.py: hanya ``params_to_classify`` yang dioptimasi
        setelah tahap ``concept``). BatchNorm backbone ikut dikunci di mode
        eval agar fitur identik dengan Tahap 1."""
        if self.parent_predictor is None:
            raise ValueError("freeze_concept_path membutuhkan parent_probability='hicem_mix'.")
        for module in (self.backbone, self.parent_predictor, self.scorer):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self._concept_path_frozen = True
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._concept_path_frozen:
            self.backbone.eval()
            self.parent_predictor.eval()
        return self

    def _aggregate_all_concepts(
        self, mu_sub: torch.Tensor, sigma2_sub: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Terapkan aggregator per konsep induk i; konsep tanpa subkonsep
        (K_i=0) jatuh kembali ke rata-rata trivial (mask kosong -> alpha nol,
        mu/sigma2 diisi dari sub tensor yang memang sudah nol) - dalam
        praktiknya K_i=0 sebaiknya dihindari di Bab IV.5.1.4 dengan
        menjamin minimal 1 subkonsep per konsep induk (mis. subkonsep
        tunggal = konsep itu sendiri saat SAE tidak menemukan struktur)."""
        batch = mu_sub.shape[0]
        mask = self.subconcept_predictor.mask.unsqueeze(0).expand(batch, -1, -1)  # (B, C, K_max)

        mu_parent = torch.zeros(batch, self.num_concepts, self.concept_dim, device=mu_sub.device)
        sigma2_parent = torch.zeros_like(mu_parent)
        alpha_all = torch.zeros(batch, self.num_concepts, mask.shape[-1], device=mu_sub.device)

        for i in range(self.num_concepts):
            agg = self.aggregator.aggregate(mu_sub[:, i], sigma2_sub[:, i], mask=mask[:, i])
            mu_parent[:, i] = agg["mu_parent"]
            sigma2_parent[:, i] = agg["sigma2_parent"]
            alpha_all[:, i] = agg["alpha"]

        return mu_parent, sigma2_parent, alpha_all

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self._concept_path_frozen:
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)

    @staticmethod
    def _anchor_targets(anchors: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """values: label intervensi 0/1, NaN = tidak diintervensi. Mengembalikan
        (mask bool, embedding anchor) dengan bentuk values + (d_c,)."""
        mask = ~torch.isnan(values)
        positive = torch.nan_to_num(values, nan=0.0) > 0.5
        target = torch.where(positive.unsqueeze(-1), anchors[..., 1, :], anchors[..., 0, :])
        return mask, target

    def forward(
        self,
        x: torch.Tensor,
        subconcept_intervention: torch.Tensor | None = None,
        subconcept_intervention_mask: torch.Tensor | None = None,
        parent_intervention: torch.Tensor | None = None,
        sub_intervention: torch.Tensor | None = None,
        sample_intervention_mask: torch.Tensor | None = None,
        parent_labels: torch.Tensor | None = None,
    ) -> Stage2Output:
        return self.forward_features(
            self.encode(x), subconcept_intervention, subconcept_intervention_mask,
            parent_intervention, sub_intervention, sample_intervention_mask, parent_labels,
        )

    def forward_features(
        self,
        h: torch.Tensor,
        subconcept_intervention: torch.Tensor | None = None,
        subconcept_intervention_mask: torch.Tensor | None = None,
        parent_intervention: torch.Tensor | None = None,
        sub_intervention: torch.Tensor | None = None,
        sample_intervention_mask: torch.Tensor | None = None,
        parent_labels: torch.Tensor | None = None,
    ) -> Stage2Output:
        """subconcept_intervention / _mask: (B, C, K_max, d_c) & (B, C, K_max)
        bool - API lama `intervention.py` (mengganti mu_ik sebelum agregasi).

        Intervensi anchor (model hicem_mix) hanya mengubah jalur classifier;
        probabilitas konsep tetap prediksi model, seperti HiCEM yang menghitung
        concept loss dari ``predicted_concept_probs``:
        - ``parent_intervention`` (B, C) dan ``sub_intervention`` (B, C, K_max):
          label 0/1, NaN = tidak diintervensi (RandInt HiCEM / evaluasi).
        - ``sample_intervention_mask`` (B, S) + ``parent_labels`` (B, C): seluruh
          konsep induk diganti anchor ground truth pada sampel MC terpilih
          (``intervention_prob`` saat training ProbCBM).
        """
        anchors_needed = parent_intervention is not None or sub_intervention is not None or sample_intervention_mask is not None
        if anchors_needed and self.parent_anchors is None:
            raise ValueError("Intervensi anchor membutuhkan parent_probability='hicem_mix'.")
        batch = h.shape[0]
        mu_sub, sigma2_sub = self.subconcept_predictor(h)

        if subconcept_intervention is not None and subconcept_intervention_mask is not None:
            mask = subconcept_intervention_mask.unsqueeze(-1)
            mu_sub = torch.where(mask, subconcept_intervention, mu_sub)
            sigma2_sub = torch.where(mask, torch.zeros_like(sigma2_sub), sigma2_sub)

        mu_parent, sigma2_parent, alpha = self._aggregate_all_concepts(mu_sub, sigma2_sub)
        mu_cls, sigma2_cls = mu_parent, sigma2_parent
        if sub_intervention is not None:
            valid = self.subconcept_predictor.mask.unsqueeze(0)
            mask, target = self._anchor_targets(self.sub_anchors.unsqueeze(0), sub_intervention)
            mask = (mask & valid).unsqueeze(-1)
            mu_cls, sigma2_cls, _ = self._aggregate_all_concepts(
                torch.where(mask, target, mu_sub), torch.where(mask, torch.zeros_like(sigma2_sub), sigma2_sub)
            )
        if parent_intervention is not None:
            mask, target = self._anchor_targets(self.parent_anchors.unsqueeze(0), parent_intervention)
            mask = mask.unsqueeze(-1)
            mu_cls = torch.where(mask, target, mu_cls)
            sigma2_cls = torch.where(mask, torch.zeros_like(sigma2_cls), sigma2_cls)

        n_samples = self.n_mc_samples_train if self.training else self.n_mc_samples_eval
        z = reparameterize_parent(mu_cls, sigma2_cls, n_samples=n_samples)  # (B, C, S, d_c)
        if sample_intervention_mask is not None:
            _, target = self._anchor_targets(self.parent_anchors.unsqueeze(0), parent_labels.float())
            z = torch.where(sample_intervention_mask[:, None, :, None], target.unsqueeze(2), z)

        bottleneck = z.permute(0, 2, 1, 3).reshape(batch, n_samples, -1)  # (B, S, C*d_c)
        task_logits_per_sample = self.classifier(bottleneck)  # (B, S, num_classes)
        task_probs = torch.softmax(task_logits_per_sample, dim=-1).mean(dim=1)  # p_bar, Bab IV.5.2.5

        valid_mask = self.subconcept_predictor.mask
        sub_probs = self.scorer(mu_sub) * valid_mask.unsqueeze(0)

        kl_loss = kl_to_standard_normal(mu_parent, sigma2_parent)
        if self.parent_probability == "hicem_mix":
            mu_own, sigma2_own = self.parent_predictor(h)
            own_probs = self.scorer.probability_from_distribution(mu_own, sigma2_own, n_samples=n_samples)
            parent_probs = 0.5 * (own_probs + hicem_soft_max(sub_probs, valid_mask))
            kl_loss = kl_loss + kl_to_standard_normal(mu_own, sigma2_own)
        else:
            # Mode lama: induk dinilai dari agregasi attention subkonsep.
            # Karena scorer linear dan sum(alpha)=1, logit induk menjadi rata-rata
            # berbobot logit subkonsep; dipertahankan hanya untuk evaluasi ulang
            # checkpoint lama.
            parent_probs = self.scorer.probability_from_distribution(mu_parent, sigma2_parent, n_samples=n_samples)

        return Stage2Output(
            mu_sub=mu_sub,
            sigma2_sub=sigma2_sub,
            mu_parent=mu_parent,
            sigma2_parent=sigma2_parent,
            attention=alpha,
            sub_concept_probs=sub_probs,
            parent_concept_probs=parent_probs,
            task_logits_per_sample=task_logits_per_sample,
            task_probs=task_probs,
            kl_loss=kl_loss,
        )
