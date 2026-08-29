"""Label predictor terstandardisasi (Keputusan Desain Eksperimen #1).

Seluruh model - CBM, CEM, ProbCBM (adaptasi), HiCEM (adaptasi), dan
HiProbCBM - memakai classifier head yang SAMA: satu lapisan
fully-connected diikuti softmax, mengambil bottleneck (konkatenasi
representasi seluruh konsep top-level) sebagai masukan.

Ini deviasi eksplisit dari mekanisme asli ProbCBM (jarak Euclidean ke
anchor kelas terlatih, Eq. 4-5 Kim dkk., 2023) - dilakukan supaya
perbedaan performa antar-model bisa diatribusikan pada perbedaan
representasi konsep, bukan arsitektur classifier. `AnchorClassifier` di
bawah tetap disediakan sebagai jalur validasi ("sanity check") untuk
mereplikasi ProbCBM asli sebelum classifier distandardisasi - lihat
`hiprobcbm/models/baselines/probcbm.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSoftmaxClassifier(nn.Module):
    """Classifier head standar yang dipakai seluruh model dalam studi ini."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dims: tuple[int, ...] = ()):
        super().__init__()
        dims = [input_dim, *hidden_dims, num_classes]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """bottleneck: (..., input_dim) -> logits: (..., num_classes)."""
        return self.net(bottleneck)

    def predict_proba(self, bottleneck: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(bottleneck), dim=-1)


class AnchorClassifier(nn.Module):
    """Mekanisme klasifikasi asli ProbCBM (Eq. 4-5, Kim dkk., 2023):
    proyeksi ke ruang embedding kelas, lalu jarak Euclidean ke anchor
    kelas terlatih. Dipakai HANYA untuk sanity-check replikasi baseline
    (lihat "Keputusan Desain Eksperimen" #1), tidak dipakai pada
    perbandingan utama antar-model.
    """

    def __init__(self, input_dim: int, num_classes: int, class_embedding_dim: int, distance_scale_init: float = 5.0):
        super().__init__()
        self.projection = nn.Linear(input_dim, class_embedding_dim)
        self.class_anchors = nn.Parameter(torch.randn(num_classes, class_embedding_dim))
        self.scale = nn.Parameter(torch.tensor(float(distance_scale_init)))

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        h = self.projection(bottleneck)  # (B, d_y)
        dist = torch.cdist(h.unsqueeze(1), self.class_anchors.unsqueeze(0)).squeeze(1)  # (B, num_classes)
        return -self.scale * dist  # logits: kelas dengan jarak terkecil -> logit terbesar

    def predict_proba(self, bottleneck: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(bottleneck), dim=-1)
