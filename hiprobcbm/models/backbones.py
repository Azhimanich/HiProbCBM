"""Feature extractor bersama untuk seluruh model (Bab IV.6 - Implementasi
Model & "Keputusan Desain Eksperimen" perihal dual backbone).

Tiga backbone didukung dengan antarmuka seragam `Backbone.forward(x) ->
Tensor[B, output_dim]`, supaya `d_h` (Bab IV.5.1.1) konsisten dipakai oleh
seluruh model pembanding maupun HiProbCBM:

- ``inception_v3``  - konvensi asli CBM/CEM/HiCEM(CUB opsional), 299x299.
- ``resnet18``      - backbone utama ProbCBM (Kim dkk., 2023), 224x224.
- ``clip_vit_l14``  - dipakai HiCEM untuk PseudoKitchens & ImageNet, TIDAK
  di-fine-tune (frozen), sejalan dengan konfigurasi HiCEM (`use foundation
  model representations`).

Dua backbone pertama diuji berdampingan untuk CUB (lihat catatan "Backbone
CUB diuji dengan Inception-v3 dan ResNet18" di Bagian 3 lembar rujukan).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models


class Backbone(nn.Module):
    def __init__(self, module: nn.Module, output_dim: int, frozen: bool = False):
        super().__init__()
        self.module = module
        self.output_dim = output_dim
        self.frozen = frozen
        if frozen:
            for p in self.module.parameters():
                p.requires_grad_(False)
            self.module.eval()

    def train(self, mode: bool = True):
        # Backbone beku (mis. CLIP) tetap di mode eval walau model induk .train()
        super().train(mode)
        if self.frozen:
            self.module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                return self._extract(x)
        return self._extract(x)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - overridden
        raise NotImplementedError


class _ResNetBackbone(Backbone):
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.module(x)
        return torch.flatten(feat, 1)


class _InceptionBackbone(Backbone):
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        # torchvision inception_v3 mengembalikan InceptionOutputs(logits, aux)
        # saat mode train dengan aux_logits=True; kita matikan aux_logits di
        # build_backbone jadi keluarannya selalu tensor tunggal (fitur pooling).
        return torch.flatten(out, 1)


class _CLIPBackbone(Backbone):
    def __init__(self, clip_model, preprocess, output_dim: int):
        super().__init__(clip_model, output_dim=output_dim, frozen=True)
        self.preprocess = preprocess  # info saja; transform citra sudah ditangani di data/transforms.py

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        features = self.module.encode_image(x)
        return features.float()


class _CachedCLIPFeatureBackbone(Backbone):
    """Identity encoder for representations materialised by feature_cache.

    The input is already a ``(B, 768)`` CLIP ViT-L/14 representation.  Keeping
    it behind the normal Backbone interface means the two HiProbCBM stages do
    not need a separate model architecture for the frozen-foundation protocol.
    """

    def __init__(self):
        super().__init__(nn.Identity(), output_dim=768, frozen=True)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.output_dim:
            raise ValueError(
                "Cache CLIP harus berbentuk (batch, 768); jangan memakai "
                "use_cached_features dengan tensor gambar mentah."
            )
        return x.float()


def _strip_classifier(model: nn.Module) -> nn.Module:
    """Ganti fc/classifier terakhir dengan Identity supaya keluaran adalah
    fitur pooled, bukan logit 1000-kelas ImageNet."""
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    return model


def build_backbone(name: str, pretrained: bool = True, use_cached_features: bool = False) -> Backbone:
    name = name.lower()
    if use_cached_features:
        if name != "clip_vit_l14":
            raise ValueError("use_cached_features hanya tersedia untuk backbone clip_vit_l14.")
        return _CachedCLIPFeatureBackbone()
    if name == "resnet18":
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = tv_models.resnet18(weights=weights)
        net = _strip_classifier(net)
        return _ResNetBackbone(net, output_dim=512)

    if name == "inception_v3":
        weights = tv_models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        net = tv_models.inception_v3(weights=weights, aux_logits=True, init_weights=not pretrained)
        net.aux_logits = False
        net.AuxLogits = None
        net = _strip_classifier(net)
        return _InceptionBackbone(net, output_dim=2048)

    if name == "clip_vit_l14":
        try:
            import clip  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "Backbone 'clip_vit_l14' membutuhkan paket opsional `clip` "
                "(pip install 'hiprobcbm[clip]'). Lihat README bagian instalasi."
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-L/14", device=device)
        return _CLIPBackbone(model, preprocess, output_dim=768)

    raise ValueError(f"Backbone tidak dikenal: {name!r}. Pilihan: resnet18, inception_v3, clip_vit_l14.")
