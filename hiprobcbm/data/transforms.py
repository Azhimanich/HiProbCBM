"""Transformasi citra per backbone (Bab IV.6 & Lampiran C.2 ProbCBM).

Kebijakan augmentasi mengikuti proposal:
- CUB + Inception-v3 / ResNet18: resize 256 -> random-crop 224, atau
  resize 341 -> random-crop 299 untuk konfigurasi 299x299.
- Kitchens + CLIP ViT-L/14: preprocessing standar CLIP (resize persegi
  224 + normalisasi CLIP), mengikuti konfigurasi HiCEM agar `foundation
  model` tidak di-fine-tune.
"""

from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def imagenet_transforms(image_size: int, train: bool) -> T.Compose:
    resize_to = int(image_size * 256 / 224)
    if train:
        return T.Compose(
            [
                T.Resize((resize_to, resize_to)),
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((resize_to, resize_to)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def clip_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def build_transform(backbone: str, image_size: int, train: bool) -> T.Compose:
    if backbone in ("inception_v3", "resnet18"):
        return imagenet_transforms(image_size, train)
    if backbone == "clip_vit_l14":
        return clip_transforms(image_size)
    raise ValueError(f"Backbone tidak dikenal untuk transform: {backbone}")
