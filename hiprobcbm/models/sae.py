"""Sparse Autoencoder untuk automatic subconcept discovery (Bab III.3.7-3.8,
Bab IV.5.1.3) - dua varian, sesuai "Keputusan Desain Eksperimen" #2.

Keduanya menerima masukan berdimensi d_c (mean konsep PER-KONSEP, dikumpulkan
lintas sampel dataset) - BUKAN vektor C-dimensi lintas-konsep, memperbaiki
Temuan #1 pada lembar rujukan formulasi. Lihat `subconcept_discovery.py`
untuk bagaimana kedua varian ini dipanggil per-konsep.

1. ``KLSparseAutoencoder`` - SAE klasik dengan regularisasi sparsitas
   berbasis KL-Divergence terhadap target aktivasi rho (Cunningham dkk.,
   2023), konfigurasi utama proposal (Bab IV.5.1.3, Eq. L_SAE).

2. ``BatchTopKSAE`` - varian yang menjaga top-k aktivasi terbesar per
   batch (Bussmann dkk., 2024), sebagaimana dipakai pada Concept
   Splitting HiCEM. Diimplementasikan ulang secara mandiri di sini
   (bukan menyalin kode repo HiCEM) mengikuti definisi pada paper aslinya:
   top-k aktivasi dipertahankan, sisanya dinolkan, plus auxiliary loss
   untuk menghidupkan kembali fitur yang "mati".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEOutput:
    reconstruction: torch.Tensor
    activations: torch.Tensor
    loss: torch.Tensor
    loss_components: dict[str, torch.Tensor]


class KLSparseAutoencoder(nn.Module):
    """SAE klasik: reconstruction loss + regularisasi KL-sparsity.

    L_SAE = ||mu - mu_hat||_2^2 + beta * sum_j D_KL(rho || rho_hat_j)
    (Bab IV.5.1.3), dengan rho_hat_j = rata-rata aktivasi neuron laten ke-j
    pada satu batch.
    """

    def __init__(self, input_dim: int, latent_dim: int, sparsity_target: float = 0.05, beta: float = 1.0):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.sparsity_target = sparsity_target
        self.beta = beta

    def forward(self, x: torch.Tensor) -> SAEOutput:
        activations = torch.sigmoid(self.encoder(x))
        reconstruction = self.decoder(activations)

        recon_loss = F.mse_loss(reconstruction, x)

        rho = self.sparsity_target
        rho_hat = activations.mean(dim=0).clamp(1e-6, 1 - 1e-6)
        kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        sparsity_loss = self.beta * kl.sum()

        loss = recon_loss + sparsity_loss
        return SAEOutput(
            reconstruction=reconstruction,
            activations=activations,
            loss=loss,
            loss_components={"reconstruction": recon_loss.detach(), "sparsity": sparsity_loss.detach()},
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.encoder(x))


class BatchTopKSAE(nn.Module):
    """BatchTopK Sparse Autoencoder (Bussmann, Leask & Nanda, 2024), dipakai
    HiCEM untuk Concept Splitting. Top-k*batch aktivasi terbesar di seluruh
    batch dipertahankan (bukan top-k per-sampel), sisanya dinolkan.
    """

    def __init__(self, input_dim: int, latent_dim: int, top_k: int = 32,
                 aux_top_k: int = 128, aux_penalty: float = 1 / 32,
                 dead_after_n_batches: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.top_k = top_k
        self.aux_top_k = aux_top_k
        self.aux_penalty = aux_penalty
        self.dead_after_n_batches = dead_after_n_batches

        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.b_enc = nn.Parameter(torch.zeros(latent_dim))
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(input_dim, latent_dim)))
        self.W_dec = nn.Parameter(self.W_enc.t().detach().clone())
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))

        self.register_buffer("batches_since_active", torch.zeros(latent_dim))

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    def forward(self, x: torch.Tensor) -> SAEOutput:
        x_centered = x - self.b_dec
        pre_act = F.relu(x_centered @ self.W_enc + self.b_enc)

        batch_size = x.shape[0]
        k_total = min(self.top_k * batch_size, pre_act.numel())
        flat = pre_act.reshape(-1)
        topk = torch.topk(flat, k_total)
        mask = torch.zeros_like(flat)
        mask.scatter_(0, topk.indices, 1.0)
        activations = (flat * mask).reshape(pre_act.shape)

        reconstruction = activations @ self.W_dec + self.b_dec

        recon_loss = (reconstruction - x).pow(2).mean()

        with torch.no_grad():
            active = (activations.sum(dim=0) > 0)
            self.batches_since_active[active] = 0
            self.batches_since_active[~active] += 1

        aux_loss = self._auxiliary_loss(x, reconstruction, pre_act)
        loss = recon_loss + aux_loss

        return SAEOutput(
            reconstruction=reconstruction,
            activations=activations,
            loss=loss,
            loss_components={"reconstruction": recon_loss.detach(), "aux": aux_loss.detach()},
        )

    def _auxiliary_loss(self, x: torch.Tensor, reconstruction: torch.Tensor, pre_act: torch.Tensor) -> torch.Tensor:
        """Menghidupkan kembali neuron yang 'mati' >= dead_after_n_batches
        dengan merekonstruksi residual memakai fitur-fitur tersebut saja
        (Bussmann dkk., 2024, auxiliary loss)."""
        dead = self.batches_since_active >= self.dead_after_n_batches
        if dead.sum() == 0:
            return torch.zeros((), device=x.device)

        residual = (x - reconstruction).detach()
        k_aux = min(self.aux_top_k, int(dead.sum().item()))
        dead_acts = pre_act[:, dead]
        topk = torch.topk(dead_acts, k_aux, dim=-1)
        aux_acts = torch.zeros_like(dead_acts).scatter(-1, topk.indices, topk.values)
        aux_reconstruction = aux_acts @ self.W_dec[dead]
        return self.aux_penalty * (aux_reconstruction - residual).pow(2).mean()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).activations


def train_sae(
    sae: nn.Module,
    features: torch.Tensor,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    max_grad_norm: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Rutin pelatihan generik untuk kedua varian SAE (Bab IV.5.1.3-5.1.4).

    `features`: (N, d_c) - mean konsep untuk SATU konsep induk, dikumpulkan
    lintas sampel dataset (lihat `subconcept_discovery.build_pseudo_hierarchy`).
    Mengembalikan aktivasi laten akhir (N, dict_size) untuk pembentukan
    pseudo subconcept label (thresholding, Bab IV.5.1.4).
    """
    sae = sae.to(device)
    features = features.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    n = features.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = features[idx]

            output = sae(batch)
            optimizer.zero_grad()
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_grad_norm)
            optimizer.step()
            if isinstance(sae, BatchTopKSAE):
                sae.normalize_decoder_()

    sae.eval()
    with torch.no_grad():
        final_activations = sae.encode(features)
    return final_activations.cpu()


def _resolve_latent_dim(input_dim: int, kwargs: dict) -> int:
    """Bab IV.6: 'rasio ekspansi dimensi laten sebesar 8 kali dimensi
    concept embedding'. `latent_dim` eksplisit (jika ada) menang atas
    `latent_dim_ratio`; default rasio adalah 8."""
    if "latent_dim" in kwargs:
        return kwargs["latent_dim"]
    ratio = kwargs.get("latent_dim_ratio", 8)
    return input_dim * ratio


def build_sae(variant: str, input_dim: int, **kwargs) -> nn.Module:
    variant = variant.lower()
    latent_dim = _resolve_latent_dim(input_dim, kwargs)
    if variant in ("kl", "kl_sparsity", "classic"):
        return KLSparseAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            sparsity_target=kwargs.get("sparsity_target", 0.05),
            beta=kwargs.get("beta", 1.0),
        )
    if variant in ("batchtopk", "batch_topk"):
        return BatchTopKSAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            top_k=kwargs.get("top_k", 32),
            aux_top_k=kwargs.get("aux_top_k", 128),
            aux_penalty=kwargs.get("aux_penalty", 1 / 32),
            dead_after_n_batches=kwargs.get("dead_after_n_batches", 5),
        )
    raise ValueError(f"Varian SAE tidak dikenal: {variant!r}. Pilihan: 'kl', 'batchtopk'.")
