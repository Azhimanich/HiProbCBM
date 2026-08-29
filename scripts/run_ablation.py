#!/usr/bin/env python
"""CLI: python scripts/run_ablation.py --variant a1 --config configs/cub_stage2.yaml \
    --stage1-log-dir train_log/cub/cub_stage1

Menjalankan salah satu varian studi ablasi HiProbCBM (Tabel 4.8):

    a1  - HiProbCBM-A1: tanpa learned attention (UniformAggregator).
    a2  - HiProbCBM-A2: tanpa regularisasi KL Divergence (lambda_kl=0).
    a3  - HiProbCBM-A3: varian Sparse Autoencoder (kl <-> batchtopk),
          diterapkan pada TAHAP 1 (run_stage1.py) - lihat catatan di bawah.

Untuk a3, jalankan run_stage1.py dua kali dengan `sae.variant` berbeda
(kl / batchtopk) pada config Tahap 1, lalu run_stage2.py masing-masing
dari `pseudo_hierarchy.pt` yang dihasilkan; skrip ini hanya mengurus
a1 & a2 yang murni parameter Tahap 2.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hiprobcbm.config import load_config, prepare_log_dir, resolve_device
from hiprobcbm.engine import train_stage2
from hiprobcbm.utils.seed import set_random_seed

ABLATION_OVERRIDES = {
    "a1": {"model": {"use_attention": False}},
    "a2": {"train": {"lambda_kl": 0.0}},
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Studi ablasi HiProbCBM (Tabel 4.8)")
    parser.add_argument("--variant", choices=["a1", "a2"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base-config", type=str, default=None)
    parser.add_argument("--stage1-log-dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.base_config)
    merged = _deep_merge(cfg.to_dict(), ABLATION_OVERRIDES[args.variant])
    merged["experiment_name"] = f"{merged.get('experiment_name', 'hiprobcbm')}_ablation_{args.variant}"
    from hiprobcbm.config import Config

    cfg = Config(merged)

    device = resolve_device(args.gpu)
    log_dir = prepare_log_dir(cfg, args.log_dir)
    set_random_seed(args.seed or cfg.get("seed", 42))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Ablasi HiProbCBM-%s | konfigurasi: %s | log: %s", args.variant.upper(), args.config, log_dir)

    train_stage2.run(cfg, device, log_dir, Path(args.stage1_log_dir))


if __name__ == "__main__":
    main()
