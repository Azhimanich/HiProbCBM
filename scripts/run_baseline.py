#!/usr/bin/env python
"""CLI: python scripts/run_baseline.py --config configs/baselines/probcbm_cub.yaml

Melatih satu model pembanding (CBM/CEM/ProbCBM/HiCEM), memakai dataset dan
classifier head yang sama dengan HiProbCBM (Tabel 4.7, "Keputusan Desain
Eksperimen" #1).
"""

from __future__ import annotations

import logging

from hiprobcbm.config import build_arg_parser, load_config, prepare_log_dir, resolve_device, apply_seed_override
from hiprobcbm.engine import train_baseline
from hiprobcbm.utils.seed import set_random_seed


def main() -> None:
    parser = build_arg_parser("Latih satu baseline (cbm/cem/probcbm/hicem)")
    args = parser.parse_args()
    if args.only_eval:
        parser.error("--only-eval tidak didukung trainer; gunakan evaluator notebook sel 18.")

    cfg = load_config(args.config, args.base_config)
    cfg = apply_seed_override(cfg, args.seed)
    device = resolve_device(args.gpu)
    log_dir = prepare_log_dir(cfg, args.log_dir)
    set_random_seed(cfg.seed)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "training.log", encoding="utf-8")],
    )
    logging.info("Baseline: %s | konfigurasi: %s | perangkat: %s | log: %s", cfg.baseline, args.config, device, log_dir)

    train_baseline.run(cfg, device, log_dir, resume=args.resume)


if __name__ == "__main__":
    main()
