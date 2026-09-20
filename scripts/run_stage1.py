#!/usr/bin/env python
"""CLI: python scripts/run_stage1.py --config configs/cub_stage1.yaml

Melatih Probabilistic Concept Predictor (Tahap 1, Bab IV.5.1) dan
menjalankan automatic subconcept discovery di akhir training.
"""

from __future__ import annotations

import logging

from hiprobcbm.config import build_arg_parser, load_config, prepare_log_dir, resolve_device, apply_seed_override
from hiprobcbm.engine import train_stage1
from hiprobcbm.utils.seed import set_random_seed


def main() -> None:
    parser = build_arg_parser("Tahap 1 HiProbCBM: probabilistic concept predictor + subconcept discovery")
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
    logging.info("Konfigurasi: %s | perangkat: %s | log: %s", args.config, device, log_dir)

    train_stage1.run(cfg, device, log_dir, resume=args.resume)


if __name__ == "__main__":
    main()
