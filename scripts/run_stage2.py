#!/usr/bin/env python
"""CLI: python scripts/run_stage2.py --config configs/cub_stage2.yaml \
    --stage1-log-dir train_log/cub/cub_stage1

Melatih Hierarchical Concept Aggregation + Hierarchical Probabilistic
Reasoning (Tahap 2, Bab IV.5.2) end-to-end memakai pseudo hierarchical
subconcept dataset yang dihasilkan Tahap 1.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hiprobcbm.config import load_config, prepare_log_dir, resolve_device, apply_seed_override
from hiprobcbm.engine import train_stage2
from hiprobcbm.utils.seed import set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Tahap 2 HiProbCBM: hierarchical aggregation + reasoning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base-config", type=str, default=None)
    parser.add_argument("--stage1-log-dir", type=str, required=True, help="Direktori berisi pseudo_hierarchy.pt dari run_stage1.py")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--resume", default="auto", help="auto | never | path checkpoint lengkap run ini")
    args = parser.parse_args()

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

    train_stage2.run(cfg, device, log_dir, Path(args.stage1_log_dir), resume=args.resume)


if __name__ == "__main__":
    main()
