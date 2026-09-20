"""Pemuat konfigurasi (YAML + override CLI) untuk seluruh skrip HiProbCBM.

Konvensi: satu `base.yaml` menampung default lintas eksperimen (Bab IV.6 -
Implementasi Model), lalu setiap file konfigurasi eksperimen hanya
menuliskan bagian yang berbeda. Ini sejalan dengan pola yang dipakai kedua
baseline (ProbCBM: config_base.yaml + config_exp.yaml; HiCEM: satu yaml per
dataset), supaya perbandingan head-to-head mudah diaudit lewat diff config.
"""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge rekursif: `override` menang, tapi kunci yang tidak disebut di
    override tetap dipertahankan dari base (bukan replace-seluruh-dict)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Config:
    """Wrapper tipis di atas dict supaya bisa diakses lewat atribut
    (`cfg.model.d_c`) maupun dict (`cfg["model"]["d_c"]`)."""

    _data: dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        try:
            value = self._data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self._data.get(item, default)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover - kosmetik
        return f"Config({self._data!r})"


def load_config(exp_config_path: str | Path, base_config_path: str | Path | None = None) -> Config:
    base = load_yaml(base_config_path or DEFAULT_BASE_CONFIG)
    exp = load_yaml(exp_config_path)
    merged = _deep_update(base, exp)
    merged.setdefault("experiment_name", Path(exp_config_path).stem)
    merged.setdefault("config_path", str(exp_config_path))
    return Config(merged)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path ke file YAML konfigurasi eksperimen.")
    parser.add_argument("--base-config", type=str, default=None, help="Override base.yaml (opsional).")
    parser.add_argument("--gpu", type=int, default=0, help="Index GPU CUDA; -1 untuk CPU.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed acak dari config.")
    parser.add_argument("--log-dir", type=str, default=None, help="Override direktori log/checkpoint.")
    parser.add_argument("--resume", type=str, default="auto", help="auto (default): validasi lalu lanjut; never: wajib direktori baru; atau path *_resume.pth run ini.")
    parser.add_argument("--only-eval", action="store_true", help="Lewati training, langsung evaluasi checkpoint.")
    return parser


def apply_seed_override(cfg: Config, seed: int | None) -> Config:
    values = cfg.to_dict()
    values["seed"] = values.get("seed", 42) if seed is None else seed
    return Config(values)


def resolve_device(gpu: int):
    import torch

    if gpu is None or gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu}")


def prepare_log_dir(cfg: Config, override: str | None = None) -> Path:
    log_dir = Path(override or cfg.get("log_dir", "train_log")) / cfg.get("dataset", "misc") / cfg.get(
        "experiment_name", "run"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
