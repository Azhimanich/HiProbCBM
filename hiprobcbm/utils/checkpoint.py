"""Validated epoch-boundary checkpoints for the single-process trainers.

The checkpoint is the authority; *_best/last.pth are repairable exports.
No mid-batch continuation is claimed. Use one writer per experiment directory.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
SCHEMA = 2


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(value, path):
    """Write beside the destination, flush, then replace (never truncate live file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_artifact(value, path):
    atomic_save(value, path)
    path = Path(path)
    atomic_text(file_hash(path), path.with_suffix(path.suffix + ".sha256"))


def load_artifact(path):
    path = Path(path)
    if file_hash(path) != path.with_suffix(path.suffix + ".sha256").read_text().strip():
        raise ValueError(f"Artifact rusak (checksum berbeda): {path}. Pulihkan salinan utuh; tidak ditimpa otomatis.")
    return torch.load(path, map_location="cpu", weights_only=True)


def capture_rng():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def require_finite(value):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError("Checkpoint/loss mengandung NaN atau Inf; checkpoint valid sebelumnya dipertahankan.")
    elif isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Checkpoint/metric mengandung NaN atau Inf.")


def run_identity(cfg, device, artifacts=()):
    """Hash metadata, not image pixels; record the exact effective config and code."""
    config = cfg.to_dict()
    for key in ("log_dir", "config_path"):
        config.pop(key, None)
    root = Path(__file__).resolve().parents[2]
    code = {str(p.relative_to(root)).replace("\\", "/"): file_hash(p)
            for p in sorted((root / "hiprobcbm").rglob("*.py"))}
    metadata = {}
    data_root = config.get("data", {}).get("data_root")
    if data_root:
        base = Path(data_root)
        candidates = [base / "manifest.csv", base / "info.json", base / "attributes.txt", base / "metadata" / "attributes.txt"]
        candidates += list((base / "metadata").glob("*.pkl"))
        metadata = {str(p.relative_to(base)).replace("\\", "/"): file_hash(p)
                    for p in candidates if p.is_file()}
        if not metadata:
            raise ValueError(f"Metadata dataset tidak ditemukan di {base}; identitas resume tidak dapat diperiksa.")
    environment = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "cuda": torch.version.cuda,
        "device": str(device),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "packages": {name: importlib.metadata.version(name) for name in ("torchvision", "Pillow")},
        "cudnn": torch.backends.cudnn.version(),
    }
    return {"config": config, "code": code, "data": metadata, "environment": environment,
            "artifacts": {Path(p).name: file_hash(p) for p in artifacts}}


class RunCheckpoint:
    def __init__(self, log_dir, name, model, optimizer, identity, total_epochs, resume="auto", scheduler=None, patience=None):
        self.directory = Path(log_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{name}_resume.pth"
        self.previous = self.directory / f"{name}_resume.prev.pth"
        self.name, self.model, self.optimizer = name, model, optimizer
        self.identity, self.total_epochs = identity, total_epochs
        self.epoch, self.best_epoch, self.best_val = -1, -1, -1.0
        self.best_model = None
        self.scheduler, self.patience = scheduler, patience
        self.bad_epochs = 0
        self.history = []
        self.current_valid = False
        if total_epochs < 1:
            raise ValueError("Jumlah epoch harus positif.")
        if resume not in ("auto", "never"):
            # Restrict to this run's authority file: prevents branching into stale artifacts.
            if Path(resume).resolve() != self.path.resolve():
                raise ValueError(f"Gunakan checkpoint lengkap {self.path}; bobot best/last bukan resume. "
                                 "Untuk pindah storage, salin seluruh direktori run lalu gunakan --resume auto.")
        exists = self.path.exists() or self.previous.exists()
        if resume == "never" and (exists or list(self.directory.glob("*.pth"))):
            raise ValueError("Direktori berisi checkpoint. Pilih log-dir baru untuk training baru.")
        if exists:
            payload = self._read()
            self._validate(payload)
            self.model.load_state_dict(payload["model"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer"])
            self.epoch, self.best_epoch = payload["epoch"], payload["best_epoch"]
            self.best_val, self.best_model = payload["best_val"], payload["best_model"]
            self.bad_epochs, self.history = payload["bad_epochs"], payload["history"]
            if scheduler is not None:
                scheduler.load_state_dict(payload["scheduler"])
            restore_rng(payload["rng"])
            logger.info("RESUME VALID: %s | epoch selesai=%d/%d | lanjut epoch=%d | best_val=%.6f (epoch=%d)",
                        self.path, self.epoch + 1, total_epochs, self.epoch + 2, self.best_val, self.best_epoch + 1)
        elif resume not in ("auto", "never"):
            raise FileNotFoundError(self.path)
        elif list(self.directory.glob("*.pth")) or (self.directory / "pseudo_hierarchy.pt").exists():
            raise ValueError("Hanya ada artifact lama tanpa checkpoint resume lengkap. Optimizer/epoch/RNG "
                             "tidak dapat direkonstruksi. Arsipkan run lama dan pilih --log-dir baru; jangan menimpanya.")
        else:
            # Also makes an interruption during the very first epoch recoverable.
            self._save()
        atomic_text(json.dumps({"schema": SCHEMA, "name": name, **identity}, indent=2),
                    self.directory / "run_manifest.json")

    def _read(self):
        errors = []
        for candidate in (self.path, self.previous):
            if not candidate.exists():
                continue
            try:
                expected = candidate.with_suffix(candidate.suffix + ".sha256").read_text().strip()
                if file_hash(candidate) != expected:
                    raise ValueError("checksum SHA256 tidak cocok")
                payload = torch.load(candidate, map_location="cpu", weights_only=True)
            except Exception as exc:
                errors.append(f"{candidate.name}: {exc}")
                continue
            self.current_valid = candidate == self.path
            if not self.current_valid:
                logger.warning("Checkpoint terbaru rusak/tidak lengkap; memakai cadangan %s. %s", candidate, errors)
            return payload
        raise ValueError("Tidak ada checkpoint utuh. " + " | ".join(errors))

    def _validate(self, payload):
        required = {"schema", "name", "identity", "epoch", "best_epoch", "best_val", "model", "best_model", "optimizer", "rng"}
        required |= {"optimizer_type", "scheduler", "bad_epochs", "history", "patience"}
        if not isinstance(payload, dict) or not required <= payload.keys() or payload["schema"] != SCHEMA:
            raise ValueError("Format checkpoint tidak lengkap/legacy; resume ditolak.")
        if payload["name"] != self.name:
            raise ValueError("Jenis stage/baseline checkpoint berbeda.")
        if payload["optimizer_type"] != type(self.optimizer).__name__ or payload["patience"] != self.patience:
            raise ValueError("Optimizer/early stopping berbeda.")
        if (payload["scheduler"] is None) != (self.scheduler is None):
            raise ValueError("Scheduler checkpoint berbeda.")
        for key in self.identity:
            if payload["identity"].get(key) == self.identity[key]:
                continue
            if key == "environment":
                # Colab dapat mengganti image/Python/CUDA di antara dua sesi.
                # Tensor model dan state optimizer tetap divalidasi di bawah;
                # perubahan ini dicatat, tetapi tidak boleh menghapus peluang
                # untuk melanjutkan run yang sama dari Drive.
                logger.warning(
                    "RUNTIME BERUBAH saat resume. Checkpoint akan dilanjutkan setelah "
                    "validasi model/optimizer/RNG; hasil tidak dijamin bitwise identik. "
                    "Sebelumnya=%r | sekarang=%r",
                    payload["identity"].get("environment"), self.identity["environment"],
                )
                continue
            raise ValueError(f"Resume ditolak: {key} berbeda (config/seed, kode, data, atau hierarchy). "
                             "Pulihkan setting semula, atau gunakan direktori run baru.")
        epoch, best_epoch = payload["epoch"], payload["best_epoch"]
        if type(epoch) is not int or not -1 <= epoch < self.total_epochs:
            raise ValueError("Epoch checkpoint tidak valid.")
        if type(best_epoch) is not int or not -1 <= best_epoch <= epoch:
            raise ValueError("Best epoch checkpoint tidak valid.")
        if (epoch >= 0) != (payload["best_model"] is not None) or (epoch >= 0 and best_epoch < 0):
            raise ValueError("Best model/epoch tidak konsisten.")
        expected = self.model.state_dict()
        for state in (payload["model"], payload["best_model"]):
            if state is None:
                continue
            if state.keys() != expected.keys() or any(
                state[k].shape != expected[k].shape or state[k].dtype != expected[k].dtype for k in expected
            ):
                raise ValueError("Key/shape/dtype bobot checkpoint tidak cocok dengan model.")
            require_finite(state)
        saved_groups = payload["optimizer"]["param_groups"]
        if len(saved_groups) != len(self.optimizer.param_groups):
            raise ValueError("Optimizer param groups berbeda.")
        for saved, current in zip(saved_groups, self.optimizer.param_groups):
            if len(saved["params"]) != len(current["params"]):
                raise ValueError("Jumlah parameter optimizer berbeda.")
            for index, param in zip(saved["params"], current["params"]):
                state = payload["optimizer"]["state"].get(index, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"):
                    if key in state and state[key].shape != param.shape:
                        raise ValueError("Shape state Adam tidak cocok.")
        require_finite(payload["optimizer"])
        if epoch >= 0 and not payload["optimizer"]["state"]:
            raise ValueError("Optimizer state kosong setelah training; bukan resume lengkap.")
        for state in payload["optimizer"]["state"].values():
            keys = {"momentum_buffer"} if isinstance(self.optimizer, torch.optim.SGD) else {"step", "exp_avg", "exp_avg_sq"}
            if not keys <= state.keys():
                raise ValueError("State Adam tidak lengkap.")
        require_finite(payload["best_val"])
        # Validate RNG on isolated generators; do not consume live RNG before load.
        rng = payload["rng"]
        random.Random().setstate(rng["python"])
        ns = rng["numpy"]
        np.random.RandomState().set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
        torch.Generator().set_state(rng["torch"])
        if len(rng["cuda"]) != torch.cuda.device_count():
            raise ValueError("Jumlah CUDA RNG berbeda; gunakan runtime semula.")
        for i, state in enumerate(rng["cuda"]):
            torch.Generator(device=f"cuda:{i}").set_state(state)

    def _save(self):
        payload = {"schema": SCHEMA, "name": self.name, "identity": self.identity,
                   "epoch": self.epoch, "best_epoch": self.best_epoch, "best_val": self.best_val,
                   "model": cpu_copy(self.model.state_dict()), "best_model": self.best_model,
                   "optimizer": cpu_copy(self.optimizer.state_dict()), "rng": capture_rng()}
        payload.update(optimizer_type=type(self.optimizer).__name__,
                       scheduler=self.scheduler.state_dict() if self.scheduler is not None else None,
                       bad_epochs=self.bad_epochs, history=self.history, patience=self.patience,
                       completed=self.completed)
        require_finite(payload["model"])
        require_finite(payload["optimizer"])
        if self.epoch >= 0 and not payload["optimizer"]["state"]:
            raise ValueError("Tidak ada update optimizer; periksa batch size/drop_last dan jumlah sampel train.")
        if self.current_valid:
            # Keep the previous *validated* generation, even after corruption recovery.
            temporary = self.previous.with_suffix(".copy.tmp")
            shutil.copyfile(self.path, temporary)
            os.replace(temporary, self.previous)
            atomic_text(self.path.with_suffix(".pth.sha256").read_text(), self.previous.with_suffix(".pth.sha256"))
        atomic_save(payload, self.path)
        atomic_text(file_hash(self.path), self.path.with_suffix(".pth.sha256"))
        self.current_valid = True

    @property
    def start_epoch(self):
        return self.epoch + 1

    @property
    def completed(self):
        return self.start_epoch == self.total_epochs or (self.patience is not None and self.bad_epochs >= self.patience)

    def save_epoch(self, epoch, val):
        require_finite(val)
        if not self.epoch < epoch < self.total_epochs:
            raise ValueError("Checkpoint harus disimpan setelah epoch lengkap dengan indeks yang meningkat.")
        self.epoch = epoch
        improved = val > self.best_val
        if improved:
            self.best_val, self.best_epoch = float(val), epoch
            self.best_model = cpu_copy(self.model.state_dict())
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val)
            else:
                self.scheduler.step()
        self.history.append({"epoch": epoch, "val": float(val), "lr": [g["lr"] for g in self.optimizer.param_groups]})
        self._save()
        if improved:
            atomic_save(self.best_model, self.directory / f"{self.name}_best.pth")
        logger.info("CHECKPOINT SAVED: %s | epoch selesai=%d/%d | best_val=%.6f", self.path, epoch + 1, self.total_epochs, self.best_val)

    def export_weights(self):
        if not self.completed:
            raise ValueError("Training belum selesai; last tidak boleh menjadi marker selesai.")
        atomic_save(self.model.state_dict(), self.directory / f"{self.name}_last.pth")
        atomic_save(self.best_model, self.directory / f"{self.name}_best.pth")
