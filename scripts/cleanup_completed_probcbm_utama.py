#!/usr/bin/env python
"""Safely reclaim resume storage from completed ProbCBM-utama runs.

Default mode is a dry run. It never touches datasets, A1/A2 inputs, best
checkpoints, manifests, logs, or result CSVs. Pass --apply only after review.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


RESUME_FILES = {
    "stage1": (
        "stage1_resume.pth", "stage1_resume.pth.sha256",
        "stage1_resume.prev.pth", "stage1_resume.prev.pth.sha256", "stage1_last.pth",
    ),
    "stage2": (
        "stage2_resume.pth", "stage2_resume.pth.sha256",
        "stage2_resume.prev.pth", "stage2_resume.prev.pth.sha256", "stage2_last.pth",
    ),
}


@dataclass(frozen=True)
class Removal:
    path: Path
    reason: str


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _format_size(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _files_to_remove(directory: Path, stage: str) -> list[Removal]:
    return [
        Removal(directory / name, f"{stage} selesai; state resume tidak lagi diperlukan")
        for name in RESUME_FILES[stage]
        if (directory / name).exists()
    ]


def make_plan(root: Path, seeds: list[int]) -> tuple[list[Removal], list[str]]:
    """Return only removals that cannot break A1/A2 or completed results."""
    root = root.resolve()
    removals: list[Removal] = []
    notes: list[str] = []

    for seed in seeds:
        seed_dir = root / f"seed_{seed}"
        if not seed_dir.is_dir():
            notes.append(f"LEWATI seed {seed}: folder tidak ditemukan: {seed_dir}")
            continue

        hierarchy_files = sorted(seed_dir.rglob("pseudo_hierarchy.pt"))
        completed_stage2 = sorted({path.parent for path in seed_dir.rglob("stage2_last.pth")})
        if len(hierarchy_files) != 1:
            notes.append(f"LEWATI seed {seed}: ditemukan {len(hierarchy_files)} pseudo_hierarchy.pt; struktur ambigu.")
            continue
        hierarchy = hierarchy_files[0]
        if not hierarchy.with_suffix(hierarchy.suffix + ".sha256").is_file():
            notes.append(f"LEWATI seed {seed}: checksum hierarchy tidak ada: {hierarchy}")
            continue
        if not completed_stage2:
            notes.append(f"LEWATI seed {seed}: belum ada stage2_last.pth; Stage 2 mungkin belum selesai.")
            continue

        stage1_dir = hierarchy.parent
        removals.extend(
            candidate for candidate in _files_to_remove(stage1_dir, "stage1") if _inside(root, candidate.path)
        )
        discovery = stage1_dir / "discovery"
        if discovery.is_dir() and _inside(root, discovery):
            removals.append(Removal(discovery, "pseudo_hierarchy.pt selesai; checkpoint SAE tidak dibutuhkan A1/A2"))
        discovery_features = stage1_dir / "discovery_features.pt"
        if discovery_features.is_file() and _inside(root, discovery_features):
            removals.append(Removal(discovery_features, "pseudo_hierarchy.pt sudah selesai"))
            checksum = discovery_features.with_suffix(discovery_features.suffix + ".sha256")
            if checksum.is_file():
                removals.append(Removal(checksum, "checksum artifact discovery yang dihapus"))

        for stage2_dir in completed_stage2:
            if not _inside(root, stage2_dir):
                notes.append(f"LEWATI direktori di luar root: {stage2_dir}")
            elif not (stage2_dir / "stage2_best.pth").is_file():
                notes.append(f"LEWATI {stage2_dir}: stage2_best.pth tidak ada.")
            else:
                removals.extend(_files_to_remove(stage2_dir, "stage2"))

    unique = {item.path.resolve(): item for item in removals}
    return list(unique.values()), notes


def remove(item: Removal, root: Path) -> None:
    if not item.path.exists() or not _inside(root, item.path):
        raise RuntimeError(f"Menolak menghapus path tidak aman: {item.path}")
    if item.path.is_dir():
        shutil.rmtree(item.path)
    else:
        item.path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("train_log/probcbm/utama"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--apply", action="store_true", help="Hapus. Tanpa flag ini hanya dry-run.")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir() or root.parts[-2:] != ("probcbm", "utama"):
        raise ValueError(f"Root harus folder train_log/probcbm/utama yang ada, bukan: {root}")

    plan, notes = make_plan(root, args.seeds)
    print("MODE:", "HAPUS" if args.apply else "DRY-RUN (tidak ada file dihapus)")
    print("ROOT:", root)
    for note in notes:
        print(note)
    if not plan:
        print("Tidak ada file yang memenuhi syarat aman untuk dibersihkan.")
        return

    total = 0
    for item in sorted(plan, key=lambda entry: str(entry.path)):
        size = _size(item.path)
        total += size
        print(f"{_format_size(size):>10}  {item.path}\n             alasan: {item.reason}")
    print(f"\nTotal kandidat: {_format_size(total)} ({len(plan)} item)")
    print("Dilindungi: pseudo_hierarchy.pt (+ .sha256), stage1/2_best.pth, manifest, CSV, log.")

    if args.apply:
        for item in plan:
            remove(item, root)
        print(f"SELESAI: {_format_size(total)} telah dihapus dari Drive mount.")
    else:
        print("Tinjau daftar. Jika seluruhnya benar, ulangi dengan --apply.")


if __name__ == "__main__":
    main()
