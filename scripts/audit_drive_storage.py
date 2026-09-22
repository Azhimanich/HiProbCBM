#!/usr/bin/env python
"""Read-only storage inventory for a mounted Google Drive.

It uses file metadata only: no file contents are read, changed, or deleted.
The scan may take several minutes for a large Drive mount.

Example:
    python scripts/audit_drive_storage.py --root /content/drive/MyDrive --top 100
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


def format_size(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def scan(root: Path, progress_every: int) -> tuple[dict[str, int], list[tuple[int, Path]], int, int]:
    """Return first-level directory totals, all files, skipped entries, and count."""
    totals: dict[str, int] = defaultdict(int)
    files: list[tuple[int, Path]] = []
    skipped = 0
    count = 0
    root = root.resolve()

    for current, dirs, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        # A symlink can point outside MyDrive or duplicate an already counted tree.
        dirs[:] = [name for name in dirs if not (current_path / name).is_symlink()]
        for name in names:
            path = current_path / name
            if path.is_symlink():
                skipped += 1
                continue
            try:
                size = path.stat().st_size
            except OSError:
                skipped += 1
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:  # pragma: no cover - defensive for unusual mounts
                skipped += 1
                continue
            group = relative.parts[0] if len(relative.parts) > 1 else "(file di root)"
            totals[group] += size
            files.append((size, path))
            count += 1
            if progress_every and count % progress_every == 0:
                print(f"Memeriksa {count:,} file...", flush=True)
    return totals, files, skipped, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive"))
    parser.add_argument("--top", type=int, default=100, help="Jumlah file terbesar yang ditampilkan.")
    parser.add_argument("--min-mib", type=float, default=100.0, help="Tampilkan file minimal sebesar ini (MiB).")
    parser.add_argument("--csv", type=Path, default=None, help="Simpan daftar file besar ke CSV opsional.")
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"Root tidak ditemukan atau bukan direktori: {root}")
    if args.top < 1 or args.min_mib < 0:
        raise ValueError("--top harus positif dan --min-mib tidak boleh negatif.")

    print(f"Memindai metadata saja (read-only): {root}")
    totals, files, skipped, count = scan(root, args.progress_every)
    total_size = sum(totals.values())
    threshold = int(args.min_mib * 1024 * 1024)
    largest = [(size, path) for size, path in files if size >= threshold]
    largest.sort(key=lambda item: item[0], reverse=True)

    print(f"\nTotal file terhitung: {count:,} | ukuran file: {format_size(total_size)} | dilewati: {skipped}")
    print("\nFolder tingkat atas terbesar:")
    for name, size in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        print(f"{format_size(size):>12}  {name}")

    print(f"\n{min(args.top, len(largest))} file terbesar (>= {args.min_mib:g} MiB):")
    for size, path in largest[:args.top]:
        print(f"{format_size(size):>12}  {path}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("size_bytes", "size_human", "path"))
            writer.writeheader()
            for size, path in largest:
                writer.writerow({"size_bytes": size, "size_human": format_size(size), "path": str(path)})
        print(f"\nCSV file besar disimpan: {args.csv}")


if __name__ == "__main__":
    main()
