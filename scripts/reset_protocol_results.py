#!/usr/bin/env python
"""Hapus seluruh hasil lama satu protokol sebelum menjalankan ulang revisi.

Revisi 11 Oktober 2026 mengubah Tahap 1 dan Tahap 2 pada protokol probcbm dan
hicem. Nama eksperimen tidak berubah, sehingga folder lama harus dihapus agar
run baru dapat menulis di lokasi yang sama (checkpoint lama ditolak oleh
validasi resume). Cache fitur CLIP di datasets/ tidak disentuh.

Default hanya dry-run. Contoh di Colab:

    !python scripts/reset_protocol_results.py --protocol probcbm
    !python scripts/reset_protocol_results.py --protocol probcbm --apply

Jangan jalankan --apply saat notebook protokol yang sama masih training.
File yang dihapus dari mount Google Drive masuk Trash; kosongkan Trash agar
kuota benar-benar berkurang.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROTOCOLS = ("probcbm", "hicem")


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--train-log", default="train_log", help="Folder train_log (symlink ke Drive di Colab).")
    parser.add_argument("--apply", action="store_true", help="Benar-benar menghapus; tanpa ini hanya dry-run.")
    args = parser.parse_args()

    train_log = Path(args.train_log)
    if train_log.name != "train_log" or not train_log.is_dir():
        raise SystemExit(f"--train-log harus menunjuk folder train_log yang ada: {train_log}")
    target = train_log / args.protocol
    if not target.exists():
        print(f"Tidak ada hasil lama: {target}")
        return

    children = sorted(p for p in target.iterdir())
    total = 0
    print("MODE:", "APPLY (menghapus)" if args.apply else "DRY-RUN (tidak ada yang dihapus)")
    for child in children:
        size = directory_size(child) if child.is_dir() else child.stat().st_size
        total += size
        print(f"{human(size):>12}  {child}")
    print(f"Total: {human(total)} dalam {target.resolve()}")

    if not args.apply:
        print("Tinjau daftar di atas. Jika benar, ulangi dengan --apply.")
        return
    shutil.rmtree(target)
    print(f"Dihapus: {target}. Kosongkan Trash Google Drive agar kuota berkurang.")


if __name__ == "__main__":
    main()
