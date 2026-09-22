# Membersihkan penyimpanan eksperimen selesai

Gunakan skrip ini hanya setelah `probcbm/utama` selesai untuk seed 42, 43,
dan 44. A1/A2 boleh belum dijalankan: skrip mempertahankan
`pseudo_hierarchy.pt` beserta checksum-nya sebagai input kedua ablasi itu.

Di notebook Colab yang sudah mount Drive dan membuat symlink `train_log`:

```bash
%cd /content/HiProbCBM
!python scripts/cleanup_completed_probcbm_utama.py
```

Perintah pertama adalah **dry-run**. Tinjau path dan ukuran yang dicetak. Jika
seluruhnya berada di `train_log/probcbm/utama/seed_42`, `seed_43`, atau
`seed_44` dan run memang selesai, baru jalankan:

```bash
!python scripts/cleanup_completed_probcbm_utama.py --apply
```

Syarat penghapusan per seed: tepat satu `pseudo_hierarchy.pt` dan checksum-nya,
serta `stage2_last.pth` dan `stage2_best.pth`. Jika syarat gagal atau struktur
folder ambigu, seed dilewati tanpa perubahan. Skrip hanya menghapus state
resume, ekspor `last`, dan artifact discovery/SAE yang tidak dibutuhkan A1/A2.
Ia tidak menyentuh best checkpoint, hierarchy, manifest, CSV, maupun log.

## Audit seluruh My Drive tanpa menghapus

Gunakan skrip read-only berikut untuk menemukan folder dan file terbesar.
Ia hanya membaca metadata ukuran; Drive besar dapat memerlukan beberapa menit.

```bash
!python scripts/audit_drive_storage.py --root /content/drive/MyDrive --top 100
```

Untuk menyimpan daftar file besar sebagai CSV di Drive:

```bash
!python scripts/audit_drive_storage.py --root /content/drive/MyDrive --top 200 \
  --csv "/content/drive/MyDrive/TESIS AZHIM/audit_drive_storage.csv"
```

Audit ini mencakup file di My Drive. Kuota Google One juga dapat mencakup
Google Photos, Gmail, dan backup perangkat; ketiganya perlu diperiksa dari
halaman Storage Google bila total hasil audit lebih kecil daripada kuota.
