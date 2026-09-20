# Resume training Colab / VS Code

Implementasi 20 September 2026. Berlaku untuk kode yang sudah diperbarui,
bukan proses Python lama yang masih berjalan. Salin/push perubahan lokal lalu
perbarui checkout `/content/HiProbCBM` sebelum memulai proses training berikutnya.
Jangan menjalankan dua proses untuk direktori eksperimen yang sama. Putus
koneksi VS Code tidak selalu berarti training Colab berhenti; periksa sesi
Colab sebelum memulai proses pengganti.

## Menjalankan kembali

Mount Drive, pulihkan symlink dataset, dan jalankan sel 14 agar `train_log`
menunjuk ke Drive. Kemudian jalankan lagi sel 15, 16, atau loop sel 17.
Ketiganya memakai `--resume auto`. Jika runtime masih hidup, tidak perlu
mengulang persiapan data yang sudah ada.
Paket repo perlu sudah terpasang di runtime (`pip install -e .`, bagian setup
notebook) sebelum memanggil `python scripts/...`.

```bash
python scripts/run_stage1.py --config configs/cub_stage1.yaml --resume auto
python scripts/run_stage2.py --config configs/cub_stage2.yaml --stage1-log-dir train_log/cub/cub_stage1_inception --resume auto
python scripts/run_baseline.py --config configs/baselines/cem_cub.yaml --resume auto
python scripts/run_ablation.py --variant a1 --config configs/cub_stage2.yaml --stage1-log-dir train_log/cub/cub_stage1_inception --resume auto
```

Saat restart, pesan `RESUME VALID` menampilkan jumlah epoch yang sudah selesai,
epoch berikutnya, best validation, dan epoch terbaik. Contoh: `epoch selesai=21/50`
berarti 21 epoch lengkap tersimpan; berikutnya epoch ke-22 (indeks log `epoch=21`).
Saat semua epoch telah selesai, loop training dilewati. Stage 1 masih memeriksa
atau menyelesaikan discovery/SAE sebelum `pseudo_hierarchy.pt` siap.

## Isi checkpoint dan pemulihan

| Artifact | Fungsi |
| --- | --- |
| `stage1_resume.pth`, `stage2_resume.pth`, `{baseline}_resume.pth` | Model saat terakhir disimpan, state Adam, epoch terakhir selesai, best val/epoch, salinan bobot terbaik, RNG Python/NumPy/Torch CPU/seluruh CUDA, konfigurasi dan identitas run |
| `*_resume.prev.pth` | Satu generasi checkpoint valid sebelumnya |
| `*.sha256` | Checksum integritas checkpoint/artifact; ikut disalin jika memindahkan direktori |
| `*_best.pth` | Bobot terbaik menurut validation; dipakai untuk evaluasi atau discovery |
| `*_last.pth` | Bobot setelah semua epoch selesai; bukan checkpoint optimizer |
| `run_manifest.json` | Konfigurasi efektif, seed, hash source Python dan metadata data, versi runtime, serta hash hierarchy untuk Stage 2 |
| `discovery_features.pt` | Fitur mean/probabilitas/path train dan RNG sesudah ekstraksi; terikat ke hash bobot parent |
| `discovery/concept_N/sae_resume.pth` | State training SAE, termasuk optimizer dan RNG |
| `discovery/concept_N/result.pt` | Hasil konsep yang telah selesai; tidak dilatih ulang |
| `pseudo_hierarchy.pt` | Hierarchy selesai; hash tetap dipertahankan saat menjalankan ulang Stage 1 selesai |
| `console.log` | Output training yang ditulis loop sel 17 ke Drive |
| `training.log` | Log trainer CLI sel 15/16/17, ditambahkan ke file yang sama saat resume |

Checkpoint utama ditulis sebelum training pertama dan pada akhir **setiap
epoch lengkap, sesudah validation**. Penyimpanan menggunakan file sementara,
flush/fsync, kemudian replace. Checksum disimpan terpisah: jika penulisan
checkpoint/checksum terputus, loader mencoba generasi sebelumnya dan mencetak
peringatan. Ketidakcocokan config/data/kode tidak dianggap korupsi yang boleh
diam-diam diatasi dengan checkpoint lama; training berhenti.

SAE disimpan setiap 10 epoch dan pada akhir training konsep, supaya ribuan
penulisan kecil ke Drive tidak mendominasi waktu. Ubah
`sae.train_kwargs.checkpoint_interval` **sebelum run baru** jika perlu.
Setelah restart, training utama tidak diulang; SAE melanjutkan checkpoint
konsep aktif dan melewati konsep yang selesai. Jika terputus sebelum cache
fitur selesai ditulis, ekstraksi fitur diulang dari model parent terbaik.

## Validasi sebelum melanjutkan

Loader memeriksa format/field checkpoint, checksum, jenis stage, konfigurasi
efektif (termasuk seed dan ablasi), hash kode Python, hash metadata dataset,
versi Python/Torch/NumPy/Torchvision/Pillow/CUDA/cuDNN, perangkat dan flag backend,
serta hash `pseudo_hierarchy.pt` untuk Stage 2. Bobot dicek key/shape/dtype dan
nilai finite; state Adam dicek struktur/shape dan nilai finite; RNG dicek dengan
generator terpisah sebelum dipulihkan. Epoch dan keberadaan bobot terbaik
juga harus konsisten.

Kebijakan saat ini **ketat**: jangan mengganti GPU, versi runtime, seed, batch,
worker, jumlah epoch, konfigurasi SAE, atau kode di tengah run lalu menganggapnya
resume identik. Jika berbeda, pulihkan setting semula atau mulai eksperimen
baru di direktori berbeda. Perubahan lokasi `--log-dir` diperbolehkan dengan
menyalin **seluruh folder run**, termasuk checkpoint/checksum/cache; konfigurasi
path data dan isi metadata tetap harus cocok. Argumen path eksplisit hanya
menerima file `*_resume.pth` milik direktori run tujuan, bukan best/last.

## Run lama yang hanya menyimpan bobot

`*_best.pth` / `*_last.pth` dari implementasi lama tidak menyimpan optimizer,
epoch, best validation, atau RNG. Semua itu tidak dapat direkonstruksi dengan
benar dari bobot saja. Kode baru menolak menimpa direktori tersebut. Arsipkan
untuk evaluasi/pilot, lalu gunakan lokasi baru, misalnya:

```bash
python scripts/run_stage1.py --config configs/cub_stage1.yaml --log-dir train_log/resume_v1 --resume auto
python scripts/run_stage2.py --config configs/cub_stage2.yaml --log-dir train_log/resume_v1 --stage1-log-dir train_log/resume_v1/cub/cub_stage1_inception --resume auto
```

Di sel 17, gunakan `TL="train_log/resume_v1"` untuk menyamakan lokasi semua
dependency. Tidak ada warm-start otomatis yang menyamarkan reset optimizer
sebagai resume. Untuk mulai baru secara eksplisit, `--resume never` hanya
diizinkan pada direktori tanpa checkpoint.

## Batas jaminan

- Resume adalah **batas epoch**, bukan batch. Pekerjaan dalam epoch yang belum
  tersimpan akan diulang. Cadangan bisa mundur satu penyimpanan lagi bila
  checkpoint terbaru rusak. Ekstraksi fitur belum mempunyai checkpoint batch.
- Penulisan atomik dan checksum mengurangi risiko file parsial, tetapi tidak
  menjamin sinkronisasi server Google Drive selesai saat VM mati. Simpan
  seluruh folder dan pastikan ruang Drive/RAM cukup; state Adam dan salinan
  best/previous lebih besar daripada checkpoint bobot saja.
- Hash dataset mencakup metadata/split/manifest, **bukan seluruh piksel gambar**.
  Jangan mengganti gambar dengan path sama selama eksperimen.
- Tes fault injection CPU membuktikan kesamaan tensor/state untuk kasus yang
  diuji. Ini bukan jaminan bitwise pada CUDA, hardware/versi berbeda, atau
  operasi nondeterministik. Tidak ada klaim bahwa sesi Colab pengguna sudah diuji.
- Trainer sekarang memakai Adam tanpa scheduler/AMP. Bila nanti menambah
  scheduler atau GradScaler, state keduanya juga harus dimasukkan checkpoint
  dan skema diubah sebelum resume dianggap lengkap.
- Baseline HiCEM internal ditahan karena supervisi subkonsep placeholder.
  Dukungan resume baseline yang bisa dilatih berlaku untuk CBM, CEM-adapted,
  dan ProbCBM-adapted; resume tidak memperbaiki validitas ilmiah baseline.

Rujukan: [checkpoint umum PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training)
dan [batas reproduksibilitas PyTorch](https://pytorch.org/docs/stable/notes/randomness.html).

## Hasil verifikasi lokal

- Suite penuh setelah implementasi resume dan regresi audit: **58 passed**.
- Setelah menambah validasi hierarchy Stage 2, tes yang terdampak dijalankan
  kembali: **25 passed**, mencakup satu tes baru (total inventaris 59 tes).
- Fault injection mencakup Stage 1, Stage 2, CBM, SAE, update parsial yang
  dibuang, file rusak, penulisan terputus, perubahan identitas, serta checkpoint
  legacy/optimizer tidak lengkap. Bobot, state Adam, RNG dan best state pada
  tes CPU sama persis antara run tanpa putus dan run yang dilanjutkan.
- CLI keempat trainer menerima `--resume`; sel notebook yang diubah lolos
  pemeriksaan sintaks, `git diff --check` lolos, dan output lama tetap disimpan.
- Multiprocessing DataLoader diuji di luar sandbox Windows karena named pipe
  diblokir dalam sandbox. Tidak ada training dataset asli atau runtime Colab
  yang dijalankan untuk verifikasi ini.
