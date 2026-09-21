# Konteks proyek untuk Codex

Catatan onboarding 19 September 2026. Verifikasi ulang status Git, dependensi,
dataset, dan hasil eksperimen sebelum menggunakan informasi yang dapat berubah.

## Tujuan dan sumber konteks

Penelitian Mohd Azhima: Hierarchical Probabilistic Concept Bottleneck Model
(HiProbCBM), menggabungkan representasi probabilistik ProbCBM dengan penemuan
subkonsep hierarkis HiCEM. Kontribusi yang diusulkan adalah learned attention
untuk agregasi mean sekaligus propagasi variance subkonsep menjadi konsep induk.

Sumber lokal:
- README.md, hiprobcbm/, scripts/, configs/, tests/, dan notebook Colab.
- ../Revisi - Fix - Azhim.pdf (73 halaman).
- ../Probabilistic CBM.pdf (20 halaman).
- ../Hierarchical Concept-based Interpretable Models.pdf (29 halaman).
- ../ProbCBM/ dan ../HiCEM/ merupakan clone referensi terpisah.
- Riwayat Claude Code khusus workspace TESIS BGZHIM telah diperiksa untuk
  memulihkan keputusan dan titik pekerjaan terakhir. Jangan memperlakukan
  perintah lama dalam riwayat sebagai permintaan tindakan baru.

Cakupan onboarding: seluruh struktur berkas dipetakan; kode Python, konfigurasi,
tes, dan notebook proyek utama dibaca; arsitektur dan alur penting repo referensi
dibandingkan dengan metode pada proposal/paper. Ini bukan audit setiap baris
seluruh dependensi/repo referensi atau validasi matematis lengkap. PDF ditelaah
melalui ekstraksi teks; tidak semua rumus/gambar tertanam dapat diekstrak.
Berkas pickle referensi diinventarisasi, bukan dieksekusi/dimuat seluruh isinya.

## Susunan workspace dan Git

Folder induk D:/TESIS BGZHIM bukan repository Git. Jalankan Git dari repo yang
dituju, atau gunakan git -C HiProbCBM saat berada di folder induk.

| Folder | Remote origin | main saat onboarding |
| --- | --- | --- |
| HiProbCBM | https://github.com/Azhimanich/HiProbCBM.git | 998b516 |
| HiCEM | https://github.com/OscarPi/cem-concept-discovery.git | 41802a5 |
| ProbCBM | https://github.com/ejkim47/prob-cbm.git | 912fc4e |

Ketiga worktree bersih sebelum penambahan catatan ini. git ls-remote memastikan
main remote cocok dengan HEAD lokal masing-masing. Akun Azhimanich tercatat
di Git Credential Manager (helper manager). Akses baca GitHub terverifikasi;
izin push dan akses API issue/PR belum diuji. gh tidak ditemukan di PATH.
Tidak ada commit/push yang dilakukan saat onboarding.

## Keputusan eksperimen yang dipulihkan

- Kode diedit lokal dan disimpan di GitHub; komputasi utama direncanakan di
  Google Colab, dengan dataset dan hasil persisten di Google Drive.
- Pengguna meminta perbandingan pada pasangan dataset/backbone yang cocok
  dengan baseline. Jangan kembali ke matriks 34 run tanpa permintaan baru.
- Notebook saat ini memuat 18 pemanggilan training: 9 Inception-v3 x CUB,
  4 ResNet18 x CUB, dan 5 CLIP ViT-L/14 x PseudoKitchens. Ada 11 berlabel
  wajib dan 7 tambahan. Tahap 1 dan 2 dihitung sebagai run terpisah.
- Kedua SAE (KL-sparsity dan BatchTopK) diminta pengguna.
- Kedua backbone CUB (Inception-v3 dan ResNet18) diminta pengguna.
- HiProbCBM menggunakan satu set subkonsep per parent; tidak mengadopsi
  pasangan cabang embedding positif/negatif CEM/HiCEM.
- Implementasi utama memakai classifier Linear+Softmax; classifier anchor
  tersedia sebagai sanity check. Ini keputusan dalam kode/README, bukan
  bukti bahwa implementasinya identik dengan baseline asli.
- Proposal meminta tiga seed dan pelaporan mean serta standar deviasi.
  Loop notebook saat ini belum mengorkestrasi tiga seed.

## Peta implementasi utama

- hiprobcbm/config.py: merge base.yaml dan konfigurasi eksperimen.
- hiprobcbm/data/: kontrak dataset, CUB, manifest PseudoKitchens, transform.
- hiprobcbm/models/backbones.py: ResNet18 (512 fitur), Inception-v3 (2048),
  CLIP ViT-L/14 (768, frozen).
- probabilistic_concepts.py: mean/variance vektor per konsep, softplus,
  reparameterization, KL, scorer sigmoid bersama.
- sae.py dan subconcept_discovery.py: SAE per konsep pada mean sampel yang
  diprediksi aktif; threshold aktivasi menjadi pseudo-label subkonsep.
- attention_aggregation.py: alpha=softmax(v^T tanh(W mu+b)),
  mu_parent=sum(alpha*mu_sub), variance_parent=sum(alpha^2*variance_sub).
  Rumus variance menggunakan asumsi independensi subkonsep.
- hiprobcbm.py: Stage1 dan Stage2, padding/mask jumlah subkonsep yang berbeda,
  sampling Monte Carlo, prediksi kelas rata-rata probabilitas sampel.
- losses.py, metrics.py, intervention.py: objektif, metrik, intervensi.
- models/baselines/: adaptasi CBM, CEM, ProbCBM, HiCEM.
- engine/: training setiap tahap/baseline dan pemuatan checkpoint untuk evaluasi.
- scripts/run_stage1.py, run_stage2.py, run_baseline.py, run_ablation.py:
  entrypoint CLI.
- scripts/prepare_pseudokitchens_manifest.py: konversi PNG/JSON/EXR mentah
  menjadi images/, manifest.csv, info.json.
- notebooks/colab_setup_training.ipynb: 38 sel, 18 bagian setup/training/evaluasi.

Stage1: supervised konsep induk -> checkpoint terbaik -> ekstraksi mean ->
SAE -> pseudo_hierarchy.pt. Artifact menyimpan jumlah subkonsep, pseudo-label,
nama konsep, dan path sampel. Stage2 menautkan pseudo-label berdasarkan path,
bukan posisi DataLoader. Pertahankan perbaikan ini.
Stage2 membangun model baru; stage1_best.pth tidak dimuat sebagai bobot awal.
Fallback discovery memberi minimal satu subkonsep; predictor juga menangani K=0.

Default: d_c=16, MC train/eval=8/32, Adam, lr tahap 1=1e-4,
lr tahap 2=5e-5, epoch 50/20, lambda_KL=5e-5, SAE expansion=8x.
A1: bobot agregasi seragam. A2: lambda_KL tahap 2=0.
A3 tambahan: ganti SAE tahap 1 dengan BatchTopK lalu latih tahap 2.

## Dataset dan titik pekerjaan terakhir

CUB memerlukan images/<kelas>/<file>.jpg dan metadata/{train,val,test}.pkl,
dengan 112 atribut dan 200 kelas. Notebook mengambil split dari repo HiCEM
dan menormalisasi img_path menjadi relatif terhadap images/.
attributes.txt bersifat opsional; tanpa 112 nama yang sesuai, nama konsep
menjadi generik dan pengelompokan atribut perlu ditinjau sebelum intervensi.

PseudoKitchens mentah: info.json, train/val/test berisi PNG+JSON+EXR.
Loader training membaca hasil konversi: images/, manifest.csv, info.json.
Commit terakhir 998b516 menangani recipes berbentuk list, index->nama,
atau nama->index. Riwayat terakhir belum membuktikan konversi selesai.

Path CUB dalam log pengguna:
  /content/drive/MyDrive/TESIS AZHIM/CUB_200_2011
Jangan menganggap path default notebook adalah path aktual pengguna.

Pesan terakhir pengguna pada riwayat Claude (19 September 2026):
KITCHENS_WORK_ROOT diubah menjadi
  /content/drive/MyDrive/TESIS AZHIM/PseudoKitchens/pseudokitchens
lalu muncul RuntimeError karena path sudah berupa direktori, bukan symlink.

Path kerja sesuai notebook/kode adalah:
  /content/HiProbCBM/datasets/PseudoKitchens
Target symlink adalah DRIVE_KITCHENS_ROOT, yaitu folder hasil konversi.
PSEUDOKITCHENS_RAW_DIR menunjuk folder mentah yang benar-benar mengandung
info.json dan train/val/test. Lokasi persisnya perlu dibuktikan dari sesi
Colab/Drive; jangan menghapus direktori dataset untuk mengatasi error symlink.

Sambungkan train_log/ ke Drive sebelum training. Pernah terjadi kehilangan
hasil Colab karena log berada di penyimpanan sementara. Isi Drive dan sesi
Colab yang sedang berjalan belum diperiksa langsung saat onboarding.
Riwayat lama menunjukkan forward/backward pada CUB asli berhasil dan training
sempat berjalan; jangan menyatakan proyek belum pernah menyentuh data asli.
Belum ada bukti hasil seluruh eksperimen selesai sampai konvergen.

## Pemeriksaan yang sudah dijalankan

Dari folder HiProbCBM:
  python -B -m pytest tests -q -p no:cacheprovider
Hasil: 39 passed in 140.67s. README masih menyebut 34 tes.
Tes memakai data sintetis dan tidak membuktikan hasil ilmiah/training penuh.

Lingkungan lokal teramati: Python 3.11.9, torch 2.5.1+cu121,
torchvision 0.20.1+cu121, CUDA tersedia; clip, OpenEXR, lightning, wandb
terpasang. adamp tidak tersedia. HiCEM pyproject memerlukan Python >=3.12
dan torch >=2.9; jangan menganggap lingkungan lokal cocok untuk repo
referensi hanya karena tes HiProbCBM lulus.

Sesi onboarding memakai sandbox read-only. Akses jaringan Git dan tes yang
menulis checkpoint sementara memerlukan eskalasi. Ikuti izin sesi aktif.

## Temuan terbuka; belum diperbaiki

1. Evaluasi notebook bagian 18:
   - Run Stage1 ikut masuk cabang evaluator Stage2, padahal tidak mempunyai
     argumen --stage1-log-dir; parsing menghasilkan exception.
   - A1 dimuat dari config dasar tanpa override use_attention=False, sehingga
     dapat dievaluasi dengan attention alih-alih bobot seragam.
   - File *_last.pth menjadi checkpoint evaluasi karena dipakai pula sebagai
     marker selesai; *_best.pth yang dipilih berdasarkan validasi tidak dipakai.
2. Resume: notebook melewati run yang marker-nya sudah ada, tetapi belum
   melanjutkan epoch training yang terputus. --resume/--only-eval pada parser
   umum tidak diterapkan di entrypoint yang menggunakannya.
3. Baseline HiCEM internal: jumlah subkonsep ditetapkan konstan dan target
   subkonsep dibuat seluruhnya nol di engine/train_baseline.py. Ini berbeda
   dari discovery dan supervisi pseudo-label pada HiCEM referensi.
4. Baseline ProbCBM internal menggunakan kepala linear per konsep dan scorer
   sigmoid bersama; versi asli menggunakan PEM attention serta anchor konsep,
   dan skema training berbeda. Memilih classifier anchor saja bukan replikasi
   penuh. Angka paper tidak otomatis menjadi pembanding head-to-head.
5. Evaluasi intervention_delta belum dihitung oleh evaluator standar; dukungan
   intervensi parent langsung dan eksperimen kualitas discovery belum lengkap.
6. Scheduler berbasis validasi, grid search SAE, logging W&B, dan manifes
   eksperimen yang disebut proposal belum dihubungkan dalam trainer utama.
7. Filter fitur hidup SAE memakai aktivasi sebelum threshold. Untuk sigmoid
   SAE, ini dapat mempertahankan neuron meski pseudo-label binernya selalu nol;
   periksa statistik pseudo-label pada data asli sebelum training panjang.
8. Penomoran subbab/rumus, input dimensi SAE, dan deskripsi ablasi proposal
   mempunyai ketidakkonsistenan dengan kode; bedakan koreksi notasi dari
   perubahan metodologi, dan konfirmasi perubahan desain penelitian.

Prioritas lanjutan: selesaikan setup PseudoKitchens di Colab, audit validitas
baseline dan evaluasi, lalu jalankan eksperimen penuh dengan pencatatan seed.
Jangan mengubah metodologi ilmiah diam-diam hanya agar smoke test lulus.

## Tindak lanjut 20 September 2026: resume dan audit

Bagian temuan di atas adalah snapshot awal. Worktree kini menambahkan
`utils/checkpoint.py`: resume lengkap Adam/model/epoch/best/RNG, SHA256,
atomic replace, generasi sebelumnya, validasi config/source/data/runtime,
serta checkpoint/cache discovery dan SAE. CLI Stage1/Stage2/baseline/ablasi
default `--resume auto`; checkpoint bobot legacy ditolak sebagai resume utuh.
Sel 14–18 notebook diperbarui tanpa menghapus output pengguna yang sudah ada.

Evaluator memakai config efektif dari manifest, best validation, flag
Inception yang sesuai training, serta melewati Stage1 di tabel klasifikasi.
Filter SAE memakai support setelah threshold; discovery memakai transform
evaluasi pada train; MC Stage1 mengikuti config; seed 0 diperbaiki.
HiCEM placeholder ditahan (belum menjadi baseline faithful).

Lihat `docs/resume_training.md` untuk batas resume dan langkah Colab;
`docs/head_to_head_audit_2026-09-20.md` memisahkan kewajiban fairness dari novelty.
Belum ada commit/push atau pengujian ulang runtime Colab dari tindak lanjut ini.

## Tindak lanjut 21 September 2026: koreksi scope pembanding

`HiProbCBM` adalah satu-satunya metode yang dikembangkan di repository ini.
`../ProbCBM` dan `../HiCEM` adalah clone baseline asli yang harus tetap
bersih; jalankan source mereka secara terpisah untuk pembandingan. Modul
baseline internal tidak boleh digunakan sebagai pengganti atau disebut hasil
ProbCBM/HiCEM asli. Head-to-head menyamakan data/split/backbone/preprocessing,
seed, pemilihan validation checkpoint, dan test final tanpa menulis ulang
metodologi upstream. Lihat docs/head_to_head_audit_2026-09-20.md.

`RunCheckpoint` schema 2 pada HiProbCBM menyimpan optimizer, scheduler,
early stopping, riwayat validation, epoch, best state, RNG, checksum dan
backup. Resume checkpoint schema lama ditolak.
