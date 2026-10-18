# Protokol pembandingan dengan hasil penelitian sebelumnya

Dokumen ini mengatur pembandingan **HiProbCBM** dengan angka yang dilaporkan
oleh ProbCBM dan HiCEM. Kedua baseline tidak dijalankan ulang dalam eksperimen
tesis ini. Karena GPU, random seed, dan implementasi proses training tidak
sama, istilah yang dipakai pada naskah adalah **perbandingan terhadap hasil
yang dilaporkan**, bukan eksperimen head-to-head literal.

## Matriks final

| ID | HiProbCBM yang dilatih | Hasil pembanding | Dataset dan backbone yang dikunci | Status klaim |
| --- | --- | --- | --- | --- |
| P | Stage 1 + Stage 2 `probcbm_reported` | ProbCBM (Kim dkk., 2023) | CUB-200-2011, split `class_attr_data_10`, **ResNet18 pretrained**, 224 px, 112 konsep, 200 kelas | Comparable terhadap hasil yang dilaporkan ProbCBM |
| H | Stage 1 + Stage 2 `hicem_reported` | HiCEM (Hill dkk., 2026) | PseudoKitchens **V2**, split official, **CLIP ViT-L/14 frozen**, 224 px, urutan parent dan recipe class official | Comparable terhadap hasil yang dilaporkan HiCEM |
| A1-A3 | Variasi HiProbCBM pada setiap baris P dan H | Model HiProbCBM utama pada kontrak P atau H yang sama | Dataset, backbone, split, seed, dan budget mengikuti baris P atau H; hanya komponen ablasi yang berubah | Ablasi internal untuk menguji kontribusi HiProbCBM, bukan angka pembanding paper |

Eksperimen CUB/Inception-v3 lama tetap dapat dipakai sebagai pilot atau studi
internal, tetapi tidak ditempatkan dalam tabel pembandingan terhadap ProbCBM
atau HiCEM.

HiCEM juga menyediakan konfigurasi CUB/CLIP di source-nya. Itu tidak masuk
tabel utama sampai tabel paper yang dikutip dan kontrak konsep/splitnya telah
diverifikasi satu-per-satu. Jangan memasangkannya dengan CUB/Inception atau
CUB/ResNet18.

## Ablasi HiProbCBM yang dijalankan pada notebook

- **A1:** mengganti learned attention dengan bobot agregasi seragam pada
  Tahap 2. Tahap 1 dan `pseudo_hierarchy.pt` dipakai bersama dengan model utama.
- **A2:** menetapkan `lambda_kl=0` pada Tahap 2. Tahap 1 dan hierarchy tetap
  sama dengan model utama.
- **A3:** mengganti SAE KL-sparsity pada Tahap 1 dengan BatchTopK SAE, lalu
  melatih Tahap 2 dari hierarchy A3 itu sendiri.

Setiap A1-A3 harus dijalankan tiga seed pada pasangan P dan/atau H yang ingin
ditulis dalam tabel ablation. Mereka menjawab kontribusi komponen HiProbCBM;
mereka bukan syarat untuk menyamakan implementasi ProbCBM atau HiCEM.

## Aturan yang wajib dibuktikan sebelum hasil dicantumkan

### P — ProbCBM

- Hash dan jumlah contoh `train/val/test` sama dengan split
  `class_attr_data_10` yang dipakai source ProbCBM.
- CUB memakai 112 atribut dalam urutan yang sama serta 200 class labels.
- HiProbCBM memakai ResNet18 ImageNet-pretrained dan input 224 px.
- Tidak ada angka dari CUB/Inception yang ditempatkan berdampingan dengan
  tabel ProbCBM.
- Source ProbCBM memakai batch 128, AdamP, cosine schedule, 50 epoch konsep
  dan 20 epoch classifier, serta MC 50/50. Perbedaan optimizer, batch,
  early stopping, atau jumlah MC HiProbCBM dicatat sebagai perbedaan metode,
  bukan disembunyikan sebagai kondisi identik.

### H — HiCEM

- Data berasal dari `pseudokitchens_V2` official, bukan varian dataset lain.
- `info.json`, daftar recipe, split, urutan parent concept dan label kelas
  harus cocok dengan loader HiCEM original.
- HiProbCBM memakai CLIP ViT-L/14 frozen dan preprocessing CLIP 224 px.
- HiCEM paper/source memakai max 300 epoch, early stopping validation loss,
  embedding 16, concept-loss weight 10, dan memecah 3 parent concept.
  HiProbCBM boleh mempertahankan discovery semua parent sebagai novelty, tetapi
  jumlah parent yang di-split serta budget training wajib dilaporkan.

## Metrik yang ditampilkan

| Kelompok | Metrik pembanding utama | Metrik HiProbCBM tambahan |
| --- | --- | --- |
| P | Task accuracy; concept accuracy/ROC-AUC dan intervention hanya bila definisinya identik dengan paper ProbCBM | ECE, uncertainty decomposition, F1, parameter, waktu |
| H | Task accuracy; discovered-subconcept ROC-AUC/matching dan intervention multi-level hanya bila protocol concept bank identik dengan HiCEM | Parent concept accuracy/ROC-AUC, ECE, F1, parameter, waktu |

Setiap tabel menampilkan sumber angka baseline, seed/protocol yang dilaporkan
paper, konfigurasi HiProbCBM, dan tanda `reported baseline`. Mean ± standard
deviation tiga seed hanya berlaku untuk HiProbCBM kecuali paper baseline juga
melaporkan pengulangan yang sama. Jangan menghitung uji signifikansi langsung
terhadap satu angka publikasi.

## Revisi 11 Oktober 2026 — Tahap 1, Tahap 2, dan intervensi

### Mengapa direvisi

Hasil run pertama (tiga seed, test) menunjukkan akurasi konsep induk turun
tajam dari Tahap 1 ke Tahap 2 pada kedua protokol:

| Protokol | Tahap 1 (val) | Tahap 2 lama (test) | Baseline yang dilaporkan |
| --- | --- | --- | --- |
| P, concept accuracy | 0,938 | 0,820 | ProbCBM 224 px: 0,949 (Tabel 1) |
| P, task accuracy | – | 0,589 | ProbCBM 224 px: 0,680 (Tabel 1) |
| H, concept ROC-AUC | 0,890–0,893 | 0,795 | HiCEM: 0,91 (Tabel 4) |
| H, task accuracy | – | 0,671 | HiCEM: 0,65 (Tabel 3) |

Hasil lengkap run lama per skenario (test, mean ± std tiga seed):
P utama 0,5885 ± 0,0050; H utama 0,6713 ± 0,0055; H A1 0,6107 ± 0,0186;
H A2 0,6720 ± 0,0098. Run lama dihapus untuk menghemat ruang Drive.

Pada H, loss konsep induk di data **train** Tahap 2 tertahan di 0,42–0,46,
sedangkan di Tahap 1 turun ke 0,285. Artinya masalahnya underfitting
struktural, bukan overfitting. Penyebabnya adalah Tahap 2 lama tidak punya
prediktor induk sendiri: logit induk sama dengan rata-rata berbobot logit
subkonsep (scorer linear, `sum(alpha) = 1`), padahal pseudo-label subkonsep
sebagian besar 0.

Audit lanjutan menunjukkan Tahap 1 juga belum mengikuti hyperparameter
sumber. Hasil Tahap 1 yang mendekati baseline tidak dianggap cukup sebagai
alasan untuk tidak menyelaraskannya.

### Perubahan dan sumbernya

| Perubahan | Berlaku | Sumber baseline |
| --- | --- | --- |
| Augmentasi dan normalisasi CUB disalin dari ProbCBM (`data.transform: probcbm`): ColorJitter(32/255, sat 0.5), flip, resize, RandomResizedCrop(0.8–1.0), normalisasi 0.5 | P, Tahap 1 dan 2 | `ProbCBM/dataloaders/cub312_datamodule.py` |
| Tahap 1: AdamP, lr backbone 1e-3 dan kepala 1e-2 (`lr_ratio: 10`), backbone beku 5 epoch pertama, grad clipping 2.0; cosine 50 epoch, batch 128, MC 50, KL 5e-5 sudah sama | P | `ProbCBM/main.py`, `run_epoch.py`, `configs/config_exp.yaml` |
| Tahap 1: Adam lr 1e-3, bobot loss konsep 10, bobot imbalance `N/positif − 1`, early stopping 75 epoch | H | Model awal HiCEM: `cemcd/training.py`, `cemcd/models/base.py`, `experiments/configs/kitchens.yaml` |
| `parent_probability: hicem_mix`: induk punya prediktor probabilistik sendiri; probabilitas induk = ½(prediksi langsung + soft-OR subkonsep `sum(softmax(200p-100)·p)`) | P dan H, Tahap 2 | `HiCEM/cemcd/models/hicem.py` |
| Tahap 2: backbone, prediktor konsep, dan scorer dari `stage1_best.pth` dibekukan; kepala baru dilatih AdamP lr 1e-3 × 10, cosine 20 epoch, clip 2.0 | P | Tahap `class` ProbCBM sequential (`main.py`, `config_exp.yaml`) |
| Tahap 2: Adam lr 1e-3; bobot 10 atas BCE gabungan top+sub dengan bobot imbalance; early stopping 75 epoch; dilatih dari awal | H | `HiCEM/cemcd/training.py` (`train_hicem`), `cemcd/models/base.py` |
| Latihan intervensi: anchor positif/negatif yang dipelajari menggantikan representasi konsep pada jalur classifier. P: tiap (sampel, sampel MC) dengan peluang 0,5 memakai anchor ground truth untuk seluruh konsep. H: RandInt Bernoulli(0,25) per langkah atas kolom induk dan subkonsep | P dan H, Tahap 2 | `ProbCBM/models/build_model_resnset.py` (`intervention_prob`), `HiCEM/cemcd/models/hicem.py` (`torch.bernoulli(0.25)`) |
| Evaluasi intervensi: P memakai 28 grup atribut CUB, urutan acak (5 kali) dan urutan uncertainty `exp(mean log σ)`; H memakai konsep induk dalam urutan indeks, kumulatif dan satu per satu. Kurva disimpan di `intervention_curves.json` | Evaluasi | `ProbCBM` `predict_class_with_gt_concepts`, Sec. 5.2.4; `HiCEM/experiments/experiment_utils.py` `get_intervention_accuracies` |
| Nama 112 atribut CUB dari `CONCEPT_SEMANTICS[SELECTED_CONCEPTS]`, sehingga terbentuk 28 grup | Data | `HiCEM/cemcd/data/cub.py`; indeks identik dengan `attr2attrlabel` ProbCBM |
| Metrik `concept_roc_auc_hicem_protocol` (kolom satu kelas memakai accuracy) | H | `HiCEM/cemcd/metrics.py`; diuji identik dengan source |

Concept accuracy HiProbCBM (`>= 0.5`, rata-rata seluruh konsep) sudah sama
dengan `ProbCBM/utils/evaluate.py`, jadi tidak diubah.

### Yang sengaja tidak diubah, dan alasannya

- **Sample filter discovery tetap memakai prediksi Tahap 1 (`p > 0.5`).**
  HiCEM melakukan hal yang sama (`concept_discovery.py`), jadi menggantinya
  dengan label asli justru menyimpang dari sumber.
- **Arsitektur HiProbCBM tetap**: kepala mean/variance linear dan scorer
  sigmoid (bukan PIENet/anchor matching ProbCBM), Tahap 1 hanya melatih
  konsep, satu cabang subkonsep tanpa embedding negatif, dan attention untuk
  agregasi. Semua ini rancangan metode yang diusulkan.
- **Perbedaan yang tetap harus dilaporkan:**
  - HiProbCBM memecah semua parent, sedangkan HiCEM memecah 3.
  - HiCEM melatih hanya subkonsep yang cocok dengan concept bank berlabel;
    HiProbCBM label-free.
  - Pemilihan checkpoint memakai validation (Tahap 1 concept accuracy,
    Tahap 2 task accuracy); ProbCBM memakai epoch terakhir tahap konsep,
    HiCEM memakai model terakhir setelah early stopping val loss.
  - Scheduler H memakai plateau pada metrik validasi; HiCEM memakai
    ReduceLROnPlateau pada train loss.
  - Intervensi subkonsep yang ditemukan (HiCEM: discovered concepts)
    memerlukan matching concept bank dan belum dihitung. Anchor subkonsep
    sudah dilatih, jadi evaluasinya dapat ditambahkan tanpa melatih ulang.

### Dampak pada run

Seluruh Tahap 1 dan Tahap 2 kedua protokol harus diulang. Nama eksperimen
tidak berubah, jadi hasil baru ditulis di lokasi lama. Sebelum menjalankan
ulang suatu protokol, hapus hasil lamanya dengan dry-run lalu `--apply`:

```bash
!python scripts/reset_protocol_results.py --protocol probcbm
!python scripts/reset_protocol_results.py --protocol probcbm --apply
```

Notebook melewati Tahap 1 hanya bila `stage1_complete()` memastikan
`pseudo_hierarchy.pt` dibuat oleh konfigurasi dan seed yang sama, sehingga
hierarchy lama tidak pernah dipakai diam-diam. Cache fitur CLIP tidak
bergantung pada kode HiProbCBM dan tetap dipakai ulang.

## Analisis tambahan: uncertainty dan subkonsep hasil discovery

Dijalankan oleh `notebooks/colab_runs/10_analisis_tambahan.ipynb` memakai
`hiprobcbm/engine/analysis.py`. Semuanya inferensi pada `stage2_best.pth`.

| Analisis | Prosedur | Sumber | Catatan |
| --- | --- | --- | --- |
| Uji cutout | Patch 64 di tengah citra ternormalisasi dinolkan; dilaporkan proporsi sampel yang rata-rata uncertainty konsepnya naik | `ProbCBM/utils/evaluate.py` `run_cutout_test` | Hanya CUB; PseudoKitchens memakai cache fitur CLIP sehingga citra mentah tidak tersedia |
| Uncertainty vs performa | 10 kelompok berukuran sama menurut uncertainty; akurasi per kelompok dan korelasi Spearman | ProbCBM Sec. 5.2.3, Fig. 10 | Kode figur tidak ada di repo ProbCBM; prosedur direkonstruksi dari teks paper |
| Urutan intervensi | Area kurva urutan uncertainty vs rata-rata urutan acak dari `intervention_curves.json` | ProbCBM Fig. 11 | Protokol CUB |
| Subkonsep hasil discovery | Pencocokan ke ingredient di train (ROC-AUC > 0,7, hanya subkonsep milik parent yang sama); rata-rata ROC-AUC test prediksi subkonsep yang cocok; intervensi kumulatif dan satu per satu dengan label test ingredient | `HiCEM/experiments/run.py` `match_discovered_concepts_to_concept_bank`, `get_accuracies`; `experiment_utils.py` `get_intervention_accuracies` | Hanya PseudoKitchens. HiCEM memakai concept bank level ingredient; level sub-sub (`SUB_SUB_CONCEPTS`) tidak dipakai pada matching HiCEM non-deep |

Uncertainty konsep = `exp(mean(log sigma))` (`get_uncertainty` ProbCBM),
dihitung dua kali: dari prediktor induk langsung (padanan ProbCBM) dan dari
variance induk hasil propagasi subkonsep `sum(alpha^2 sigma2_sub)` (klaim
HiProbCBM). Uncertainty kelas memakai entropi prediktif ternormalisasi;
ProbCBM memakai `get_class_uncertainty` yang khusus untuk head anchor, jadi
ini adaptasi. Pada protokol CUB, prediktor induk langsung adalah bobot Tahap 1
yang dibekukan, sehingga uncertainty `own` mencerminkan Tahap 1.

