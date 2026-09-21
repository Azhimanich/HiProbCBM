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
