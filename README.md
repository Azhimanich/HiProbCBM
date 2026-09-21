# HiProbCBM

**Hierarchical Probabilistic Concept Bottleneck Model** melalui Agregasi
Subkonsep Berbasis Learned Attention.

Implementasi ini mengikuti Usulan Penelitian S2 *"Model Concept Bottleneck
Hierarkis Probabilistik melalui Agregasi Subkonsep Berbasis Learned
Attention"* (Mohd Azhima, Program Magister Kecerdasan Artifisial, DTETI
UGM, 2026), yang mengintegrasikan:

- **Representasi konsep probabilistik** dari *Probabilistic Concept
  Bottleneck Model* / ProbCBM (Kim dkk., ICML 2023).
- **Automatic hierarchical subconcept discovery** dari *Hierarchical
  Concept Embedding Model* / HiCEM (Hill, Espinosa Zarlenga & Jamnik,
  ICLR 2026).

Novelty utamanya: mekanisme **hierarchical concept aggregation berbasis
learned attention** (bukan bobot mixture dari probabilitas subkonsep
seperti HiCEM) yang mempelajari kontribusi relatif setiap subkonsep
terhadap konsep induknya, sekaligus **mempropagasikan ketidakpastian**
(variance) melalui hierarki tersebut - sesuatu yang tidak dimiliki ProbCBM
(flat) maupun HiCEM (deterministik).

## Status implementasi

Modul inti dan orkestrasi training diuji dengan data sintetis di CPU,
termasuk tes putus-sambung checkpoint. Tes ini tidak membuktikan kesetaraan
ilmiah terhadap metode asli atau konvergensi pada data pengguna.
HiProbCBM adalah model yang diimplementasikan dan dilatih di repository ini.
ProbCBM dan HiCEM pembanding utama dijalankan dari repository referensi
terpisah tanpa mengubah kode upstream; modul baseline internal bukan pengganti
hasil source tersebut. Protokol evaluasi intervensi dan discovery held-out
belum lengkap.

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Pengguna sudah menjalankan persiapan dataset dan training CUB di Colab;
penyelesaian seluruh matriks eksperimen belum diverifikasi. Lihat
[audit head-to-head dan novelty](docs/head_to_head_audit_2026-09-20.md)
untuk status pembanding, serta [panduan resume](docs/resume_training.md)
sebelum menjalankan sel 15/16/17. Checkpoint lengkap tersimpan per epoch;
bobot dari kode lama tidak otomatis menjadi checkpoint resume penuh.

## Peta subbab proposal -> kode

| Subbab proposal | Isi | File |
|---|---|---|
| III.3.2 | Concept Bottleneck Model (baseline) | `hiprobcbm/models/baselines/cbm.py` |
| III.2.1 (Tabel 2.1) | Concept Embedding Model (baseline) | `hiprobcbm/models/baselines/cem.py` |
| III.3.3 | Probabilistic Concept Bottleneck Model (baseline) | `hiprobcbm/models/baselines/probcbm.py` |
| III.3.4 | Hierarchical Concept Embedding Model (baseline) | `hiprobcbm/models/baselines/hicem.py` |
| III.3.6, IV.5.1.2 | Representasi konsep probabilistik (mu, sigma^2), reparameterization, KL | `hiprobcbm/models/probabilistic_concepts.py` |
| III.3.7-3.8, IV.5.1.3-4.5.1.4 | Sparse Autoencoder (KL-sparsity & BatchTopK) + pembentukan pseudo subconcept label | `hiprobcbm/models/sae.py`, `hiprobcbm/models/subconcept_discovery.py` |
| IV.5.2.1-4.5.2.4 | Learned attention + agregasi mean & propagasi variance | `hiprobcbm/models/attention_aggregation.py` |
| IV.5.2.5 | Hierarchical probabilistic reasoning (MC sampling + classifier) | `hiprobcbm/models/hiprobcbm.py` (`HiProbCBMStage2.forward`) |
| IV.5.4 | Concept intervention (subkonsep & konsep induk) | `hiprobcbm/intervention.py` |
| IV.6 | Fungsi objektif setiap model | `hiprobcbm/losses.py` |
| IV.7.1 | Task Accuracy, F1, Concept Accuracy, ROC-AUC, ECE, Intervention Performance | `hiprobcbm/metrics.py` |
| IV.7.3, Tabel 4.7-4.8 | Trainer baseline & studi ablasi | `hiprobcbm/engine/`, `scripts/run_baseline.py`, `scripts/run_ablation.py` |

## Keputusan desain eksperimen

Empat keputusan berikut (hasil diskusi sebelum implementasi) sudah
tertanam di kode - lihat juga lembar rujukan [*Peta Formulasi
HiProbCBM*](#) yang dibuat sebagai pendamping Bab III/IV:

1. **Classifier head terstandardisasi.** Seluruh model (CBM, CEM, ProbCBM,
   HiCEM, HiProbCBM) memakai `LinearSoftmaxClassifier`
   (`hiprobcbm/models/classifier.py`) pada protokol adaptasi bersama.
   Ini mengontrol head, tetapi tidak membuktikan seluruh selisih hasil hanya
   berasal dari representasi konsep. `AnchorClassifier` tersedia sebagai
   ablasi head pada adaptasi ProbCBM, bukan replikasi penuh paper - lihat
   `configs/baselines/probcbm_cub_anchor_sanity_check.yaml`.
2. **Dua varian Sparse Autoencoder** - KL-sparsity (default) dan
   BatchTopK - keduanya diimplementasikan di `hiprobcbm/models/sae.py` dan
   bisa dipilih lewat `sae.variant` di config (`kl` / `batchtopk`), sesuai
   ablasi HiProbCBM-A3 pada Tabel 4.8.
3. **Tanpa cabang positif/negatif ala HiCEM.** HiProbCBM tetap memakai
   satu set subkonsep per konsep induk (bukan `ĉ+`/`ĉ-` seperti CEM/HiCEM)
   - konsisten dengan fondasi ProbCBM yang tidak mengenal pasangan
   embedding aktif/tidak-aktif.
4. **Dual backbone untuk CUB.** `inception_v3` (konvensi CBM/CEM asli) dan
   `resnet18` (backbone utama ProbCBM) sama-sama tersedia lewat
   `model.backbone` di config - lihat `configs/cub_stage1.yaml` vs
   `configs/cub_stage1_resnet18.yaml`.

## Struktur repository

```
hiprobcbm/
  config.py                  # loader YAML + override CLI
  data/                       # CUB-200-2011 & PseudoKitchens
  models/
    backbones.py             # Inception-v3 / ResNet18 / CLIP ViT-L/14
    probabilistic_concepts.py
    sae.py                   # KLSparseAutoencoder & BatchTopKSAE
    subconcept_discovery.py
    attention_aggregation.py # << novelty utama HiProbCBM
    classifier.py
    hiprobcbm.py             # HiProbCBMStage1 & HiProbCBMStage2
    baselines/                # CBM, CEM, ProbCBM, HiCEM
  losses.py
  metrics.py
  intervention.py
  engine/                     # loop training & evaluasi
scripts/                       # entrypoint CLI
configs/                       # konfigurasi eksperimen (YAML)
tests/                         # 34 smoke test (lihat "Status implementasi")
```

## Instalasi

```bash
cd HiProbCBM
pip install -e .

# Opsional: backbone CLIP ViT-L/14 (dipakai untuk PseudoKitchens & HiCEM)
pip install -e ".[clip]"

# Opsional: logging W&B
pip install -e ".[logging]"

# Untuk menjalankan test suite
pip install -e ".[dev]"
```

Diuji dengan Python 3.10+, PyTorch 2.5 (CUDA 12.1) - sejalan dengan
Bab IV.3 (Alat & Bahan): GPU kelas workstation tunggal (mis. NVIDIA GTX
1660 Ti 6GB), cukup untuk backbone ResNet18/Inception-v3; CLIP ViT-L/14
dijalankan *frozen* (tanpa fine-tuning) sehingga tidak butuh VRAM training
tambahan pada backbone-nya.

## Menyiapkan dataset

### CUB-200-2011

Ikuti prosedur pembersihan atribut standar (Koh dkk., 2020;
lihat juga README `ejkim47/prob-cbm`), lalu letakkan:

```
datasets/CUB_200_2011/
  images/<kelas>/<nama_file>.jpg
  metadata/train.pkl   # list[dict] {img_path, class_label, attribute_label, attribute_certainty}
  metadata/val.pkl
  metadata/test.pkl
  metadata/attributes.txt   # opsional, untuk nama & pengelompokan 112 konsep
```

### PseudoKitchens

Dataset asli (Hill dkk., 2026) disebar dalam format render 3D mentah.
Unduh lewat tautan pada README `OscarPi/cem-concept-discovery`, lalu
ratakan formatnya:

```bash
pip install OpenEXR  # dependensi opsional, hanya untuk konversi ini
python scripts/prepare_pseudokitchens_manifest.py \
    --raw-dir /path/ke/pseudokitchens_V2 \
    --out-dir datasets/PseudoKitchens
```

## Menjalankan eksperimen

```bash
# Tahap 1: probabilistic concept predictor + automatic subconcept discovery
python scripts/run_stage1.py --config configs/cub_stage1.yaml

# Tahap 2: hierarchical aggregation + hierarchical probabilistic reasoning
python scripts/run_stage2.py --config configs/cub_stage2.yaml \
    --stage1-log-dir train_log/cub/cub_stage1_inception

# Baseline pembanding (Tabel 4.7)
python scripts/run_baseline.py --config configs/baselines/probcbm_cub.yaml
python scripts/run_baseline.py --config configs/baselines/hicem_kitchens.yaml

# Studi ablasi (Tabel 4.8)
python scripts/run_ablation.py --variant a1 --config configs/cub_stage2.yaml \
    --stage1-log-dir train_log/cub/cub_stage1_inception
python scripts/run_ablation.py --variant a2 --config configs/cub_stage2.yaml \
    --stage1-log-dir train_log/cub/cub_stage1_inception
# a3 (varian SAE) dijalankan lewat run_stage1.py dua kali dengan sae.variant
# berbeda (kl vs batchtopk), lihat configs/cub_stage1_batchtopk_sae.yaml
```

Setiap eksperimen sebaiknya dijalankan dengan tiga variasi seed (`--seed`)
mengikuti Bab IV.7.3, lalu hasilnya dirata-rata beserta standar deviasi.

## Kredit & referensi

Repository ini adalah tulisan ulang mandiri (bukan salinan langsung) yang
mempelajari arsitektur dari dua repo baseline sebagai rujukan, sesuai izin
lisensi MIT masing-masing:

- Kim, E., Jung, D., Park, S., Kim, S., & Yoon, S. (2023). *Probabilistic
  Concept Bottleneck Models.* ICML 2023.
  [`github.com/ejkim47/prob-cbm`](https://github.com/ejkim47/prob-cbm)
- Hill, O., Espinosa Zarlenga, M. E., & Jamnik, M. (2026). *Hierarchical
  Concept-based Interpretable Models.* ICLR 2026.
  [`github.com/OscarPi/cem-concept-discovery`](https://github.com/OscarPi/cem-concept-discovery)
- Koh, P. W., dkk. (2020). *Concept Bottleneck Models.* ICML 2020.
- Espinosa Zarlenga, M. E., dkk. (2022). *Concept Embedding Models.*
  NeurIPS 2022.
- Bussmann, B., Leask, P., & Nanda, N. (2024). *BatchTopK Sparse
  Autoencoders.*
- Cunningham, H., dkk. (2023). *Sparse Autoencoders Find Highly
  Interpretable Features in Language Models.*
