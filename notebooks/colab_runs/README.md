# Notebook Colab per skenario

Jalankan satu notebook sampai selesai, lalu pindah ke notebook berikutnya. Semua checkpoint memakai `train_log/` di Google Drive dan aman untuk resume.

| Urutan | Notebook | Prasyarat |
| --- | --- | --- |
| 1 | `01_probcbm_utama.ipynb` | CUB di Drive |
| 2 | `02_probcbm_a1.ipynb` | ProbCBM utama selesai tiga seed |
| 3 | `03_probcbm_a2.ipynb` | ProbCBM utama selesai tiga seed |
| 4 | `04_probcbm_a3.ipynb` | CUB di Drive |
| 5 | `05_hicem_utama.ipynb` | PseudoKitchens processed di Drive |
| 6 | `06_hicem_a1.ipynb` | HiCEM utama selesai tiga seed |
| 7 | `07_hicem_a2.ipynb` | HiCEM utama selesai tiga seed |
| 8 | `08_hicem_a3.ipynb` | PseudoKitchens processed di Drive |

A1 dan A2 hanya melatih Tahap 2; keduanya memverifikasi lalu memakai `pseudo_hierarchy.pt` dari notebook utama. A3 menjalankan Tahap 1 dan Tahap 2 karena hierarchy BatchTopK berbeda.

Untuk kelompok HiCEM/PseudoKitchens, pemanggilan pertama membuat cache fitur
CLIP ViT-L/14 untuk split train dan validation di
`PseudoKitchens/representation_cache/clip_vitl14/`. Cache memakai preprocessing
resmi CLIP, checksum SHA-256, dan menjadi bagian dari identitas checkpoint.
Training berikutnya (termasuk A1/A2/A3) memakai tensor tersebut, sehingga tidak
menjalankan encoder CLIP atau membaca gambar Drive di setiap epoch. Cache test
baru dibuat saat evaluasi final.
