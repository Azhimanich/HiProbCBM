# Audit kesetaraan eksperimen HiProbCBM — 20–21 September 2026

## Koreksi ruang lingkup baseline — 21 September 2026

Objek penelitian dan implementasi yang dilatih di repository ini adalah
**HiProbCBM**. Baseline pembanding utama adalah repository referensi terpisah
`../ProbCBM` (commit `912fc4e`) dan `../HiCEM` (commit `41802a5`), yang
dipertahankan tanpa perubahan. Kode `hiprobcbm/models/baselines/` bukan
pengganti untuk menjalankan atau mengklaim hasil kedua repository tersebut.

Head-to-head yang benar berarti menjalankan sumber asli masing-masing baseline,
lalu menyamakan hanya hal yang berada di luar metodologi: versi/split dataset,
urutan konsep dan kelas, backbone/weights/freeze state, preprocessing, seed,
aturan validation checkpoint, dan test final. Arsitektur, loss, scheduler,
intervensi, discovery, serta prosedur training milik ProbCBM/HiCEM tidak
distandardisasi atau ditulis ulang ke HiProbCBM. Output setiap repository
dicatat terpisah dan baru diagregasi setelah seluruh seed selesai.

Saat ini notebook `HiProbCBM` menjalankan HiProbCBM beserta ablasinya. Entry
baseline internal yang tidak identik dengan source referensi tetap ditahan;
runner untuk menjalankan dua repository referensi secara terpisah perlu
menetapkan adapter data dan environment tanpa mengubah kode upstream.

Matriks yang dikunci untuk membandingkan dengan **angka yang dilaporkan** dan
empat YAML HiProbCBM terkait ada di
[reported_baseline_protocol.md](reported_baseline_protocol.md). CUB/Inception
tidak lagi menjadi pasangan ProbCBM; pasangan itu adalah CUB/ResNet18.

## Pembaruan implementasi dan klasifikasi keputusan

Audit diperiksa ulang pada 20 September 2026 setelah permintaan resume Colab.
Bagian A–H di bawah menyimpan temuan **snapshot sebelum perbaikan**; status
terbaru pada tabel ini mendahului keterangan historis di bagian tersebut.
Perubahan kode ini belum berarti seluruh baseline sudah menjadi replikasi paper.

| Temuan | Penyesuaian sekarang | Batas yang masih berlaku |
| --- | --- | --- |
| E06: resume | Checkpoint model + Adam + epoch + best val/epoch + bobot best + RNG Python/NumPy/Torch/CUDA, validasi identitas, checksum, atomic replace dan satu backup; sel 15/16/17 memakai resume | Checkpoint bobot lama tidak bisa dipulihkan sebagai resume penuh; batas epoch, bukan batch; sinkronisasi Drive belum diuji langsung |
| Discovery setelah Stage 1 | Cache fitur terikat bobot parent; hasil konsep disimpan; SAE mendapat checkpoint optimizer/RNG tiap 10 epoch dan akhir | Ekstraksi fitur yang belum selesai diulang; artifact rusak ditolak, bukan dianggap selesai |
| A02, E01, E02 | Evaluasi memuat config efektif dari manifest, memakai best validation, melewati Stage 1, membatasi RUN_TAGS; A1 tetap uniform | Checkpoint legacy perlu konfigurasi eksplisit/audit manual |
| A03 | Evaluator mengembalikan flag Inception `transform_input` sesuai pilihan pretrained saat training | Belum mengukur perubahan accuracy checkpoint pengguna |
| A04 | Filter child dilakukan setelah threshold label biner; fitur tanpa contoh positif dibuang dan fallback tetap tersedia | Fallback parent tanpa sampel positif dapat tetap berlabel nol; ini dilaporkan sebagai tidak ada bukti discovery, bukan child baru yang berhasil ditemukan |
| Validasi sebelum Stage 2 | Memeriksa checksum artifact baru, shape, label biner, padding, path unik, jumlah parent dan support positif child; artifact legacy dengan child mati ditolak | Identitas piksel gambar belum di-hash; fallback tunggal tanpa support diperingatkan, bukan disebut keberhasilan discovery |
| B09 | Forward train/validation Stage 1 membaca MC train/eval dari YAML | Default 8/32 tetap adaptasi; bukan 50/50 paper ProbCBM |
| C01 | Ablasi anchor ResNet18 memakai 224 seperti adaptasi linear ResNet18 | Anchor internal tetap bukan PEM/anchor/scorer lengkap dari paper |
| D02 | Ekstraksi discovery memakai transform evaluasi pada **sampel train**, tanpa augmentasi acak | Sampling MC masih acak dan RNG disimpan; data test tidak masuk discovery |
| E05 | Seed 0 ditangani benar; seed efektif disimpan dan pergantian seed pada folder lama ditolak; notebook memberi pola direktori per seed | Tiga seed belum diluncurkan otomatis; hasil mean/std belum tersedia |
| A01/B05 | Training/evaluasi HiCEM internal placeholder ditolak; entrinya ditandai DITAHAN dalam notebook | Baseline yang digunakan untuk pembanding utama adalah repository `../HiCEM` asli, bukan implementasi internal ini |
| C04/E07 | Jalur PseudoKitchens mematerialkan fitur CLIP ViT-L/14 sekali dengan transform resmi `clip.load`, lalu Stage 1/2 membaca tensor cache | Cache bukan hasil ilmiah; laporkan waktu extraction terpisah. Kesetaraan penuh dengan HiCEM asli tetap memerlukan runner baseline asli dan verifikasi data/split |

Panduan operasional: [resume_training.md](resume_training.md). Output eksekusi
lama notebook dipertahankan; output tersebut bukan bukti kode baru telah
dijalankan pada Colab. Audit ulang tetap mengacu pada paper dan commit lokal
yang tercantum di bawah; source terbaru referensi tidak otomatis sama dengan
snapshot eksperimen paper.

Verifikasi tindak lanjut: suite penuh **58 passed**; setelah tambahan validasi
hierarchy Stage 2, **25 tes terdampak passed**, termasuk satu tes baru (59 tes
dalam inventaris). Tes memakai CPU/data sintetis, bukan hasil eksperimen
ilmiah CUB/PseudoKitchens. Rincian fault injection ada di panduan resume.

## Mana yang wajib sama, mana yang memang berbeda

| Komponen | Keputusan untuk perbandingan terkontrol tesis | Alasan / tindakan |
| --- | --- | --- |
| Dataset, versi, split, kelas, anotasi dan urutan parent | **Wajib sama** dalam pasangan pembanding | Hash metadata sekarang tercatat; identitas aktual data Drive dan disjoint split masih perlu pemeriksaan dataset |
| Backbone, pretrained weights, frozen/fine-tuned, resolusi dan preprocessing | **Wajib sama** dalam kelompok yang mengisolasi perubahan concept model | Jangan membandingkan CUB 112 atribut dengan CUB 32 konsep warna HiCEM sebagai tugas yang sama |
| Akses label train/val/test | **Wajib setara** | Test hanya untuk penilaian final; jangan memilih epoch, child, threshold, hyperparameter atau mapping berdasarkan test |
| Pemilihan checkpoint dan metrik | **Wajib konsisten** untuk model final | Model final dipilih dengan validation task accuracy; Stage 1 dipilih dengan validation concept accuracy karena belum mempunyai task head |
| Seed, jumlah pengulangan dan pelaporan biaya | **Wajib protokol yang sama** | Tiga seed yang ditetapkan lebih dulu, mean/std; laporkan seluruh biaya termasuk Stage 1/SAE/feature extraction |
| Optimizer, LR, batch size, maksimum epoch, scheduler, early stopping | **Tidak harus identik angkanya** | Pertahankan protokol paper untuk replikasi; pada adaptasi, beri anggaran tuning validasi yang sebanding dan periksa konvergensi masing-masing metode |
| Predictor/scorer/objektif dan prosedur training baseline asli | **Harus dipertahankan bila memakai nama metode asli** | Menghilangkan RandInt, PEM, anchor konsep, atau discovery HiCEM bukan novelty HiProbCBM dan tidak boleh disembunyikan sebagai standardisasi |
| Satu cabang subkonsep pada HiProbCBM, bukan pasangan positif/negatif HiCEM | **Perbedaan metode yang disengaja** | Pertahankan pada HiProbCBM; jangan memaksa baseline HiCEM kehilangan dua cabangnya |
| Mean/variance probabilistik dan discovery dari mean | **Bagian rancangan HiProbCBM** | Pertahankan; kontribusinya diuji melalui pembanding/ablasi yang sah, bukan kesamaan arsitektur total |
| Learned attention untuk mean dan propagasi variance dengan alpha kuadrat | **Kontribusi yang diusulkan** | A1 menguji uniform; nyatakan asumsi independensi subkonsep untuk rumus variance |
| KL-SAE vs BatchTopK dan ukuran dictionary | **Pilihan discovery/ablasi** | Tidak wajib menyalin dictionary 12,288 ke input mean 16 dimensi; laporkan jumlah child, parameter, label support dan biaya. Untuk ablasi murni, parent checkpoint/fitur harus sama |
| Head Linear+Softmax bersama | **Kontrol eksperimen yang diizinkan** | Sebut baris yang berbeda dari metode asli sebagai adaptasi. Ablasi head anchor internal bukan replikasi ProbCBM |
| Jumlah parameter total | **Tidak wajib sama persis** | Struktur hierarki mengubah kapasitas; ukur/laporkan dan jangan mengatribusikan semua peningkatan pada attention saja |
| A2 tanpa KL | **Klaim terbatas pada Stage 2** | Discovery masih berasal dari Stage 1 dengan KL; bukan eksperimen tanpa KL global |

**Keputusan epoch:** catatan 50 Stage 1/20 Stage 2 di snapshot awal adalah
budget pilot, bukan angka yang telah terbukti adil atau konvergen. Konfigurasi
PseudoKitchens/CLIP aktif sekarang memakai maksimum 300 epoch dengan early
stopping patience 15 dan batch 256 setelah cache fitur, selaras batas umum
protokol HiCEM. Menyamakan semuanya ke 300 tanpa memperbaiki baseline dan tanpa
kurva validasi tetap tidak menghasilkan head-to-head yang sah. Untuk klaim
terhadap metode asli, selesaikan baseline faithful dan jalankan protokol
referensinya; untuk klaim adaptasi terkontrol, tetapkan rentang tuning dan
kriteria berhenti sebelum test. Resume tidak dipakai untuk diam-diam menambah
epoch setelah melihat test.

**Yang belum selesai untuk tabel ilmiah final:** adapter/runner data dan
environment untuk menjalankan source asli ProbCBM serta HiCEM; validasi dataset
Drive; konvergensi/tuning; tiga seed; discovery held-out; intervensi dan
uncertainty. Entry HiCEM internal tetap ditahan dan tidak menjadi hasil
pembanding. Tidak ada penggantian novelty atau penulisan ulang baseline secara
diam-diam.

## Kesimpulan dan batas pemeriksaan awal

**Matriks eksperimen saat ini belum siap dipakai untuk klaim keunggulan terhadap metode asli.** Pasangan dataset/backbone di beberapa kelompok sudah disamakan, tetapi ada bug evaluasi, baseline yang berubah substansial, protokol training berbeda, dan pengujian kontribusi penelitian yang belum terhubung.

Audit mencakup konfigurasi efektif 18 run di notebook, loader dan konversi data,
kedua tahap HiProbCBM, SAE, ablasi, evaluasi, intervensi, pencatatan eksperimen,
serta kontrak source dua repository referensi. Ini audit statis dengan probe CPU
kecil, bukan validasi seluruh eksperimen pada dataset asli. Isi Drive, versi
paket server Colab, checkpoint aktif, dan distribusi pseudo-label aktual belum
diperiksa langsung.

Snapshot awal: HiProbCBM `998b516a4a4aac470b02b49aa358de559c900693`; HiCEM `41802a5641917b44ab1b312c4dfe03b3040d6895`; ProbCBM `912fc4e49d96533b30253314519614c2f3df65e0`. Notebook lokal sudah berstatus modified sebelum audit. Audit pertama tidak mengubah implementasi; tindak lanjut resume mengubah worktree sebagaimana tabel pembaruan di atas. Sesi Colab tidak diubah langsung.

Ada dua jenis klaim yang harus dibedakan:

- **Replikasi paper:** cocokkan versi metode, data, preprocessing, dan protokol aslinya sebelum membandingkan angka paper.
- **Perbandingan terkontrol dalam tesis:** boleh memakai backbone atau classifier bersama, tetapi labeli model yang diubah sebagai adaptasi; pertahankan komponen esensial metode dan berikan anggaran tuning/validasi yang layak untuk semua model.

Kesetaraan tidak berarti semua arsitektur, jumlah parameter, optimizer, atau epoch harus identik. Perbedaannya harus sesuai pertanyaan penelitian, didokumentasikan, dan tidak secara sepihak melemahkan baseline. Menyamakan semua model menjadi 300 epoch tidak menyelesaikan temuan di bawah.

## Matriks yang benar-benar dijalankan

Sumber: [notebook](../notebooks/colab_setup_training.ipynb), [default](../configs/base.yaml). Nilai diperiksa setelah penggabungan YAML base dan eksperimen.

| Kelompok | Run per seed | Input / batch | Epoch |
| --- | ---: | --- | --- |
| CUB / Inception-v3 | 9 | 299 / 32 | CEM, ProbCBM, CBM masing-masing 50; dua Stage1 masing-masing 50; empat Stage2 masing-masing 20 |
| CUB / ResNet18 | 4 | 224 / 32; anchor sanity check 299 / 32 | Dua baseline 50; Stage1 50; Stage2 20 |
| PseudoKitchens / CLIP ViT-L/14 | 5 | 224 / 256 setelah cache fitur CLIP | Stage 1/2 maksimum 300 dengan early stopping patience 15; extraction CLIP sekali per split |

Ada 18 pemanggilan training, termasuk empat Stage1 yang menghasilkan discovery, sehingga tidak semuanya merupakan baris hasil klasifikasi final. Total 660 epoch per seed, di luar SAE. Sebelas run berlabel wajib dan tujuh tambahan. Notebook belum mengulang otomatis tiga seed.

## A. Bug yang harus dibereskan sebelum tabel perbandingan final

### A01 — Baseline HiCEM tidak mendapat supervisi subkonsep yang benar

**Kritis; terkonfirmasi kode.** [train_baseline.py](../hiprobcbm/engine/train_baseline.py), `build_model` dan `training_step`: jumlah subkonsep dibuat empat untuk setiap cabang positif dan negatif di semua parent; `sub_labels = torch.zeros_like(...)` digunakan sebagai target BCE. Tidak ada pemuatan hasil discovery HiCEM. Label nol berarti target negatif, bukan label yang tidak diketahui, dan bukan mekanisme self-supervised discovery.

Akibatnya pembanding bernama HiCEM mendapat objektif yang berbeda secara mendasar. Perbaikan memerlukan discovery/label subkonsep yang sah, identitas cabang, dan aturan masking; menaikkan epoch tidak memperbaikinya. Tentukan apakah baris yang direplikasi adalah HiCEM dengan discovered labels atau oracle ground-truth hierarchy; keduanya tidak boleh dicampur.

### A02 — A1 dapat dievaluasi sebagai model dengan attention tanpa error

**Kritis; dibuktikan probe.** [run_ablation.py](../scripts/run_ablation.py) melatih A1 dengan `use_attention=False`, tetapi sel 18 hanya memuat YAML dasar. [evaluate.py](../hiprobcbm/engine/evaluate.py) mengembalikan default attention aktif.

[UniformAggregator](../hiprobcbm/models/attention_aggregation.py) mewarisi parameter W dan v dari learned aggregator. Karena itu `load_state_dict` dapat sukses meskipun perilakunya salah. Probe CPU memuat state uniform ke learned aggregator: seluruh key cocok, tetapi bobot `[0.5, 0.5]` berubah menjadi `[0.2761, 0.7239]` untuk contoh yang sama. Parameter attention tersebut tidak digunakan saat training A1. Simpan dan pulihkan konfigurasi efektif, termasuk override ablasi.

### A03 — Preprocessing internal Inception berbeda saat evaluasi HiProbCBM

**Kritis untuk evaluasi CUB/Inception; terkonfirmasi source lokal dan upstream.** [backbones.py](../hiprobcbm/models/backbones.py) tidak menetapkan `transform_input`. Torchvision mengaktifkannya secara default saat weights pretrained diberikan, sedangkan constructor tanpa weights mempunyai default False. Training memakai pretrained=True; [evaluator Stage2](../hiprobcbm/engine/evaluate.py) membangun model dengan pretrained=False lalu memuat state dict. Flag ini bukan tensor checkpoint, sehingga transformasi input tidak ikut dipulihkan.

Jalur evaluasi baseline membangun model dari `cfg.model.pretrained`, sehingga bug ini dapat membuat preprocessing baseline dan HiProbCBM tidak simetris. Tetapkan flag eksplisit yang konsisten dengan checkpoint dan preprocessing training; jangan menggantinya diam-diam sehingga checkpoint lama berubah makna. Sumber: [Torchvision Inception](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/inception.html). Dampak numerik pada checkpoint Colab belum diukur.

### A04 — SAE KL bisa mempertahankan subkonsep yang labelnya seluruhnya nol

**Kritis untuk discovery; dibuktikan sebagai kasus yang mungkin, belum diukur pada data pengguna.** [subconcept_discovery.py](../hiprobcbm/models/subconcept_discovery.py) menyaring fitur dengan `activations.sum(0)>0` sebelum threshold 0.5. Aktivasi sigmoid umumnya positif meskipun semuanya di bawah threshold.

Probe dengan aktivasi sigmoid sekitar 0.0474 mempertahankan 128 fitur, tetapi menghasilkan nol kolom label biner yang mempunyai contoh positif. Fallback satu subkonsep tidak terpicu. Ini dapat merusak supervisi Stage2 dan mengacaukan ablasi KL versus BatchTopK.

Periksa support positif/negatif, fitur konstan, redundansi, dan jumlah subkonsep per parent pada `pseudo_hierarchy.pt`. Filter setelah pembentukan label, disertai kriteria yang ditetapkan dari train/validation. Jangan menyimpulkan discovery valid dari jumlah kolom saja.

Implikasi kapasitas: jika seluruh 112 parent CUB masing-masing mempertahankan 128 fitur, dua head linear per subkonsep di [SubconceptPredictor](../hiprobcbm/models/hiprobcbm.py) membutuhkan `112*128*2*(2048+1)*16 = 939,982,848` parameter head. Ini perhitungan kondisional, bukan ukuran model pengguna yang sudah terukur. Jumlah subkonsep perlu diperiksa sebelum Stage2, termasuk untuk menilai kecukupan T4.

## B. Identitas model dan prosedur training baseline

### B01 — ProbCBM internal adalah adaptasi, termasuk mode anchor

[Baseline internal](../hiprobcbm/models/baselines/probcbm.py) memakai [head mean/variance linear dan scorer sigmoid bersama](../hiprobcbm/models/probabilistic_concepts.py). [Referensi ProbCBM](../../ProbCBM/models/build_model_resnset.py) memakai modul embedding dengan attention, normalisasi mean, serta pencocokan ke anchor konsep. Classifier linear utama juga memang keputusan desain yang berbeda dari model paper.

Mengganti hanya classifier kelas menjadi `anchor` tidak memulihkan predictor konsep, scorer, atau training asli. Jarak/skalanya juga perlu dicocokkan: implementasi internal memakai Euclidean norm, referensi kelas memakai akar rata-rata kuadrat dengan scale tersendiri. Jadi nama `anchor sanity-check replikasi` saat ini terlalu kuat; gunakan sebagai ablation head pada adaptasi, atau bangun jalur replikasi yang benar. [Sumber paper ProbCBM](https://proceedings.mlr.press/v202/kim23g/kim23g.pdf).

### B02 — ProbCBM internal dilatih joint, bukan sequential 50 + 20

[Trainer baseline](../hiprobcbm/engine/train_baseline.py) mengoptimalkan loss kelas, konsep, dan KL bersamaan selama 50 epoch. [main.py referensi](../../ProbCBM/main.py) dan [config_exp.yaml](../../ProbCBM/configs/config_exp.yaml) memakai 50 epoch konsep lalu 20 epoch kelas, pembekuan awal backbone lima epoch, dan penggantian konsep saat melatih classifier dengan probabilitas 0.5. Prosedur tersebut belum diterapkan pada baseline internal.

Angka 50 + 20 pada HiProbCBM tidak menggantikan dua tahap baseline: Stage1 kita untuk parent/discovery, sedangkan Stage2 membangun model hierarkis baru, bukan meneruskan bobot concept predictor Stage1. Perbedaan ini harus dilaporkan sebagai desain HiProbCBM.

### B03 — Objektif Monte Carlo kelas ProbCBM berubah

[losses.py](../hiprobcbm/losses.py) merata-ratakan cross-entropy tiap sampel Monte Carlo. [run_epoch.py referensi](../../ProbCBM/run_epoch.py) menerapkan NLL pada probabilitas kelas yang sudah dirata-ratakan. Secara umum `mean(-log p_s)` berbeda dari `-log(mean p_s)`. Ini perlu diputuskan secara eksplisit, bukan dianggap hanya beda implementasi numerik.

### B04 — CEM internal menghilangkan bagian scorer dan RandInt

[cem.py internal](../hiprobcbm/models/baselines/cem.py) menilai probabilitas dari embedding positif saja, dengan input scorer berdimensi d. CEM asli menilai gabungan embedding positif dan negatif, berdimensi 2d. Baseline internal juga tidak menerima label konsep untuk RandInt saat training. Ini memengaruhi kemampuan prediksi konsep dan intervensi, bukan sekadar jumlah epoch. Bukti: [paper CEM, §3](https://arxiv.org/pdf/2209.09056) dan [CEM dalam repo referensi](../../HiCEM/cemcd/models/cem.py).

### B05 — HiCEM internal juga berbeda struktur dan identitas cabang

Selain A01, [hicem.py internal](../hiprobcbm/models/baselines/hicem.py) menggabungkan probabilitas child positif dan negatif dengan `maximum` pada indeks yang sama. Ini menghilangkan identitas dua child yang semestinya terpisah. Top generator internal hanya Linear, sedangkan [referensi](../../HiCEM/cemcd/models/hicem.py) menambahkan LeakyReLU. RandInt/hierarchical intervention belum diterapkan dalam forward baseline internal.

### B06 — CBM harus diberi nama varian yang tepat

[cbm.py](../hiprobcbm/models/baselines/cbm.py) memberikan ground-truth concepts ke classifier saat training dan probabilitas prediksi saat evaluasi. Ini merupakan pilihan independent-style, bukan joint CBM. Kedua bagian dioptimalkan dalam loop yang sama tetapi loss kelas tidak melewati concept predictor pada jalur training tersebut.

Ini tidak otomatis bug. Namun tabel tidak boleh menamakannya hanya CBM lalu membandingkan dengan angka joint/sequential CBM. Head linear juga merupakan adaptasi terhadap model dengan head lebih kompleks; perlu disebutkan. [Paper CBM](https://proceedings.mlr.press/v119/koh20a/koh20a.pdf).

### B07 — Anggaran dan hyperparameter training belum mengikuti acuan

| Aspek | Referensi yang diperiksa | Internal |
| --- | --- | --- |
| ProbCBM | 50 konsep + 20 kelas; AdamP; cosine; LR backbone 1e-3 dan head 1e-2; batch 128 | Joint 50; Adam 1e-4; batch 32 |
| CEM pada CUB | Maksimal 300; early stopping patience 15; SGD momentum 0.9; LR 1e-2; batch 128; penurunan LR berbasis validation loss | 50; Adam 1e-4; batch 32; tanpa early stopping/scheduler |
| HiCEM pada Kitchens | Maksimal 300; berhenti setelah 75 epoch tanpa perbaikan validation loss; Adam 1e-3; batch 256 | 50; Adam 1e-4; batch 64; tanpa early stopping/scheduler |
| HiProbCBM | Metode baru, perlu validasi konvergensi sendiri | Stage1 50/1e-4; Stage2 20/5e-5 |

Acuan: [ProbCBM Appendix D](https://proceedings.mlr.press/v202/kim23g/kim23g.pdf), [CEM Appendix A.6](https://arxiv.org/pdf/2209.09056), [HiCEM Appendix E](https://arxiv.org/html/2602.23947v1#A5). Batas 300 bukan bukti semua run penelitian asli selesai tepat pada epoch 300. Jumlah update per epoch bergantung batch size; jangan menafsirkan selisih epoch sebagai rasio komputasi yang sama.

### B08 — Bobot dan penanganan imbalance konsep berbeda

[Loss internal](../hiprobcbm/losses.py) memakai BCE tanpa bobot kelas dan default concept weight 1. Acuan CEM CUB memakai bobot konsep 5; HiCEM 10 dengan pembobotan untuk ketidakseimbangan konsep. Referensi HiCEM menghitung loss pada gabungan konsep, sedangkan baseline internal menjumlahkan dua mean BCE parent dan child. Perubahan denominator/bobot juga mengubah skala objektif. Perbandingan terkontrol tetap perlu tuning validasi yang layak, bukan otomatis memaksakan satu nilai yang sama.

### B09 — MC sampling belum konsisten dengan acuan maupun seluruh config

Default internal train/eval 8/32 versus ProbCBM 50/50. [Stage1.forward](../hiprobcbm/models/hiprobcbm.py) memakai default argumen `n_samples=8`; trainer dan validator tidak meneruskan nilai YAML untuk forward ini. Ekstraksi discovery meneruskan jumlah sampel eval, sehingga validation Stage1 dan discovery dapat menggunakan jumlah sampel berbeda. Perubahan YAML saja belum menjamin Stage1 mengikuti protokol baru.

## C. Data, preprocessing, dan cakupan pembanding

### C01 — Sebagian pasangan internal sudah cocok, tetapi bukan seluruh angka paper

Kelompok Inception/CUB mempunyai backbone, ukuran input, loader, dan batch size yang sama antar-run. Kelompok ResNet/CUB reguler juga demikian; kelompok Kitchens memakai CLIP beku yang sama. Ini fondasi yang benar untuk perbandingan internal.

Namun ProbCBM/Inception adalah perubahan backbone dari acuan ResNet18. Anchor sanity check ResNet18 memakai 299, sedangkan ResNet18 reguler dan HiProbCBM-ResNet18 memakai 224. Anchor versus linear pada konfigurasi tersebut bukan ablasi head murni karena resolusi ikut berubah. Tidak ada HiCEM/CUB dalam matriks 18 run; ini keputusan cakupan, bukan bug yang mengharuskan penambahan eksperimen tanpa persetujuan.

### C02 — CUB HiCEM merupakan tugas konsep berbeda

Kita membaca 112 atribut dari pkl. [Loader HiCEM](../../HiCEM/cemcd/data/cub.py) membentuk 32 konsep warna terang/gelap dari anotasi kelas dan menggunakan warna terperinci sebagai concept bank. Backbone paper HiCEM adalah CLIP beku, termasuk CUB. Maka angka konsep/discovery CUB HiCEM tidak bisa dibandingkan langsung dengan tugas 112 konsep Inception/ResNet kita. Memakai file split dari repo HiCEM tidak otomatis memakai definisi konsep HiCEM. [Paper, Appendix C.3/E](https://arxiv.org/html/2602.23947v1).

### C03 — Identitas dataset aktual belum dibuktikan

[Konverter Kitchens](../scripts/prepare_pseudokitchens_manifest.py) mengikuti pembacaan cryptomatte r/b dan [loader](../hiprobcbm/data/kitchens.py) membentuk parent dengan OR anggota grup, sesuai jalur referensi. Namun belum ada pemeriksaan jumlah file, hash split, label per-sampel, versi dataset, dan daftar recipe/concept pada Drive pengguna. Referensi lokal menyebut `pseudokitchens_V2`; nama folder pengguna saja tidak membuktikan versinya sama atau berbeda.

Untuk CUB, verifikasi jumlah sampel, disjoint split, pemetaan 112 atribut, dan nama atribut. [Loader CUB](../hiprobcbm/data/cub.py) hanya menerima `attributes.txt` jika panjangnya persis 112; bila tidak, nama menjadi `concept_i`. Akibatnya grouping intervensi tidak lagi merepresentasikan kelompok atribut semantik. Pkl 112 atribut dan file nama 312 atribut perlu pemetaan eksplisit.

### C04 — Transform CLIP resmi dipakai pada jalur cache

[clip_transforms](../hiprobcbm/data/transforms.py) lama langsung meresize ke persegi 224 dengan default interpolasi torchvision. Jalur PseudoKitchens yang memakai `use_cached_features: true` sekarang mengganti transform tersebut dengan preprocess dari `clip.load("ViT-L/14")` saat mematerialkan cache train, validation, dan test. Ini mengikuti mekanisme [referensi](../../HiCEM/cemcd/data/base.py), yang juga menyimpan representasi foundation model. Checksum cache dan hash manifest/info.json masuk ke identitas checkpoint.

Transform CUB internal juga berbeda dari [ProbCBM](../../ProbCBM/dataloaders/cub312_datamodule.py): ColorJitter, urutan augmentasi, dan konstanta normalisasi tidak identik. Tetapkan apakah ingin replikasi preprocessing asli atau protokol bersama yang dilaporkan sebagai adaptasi.

## D. Discovery, kapasitas, dan ablasi

### D01 — BatchTopK kita belum mereplikasi konfigurasi HiCEM

[SAE internal](../hiprobcbm/models/sae.py) menggunakan dictionary 128 (16*8), batch 256, 100 epoch, threshold pseudo-label 0.5, Adam default betas, dan grad clip 1. [Konfigurasi referensi](../../HiCEM/experiments/configs/kitchens.yaml) pada bagian BatchTopK memakai dictionary 12,288, batch 50,000, 300 epoch, beta2 0.99, dan input normalization; [discovery referensi](../../HiCEM/cemcd/concept_discovery.py) membinerkan aktivasi dengan >0. Decoder gradient normalization dan auxiliary-loss details juga berbeda di [sae.py referensi](../../HiCEM/cemcd/sae.py).

Varian KL, penggunaan mean probabilistik, satu cabang subkonsep, dan attention propagasi variance adalah desain HiProbCBM yang memang disengaja. Tidak perlu dihapus agar identik dengan HiCEM, tetapi BatchTopK internal harus disebut adaptasi, dan ablasinya perlu memastikan label tidak rusak oleh A04.

### D02 — Ekstraksi discovery CUB masih memakai augmentasi acak

[train_stage1.py](../hiprobcbm/engine/train_stage1.py) membuat discovery loader dari split train, hanya mengubah shuffle/drop_last. Transform tetap transform training sehingga crop/jitter dapat berubah. Pseudo-label yang dibuat sekali kemudian ditautkan ke path citra untuk view lain saat Stage2. Penautan berdasarkan path sudah benar, tetapi konsistensi label terhadap augmentasi perlu diuji atau gunakan transform ekstraksi deterministik. CLIP saat ini deterministik sehingga kekhawatiran augmentasi ini khusus jalur non-CLIP.

### D03 — SAE dan pemetaan discovery belum menjadi artifact yang dapat diterapkan ke test

Artifact `pseudo_hierarchy.pt` menyimpan jumlah child, label train, path, dan nama parent; bobot SAE, mapping fitur hidup, serta konfigurasi efektif tidak disimpan. Fungsi [subconcept_discovery_roc_auc](../hiprobcbm/metrics.py) belum dipanggil evaluator dan menerima satu target ground-truth tanpa protokol mapping/freeze/evaluasi lengkap. Loader Kitchens menyimpan informasi hierarki tetapi batch biasa hanya berisi parent.

Tentukan matching memakai train/validation, bekukan mapping, lalu ukur pada test yang tidak digunakan memilih fitur. Jangan menganggap fungsi best-AUC pada set arbitrer sudah mereplikasi keseluruhan penilaian discovery paper.

### D04 — A2 dan A3 perlu label klaim yang lebih sempit

A2 menghapus KL hanya di Stage2; artifact discovery tetap berasal dari Stage1 dengan KL. Nama yang tepat saat ini adalah **tanpa KL Stage2**, bukan tanpa KL di seluruh metode. Jika hendak menguji penghapusan KL global, perlu protokol Stage1 tersendiri dan biaya berbeda.

A3 melatih Stage1 kembali untuk varian SAE lain. Seed sama tidak membuktikan checkpoint parent identik karena ada potensi nondeterminisme. Untuk mengisolasi pengaruh SAE, gunakan checkpoint parent dan fitur yang sama atau verifikasi hash. Jumlah child yang berbeda juga mengubah kapasitas Stage2; laporkan jumlah parameter dan child, bukan mengatribusikan semua selisih hanya pada kualitas SAE.

## E. Evaluasi, intervensi, dan reproduksibilitas

### E01 — Pemilihan checkpoint final berbeda dari checkpoint validasi terbaik

Trainer menyimpan `_best.pth` dan `_last.pth`, tetapi sel 18 mengevaluasi marker `_last.pth`. Ini konsisten sebagai kebijakan last bila sengaja dipilih, tetapi tidak sesuai klaim evaluasi checkpoint terbaik. Stage1 sendiri memakai best concept accuracy untuk discovery; Stage2/baseline best task accuracy. Tetapkan kebijakan per tahap dan gunakan validation, bukan test, untuk pemilihan model.

### E02 — Evaluator notebook memasukkan Stage1 ke jalur Stage2

Sel 18 menganggap semua command selain baseline adalah Stage2. Command Stage1 tidak mempunyai `--stage1-log-dir`, sehingga ekstraksi regex gagal lalu baris dilewati. Pisahkan stage preparation/discovery dari hasil klasifikasi. Evaluator juga tidak merekonstruksi override ablasi (A02) dan tidak membatasi daftar dengan RUN_TAGS; artifact lama dapat ikut tercampur.

### E03 — Intervensi belum dievaluasi setara

[intervention.py](../hiprobcbm/intervention.py) berisi helper, tetapi evaluator tidak menjalankan kurva intervensi atau mengisi `intervention_delta`. Baseline CEM/ProbCBM/HiCEM internal tidak mempunyai jalur intervensi lengkap yang sesuai representasinya; mengganti tensor probabilitas saja tidak cukup bila classifier memakai embedding. HiProbCBM forward mendukung penggantian mean subkonsep dan variance nol, belum API parent langsung serta pipeline estimasi anchor yang lengkap.

Perlu menetapkan budget intervensi, tingkat parent/child, unit grup, urutan acak/uncertainty, jumlah pengulangan, ground-truth yang tersedia, dan penanganan uncertainty yang konsisten. Klaim interpretabilitas/intervensi belum dapat diambil dari accuracy biasa.

### E04 — Definisi metrik dan pengujian uncertainty belum lengkap

Internal macro concept AUC melewati kolom satu kelas; [HiCEM metrics](../../HiCEM/cemcd/metrics.py) mengganti AUC kolom tersebut dengan accuracy. Threshold juga >=0.5 versus >0.5. Untuk perbandingan internal gunakan evaluator bersama dengan definisi tertulis; untuk angka paper jangan menganggap identik.

ECE internal adalah kalibrasi confidence kelas, bukan seluruh evaluasi uncertainty ProbCBM. [Referensi ProbCBM](../../ProbCBM/utils/evaluate.py) mempunyai pengujian uncertainty/cutout dan intervensi yang belum diorkestrasi di notebook. MC evaluasi juga belum memakai seed/generator tetap dalam evaluator notebook, sehingga hasil dapat bergantung urutan run dan state RNG.

### E05 — Tiga seed belum aman dijalankan

Notebook tidak menggunakan loop tiga seed. [prepare_log_dir](../hiprobcbm/config.py) tidak memasukkan seed ke path; menjalankan ulang seed lain dapat menimpa artifact atau dilewati marker lama. Semua CLI menggunakan `args.seed or cfg.seed`: seed 0 berubah menjadi seed konfigurasi. Ini relevan karena config ProbCBM referensi menggunakan seed 0. Simpan seed efektif, pisahkan direktori, dan hitung mean/std dari run berbeda yang terverifikasi.

### E06 — Resume, identitas run, dan logging belum cukup untuk audit hasil

Argumen `--resume` di parser umum belum diterapkan; checkpoint hanya state dict tanpa optimizer/epoch/RNG. `_last` baru disimpan setelah seluruh loop training. File marker hanya diperiksa keberadaannya, tidak dicocokkan dengan hash config/kode/dataset. Mengubah protokol tetapi memakai direktori lama dapat membuat eksperimen baru dilewati sebagai selesai.

Logging trainer memakai basicConfig ke output proses; tidak ada penyimpanan otomatis config efektif, versi paket, GPU, riwayat metrik terstruktur, dan commit dalam artifact. Output notebook yang tersimpan membantu tetapi bukan pengganti manifest eksperimen. Ini masalah reproduksibilitas/biaya, bukan bukti otomatis bias pada setiap run yang selesai.

### E07 — Waktu komputasi belum dapat dibandingkan langsung

HiCEM referensi menyimpan fitur CLIP untuk dipakai kembali. Jalur HiProbCBM PseudoKitchens kini melakukan hal yang sama: cache train/validation dibuat satu kali, sedangkan cache test baru dibuat saat evaluasi final. Pembacaan gambar via Drive, jumlah worker, precision, dan GPU tetap memengaruhi waktu extraction. Jika melaporkan efisiensi, pisahkan extraction, discovery, training head, evaluasi, dan total; catat parameter, memori puncak, serta compute units. Caching saja bukan peningkatan accuracy dan bukan pengganti perbaikan metodologi.

## F. Paper dan repository referensi tidak selalu sama

Paper HiCEM yang diperiksa menyebut classifier linear dan BatchTopK. Clone lokal saat ini berisi head MLP dua hidden layer 128 pada CEM/HiCEM, default discovery `hisae`, konfigurasi HiSAE 100 epoch, dukungan deep hierarchy, dan rujukan dataset V2. Artinya source terbaru tidak boleh langsung dianggap snapshot eksperimen paper ICLR tersebut. Pilih commit/config yang merepresentasikan target paper; dokumentasikan selisih yang tidak dapat dipulihkan. Jangan mengganti head linear internal menjadi MLP hanya karena branch sekarang memakai MLP.

Penelitian baru boleh memperkenalkan mean/variance linear, attention, KL-SAE, dan head bersama. Namun baseline yang dimodifikasi harus diberi nama adaptasi; perbandingan faithful baseline dan ablation komponen bersama menjawab pertanyaan ilmiah yang berbeda. Kesimpulan bahwa perbedaan hasil hanya disebabkan hierarki belum didukung jika scorer, training, budget, dan kapasitas ikut berubah.

## G. Verifikasi yang benar-benar dilakukan

- Membaca source terkait dan mengekstrak matriks 18 command beserta konfigurasi efektif secara programatis.
- Membandingkan Appendix paper dan source lokal; mencatat commit yang diperiksa.
- Probe CPU A1: state dict cocok seluruhnya meskipun bobot agregasi dan perilaku berbeda.
- Probe CPU SAE: 128 fitur lolos filter alive dengan nol fitur label biner positif pada konstruksi sederhana. Tidak mengklaim ini sudah terjadi pada seluruh dataset pengguna.
- Memeriksa factory Inception terpasang dan source upstream: `transform_input` tergantung pemberian weights. Tidak melakukan forward checkpoint Colab atau mengukur dampak akurasinya.
- Tidak menjalankan training baru, tidak mengunduh bobot/dataset, tidak memakai CU Colab, dan tidak mengubah implementasi selama audit. Tes 39 passed pada catatan onboarding bukan pengujian baru audit ini dan bukan bukti kesetaraan ilmiah.

## H. Urutan perbaikan yang disarankan

1. Amankan artifact run yang sedang berlangsung sebagai eksperimen awal beserta log dan versi kode. Sebelum Stage2, periksa distribusi pseudo-label/jumlah child (A04).
2. Perbaiki evaluator A1, flag Inception, routing Stage1, dan pemilihan checkpoint (A02/A03/E01/E02). Beberapa masalah evaluasi dapat diperbaiki tanpa melatih ulang checkpoint yang valid.
3. Tetapkan target paper/commit dan pisahkan label faithful baseline dari adaptasi. Perbaiki HiCEM labels/branch, scorer CEM/RandInt, dan jalur ProbCBM yang dipilih.
4. Tetapkan data/split/concepts/preprocessing, protokol validasi dan tuning, serta anggaran per tahap. Periksa hasil konvergensi sebelum memilih epoch final.
5. Lengkapi discovery held-out, intervensi, uncertainty, dan pembatasan klaim A2/A3.
6. Terapkan seed terpisah, config/hash manifest, resume lengkap, dan pencatatan hasil sebelum menjalankan seluruh matriks tiga seed.

Tidak semua hasil yang sedang dihitung harus dibuang. Checkpoint Stage1 dapat tetap berguna; keputusan menggunakan ulang atau melatih ulang bergantung pada perbaikan yang akhirnya dipilih. Matriks 18 run tidak diperluas dalam audit ini.
