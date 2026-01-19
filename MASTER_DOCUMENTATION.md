# MASTER DOKUMENTASI PROYEK ANALISIS SENTIMEN HYBRID

Dokumen ini menggabungkan seluruh aspek dokumentasi teknis, laporan, panduan operasional, dan materi presentasi dalam satu file terpadu.

---

## DAFTAR ISI
1.  **Ringkasan Eksekutif & Laporan Utama** (Executive Summary)
2.  **Dokumentasi Teknis & Pipeline** (Technical Documentation)
3.  **Panduan Operasional (Walkthrough)**
4.  **Panduan Presentasi** (Presentation Guide)

---

# BAGIAN 1: RINGKASAN EKSEKUTIF & LAPORAN UTAMA

## 1. Judul Proyek
"Analisis Sentimen Publik Menggunakan **Hybrid IndoBERT** serta Pengembangan Dashboard Web Interaktif untuk Monitoring Layanan Informasi Kota Semarang"

## 2. Ringkasan Sistem (Abstract)
Sistem ini dirancang untuk memantau opini publik di media sosial Instagram Pemerintah Kota Semarang secara otomatis. Menggunakan algoritma **IndoBERT (Transformer)** yang telah dioptimasi dengan teknik **Class-Weighted Fine-Tuning** dan **Keyword-Based Hybrid Inference**, sistem ini mampu mengklasifikasikan komentar warga (Positif, Negatif, Netral) secara akurat, bahkan untuk teks yang mengandung bahasa daerah (Jawa) dan slang.

---

## 3. Arsitektur & Alur Kerja (Pipeline)

Sistem bekerja dalam 4 komponen utama yang dieksekusi secara berurutan:

#### 1. `indo_utils.py` (Pondasi / Helper)
*   **Peran**: Modul utilitas yang tidak dieksekusi sendiri, namun **di-import** oleh semua skrip lain.
*   **Isi Utama**:
    *   **Kamus Normalisasi**: Mengubah bahasa alay/singkatan menjadi baku (contoh: *gk -> tidak*).
    *   **Keyword Dictionary**: Daftar kata kunci `POSITIF`, `NEGATIF`, dan `NETRAL` (Termasuk bahasa Jawa/Slang Semarang) untuk validasi hibrida.
    *   **Fungsi Preprocessing**: `clean_text()` dan `normalize_text()` untuk membersihkan input mentah.

#### 2. `dapt.py` (Langkah 1: Belajar Bahasa / Domain Adaptation)
*   **Tujuan**: Mengajarkan model IndoBERT tentang gaya bahasa spesifik warga Semarang.
*   **Proses**: Masked Language Modeling (MLM) pada seluruh data komentar (Labeled + Unlabeled).
*   **Output**: Model cerdas `model_output/dapt_indobert` yang paham konteks lokal (misal: "cumi darat" = asap bus).

#### 3. `train_kfold.py` (Langkah 2: Latihan & Validasi)
*   **Tujuan**: Melatih model untuk membedakan sentimen.
*   **Strategi**: Menggunakan *Class Weights* (Netral: 1.3, Negatif: 1.2) dan *Early Stopping* (Epoch 3).
*   **Output**: `oof_results.pkl` (Rekap performa validasi) dan file model `.bin`.

#### 4. `ensemble_predict.py` (Langkah 3: Inference / Prediksi Final)
*   **Tujuan**: Menghaskan label akhir untuk data tak berlabel.
*   **Fitur Utama**: **Keyword Refinement (Hybrid)**. Mengecek kamus `indo_utils.py`. Jika ditemukan kata kunci kuat (misal: "pengen", "mau" -> Netral), prediksi model dikoreksi dengan boost **+0.8**.
*   **Output**: `output_comments.csv` (Data lengkap), `output_trends.csv`, `output_keywords.csv`.

---

## 4. Hasil & Analisis Performa

### Hasil Statistik (Validasi)
*   **Akurasi Global**: ~70-71%
*   **F1-Score**: 
    *   Positif: 0.77 (Sangat Baik)
    *   Negatif: 0.73 (Baik)
    *   Netral: 0.62 (Cukup Baik untuk kelas tersulit)

### Inovasi Penyelesaian Masalah
1.  **Handling Slang/Jawa**: Menggunakan DAPT dan Kamus Normalisasi.
2.  **Safety Net (Hybrid)**: Kasus ambigu seperti "Pengen coba" yang awalnya salah dideteksi Negatif, kini sukses dilabeli Netral berkat *Keyword Boosting*.

---

# BAGIAN 2: DOKUMENTASI TEKNIS DETIL

## 1. Flowchart Sistem

```mermaid
graph TD
    A[Raw Data: Instagram] --> B(Preprocessing: indo_utils.py);
    B --> C{Jalankan DAPT?};
    C -- Yes --> D[DAPT: dapt.py];
    C -- No --> E[Base Model: IndoBERT];
    D --> E;
    
    subgraph Training
    E --> F[Train K-Fold];
    F --> G[Weighted Loss Update];
    G --> H[Model Checkpoints (.bin)];
    end
    
    subgraph Inference Hybrid
    H --> M[Ensemble Prediction];
    M --> N[Soft Voting Probabilities];
    N --> O[Keyword Refinement +0.8 Bias];
    O --> P[Final Label];
    end
    
    P --> Q[CSV Outputs];
```

## 2. Pustaka Utama
*   **PyTorch & Transformers**: Core Deep Learning (IndoBERT).
*   **Scikit-Learn**: Metrik evaluasi dan K-Fold splitting.
*   **Pandas & NumPy**: Manipulasi data dan operasi matriks.

---

# BAGIAN 3: PANDUAN OPERASIONAL (WALKTHROUGH)

## Persiapan & Eksekusi Pipeline

Pastikan environment Python sudah siap dengan `requirements.txt` terinstall.

### Langkah 1: Persiapan
Pastikan file `indo_utils.py` sudah berisi daftar keyword terbaru (termasuk "pengen", "coba", dll).

### Langkah 2: Pretraining (DAPT)
Jalankan perintah ini untuk melatih model memahami bahasa lokal:
```bash
python dapt.py
```
*Output: Folder `model_output/dapt_indobert`.*

### Langkah 3: Training & Validasi (K-Fold)
Melatih model dengan data berlabel manual.
*   **Fitur**: Class Weights `[1.3, 1.2, 1.0]`, Dropout 0.3, Weight Decay 0.1, Learning Rate 2e-5, Early Stopping (Epoch 3).
```bash
python train_kfold.py
```
*Output: `model_output/model_fold_0.bin`.*

### Langkah 4: Inference (Prediksi)
Menjalankan prediksi pada data baru dengan logika Hybrid:
```bash
python ensemble_predict.py
```
*Output: `output_comments.csv`.*

## Troubleshooting
*   **Warning "Some weights not initialized"**: Abaikan saat menjalankan `ensemble_predict.py`. Ini normal saat memuat arsitektur model sebelum menimpa bobotnya.
*   **Akurasi Kosong di Report**: Pada `classification_report`, baris `accuracy` memang kolom precision/recall-nya kosong. Nilai akurasi ada di kolom f1-score (tengah).

---

# BAGIAN 4: PANDUAN PRESENTASI

## Slide Outline (Saran)

#### Slide 1: Judul
*   Judul: "Analisis Sentimen Publik Menggunakan **Hybrid IndoBERT**"
*   Visual: Logo Instansi & Background Tugu Muda.

#### Slide 2: Masalah & Solusi
*   Masalah: Komentar ribuan, bahasa campur (Jawa/Indo), sulit dipantau manual.
*   Solusi: AI Otomatis yang paham konteks lokal (Semarang).

#### Slide 3: Metodologi (The Secret Sauce)
*   Jelaskan 3 Lapisan Kecerdasan:
    1.  **Preprocessing**: Normalisasi "alay".
    2.  **IndoBERT**: Otak utama (Deep Learning).
    3.  **Hybrid Logic**: Koreksi manual via Kamus Keyword (agar tidak baper/salah paham).

#### Slide 4: Hasil & Performa
*   Tampilkan Grafik `loss_curve` (menurun = bagus).
*   Highlight kemampuan mendeteksi istilah lokal seperti "cumi darat" (asap hitam) sbg Negatif.

#### Slide 5: Demo / Output
*   Tampilkan Screenshot `output_comments.csv` atau Dashboard (jika ada).
*   Tunjukkan contoh: "Pengen coba" -> Terdeteksi **Netral** (Benar).

#### Slide 6: Kesimpulan
*   Sistem siap pakai, akurat (70%+), dan peka terhadap aspirasi warga.

---
**Dokumen ini dibuat secara otomatis menggabungkan seluruh aspek teknis proyek Analisis Sentimen.**
