# 🫁 RespiScan: Lightweight AI for Pulmonary Tuberculosis Screening

![Status](https://img.shields.io/badge/Status-Research_Phase-blue)
![Target](https://img.shields.io/badge/Target-Mobile_Edge_Deployment-success)
![Accuracy](https://img.shields.io/badge/Patient_Level_Accuracy-83.87%25-brightgreen)

## 📌 Deskripsi Proyek
**RespiScan** adalah sistem kecerdasan buatan (AI) berbasis pengolahan sinyal audio yang dirancang untuk melakukan *screening* awal penyakit Tuberkulosis (TBC) melalui rekaman suara batuk pasif. 

Proyek ini mengatasi kelemahan model AI medis pendahulu yang terlalu berat untuk dijalankan pada perangkat berspesifikasi rendah. Dengan menggunakan arsitektur **MobileNetV2** yang sangat efisien dan ekstraksi fitur waktu-frekuensi 2D (**Scalogram**), RespiScan dirancang khusus untuk implementasi *offline* pada *smartphone* di fasilitas kesehatan tingkat pertama atau daerah terpencil dengan keterbatasan akses internet.

## 🚀 Fitur Utama & Inovasi
* **Optimasi Fitur Akustik (Scalogram):** Mengubah sinyal audio `.wav` menjadi representasi *Continuous Wavelet Transform* (CWT) Scalogram pada rentang frekuensi spesifik (10 Hz - 4 kHz) yang paling kaya akan informasi klinis TBC, menggantikan pendekatan MFCC konvensional.
* **Lightweight Architecture:** Menggunakan MobileNetV2 (~3.4 juta parameter) yang memangkas beban komputasi secara drastis dibandingkan model SOTA seperti ResNet18 (~11 juta parameter), tanpa mengorbankan sensitivitas diagnostik.
* **Patient-Level Evaluation (Majority Vote):** Evaluasi ketat berstandar klinis yang menghitung akurasi berdasarkan identitas pasien, bukan sekadar potongan gambar batuk, guna menghindari bias *Data Leakage* dan *Acoustic Masking*.

## 📊 Performa Model
Evaluasi dilakukan menggunakan metode **Majority Vote** pada data *Test Set* yang sepenuhnya terisolasi (31 pasien unik, 442 rekaman batuk).

| Metrik Evaluasi (Patient-Level) | Skor | Deskripsi Klinis |
| :--- | :--- | :--- |
| **Sensitivity (Recall)** | **88.89%** | Kemampuan sistem mendeteksi pasien positif TBC dengan benar. |
| **Specificity** | **76.92%** | Kemampuan sistem mengenali pasien dengan batuk non-TBC (sehat/penyakit lain). |
| **Accuracy Total** | **83.87%** | Tingkat akurasi keseluruhan pada pengujian tingkat pasien. |

> **Komparasi SOTA:** Sensitivitas diagnostik RespiScan (88.89%) sangat kompetitif dan melampaui kemampuan identifikasi pasien dari *baseline* paper referensi *TBscreen* (Science Advances, 2024), dengan keuntungan ukuran model yang jauh lebih ringan.

## 🔬 Dataset & Preprocessing
Dataset penelitian diadaptasi dari repositori klinis dengan protokol pembersihan data yang ketat:
1. **Total Data:** 3.322 gambar Scalogram.
2. **Proporsi Pembagian (Split Ratio):**
   * **TRAIN:** 69.6% (2.313 gambar)
   * **VAL:** 17.1% (567 gambar)
   * **TEST:** 13.3% (442 gambar)
3. **Data Isolation:** Isolasi tingkat subjek (pasien) dijamin 100%. Pasien yang berada di folder `TEST` tidak memiliki satu pun rekaman batuk di folder `TRAIN` untuk memastikan validitas pengujian dunia nyata (*Real-World Validation*).

## 🛠️ Pipeline Arsitektur
1. **Input:** Rekaman audio batuk pasif pasien (format `.wav`, *Sampling Rate* 44.1 kHz).
2. **Preprocessing:** Transformasi sinyal 1D ke 2D menggunakan algoritma CWT (Complex Morlet Wavelet).
3. **Feature Extraction:** Gambar Scalogram beresolusi tinggi diekstraksi ke dalam matriks RGB.
4. **Classification:** Proses *forward pass* melalui jaringan MobileNetV2.
5. **Decision:** Klasifikasi akhir biner (`TB` / `Healthy`) berdasarkan agregasi *Majority Vote* per pasien.

## 💻 Teknologi & Modul
* `PyTorch` & `Torchvision` (Deep Learning Framework)
* `Torchaudio` (Audio Signal Processing)
* `Scikit-learn` (Metrics & Evaluation)
* `Pandas` & `NumPy` (Data & Metadata Manipulation)
* `Matplotlib` / `Seaborn` (Visualization)

---
*Dikembangkan untuk kompetisi inovasi mahasiswa dan publikasi riset.*
