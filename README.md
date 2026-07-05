# 🌊 Flood Detection

Aplikasi klasifikasi citra berbasis deep learning untuk mengidentifikasi gambar **Flood** dan **Non Flood** menggunakan arsitektur **EfficientNetV2B0**.

Project ini dikembangkan sebagai aplikasi berbasis web menggunakan **Streamlit**, sehingga pengguna dapat mengunggah gambar dan memperoleh hasil klasifikasi beserta confidence score dari model.

---

## ✨ Fitur

- Upload gambar JPG, JPEG, atau PNG
- Preview gambar
- Prediksi Flood / Non Flood
- Confidence score
- Tampilan aplikasi berbasis Streamlit
- Evaluasi model menggunakan:
  - Confusion Matrix
  - Precision
  - Recall
  - F1-score
  - Classification Report

---

## 🧠 Model

Model menggunakan:

- **Architecture:** EfficientNetV2B0
- **Input Size:** 224 × 224 pixels
- **Classification Type:** Binary Classification
- **Classes:**
  - Flood
  - Non Flood

Model dikembangkan menggunakan pendekatan transfer learning dengan EfficientNetV2B0 sebagai feature extractor.

---

## 📊 Hasil Evaluasi

Evaluasi dilakukan pada internal validation split sebanyak **2.608 gambar**.

### Classification Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Flood | 0.9989 | 0.9989 | 0.9989 |
| Non Flood | 0.9974 | 0.9974 | 0.9974 |

**Accuracy: 99.85%**

### Confusion Matrix

```text
[[1838    2]
 [   2  766]]
```

File hasil evaluasi tersedia pada folder:

```text
outputs/
├── classification_report.txt
└── confusion_matrix.png
```

> Catatan: hasil evaluasi di atas berasal dari internal validation split pada dataset yang sama. Nilai tersebut tidak otomatis merepresentasikan performa pada seluruh gambar dunia nyata.

---

## 🛠️ Teknologi

- Python
- TensorFlow
- Streamlit
- NumPy
- Pillow
- Matplotlib
- Scikit-learn
- EfficientNetV2B0

---

## 📁 Struktur Project

```text
FloodVision/
├── app.py
├── train_model.py
├── evaluate_model.py
├── requirements.txt
├── README.md
├── model/
│   └── efficientnetv2_flood_model.keras
├── outputs/
│   ├── classification_report.txt
│   └── confusion_matrix.png
├── sample_images/
├── utils/
└── assets/
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Clone repository

```bash
git clone https://github.com/adamfirdausyah10-sys/flood-detection.git
```

### 2. Masuk ke folder project

```bash
cd flood-detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
python -m streamlit run app.py
```

Setelah itu aplikasi akan terbuka melalui browser.

---

## 🔬 Menjalankan Evaluasi Model

Untuk menjalankan evaluasi:

```bash
python evaluate_model.py
```

Hasil evaluasi akan disimpan pada folder:

```text
outputs/
```

---

## ⚠️ Keterbatasan

Model merupakan binary image classifier yang hanya membedakan:

- Flood
- Non Flood

Model tidak melakukan:

- Object detection
- Bounding box prediction
- Segmentasi area banjir
- Estimasi kedalaman air

Hasil prediksi juga dapat mengalami kesalahan pada gambar yang berbeda secara signifikan dari distribusi data training.

---

## 📌 Status Project

Project masih dalam tahap pengembangan.

Pengembangan selanjutnya dapat mencakup:

- Pengujian tambahan menggunakan gambar di luar dataset
- Analisis kesalahan klasifikasi
- Peningkatan antarmuka aplikasi
- Explainability menggunakan Grad-CAM

---

## 👥 Project

Project dikembangkan untuk kebutuhan pembelajaran dan tugas kelompok klasifikasi citra menggunakan deep learning.