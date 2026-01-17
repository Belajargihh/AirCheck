# 🌬️ AirCheck - Sistem Pakar Kualitas Udara

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

AirCheck adalah aplikasi web **Sistem Pakar berbasis AI** untuk menganalisis dan memprediksi kualitas udara berdasarkan kondisi lingkungan yang diinput oleh pengguna. Aplikasi ini menggunakan **Natural Language Processing (NLP)** dan algoritma **Naive Bayes** untuk memberikan prediksi akurat beserta rekomendasi kesehatan.

## ✨ Fitur Utama

- 🔍 **Analisis Kualitas Udara** - Prediksi kualitas udara (Baik/Sedang/Tidak Sehat) berdasarkan deskripsi kondisi lingkungan
- 🗣️ **Input Suara** - Dukungan Speech-to-Text untuk input menggunakan suara (Bahasa Indonesia)
- 🤖 **NLP Processing** - Pemrosesan bahasa alami dengan stemming Bahasa Indonesia (Sastrawi)
- 💡 **Rekomendasi Kesehatan** - Saran kesehatan yang relevan berdasarkan hasil prediksi
- 📊 **Probabilitas Prediksi** - Menampilkan tingkat kepercayaan untuk setiap kategori
- 🌙 **Dark/Light Mode** - Tampilan responsif dengan tema gelap dan terang

## 🛠️ Teknologi yang Digunakan

| Komponen | Teknologi |
|----------|-----------|
| Backend | Flask (Python) |
| Machine Learning | Scikit-learn (Naive Bayes) |
| NLP | NLTK, Sastrawi |
| Data Processing | Pandas, NumPy |
| Frontend | HTML, CSS, JavaScript |

## 📁 Struktur Proyek

```
AirCheck/
├── app.py                 # Server Flask utama
├── model_training.py      # Script untuk training model
├── nlp_processor.py       # Modul preprocessing NLP
├── dataset_udara.csv      # Dataset training
├── model_nb.pkl           # Model Naive Bayes (hasil training)
├── vectorizer.pkl         # TF-IDF Vectorizer (hasil training)
├── requirements.txt       # Dependencies Python
├── templates/             # Template HTML
│   ├── index.html         # Halaman beranda
│   ├── analisis.html      # Halaman form analisis
│   ├── hasil.html         # Halaman hasil prediksi
│   └── tentang.html       # Halaman tentang
└── static/                # File statis (CSS, JS, gambar)
```

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/Belajargihh/AirCheck.git
cd AirCheck
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Training Model (Opsional)

Jika ingin melatih ulang model:

```bash
python model_training.py
```

### 4. Jalankan Aplikasi

```bash
python app.py
```

### 5. Akses Aplikasi

Buka browser dan akses: **http://localhost:5000**

## 📖 Cara Penggunaan

1. Buka halaman **Analisis** dari menu navigasi
2. Jawab pertanyaan tentang kondisi udara di sekitar Anda:
   - Kondisi kabut/asap
   - Bau di udara
   - Kondisi pernapasan
3. Tambahkan deskripsi tambahan jika diperlukan (bisa menggunakan input suara)
4. Klik **"Analisis Sekarang"**
5. Lihat hasil prediksi dan rekomendasi kesehatan

## 🎯 Kategori Kualitas Udara

| Kategori | Deskripsi |
|----------|-----------|
| ✅ **Baik** | Udara bersih dan sehat untuk semua aktivitas |
| ⚠️ **Sedang** | Perlu sedikit perhatian, terutama bagi yang sensitif |
| 🚨 **Tidak Sehat** | Berbahaya, hindari aktivitas luar ruangan |

## 👥 Tim Pengembang

Dikembangkan oleh mahasiswa **UNAMIN**:

- Desyia Tatuhey
- Nurosida Sebualamo
- Yulio Delvin Kambu

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ for a better air quality awareness
</p>
