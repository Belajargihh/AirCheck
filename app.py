"""
Flask Web Application untuk Sistem Pakar Kualitas Udara
Server utama yang menghubungkan frontend dengan model AI
"""

from flask import Flask, render_template, request, redirect, url_for
import pickle
import os
import random

# Import modul NLP
from nlp_processor import preprocess

app = Flask(__name__)

# Path ke model dan vectorizer
MODEL_PATH = 'model_nb.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

# Load model dan vectorizer saat startup
model = None
vectorizer = None


def load_model():
    """
    Memuat model dan vectorizer dari file pickle
    """
    global model, vectorizer
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print("[INFO] Model dan vectorizer berhasil dimuat!")
        return True
    else:
        print("[WARNING] Model atau vectorizer tidak ditemukan!")
        print("[INFO] Jalankan 'python model_training.py' terlebih dahulu.")
        return False


def get_saran(kualitas):
    """
    Memberikan saran berdasarkan kualitas udara dengan variasi random
    
    Args:
        kualitas (str): Hasil prediksi kualitas udara
        
    Returns:
        dict: Saran dan informasi terkait (dengan random pick)
    """
    saran_pool = {
        "Baik": {
            "warna": "#28a745",  # Hijau
            "icon": "✅",
            "deskripsi": "Kualitas udara sangat baik dan sehat untuk semua aktivitas.",
            "saran_all": [
                "Aman untuk beraktivitas di luar ruangan sepanjang hari",
                "Cocok untuk olahraga dan kegiatan outdoor",
                "Nikmati udara segar dengan membuka jendela rumah",
                "Tidak perlu menggunakan masker",
                "Waktu yang tepat untuk jogging atau lari pagi",
                "Ideal untuk bersepeda di taman atau jalanan",
                "Bagus untuk piknik bersama keluarga",
                "Aman untuk anak-anak bermain di luar",
                "Cocok untuk senam pagi di area terbuka",
                "Manfaatkan untuk menjemur pakaian",
                "Baik untuk berkebun atau aktivitas halaman",
                "Udara segar baik untuk kesehatan mental",
                "Sempurna untuk hiking atau mendaki",
                "Aman untuk lansia beraktivitas outdoor",
                "Waktu tepat untuk olahraga bersama komunitas"
            ]
        },
        "Sedang": {
            "warna": "#ffc107",  # Kuning
            "icon": "⚠️",
            "deskripsi": "Kualitas udara cukup baik, namun perlu sedikit perhatian.",
            "saran_all": [
                "Kurangi aktivitas berat di luar ruangan",
                "Gunakan masker jika sensitif terhadap polusi",
                "Perhatikan gejala seperti batuk atau mata perih",
                "Tutup jendela jika polusi meningkat",
                "Batasi waktu olahraga outdoor maksimal 1 jam",
                "Hindari area dengan lalu lintas padat",
                "Minum air putih lebih banyak dari biasanya",
                "Pertimbangkan olahraga indoor hari ini",
                "Waspadai jika memiliki riwayat asma",
                "Kurangi aktivitas fisik saat siang hari",
                "Pantau kondisi udara secara berkala",
                "Siapkan masker untuk berjaga-jaga",
                "Hindari area konstruksi atau pembangunan",
                "Batasi aktivitas outdoor untuk anak kecil",
                "Pertimbangkan untuk bekerja dari rumah"
            ]
        },
        "Tidak Sehat": {
            "warna": "#dc3545",  # Merah
            "icon": "🚨",
            "deskripsi": "Kualitas udara buruk dan berbahaya bagi kesehatan.",
            "saran_all": [
                "Hindari aktivitas di luar ruangan",
                "WAJIB gunakan masker N95 jika harus keluar",
                "Tutup semua jendela dan pintu rapat-rapat",
                "Gunakan air purifier jika tersedia",
                "Segera ke dokter jika mengalami sesak napas",
                "Jangan biarkan anak-anak bermain di luar",
                "Batasi aktivitas fisik seminimal mungkin",
                "Nyalakan AC dengan mode recirculate",
                "Siapkan obat-obatan pernapasan darurat",
                "Hindari memasak dengan pembakaran terbuka",
                "Basahi kain dan letakkan di ventilasi",
                "Minum air hangat untuk meredakan tenggorokan",
                "Tetap di dalam ruangan sebisa mungkin",
                "Hubungi layanan kesehatan jika gejala memburuk",
                "Evakuasi jika kondisi sangat parah",
                "Pantau informasi dari BMKG dan Dinkes",
                "Jangan membakar sampah atau apapun"
            ]
        }
    }
    
    pool = saran_pool.get(kualitas, saran_pool["Sedang"])
    
    # Random pick 4-5 saran dari pool
    num_saran = random.randint(4, 5)
    selected_saran = random.sample(pool["saran_all"], min(num_saran, len(pool["saran_all"])))
    
    return {
        "warna": pool["warna"],
        "icon": pool["icon"],
        "deskripsi": pool["deskripsi"],
        "saran": selected_saran
    }


@app.route('/')
def index():
    """
    Halaman Beranda
    """
    return render_template('index.html')


@app.route('/analisis')
def analisis():
    """
    Halaman Form Analisis
    """
    return render_template('analisis.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk melakukan prediksi kualitas udara (returns JSON)
    """
    from flask import jsonify
    
    if model is None or vectorizer is None:
        return jsonify({
            'error': True,
            'message': 'Model belum dimuat. Jalankan training terlebih dahulu.'
        })
    
    # Ambil input dari form
    kondisi_kabut = request.form.get('kondisi_kabut', '')
    kondisi_bau = request.form.get('kondisi_bau', '')
    kondisi_pernapasan = request.form.get('kondisi_pernapasan', '')
    deskripsi_tambahan = request.form.get('deskripsi', '')
    
    # Gabungkan semua input menjadi satu teks
    input_text = f"{kondisi_kabut} {kondisi_bau} {kondisi_pernapasan} {deskripsi_tambahan}"
    
    if not input_text.strip():
        return jsonify({
            'error': True,
            'message': 'Silakan isi minimal satu pertanyaan.'
        })
    
    # Preprocessing
    processed_text = preprocess(input_text)
    
    # Vectorize
    text_tfidf = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    # Buat dictionary probabilitas
    prob_dict = dict(zip(model.classes_, [round(p * 100, 1) for p in probability]))
    
    # Dapatkan saran
    saran_info = get_saran(prediction)
    
    return jsonify({
        'error': False,
        'kualitas': prediction,
        'input_text': input_text,
        'processed_text': processed_text,
        'probabilitas': prob_dict,
        'saran': saran_info['saran'],
        'deskripsi': saran_info['deskripsi']
    })


@app.route('/tentang')
def tentang():
    """
    Halaman Tentang
    """
    return render_template('tentang.html')


if __name__ == '__main__':
    # Coba load model saat startup
    if not load_model():
        print("\n" + "=" * 60)
        print("PERINGATAN: Model belum tersedia!")
        print("Jalankan perintah berikut untuk melatih model:")
        print("  python model_training.py")
        print("=" * 60 + "\n")
    
    # Jalankan Flask server
    print("\n[INFO] Memulai server Flask...")
    print("[INFO] Buka browser dan akses: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
