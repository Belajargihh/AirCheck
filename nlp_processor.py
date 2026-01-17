"""
NLP Processor Module untuk Sistem Pakar Kualitas Udara
Modul ini berisi fungsi-fungsi preprocessing teks bahasa Indonesia
menggunakan teknik NLP: Case Folding, Tokenizing, Stopword Removal, dan Stemming
"""

import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi Sastrawi Stemmer dan Stopword Remover
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Daftar stopwords tambahan khusus untuk domain kualitas udara
CUSTOM_STOPWORDS = {
    'dan', 'atau', 'yang', 'di', 'ke', 'dari', 'ini', 'itu', 
    'dengan', 'untuk', 'pada', 'adalah', 'juga', 'sudah',
    'akan', 'bisa', 'ada', 'tidak', 'saya', 'kami', 'kita',
    'mereka', 'dia', 'ia', 'sangat', 'sekali', 'terasa', 'seperti'
}


def case_folding(text):
    """
    Mengubah semua karakter menjadi huruf kecil (lowercase)
    
    Args:
        text (str): Teks input
        
    Returns:
        str: Teks dalam huruf kecil
    """
    if not isinstance(text, str):
        return ""
    return text.lower()


def remove_punctuation(text):
    """
    Menghapus tanda baca dari teks
    
    Args:
        text (str): Teks input
        
    Returns:
        str: Teks tanpa tanda baca
    """
    # Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    """
    Memecah kalimat menjadi token (kata-kata)
    
    Args:
        text (str): Teks input
        
    Returns:
        list: Daftar kata hasil tokenisasi
    """
    return text.split()


def remove_stopwords(tokens):
    """
    Menghapus stopwords (kata-kata umum yang tidak bermakna)
    
    Args:
        tokens (list): Daftar kata
        
    Returns:
        list: Daftar kata tanpa stopwords
    """
    # Gabungkan tokens menjadi string untuk Sastrawi
    text = ' '.join(tokens)
    
    # Gunakan Sastrawi stopword remover
    text = stopword_remover.remove(text)
    
    # Filter tambahan dengan custom stopwords
    result_tokens = text.split()
    result_tokens = [word for word in result_tokens if word not in CUSTOM_STOPWORDS]
    
    return result_tokens


def stem(tokens):
    """
    Melakukan stemming (mengubah kata ke bentuk dasar)
    menggunakan algoritma Sastrawi untuk Bahasa Indonesia
    
    Args:
        tokens (list): Daftar kata
        
    Returns:
        list: Daftar kata dalam bentuk dasar
    """
    return [stemmer.stem(word) for word in tokens]


def preprocess(text):
    """
    Pipeline lengkap preprocessing NLP:
    1. Case Folding
    2. Remove Punctuation
    3. Tokenizing
    4. Stopword Removal
    5. Stemming
    
    Args:
        text (str): Teks input mentah
        
    Returns:
        str: Teks yang sudah diproses (cleaned text)
    """
    # Step 1: Case Folding
    text = case_folding(text)
    
    # Step 2: Remove Punctuation
    text = remove_punctuation(text)
    
    # Step 3: Tokenizing
    tokens = tokenize(text)
    
    # Step 4: Stopword Removal
    tokens = remove_stopwords(tokens)
    
    # Step 5: Stemming
    tokens = stem(tokens)
    
    # Gabungkan kembali menjadi string
    return ' '.join(tokens)


def preprocess_batch(texts):
    """
    Preprocessing untuk batch/list teks
    
    Args:
        texts (list): Daftar teks input
        
    Returns:
        list: Daftar teks yang sudah diproses
    """
    return [preprocess(text) for text in texts]


# Testing module jika dijalankan langsung
if __name__ == "__main__":
    # Contoh penggunaan
    test_texts = [
        "Udara terasa SEGAR, tidak ada bau, langit cerah!",
        "Bau asap menyengat, dada sesak, langit gelap...",
        "Sedikit berdebu, mata agak perih, jarak pandang normal."
    ]
    
    print("=" * 60)
    print("Testing NLP Processor Module")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        print(f"Processed: {preprocess(text)}")
        print("-" * 40)
