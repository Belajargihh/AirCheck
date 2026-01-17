"""
Model Training Script untuk Sistem Pakar Kualitas Udara
Script ini melatih model Naive Bayes menggunakan dataset kualitas udara
dengan preprocessing NLP dan TF-IDF Vectorization
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

# Import modul NLP kita
from nlp_processor import preprocess, preprocess_batch


def load_dataset(filepath='dataset_udara.csv'):
    """
    Memuat dataset dari file CSV
    
    Args:
        filepath (str): Path ke file CSV
        
    Returns:
        pandas.DataFrame: Dataset yang dimuat
    """
    print(f"[INFO] Memuat dataset dari {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset berhasil dimuat: {len(df)} sampel")
    print(f"[INFO] Distribusi label:\n{df['label_kualitas'].value_counts()}")
    return df


def preprocess_dataset(df):
    """
    Melakukan preprocessing pada seluruh dataset
    
    Args:
        df (pandas.DataFrame): Dataset input
        
    Returns:
        pandas.DataFrame: Dataset dengan kolom teks yang sudah diproses
    """
    print("\n[INFO] Memulai preprocessing NLP...")
    df['processed_text'] = preprocess_batch(df['jawaban_user'].tolist())
    print("[INFO] Preprocessing selesai!")
    
    # Tampilkan contoh hasil preprocessing
    print("\n[SAMPLE] Contoh hasil preprocessing:")
    for i in range(min(3, len(df))):
        print(f"  Original: {df['jawaban_user'].iloc[i][:50]}...")
        print(f"  Processed: {df['processed_text'].iloc[i]}")
        print()
    
    return df


def train_model(df):
    """
    Melatih model Naive Bayes dengan TF-IDF Vectorization
    
    Args:
        df (pandas.DataFrame): Dataset yang sudah dipreprocess
        
    Returns:
        tuple: (model, vectorizer, accuracy)
    """
    print("\n" + "=" * 60)
    print("TRAINING MODEL NAIVE BAYES")
    print("=" * 60)
    
    # Pisahkan fitur dan label
    X = df['processed_text']
    y = df['label_kualitas']
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Data Training: {len(X_train)} sampel")
    print(f"[INFO] Data Testing: {len(X_test)} sampel")
    
    # TF-IDF Vectorization
    print("\n[INFO] Melakukan TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Maksimal 1000 fitur
        ngram_range=(1, 2),  # Unigram dan Bigram
        min_df=1  # Minimal muncul di 1 dokumen
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"[INFO] Jumlah fitur TF-IDF: {len(vectorizer.get_feature_names_out())}")
    
    # Training Multinomial Naive Bayes
    print("\n[INFO] Training Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=1.0)  # Laplace smoothing
    model.fit(X_train_tfidf, y_train)
    print("[INFO] Training selesai!")
    
    # Evaluasi model
    print("\n" + "-" * 60)
    print("EVALUASI MODEL")
    print("-" * 60)
    
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n[RESULT] Akurasi: {accuracy * 100:.2f}%")
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n[RESULT] Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    labels = model.classes_
    print(f"Labels: {labels}")
    print(cm)
    
    return model, vectorizer, accuracy


def save_model(model, vectorizer, model_path='model_nb.pkl', vectorizer_path='vectorizer.pkl'):
    """
    Menyimpan model dan vectorizer ke file pickle
    
    Args:
        model: Model Naive Bayes yang sudah ditraining
        vectorizer: TF-IDF Vectorizer yang sudah di-fit
        model_path (str): Path untuk menyimpan model
        vectorizer_path (str): Path untuk menyimpan vectorizer
    """
    print("\n[INFO] Menyimpan model dan vectorizer...")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[INFO] Model disimpan ke: {model_path}")
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Vectorizer disimpan ke: {vectorizer_path}")
    
    print("\n[SUCCESS] Model dan vectorizer berhasil disimpan!")


def test_prediction(model, vectorizer):
    """
    Menguji prediksi model dengan contoh input
    
    Args:
        model: Model Naive Bayes
        vectorizer: TF-IDF Vectorizer
    """
    print("\n" + "=" * 60)
    print("TEST PREDIKSI")
    print("=" * 60)
    
    test_inputs = [
        "Udara sangat segar dan bersih, langit cerah",
        "Ada sedikit kabut dan debu di udara",
        "Asap tebal menyengat, susah napas, mata perih parah"
    ]
    
    for text in test_inputs:
        # Preprocess
        processed = preprocess(text)
        # Vectorize
        text_tfidf = vectorizer.transform([processed])
        # Predict
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0]
        
        print(f"\nInput: {text}")
        print(f"Processed: {processed}")
        print(f"Prediksi: {prediction}")
        print(f"Probabilitas: {dict(zip(model.classes_, [f'{p:.2%}' for p in probability]))}")


def main():
    """
    Fungsi utama untuk menjalankan pipeline training
    """
    print("=" * 60)
    print("SISTEM PAKAR KUALITAS UDARA - MODEL TRAINING")
    print("Metode: NLP + Naive Bayes Classification")
    print("=" * 60)
    
    # 1. Load dataset
    df = load_dataset()
    
    # 2. Preprocessing
    df = preprocess_dataset(df)
    
    # 3. Training
    model, vectorizer, accuracy = train_model(df)
    
    # 4. Save model
    save_model(model, vectorizer)
    
    # 5. Test prediction
    test_prediction(model, vectorizer)
    
    print("\n" + "=" * 60)
    print(f"TRAINING SELESAI! Akurasi: {accuracy * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
