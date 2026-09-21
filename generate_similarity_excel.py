"""
Generate Excel File untuk Perhitungan Cosine Similarity Manual
Berdasarkan format referensi dari Similarity/manual.py

Format Excel:
- Kolom A: Dimensi (Dimensi 1, Dimensi 2, dst)
- Kolom B: Vector Input Query (Vector Embedding hasil dari input user)
- Kolom C dst: Vector Dataset (Vector dari dokumen training)
- Baris terakhir: Hasil Cosine Similarity

PERBAIKAN: Hanya menampilkan dimensi dengan nilai non-zero
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter
import os

# Import NLP processor
from nlp_processor import preprocess, preprocess_batch


def create_similarity_excel():
    """
    Membuat file Excel dengan perhitungan Cosine Similarity manual
    Format mengikuti referensi Similarity/manual.py
    """
    print("=" * 60)
    print("GENERATE EXCEL COSINE SIMILARITY MANUAL")
    print("Format: Dimensi x Vector (seperti referensi)")
    print("=" * 60)
    
    # 1. Load dan preprocess dataset
    print("\n[INFO] Memuat dataset...")
    df = pd.read_csv('dataset_udara.csv')
    
    # Preprocess semua data
    print("[INFO] Preprocessing teks...")
    df['processed_text'] = preprocess_batch(df['jawaban_user'].tolist())
    
    # 2. TF-IDF Vectorization untuk seluruh dataset
    print("[INFO] Menghitung TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1
    )
    
    # Fit vectorizer dengan semua data
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"[INFO] Jumlah fitur TF-IDF (dimensi): {len(feature_names)}")
    print(f"[INFO] Jumlah dokumen: {len(df)}")
    
    # 3. Input Query untuk testing
    test_query = "udara berkabut tebal, asap menyengat dan sesak napas"
    processed_query = preprocess(test_query)
    query_vector = vectorizer.transform([processed_query]).toarray()[0]
    
    print(f"\n[INFO] Query: {test_query}")
    print(f"[INFO] Query (processed): {processed_query}")
    
    # 4. Hitung Cosine Similarity dengan semua dokumen
    similarities = []
    for idx in range(len(df)):
        doc_vector = tfidf_matrix[idx].toarray()[0]
        sim = cosine_similarity([query_vector], [doc_vector])[0][0]
        similarities.append({
            'idx': idx,
            'text': df['jawaban_user'].iloc[idx],
            'label': df['label_kualitas'].iloc[idx],
            'similarity': sim,
            'vector': doc_vector
        })
    
    # Sort by similarity (descending)
    similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    
    # Ambil top 10 dokumen paling mirip
    top_k = 10
    top_docs = similarities[:top_k]
    
    print(f"\n[INFO] Top {top_k} dokumen dengan similarity tertinggi:")
    for i, doc in enumerate(top_docs):
        print(f"  {i+1}. Similarity: {doc['similarity']:.4f} | Label: {doc['label']} | {doc['text'][:50]}...")
    
    # ========================================
    # FILTER: Hanya dimensi dengan nilai NON-ZERO
    # ========================================
    
    # Gabungkan semua vektor (query + top docs)
    all_vectors = [query_vector] + [doc['vector'] for doc in top_docs]
    
    # Cari indeks dimensi yang memiliki setidaknya 1 nilai non-zero
    non_zero_dims = []
    for dim in range(len(feature_names)):
        has_nonzero = any(vec[dim] != 0 for vec in all_vectors)
        if has_nonzero:
            non_zero_dims.append(dim)
    
    print(f"\n[INFO] Dimensi dengan nilai non-zero: {len(non_zero_dims)} dari {len(feature_names)}")
    
    # 5. Buat Excel dengan format seperti referensi
    print("\n[INFO] Membuat file Excel...")
    wb = Workbook()
    ws = wb.active
    ws.title = "Similarity Manual"
    
    # Styling
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # ===== HEADER ROW =====
    headers = ['Subjudul', 'Vector Dataset']
    for i in range(top_k):
        headers.append(f'Vector Embeddings {i+1}')
    
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = thin_border
    
    # ===== DATA ROWS (Dimensi NON-ZERO saja) =====
    for row_idx, dim in enumerate(non_zero_dims):
        row = row_idx + 2  # Mulai dari baris 2 (setelah header)
        
        # Kolom A: Nama dimensi dengan nama fitur (kata)
        feature_name = feature_names[dim]
        cell = ws.cell(row=row, column=1, value=f"Dimensi {row_idx+1} ({feature_name})")
        cell.border = thin_border
        
        # Kolom B: Vector Query (Input)
        cell = ws.cell(row=row, column=2, value=query_vector[dim])
        cell.border = thin_border
        cell.number_format = '0.000000000'
        
        # Kolom C dst: Vector Dokumen
        for doc_idx, doc in enumerate(top_docs):
            cell = ws.cell(row=row, column=3 + doc_idx, value=doc['vector'][dim])
            cell.border = thin_border
            cell.number_format = '0.000000000'
    
    # ===== BARIS COSINE SIMILARITY =====
    sim_row = len(non_zero_dims) + 3
    
    # Label baris
    cell = ws.cell(row=sim_row, column=1, value="Cosine Similarity")
    cell.font = Font(bold=True)
    cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    cell.border = thin_border
    
    # Similarity dengan diri sendiri = 1
    cell = ws.cell(row=sim_row, column=2, value=1.0)
    cell.font = Font(bold=True)
    cell.fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
    cell.border = thin_border
    cell.number_format = '0.0000'
    
    # Similarity dengan setiap dokumen
    for doc_idx, doc in enumerate(top_docs):
        cell = ws.cell(row=sim_row, column=3 + doc_idx, value=doc['similarity'])
        cell.font = Font(bold=True)
        cell.border = thin_border
        cell.number_format = '0.0000'
        
        # Color coding berdasarkan similarity
        if doc['similarity'] >= 0.7:
            cell.fill = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
        elif doc['similarity'] >= 0.4:
            cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
        else:
            cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    
    # ===== BARIS FORMULA EXCEL =====
    formula_row = sim_row + 2
    ws.cell(row=formula_row, column=1, value="Formula Excel:").font = Font(bold=True)
    
    last_dim_row = len(non_zero_dims) + 1
    example_formula = f"=SUMPRODUCT($B$2:$B${last_dim_row},C2:C{last_dim_row})/(SQRT(SUMSQ($B$2:$B${last_dim_row}))*SQRT(SUMSQ(C2:C{last_dim_row})))"
    ws.cell(row=formula_row + 1, column=1, value=example_formula)
    
    # ===== BARIS INFO DOKUMEN =====
    info_row = formula_row + 4
    ws.cell(row=info_row, column=1, value="Informasi Dokumen:").font = Font(bold=True, size=12)
    
    # Header info
    ws.cell(row=info_row + 1, column=1, value="Dokumen").font = Font(bold=True)
    ws.cell(row=info_row + 1, column=2, value="Label").font = Font(bold=True)
    ws.cell(row=info_row + 1, column=3, value="Similarity").font = Font(bold=True)
    ws.cell(row=info_row + 1, column=4, value="Teks").font = Font(bold=True)
    
    for col in range(1, 5):
        ws.cell(row=info_row + 1, column=col).fill = header_fill
        ws.cell(row=info_row + 1, column=col).font = header_font
        ws.cell(row=info_row + 1, column=col).border = thin_border
    
    # Data info dokumen
    for doc_idx, doc in enumerate(top_docs):
        row = info_row + 2 + doc_idx
        ws.cell(row=row, column=1, value=f"Vector Embeddings {doc_idx+1}").border = thin_border
        ws.cell(row=row, column=2, value=doc['label']).border = thin_border
        cell = ws.cell(row=row, column=3, value=doc['similarity'])
        cell.border = thin_border
        cell.number_format = '0.0000'
        ws.cell(row=row, column=4, value=doc['text'][:80] + "..." if len(doc['text']) > 80 else doc['text']).border = thin_border
    
    # ===== QUERY INFO =====
    query_row = info_row + 2 + top_k + 2
    ws.cell(row=query_row, column=1, value="Query Input:").font = Font(bold=True)
    ws.cell(row=query_row, column=2, value=test_query)
    ws.cell(row=query_row + 1, column=1, value="Query Processed:").font = Font(bold=True)
    ws.cell(row=query_row + 1, column=2, value=processed_query)
    
    # Adjust column widths
    ws.column_dimensions['A'].width = 30
    ws.column_dimensions['B'].width = 18
    for col in range(3, top_k + 3):
        ws.column_dimensions[get_column_letter(col)].width = 18
    
    # ========== SHEET 2: Matrix Similarity Lengkap ==========
    ws2 = wb.create_sheet("Matrix Similarity")
    
    # Header
    ws2.cell(row=1, column=1, value="Dokumen").fill = header_fill
    ws2.cell(row=1, column=1).font = header_font
    ws2.cell(row=1, column=2, value="Label").fill = header_fill
    ws2.cell(row=1, column=2).font = header_font
    ws2.cell(row=1, column=3, value="Cosine Similarity").fill = header_fill
    ws2.cell(row=1, column=3).font = header_font
    ws2.cell(row=1, column=4, value="Teks Original").fill = header_fill
    ws2.cell(row=1, column=4).font = header_font
    
    for col in range(1, 5):
        ws2.cell(row=1, column=col).border = thin_border
    
    # Semua dokumen sorted by similarity
    for idx, doc in enumerate(similarities):
        row = idx + 2
        ws2.cell(row=row, column=1, value=f"Doc_{doc['idx']+1}").border = thin_border
        ws2.cell(row=row, column=2, value=doc['label']).border = thin_border
        cell = ws2.cell(row=row, column=3, value=doc['similarity'])
        cell.border = thin_border
        cell.number_format = '0.0000'
        ws2.cell(row=row, column=4, value=doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']).border = thin_border
    
    ws2.column_dimensions['A'].width = 10
    ws2.column_dimensions['B'].width = 15
    ws2.column_dimensions['C'].width = 18
    ws2.column_dimensions['D'].width = 80
    
    # 6. Save file
    output_path = 'Hasil_Similarity_Manual_v2.xlsx'
    wb.save(output_path)
    
    print(f"\n[SUCCESS] File Excel berhasil dibuat: {output_path}")
    print("\n" + "=" * 60)
    print("FORMAT FILE EXCEL (Seperti Referensi):")
    print("=" * 60)
    print("Sheet 1: Similarity Manual")
    print(f"  - Hanya menampilkan {len(non_zero_dims)} dimensi NON-ZERO")
    print("  - Kolom A: Dimensi + Nama Fitur (kata)")
    print("  - Kolom B: Vector Dataset (Query/Input)")
    print("  - Kolom C-L: Vector Embeddings 1-10 (Top Dokumen)")
    print("  - Baris terakhir: Cosine Similarity")
    print("\nSheet 2: Matrix Similarity")
    print("  - Daftar semua dokumen dengan similarity score")
    print("=" * 60)
    
    return output_path


if __name__ == "__main__":
    create_similarity_excel()
