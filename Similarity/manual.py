# RAG application built on gemini with LangChain Memory
# Memuat semua library yang diperlukan
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pandas as pd

# Memuat variabel lingkungan dari file .env
load_dotenv()
folder_path = "source_data"  # Ganti dengan path folder Anda

# Mengambil semua file PDF dalam folder
pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

all_data = []
for pdf_file in pdf_files:
    try:
        file_path = os.path.join(folder_path, pdf_file)
        loader = PyPDFLoader(file_path)
        data = loader.load() 
        all_data.extend(data)
        print(f"Berhasil memuat {len(data)} halaman dari {pdf_file}")
    except Exception as e:
        print(f"Kesalahan saat memuat PDF {pdf_file}: {e}")

if all_data:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(all_data)
    print(f"Jumlah total bagian dokumen: {len(docs)}")
else:
    print("Tidak ada data untuk diproses.")

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/bert-base-nli-max-tokens")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="data")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    print("Penyimpanan vektor berhasil dibuat dan disimpan.")
except Exception as e:
    print(f"Kesalahan saat menginisialisasi embeddings atau penyimpanan vektor: {e}")

# Menginisialisasi memory untuk percakapan
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None)

    # Menggunakan LangChain Memory pada prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "Anda adalah CelerySens, sebuah Asisten tanaman seledri yang siap membantu Anda merawat tanaman seledri dengan solusi yang tepat dan efektif. Tugas Anda adalah mendiagnosa gejala yang disebutkan dengan cepat dan jelas, termasuk menyebutkan nama ilmiah penyakit jika relevan sesuai dataset"
                "Anda menjelaskan penyebab masalah tanaman dengan cara yang mudah dipahami."
                "Jawaban Anda harus ringkas, langsung ke inti dan tidak butuh data berupa gambar."
                "Anda berkomunikasi dengan ramah dan memberikan motivasi agar Anda semangat merawat tanaman Anda."
                "Anda akan mengingat detail yang Anda sebutkan di percakapan kita."
                "Anda dikembangkan oleh Tirta Romadhon Cipta Saputra dan Billy Jes, mahasiswa Universitas Muhammadiyah Sorong Program Studi Teknik Informatika."
                "Sumber data yang digunakan berasal dari seorang pakar bernama Han Aldianova, mahasiswa Universitas Brawijaya yang telah tersertifikasi BNSP di bidang Pertanian."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )
    
    # Mengintegrasikan memory ke dalam LLMChain
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    # Mengintegrasikan conversation chain ke dalam RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("RAG chain dengan memory berhasil diinisialisasi.")
except Exception as e:
    print(f"Kesalahan saat menginisialisasi RAG chain: {e}")

# Menguji query dan menampilkan similarity
try:
    query = "Seledri saya terdapat bercak cokelat kemerahan kecil, tersebar di daun seledri dan menyebabkan layu, apa nama penyakitnya dan bagaimana solusinya?"
    
    # Mendapatkan dokumen yang relevan
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Membuat respons dari model menggunakan memory
    response = conversation_chain.invoke({"input": query})  # Tidak ada ["answer"], langsung cetak respons
    print("Respons:", response)  # Menggunakan respons langsung
    
    # Menghitung embedding untuk query
    query_embedding = embeddings.embed_query(query)
    similarities = []
    
    for doc in retrieved_docs:
        doc_embedding = embeddings.embed_query(doc.page_content)
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        similarities.append((doc.page_content, similarity))
    
    # Mengurutkan hasil berdasarkan similarity
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    print("Hasil pencarian dengan tingkat kemiripan:")
    for content, score in similarities:
        print(f"Tingkat kemiripan: {score:.4f}")
    
    # Menyusun data untuk ditulis ke Excel
    vector_embeddings = np.array([query_embedding])  # Misal embedding hasil model LLM
    dataset_vectors = np.array([embeddings.embed_query(doc[0]) for doc in similarities])

    # Menyusun data dalam bentuk dataframe
    columns = ['Subjudul', 'Vector Dataset'] + [f'Vector Embeddings {i+1}' for i in range(len(similarities))]
    rows = []

    # Menyusun dimensi dan vektor ke dalam bentuk yang sesuai
    for dim in range(len(vector_embeddings[0])):  # Misalkan dimensi 768
        row = ['Dimensi ' + str(dim+1), vector_embeddings[0][dim]]  # Vector Hasil Embedding
        for doc_vector in dataset_vectors:
            row.append(doc_vector[dim])  # Vector Dataset
        rows.append(row)

    # Membuat DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Menulis DataFrame ke Excel
    output_file = "manual.xlsx"
    df.to_excel(output_file, index=False)

    print(f"File Excel disimpan di {output_file}")
    
except Exception as e:
    print(f"Kesalahan saat memproses query: {e}")

# Mengekspor objek penting untuk digunakan di modul lain
_all_ = ["retriever", "embeddings", "conversation_chain"]

if not (retriever and embeddings and conversation_chain):
    raise RuntimeError("Gagal menginisialisasi retriever, embeddings, atau conversation_chain.")

import matplotlib.pyplot as plt
import numpy as np

# Ekstraksi data dari hasil similarity
similarity_scores = [score for _, score in similarities]
documents = [f"Doc {i+1}" for i in range(len(similarities))]

# Membuat grafik
plt.figure(figsize=(10, 6))
plt.barh(documents, similarity_scores, color='skyblue')
plt.xlabel('Similarity Score', fontsize=12)
plt.ylabel('Documents', fontsize=12)
plt.title('Similarity Scores for Retrieved Documents', fontsize=14)
plt.gca().invert_yaxis()  # Membalikkan sumbu Y agar dokumen dengan similarity tertinggi berada di atas
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Menambahkan nilai pada grafik
for i, v in enumerate(similarity_scores):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=10)

# Menampilkan grafik
plt.tight_layout()
plt.show()
