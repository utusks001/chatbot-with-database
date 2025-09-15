# 🤖 Gemini + Groq Chatbot — Multi-files + OCR + LangGraph + LangSmith

Chatbot berbasis LangChain Graph (LangGraph) dengan LangSmith untuk observabilitas dan LLMOps.
Mendukung upload banyak file (PDF, TXT, DOCX, PPTX, Images) dengan OCR.Space untuk ekstraksi teks dari gambar.
Dilengkapi pilihan LLM Provider:

Google Gemini 2.5 Flash

Groq LLaMA-3.3-70B Versatile

🚀 Fitur

Upload banyak file dokumen & gambar.

OCR otomatis untuk file gambar (JPEG, PNG, GIF, BMP, dll).

Penyimpanan embedding dengan FAISS (local vector store).

Orkestrasi query dengan LangGraph.

Observabilitas & tracing pipeline dengan LangSmith.

Pilihan model: Gemini (Google) atau LLaMA via Groq.

📦 Instalasi
1. Clone Repo
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

2. Buat Virtual Environment
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows

3. Install Requirements
pip install -r requirements.txt

🔑 Konfigurasi API Keys

Buat file .env di root project:

# API Key untuk OCR.Space
OCR_SPACE_API_KEY=your_ocr_space_api_key

# API Key untuk Google Gemini
GOOGLE_API_KEY=your_google_api_key

# API Key untuk Groq
GROQ_API_KEY=your_groq_api_key

# API Key untuk LangSmith (opsional, untuk observabilitas)
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true


OCR.Space API Key → https://ocr.space/OCRAPI

Google Gemini API Key → https://aistudio.google.com/apikey

Groq API Key → https://console.groq.com/keys

LangSmith API Key → https://smith.langchain.com

▶️ Menjalankan Aplikasi
streamlit run main.py


Akses di browser: http://localhost:8501

📂 Alur Pemakaian

Upload file dokumen / gambar dari sidebar.

Klik Build Vector Store untuk membuat index.

Pilih model LLM (Gemini atau Groq).

Masukkan pertanyaan di input box.

Chatbot akan menjalankan pipeline LangGraph (retrieval → reasoning → LLM).

Jika LangSmith aktif, semua eksekusi otomatis ditrace di dashboard LangSmith.

📊 Observabilitas dengan LangSmith

Jika LANGCHAIN_API_KEY di .env valid:

Semua eksekusi pipeline (GraphState, retrieval, LLM) otomatis terekam.

Bisa dimonitor di dashboard: https://smith.langchain.com
.

Berguna untuk debugging, evaluasi kualitas jawaban, dan kolaborasi tim.

🛠️ Deploy ke Streamlit Cloud

Push project ke GitHub.

Buka Streamlit Cloud
.

Deploy repository Anda.

Tambahkan Secrets di menu Settings → Secrets:

OCR_SPACE_API_KEY="your_ocr_space_api_key"

GOOGLE_API_KEY="your_google_api_key"

GROQ_API_KEY="your_groq_api_key"

LANGCHAIN_API_KEY="your_langsmith_api_key"

LANGCHAIN_TRACING_V2="true"


Jalankan, aplikasi langsung tersedia online.

📜 Lisensi

MIT License.
