# app-langchain.py

import streamlit as st
import os, io, pandas as pd, requests
from pathlib import Path
from dotenv import load_dotenv, set_key
import toml
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import faiss
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ====== Load .env ======
dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# ====== Sidebar: API Key ======
st.sidebar.header("🔑 Konfigurasi API Key")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = st.text_input(
        "Masukkan GOOGLE_API_KEY (https://aistudio.google.com/apikey)",
        type="password"
    )
    if GOOGLE_API_KEY:
        st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        st.success("✅ API Key berhasil dimasukkan")
        save_choice = st.radio("Simpan key ke mana?", ["Jangan simpan", ".env", "secrets.toml"])
        if st.button("💾 Simpan API Key"):
            if save_choice == ".env":
                set_key(dotenv_path, "GOOGLE_API_KEY", GOOGLE_API_KEY)
                st.success("✅ API Key disimpan ke .env")
            elif save_choice == "secrets.toml":
                secrets_path = Path(".streamlit/secrets.toml")
                secrets_path.parent.mkdir(exist_ok=True)
                secrets = {}
                if secrets_path.exists():
                    secrets = toml.load(secrets_path)
                secrets["GOOGLE_API_KEY"] = GOOGLE_API_KEY
                with open(secrets_path, "w") as f:
                    toml.dump(secrets, f)
                st.success("✅ API Key disimpan ke .streamlit/secrets.toml")
else:
    st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    st.success("✅ GOOGLE_API_KEY berhasil dimuat")

# ====== Helper Functions ======
def safe_describe(df: pd.DataFrame):
    try:
        return df.describe(include="all").transpose()
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def df_info_text(df: pd.DataFrame):
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

def detect_data_types(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numeric_cols

# ====== OCR.Space Helper ======
def ocr_image(file):
    OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")
    if not OCR_SPACE_API_KEY:
        return ""
    try:
        file.seek(0)
        resp = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": file},
            data={"apikey": OCR_SPACE_API_KEY, "language": "eng"}
        )
        data = resp.json()
        if data.get("ParsedResults"):
            return data["ParsedResults"][0].get("ParsedText", "")
    except Exception as e:
        st.warning(f"⚠️ OCR.Space error: {e}")
    return ""

# ====== Build Vectorstore using sentence-transformers + FAISS ======
def build_vectorstore(files):
    docs = []
    for file in files:
        name = file.name
        text = ""
        try:
            if name.lower().endswith(".txt"):
                text = file.read().decode("utf-8")
            elif name.lower().endswith((".pdf", ".csv", ".xlsx", ".xls", ".docx", ".pptx")):
                text = f"[{name}] content placeholder"
            elif name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                text = ocr_image(file)
            else:
                text = file.read().decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": name}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # HuggingFace embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = [model.encode(doc.page_content) for doc in split_docs]

    # Build FAISS index
    dim = len(embeddings[0]) if embeddings else 384
    index = faiss.IndexFlatL2(dim)
    if embeddings:
        import numpy as np
        index.add(np.array(embeddings).astype("float32"))

    # Attach documents
    vectorstore = {"index": index, "docs": split_docs}
    return vectorstore

# ====== Main App ======
st.set_page_config(page_title="Data & Document RAG Chatbot", layout="wide")
st.title("📊🤖 Chatbot Analisis Data & Dokumen (RAG + HuggingFace Fallback)")

tab1, tab2 = st.tabs(["📊 Data Analysis", "📑 RAG Advanced"])

# ----- Tab 1: Data Analysis -----
with tab1:
    uploaded_file = st.file_uploader("Upload file Excel/CSV untuk analisa data", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.session_state.dfs = {"CSV": df}
        else:
            xls = pd.ExcelFile(uploaded_file)
            st.session_state.dfs = {sheet: pd.read_excel(uploaded_file, sheet_name=sheet) for sheet in xls.sheet_names}

        st.subheader("📑 Pilih Sheet")
        sheet_names = list(st.session_state.dfs.keys())
        selected_sheets = st.multiselect("Sheet Aktif", sheet_names, default=sheet_names[:1])
        if not selected_sheets:
            st.warning("Pilih minimal satu sheet untuk analisis.")
            st.stop()

        if len(selected_sheets) == 1:
            df = st.session_state.dfs[selected_sheets[0]]
        else:
            df_list = []
            for s in selected_sheets:
                temp = st.session_state.dfs[s].copy()
                temp["SheetName"] = s
                df_list.append(temp)
            df = pd.concat(df_list, ignore_index=True)

        st.markdown(f"### 📄 Analisa: {uploaded_file.name}")
        st.dataframe(df.head(10))
        categorical_cols, numeric_cols = detect_data_types(df)
        st.write(f"Kolom Numerik: {numeric_cols}")
        st.write(f"Kolom Kategorikal: {categorical_cols}")
        st.text(df_info_text(df))
        st.write(f"**Data shape:** {df.shape}")
        st.dataframe(safe_describe(df))

        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            st.write("**Correlation Heatmap**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_query = st.chat_input("Tanyakan sesuatu tentang data...")
        if user_query:
            st.chat_message("user").markdown(user_query)
            st.session_state.chat_history.append(("user", user_query))
            response = f"🤖 (HuggingFace fallback) Jawaban untuk: {user_query[:50]}..."
            st.chat_message("assistant").markdown(response)
            st.session_state.chat_history.append(("assistant", response))

        for role, msg in st.session_state.chat_history:
            st.chat_message(role).markdown(msg)

# ----- Tab 2: RAG Advanced -----
with tab2:
    uploaded_files = st.file_uploader(
        "Upload dokumen (PDF, TXT, DOCX, PPTX, CSV, XLSX, gambar) → multi-file",
        type=["pdf","txt","docx","pptx","csv","xls","xlsx","png","jpg","jpeg","bmp"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.markdown("### 📂 Dokumen yang diupload:")
        for f in uploaded_files:
            st.write("- " + f.name)

        vectorstore = build_vectorstore(uploaded_files)

        if "rag_history" not in st.session_state:
            st.session_state.rag_history = []

        user_query = st.chat_input("Tanyakan sesuatu tentang dokumen...")
        if user_query:
            st.chat_message("user").markdown(user_query)
            st.session_state.rag_history.append(("user", user_query))
            answer = f"🤖 (HuggingFace fallback) Ringkasan jawaban untuk: {user_query[:50]}..."
            st.chat_message("assistant").markdown(answer)
            st.session_state.rag_history.append(("assistant", answer))

        for role, msg in st.session_state.rag_history:
            st.chat_message(role).markdown(msg)
