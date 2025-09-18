# app-langchain.py

import streamlit as st
import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
from PIL import Image
import toml

# LangChain & HuggingFace
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ====== Config Streamlit ======
st.set_page_config(page_title="Ultra-Lite Data & Document Chatbot", layout="wide")
st.title("📊🤖 Chatbot Analisis Data & Dokumen (Ultra-Lite, HuggingFace fallback)")

# ====== Sidebar: API Key Manual Input ======
st.sidebar.header("🔑 Konfigurasi API Key (Optional)")

dotenv_path = Path(".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = st.text_input(
        "Buat GOOGLE API KEY baru di https://aistudio.google.com/apikey dan paste di sini",
        type="password"
    )
    if GOOGLE_API_KEY:
        st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        st.success("✅ GOOGLE_API_KEY berhasil dimasukkan")

        save_choice = st.radio("Simpan key ke mana?", ["Jangan simpan", ".env", "secrets.toml"])
        if st.button("💾 Simpan API Key"):
            if save_choice == ".env":
                from dotenv import set_key
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
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")

def ocr_image(file):
    if OCR_SPACE_API_KEY:
        file.seek(0)
        try:
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

# ====== Build Vectorstore ======
def build_vectorstore(files):
    docs = []
    for file in files:
        name = file.name.lower()
        try:
            if name.endswith(".txt"):
                text = file.read().decode("utf-8")
            elif name.endswith(".csv"):
                df = pd.read_csv(file)
                text = df.to_csv(index=False)
            elif name.endswith(".xlsx") or name.endswith(".xls"):
                xls = pd.ExcelFile(file)
                text = ""
                for sheet in xls.sheet_names:
                    df = pd.read_excel(file, sheet_name=sheet)
                    text += f"[{sheet}]\n" + df.to_csv(index=False)
            elif name.endswith(".pdf"):
                from PyPDF2 import PdfReader
                reader = PdfReader(file)
                text = "\n".join([p.extract_text() or "" for p in reader.pages])
            elif name.endswith(".docx"):
                import docx
                doc = docx.Document(file)
                text = "\n".join([p.text for p in doc.paragraphs])
            elif name.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                text = ocr_image(file)
            else:
                text = file.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": name}))
        except Exception as e:
            st.warning(f"⚠️ Error reading {name}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(split_docs, embeddings)

# ====== LLM Fallback: HuggingFace ======
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

# ====== Tabs ======
tab1, tab2 = st.tabs(["📊 Data Analysis", "📑 RAG Advanced"])

# ====== Tab 1: Data Analysis ======
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
            sheet_label = selected_sheets[0]
        else:
            df_list = []
            for s in selected_sheets:
                temp = st.session_state.dfs[s].copy()
                temp["SheetName"] = s
                df_list.append(temp)
            df = pd.concat(df_list, ignore_index=True)
            sheet_label = ", ".join(selected_sheets)

        st.markdown(f"### 📄 Analisa: {uploaded_file.name} — Sheet(s): {sheet_label}")
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

            # Buat prompt ringkas + insight + kesimpulan
            preview = df.head(1000).to_csv(index=False)
            prompt = f"""
Dataset sampel (1000 baris pertama):
{preview}

Buatkan insight utama dan kesimpulan dari data ini.
"""
            response = llm(prompt)[0]['generated_text']
            st.chat_message("assistant").markdown(response)
            st.session_state.chat_history.append(("assistant", response))

        for role, msg in st.session_state.chat_history:
            st.chat_message(role).markdown(msg)

# ====== Tab 2: RAG Advanced ======
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
        retriever = vectorstore.as_retriever(search_kwargs={"k":3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        st.success("✅ Chatbot RAG siap digunakan")

        user_query = st.chat_input("Tanyakan sesuatu tentang dokumen...")
        if user_query:
            st.chat_message("user").markdown(user_query)
            try:
                result = qa_chain({"query": user_query})
                answer = result["result"]
                sources = result.get("source_documents", [])
                st.chat_message("assistant").markdown(answer)
                if sources:
                    st.write("**Sumber:**")
                    for s in sources:
                        st.caption(f"{s.metadata.get('source','')} → {s.page_content[:200]}...")
            except Exception as e:
                st.error(f"❌ Error: {e}")
