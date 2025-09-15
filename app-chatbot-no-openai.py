import streamlit as st
from utils import load_excel, detect_column_types, chunk_dataframe
import pandas as pd
import plotly.express as px
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langsmith import Client

st.set_page_config(page_title="Advanced RAG Data Chatbot", layout="wide")

# --- Sidebar ---
st.sidebar.title("Riwayat Chat")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

provider = st.sidebar.selectbox(
    "Pilih LLM Provider",
    ["Google Gemini 2.5 Flash", "GROQ LLaMA 3.3 70B", "Langsmith", "HuggingFace-local"]
)

uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["csv","xls","xlsx"])

# --- Langsmith LLMOps ---
LANGSMITH_API_KEY = st.secrets.get("LANGSMITH_API_KEY", "")
langsmith_client = Client(api_key=LANGSMITH_API_KEY)

# --- Load data ---
if uploaded_file:
    try:
        sheets = load_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    sheet_names = list(sheets.keys())
    selected_sheet = st.selectbox("Pilih Sheet", sheet_names)
    df = sheets[selected_sheet]

    st.write("### Preview Data")
    st.dataframe(df.head())

    numeric_cols, categorical_cols = detect_column_types(df)
    st.write(f"**Kolom Numerik:** {numeric_cols}")
    st.write(f"**Kolom Kategori:** {categorical_cols}")

    # --- Advanced RAG: Chunking + Embeddings ---
    if "vectorstore" not in st.session_state:
        st.write("Membuat embeddings HuggingFace untuk Advanced RAG...")
        all_docs = []
        chunks = chunk_dataframe(df, chunk_size=5000)
        for c in chunks:
            records = c.to_dict(orient="records")
            docs = [Document(page_content=str(r)) for r in records]
            all_docs.extend(docs)

        hf_model = SentenceTransformer('all-MiniLM-L6-v2')

        # --- FAISS fixed: from_documents dengan embedding callable ---
        st.session_state.vectorstore = FAISS.from_documents(
            documents=all_docs,
            embedding=hf_model.encode
        )
        st.session_state.vectorstore_version = 1

    # --- Setup LLM ---
    if provider == "HuggingFace-local":
        from transformers import pipeline
        llm = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=0)
    else:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

    st.write("---")
    st.write("### Tanyakan Analisis Data ke Chatbot (Plot/ Pivot/ Statistik)")
    user_input = st.text_input("Masukkan pertanyaan anda:")

    if user_input:
        # --- Advanced RAG: similarity search ---
        docs = st.session_state.vectorstore.similarity_search(user_input, k=5)
        context_text = "\n".join([d.page_content for d in docs])

        prompt_template = f"""
Kamu adalah asisten analisis data ultra-interaktif.
Data sheet '{selected_sheet}':
{context_text}

Berdasarkan data ini, deteksi tipe plot terbaik (scatter, line, bar, histogram, pivot) untuk menjawab pertanyaan:
{user_input}

Buat kode Python Plotly atau Pivot Table sesuai pertanyaan.
Sertakan filter dropdown untuk kolom agar chart interaktif.
"""

        # --- Langsmith LLMOps run terbaru ---
        run = langsmith_client.runs.create(
            name="AdvancedRAGQuery",
            description="Query data sheet dengan Advanced RAG",
            metadata={"sheet": selected_sheet, "provider": provider}
        )

        try:
            if provider == "HuggingFace-local":
                response = llm(prompt_template, max_new_tokens=300)[0]["generated_text"]
            else:
                chain = LLMChain(
                    llm=llm,
                    prompt=PromptTemplate(template="{input}", input_variables=["input"])
                )
                response = chain.run(prompt_template)

            # --- Log ke Langsmith ---
            run.add_message("user", user_input)
            run.add_message("assistant", response)
            run.complete()

            # --- Simpan chat history ---
            st.session_state.chat_history.append((user_input, response))
            st.write("### Kode yang di-generate Chatbot")
            st.code(response, language="python")

            # --- Execute kode ---
            local_vars = {"df": df, "px": px, "st": st, "pd": pd}
            exec(response, {}, local_vars)

        except Exception as e:
            run.complete(error=str(e))
            st.error(f"Error menjalankan kode: {e}")

    # --- Chat History Sidebar ---
    st.sidebar.markdown("### Riwayat Chat")
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
        st.sidebar.markdown(f"**Q{i+1}:** {q}")
        st.sidebar.markdown(f"**A{i+1}:** {a}")
