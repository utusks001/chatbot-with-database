import streamlit as st
from utils import load_excel, detect_column_types
import pandas as pd
import plotly.express as px
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langsmith import Client
from dotenv import load_dotenv
import os

# --- Load env ---
load_dotenv()
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY","")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY","")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY","")
LANGSMITH_API_KEY = st.secrets.get("LANGSMITH_API_KEY","")

st.set_page_config(page_title="Advanced Interactive Data Chatbot", layout="wide")

# --- Sidebar: Chat History & Provider ---
st.sidebar.title("Riwayat Chat")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

provider = st.sidebar.selectbox("Pilih LLM Provider", ["OpenAI GPT-4", "Google Gemini 2.5 Flash", "GROQ LLaMA 3.3 70B", "Langsmith"])

uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["csv","xls","xlsx"])
if uploaded_file:
    sheets = load_excel(uploaded_file)
    sheet_names = list(sheets.keys())
    selected_sheet = st.selectbox("Pilih Sheet", sheet_names)
    df = sheets[selected_sheet]

    st.write("### Preview Data")
    st.dataframe(df.head())

    numeric_cols, categorical_cols = detect_column_types(df)
    st.write(f"**Kolom Numerik:** {numeric_cols}")
    st.write(f"**Kolom Kategori:** {categorical_cols}")

    # --- Setup RAG / FAISS ---
    if "vectorstore" not in st.session_state:
        records = df.head(1000).to_dict(orient='records')
        docs = [Document(page_content=str(r)) for r in records]
        embeddings = OpenAIEmbeddings(openai_api_key=GEMINI_API_KEY)
        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

    # --- LLM Initialization berdasarkan provider ---
    if provider == "OpenAI GPT-4":
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=GEMINI_API_KEY)
    elif provider == "Google Gemini 2.5 Flash":
        llm = ChatOpenAI(model_name="gemini-2.5", temperature=0.2, openai_api_key=GOOGLE_API_KEY)  # placeholder
    elif provider == "GROQ LLaMA 3.3 70B":
        llm = ChatOpenAI(model_name="llama-3.3-70b-versatile", temperature=0.2, openai_api_key=GROQ_API_KEY)  # placeholder
    elif provider == "Langsmith":
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=LANGSMITH_API_KEY)  # bisa gunakan Langsmith client untuk tracking

    langsmith_client = Client(api_key=LANGSMITH_API_KEY)

    st.write("---")
    st.write("### Tanyakan Analisis Data ke Chatbot (Plot/ Pivot/ Statistik)")

    user_input = st.text_input("Masukkan pertanyaan anda:")

    if user_input:
        docs = st.session_state.vectorstore.similarity_search(user_input, k=3)
        context_text = "\n".join([d.page_content for d in docs])

        prompt_template = f"""
        Kamu adalah asisten analisis data canggih.
        Data sheet '{selected_sheet}':
        {context_text}

        Buat kode Python Plotly atau Pivot Table sesuai pertanyaan berikut: {user_input}
        Tampilkan juga hasil visualisasi atau pivot di Streamlit.
        """
        chain = LLMChain(llm=llm, prompt=PromptTemplate(template="{input}", input_variables=["input"]))
        response = chain.run(prompt_template)

        # --- Execute code secara aman ---
        st.session_state.chat_history.append((user_input, response))
        st.write("### Kode dan Hasil Chatbot")
        st.code(response, language="python")

        try:
            local_vars = {"df": df, "px": px, "st": st, "pd": pd}
            exec(response, {}, local_vars)
        except Exception as e:
            st.error(f"Error menjalankan kode: {e}")

    # --- Tampilkan Chat History di Sidebar ---
    st.sidebar.markdown("### Riwayat Chat")
    for i, (q,a) in enumerate(st.session_state.chat_history[::-1]):
        st.sidebar.markdown(f"**Q{i+1}:** {q}")
        st.sidebar.markdown(f"**A{i+1}:** {a}")
