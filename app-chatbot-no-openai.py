import streamlit as st
from utils import load_excel, detect_column_types, suggest_visualizations
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
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

st.set_page_config(page_title="Advanced Data Analysis Chatbot", layout="wide")

# --- Sidebar Chat History ---
st.sidebar.title("Riwayat Chat")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- File Upload ---
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

    st.write("### Rekomendasi Visualisasi")
    vis_suggestions = suggest_visualizations(df)
    for s in vis_suggestions:
        st.write(f"- {s}")

    # --- Rekomendasi Visualisasi Interaktif ---
    st.write("### Visualisasi Interaktif")
    if numeric_cols:
        col = st.selectbox("Pilih kolom numerik untuk histogram", numeric_cols)
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

    if numeric_cols and categorical_cols:
        x_col = st.selectbox("Kolom kategori (x-axis)", categorical_cols)
        y_col = st.selectbox("Kolom numerik (y-axis)", numeric_cols)
        fig2 = px.box(df, x=x_col, y=y_col)
        st.plotly_chart(fig2)

    # --- Setup LLM + Langsmith ---
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY
    )
    langsmith_client = Client(api_key=OPENAI_API_KEY)

    # --- Setup RAG / FAISS Embedding ---
    if "vectorstore" not in st.session_state:
        # buat embedding dari dataset
        records = df.head(1000).to_dict(orient='records')  # ambil sample
        docs = [Document(page_content=str(r)) for r in records]
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

    # --- Chatbot ---
    st.write("---")
    st.write("### Tanyakan Analisis Data ke Chatbot")
    user_input = st.text_input("Masukkan pertanyaan anda:")

    if user_input:
        # RAG: cari similarity di vector store
        docs = st.session_state.vectorstore.similarity_search(user_input, k=3)
        context_text = "\n".join([d.page_content for d in docs])

        prompt_template = f"""
        Kamu adalah asisten analisis data canggih.
        Berikut sebagian data dari sheet '{selected_sheet}':
        {context_text}

        Jawab pertanyaan berikut dengan jelas dan gunakan konteks data:
        {user_input}
        """
        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(template="{input}", input_variables=["input"])
        )
        response = chain.run(prompt_template)

        st.session_state.chat_history.append((user_input, response))
        st.write("### Jawaban Chatbot")
        st.write(response)

    # --- Tampilkan Chat History di Sidebar ---
    st.sidebar.markdown("### Riwayat Chat")
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
        st.sidebar.markdown(f"**Q{i+1}:** {q}")
        st.sidebar.markdown(f"**A{i+1}:** {a}")
