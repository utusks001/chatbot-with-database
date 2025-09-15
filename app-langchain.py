# app-langchain.py
import streamlit as st
import pandas as pd
import io
from utils1 import detect_data_types, recommend_and_plot
import matplotlib.pyplot as plt

# LangChain and OpenAI imports
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Set up LangSmith for tracing (optional but recommended)
# Pastikan Anda telah menginstal langsmith dan mengatur LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

# Main Streamlit App
st.set_page_config(page_title="DataViz Chatbot", layout="wide")
st.title("🤖 Chatbot Otomasi Analisis Data")

# Sidebar untuk riwayat chat
with st.sidebar:
    st.header("Riwayat Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Bagian Utama ---

uploaded_file = st.file_uploader("Upload file Excel (.xls, .xlsx) atau CSV (.csv)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    
    # Inisialisasi DataFrame di session_state
    if "df" not in st.session_state:
        st.session_state.df = None
    
    # Mendeteksi dan memuat file
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            
            selected_sheet = st.selectbox("Pilih Sheet", sheet_names)
            st.session_state.df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {e}")
        st.stop()

    # Tampilkan Data Frame
    st.subheader("Data yang Diunggah")
    st.write(st.session_state.df.head())
    
    # Deteksi dan tampilkan ringkasan
    categorical_cols, numeric_cols = detect_data_types(st.session_state.df)
    st.subheader("Ringkasan Kolom")
    st.write(f"Kolom Numerik: {numeric_cols}")
    st.write(f"Kolom Kategorikal: {categorical_cols}")

    # Rekomendasi visualisasi
    st.subheader("Visualisasi Otomatis")
    visualizations = recommend_and_plot(st.session_state.df, categorical_cols, numeric_cols)
    
    for title, fig in visualizations.items():
        st.write(f"**{title}**")
        st.pyplot(fig)
        plt.close(fig) # Penting untuk menghemat memori

    # Inisialisasi LangChain Agent
    if "agent_initialized" not in st.session_state:
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
            st.session_state.agent = create_pandas_dataframe_agent(
                llm,
                st.session_state.df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True # <-- Tambahkan baris ini
            )
           
            st.session_state.agent_initialized = True
            st.success("Chatbot siap! Silakan ajukan pertanyaan tentang data Anda.")
        except Exception as e:
            st.error(f"Gagal menginisialisasi chatbot. Pastikan API key Anda benar: {e}")
            st.stop()
            
    # Chatbot
    st.subheader("Tanyakan Sesuatu Tentang Data")
    user_query = st.chat_input("Contoh: 'Berapa rata-rata pendapatan?'")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Mendapatkan respons dari agent
        with st.spinner("Memproses..."):
            try:
                # Perbaikan: Menggunakan st.session_state.agent untuk memanggil agent
                response = st.session_state.agent.run(user_query)
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Maaf, terjadi kesalahan saat memproses permintaan: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Maaf, terjadi kesalahan saat memproses permintaan."})
