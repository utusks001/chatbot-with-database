# app-langchain.py

import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import os

from utils1 import detect_data_types, recommend_and_plot

# LangChain and Google Generative AI imports
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType

# Setup LangSmith (opsional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

# ========= Helper Functions =========
def safe_describe(df: pd.DataFrame):
    """Cegah error jika describe gagal."""
    try:
        return df.describe(include="all").transpose()
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def df_info_text(df: pd.DataFrame):
    """Ambil info() dataframe sebagai string."""
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

# ========= Main Streamlit App =========
st.set_page_config(page_title="DataViz Chatbot", layout="wide")
st.title("🤖 Chatbot Otomasi Analisis Data (didukung Google Gemini)")

# Sidebar for chat history
with st.sidebar:
    st.header("Riwayat Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

uploaded_file = st.file_uploader(
    "Upload file Excel (.xls, .xlsx) atau CSV (.csv)", 
    type=["csv", "xls", "xlsx"]
)

if uploaded_file is not None:
    if "df" not in st.session_state:
        st.session_state.df = None
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
            selected_sheet = None
        else:
            xls = pd.ExcelFile(uploaded_file)
            selected_sheet = st.selectbox("Pilih Sheet", xls.sheet_names)
            st.session_state.df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {e}")
        st.stop()

    # Info file
    if selected_sheet:
        st.markdown(f"### 📄 Analisa: {uploaded_file.name} — Sheet: {selected_sheet}")
    else:
        st.markdown(f"### 📄 Analisa: {uploaded_file.name}")

    df = st.session_state.df

    # Preview data
    st.write("**Head (10):**")
    st.dataframe(df.head(10))
    st.write("**Tail (10):**")
    st.dataframe(df.tail(10))

    st.write("**describe():**")
    st.dataframe(safe_describe(df))

    st.write("**info():**")
    st.text(df_info_text(df))

    # Ringkasan kolom
    categorical_cols, numeric_cols = detect_data_types(df)
    st.subheader("Ringkasan Kolom")
    st.write(f"Kolom Numerik: {numeric_cols}")
    st.write(f"Kolom Kategorikal: {categorical_cols}")

    # Check missing values
    missing_values = df.isnull().sum()
    st.write(missing_values)
    st.write("                                             ") 

    # Display the number of duplicates removed
    duplicates_count = df.duplicated().sum()
    st.write(f"Number of Duplicates : {duplicates_count}")
    st.write("                                             ")  
        
    # Remove duplicates
    duplicates_removed = df.drop_duplicates(inplace=True)
    st.write(f"Number of Duplicates Removed: {duplicates_removed}")
    st.write("                                             ")  
    
    # Display summary statistics of the DataFrame
    st.write("**Summary Statistics:**")
    st.write(df.describe(include="all"))
    
    # Inisialisasi Chatbot
    if "agent_initialized" not in st.session_state:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )
            st.session_state.agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True
            )
            st.session_state.agent_initialized = True
            st.success("Chatbot siap! Silakan ajukan pertanyaan tentang data Anda.")
        except Exception as e:
            st.error(f"Gagal menginisialisasi chatbot. Pastikan API key Anda benar: {e}")
            st.stop()
            
    # Chat input
    st.subheader("Tanyakan Sesuatu Tentang Data")
    user_query = st.chat_input("Contoh: 'Berapa rata-rata pendapatan?'")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Memproses..."):
            try:
                response = st.session_state.agent.run(user_query)
                with st.chat_message("assistant"):
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Maaf, terjadi kesalahan saat memproses permintaan: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Maaf, terjadi kesalahan saat memproses permintaan."}
                )
