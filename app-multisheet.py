# app.py

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
    # ===== Load data =====
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.session_state.dfs = {"CSV": df}
    else:
        xls = pd.ExcelFile(uploaded_file)
        st.session_state.dfs = {sheet: pd.read_excel(uploaded_file, sheet_name=sheet) for sheet in xls.sheet_names}

    # ===== Sidebar pilih sheet =====
    with st.sidebar:
        st.subheader("Pilih Sheet")
        sheet_names = list(st.session_state.dfs.keys())
        selected_sheet = st.selectbox("Sheet Aktif", sheet_names)

    df = st.session_state.dfs[selected_sheet]

    # ===== Info file =====
    st.markdown(f"### 📄 Analisa: {uploaded_file.name} — Sheet: {s_
