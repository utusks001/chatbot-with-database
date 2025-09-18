# app-multisheet.py

import streamlit as st
import io
import os

# Data analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils1 import detect_data_types, recommend_and_plot

# LangChain and Google Generative AI imports
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType

# ========= Setup LangSmith (opsional, pakai fallback) =========
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")

# ========= Sidebar for chat history + API Key =========
with st.sidebar:
    st.header("🔑 Konfigurasi")
    # Coba ambil dari secrets dulu
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    # Kalau kosong, izinkan user isi manual
    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = st.text_input("Buat GOOGLE API KEY baru pada https://aistudio.google.com/apikey kemudian copy dan paste disini", type="password")
        if GOOGLE_API_KEY:
            st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    else:
        st.success("GOOGLE_API_KEY dari secrets berhasil dimuat ✅")

    st.header("Riwayat Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ========= Helper Functions =========
def safe_describe(df: pd.DataFrame):
    try:
        return df.describe(include="all").transpose()
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def df_info_text(df: pd.DataFrame):
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()

# ========= Main Streamlit App =========
st.set_page_config(page_title="DataViz Chatbot", layout="wide")
st.title("🤖 Chatbot Otomasi Analisis Data (didukung Google Gemini)")

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

    # ===== Sidebar multi-select =====
    with st.sidebar:
        st.subheader("📑 Pilih Sheet")
        sheet_names = list(st.session_state.dfs.keys())
        selected_sheets = st.multiselect("Sheet Aktif", sheet_names, default=sheet_names[:1])

    if not selected_sheets:
        st.warning("Pilih minimal satu sheet untuk analisis.")
        st.stop()

    # ===== Gabung jika multi-sheet =====
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

    # ===== Info file =====
    st.markdown(f"### 📄 Analisa: {uploaded_file.name} — Sheet(s): {sheet_label}")
    
    # ===== Preview data =====
    st.write("**Head (10):**")
    st.dataframe(df.head(10))
    st.write("**Tail (10):**")
    st.dataframe(df.tail(10))
    
    # ===== Ringkasan =====
    categorical_cols, numeric_cols = detect_data_types(df)
    st.write("**Ringkasan Kolom**")
    st.write(f"Kolom Numerik: {numeric_cols}")
    st.write(f"Kolom Kategorikal: {categorical_cols}")
    
    st.write("**Info():**")
    st.text(df_info_text(df))
    
    st.write(f"**Data shape:** {df.shape}")
    
    st.write("**Data information:**")
    for index, (col, dtype) in enumerate(zip(df.columns, df.dtypes)):
        non_null_count = df[col].count()
        st.write(f"{index} | {col}   | {non_null_count} non-null  |  {dtype}") 
    
    missing_values = df.isnull().sum()
    st.write("**Missing Values:**")
    st.write(missing_values)
    
    duplicates_count = df.duplicated().sum()
    st.write(f"**Number of Duplicates :** {duplicates_count}")
    
    df_no_dupes = df.drop_duplicates()
    st.write(f"**Number of Duplicates Removed:** {len(df) - len(df_no_dupes)}")
    
    st.write("**Describe():**")
    st.dataframe(safe_describe(df))
    
    st.write("**Summary Statistics:**")
    st.write(df.describe(include="all"))

    # ===== Correlation Heatmap =====
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        st.write("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    
    # ===== Inisialisasi Chatbot (per kombinasi sheet) =====
    agent_key = f"agent_{sheet_label}"
    if agent_key not in st.session_state and GOOGLE_API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=GOOGLE_API_KEY
            )
            st.session_state[agent_key] = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True
            )
            st.success(f"Chatbot siap! (Sheet: {sheet_label})")
        except Exception as e:
            st.error(f"Gagal inisialisasi chatbot: {e}")
            st.stop()
    elif not GOOGLE_API_KEY:
        st.warning("⚠️ GOOGLE_API_KEY belum diisi. Masukkan di sidebar agar chatbot aktif.")

    # ===== Chat input =====
    if GOOGLE_API_KEY:
        st.subheader(f"Tanyakan Sesuatu Tentang Data (Sheet: {sheet_label})")
        user_query = st.chat_input("Contoh: 'Berapa rata-rata pendapatan?'")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.spinner("Memproses..."):
                try:
                    response = st.session_state[agent_key].run(user_query)
                    with st.chat_message("assistant"):
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Maaf, terjadi kesalahan saat memproses permintaan: {e}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "Maaf, terjadi kesalahan saat memproses permintaan."}
                    )
