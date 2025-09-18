# app-langchain.py

import streamlit as st
import io
import os

# Data analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils1 import detect_data_types, recommend_and_plot
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType

from pathlib import Path
import toml
from dotenv import load_dotenv, set_key

# ========== Load .env kalau ada ==========
dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# ========== Setup LangSmith ==========
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))

# ========== Sidebar ==========
with st.sidebar:
    st.header("🔑 Konfigurasi API Key")
    
    # Ambil key dari secrets atau .env
    GOOGLE_API_KEY = (
        st.secrets.get("GOOGLE_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
    )
    
    # Kalau kosong, user bisa input manual
    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = st.text_input("Buat GOOGLE API KEY baru pada https://aistudio.google.com/apikey kemudian copy dan paste disini", type="password")
        if GOOGLE_API_KEY:
            st.session_state["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            st.success("GOOGLE_API_KEY berhasil dimasukkan ✅")

            # Opsi simpan
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
        st.success("GOOGLE_API_KEY berhasil dimuat ✅")

    # ===== Chat history =====
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

# ========= Main =========
st.set_page_config(page_title="DataViz Chatbot", layout="wide")
st.title("🤖 Chatbot Otomasi Analisis Data (Google Gemini)")

uploaded_file = st.file_uploader(
    "Upload file Excel (.xls, .xlsx) atau CSV (.csv)", 
    type=["csv", "xls", "xlsx"]
)

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.session_state.dfs = {"CSV": df}
    else:
        xls = pd.ExcelFile(uploaded_file)
        st.session_state.dfs = {sheet: pd.read_excel(uploaded_file, sheet_name=sheet) for sheet in xls.sheet_names}

    # Multi-select sheet
    with st.sidebar:
        st.subheader("📑 Pilih Sheet")
        sheet_names = list(st.session_state.dfs.keys())
        selected_sheets = st.multiselect("Sheet Aktif", sheet_names, default=sheet_names[:1])

    if not selected_sheets:
        st.warning("Pilih minimal satu sheet untuk analisis.")
        st.stop()

    # Gabung data
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

    # Info dataset
    st.markdown(f"### 📄 Analisa: {uploaded_file.name} — Sheet(s): {sheet_label}")
    st.dataframe(df.head(10))
    st.write("**Info():**")
    st.text(df_info_text(df))
    st.write("**Describe():**")
    st.dataframe(safe_describe(df))

    # Correlation Heatmap
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        st.write("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Inisialisasi agent
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
        st.warning("⚠️ GOOGLE_API_KEY belum diisi.")

    # Chat input
    if GOOGLE_API_KEY:
        user_query = st.chat_input("Tanyakan sesuatu tentang data...")
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
                    st.error(f"Error: {e}")
