import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

# LangChain
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# LLM Providers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# --- Load environment ---
load_dotenv()

# Default keys from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot Multiprovider", layout="wide")
st.title("🤖 Chatbot Multiprovider (Google Gemini + Groq fallback)")

st.sidebar.header("⚙️ Pengaturan Provider")
provider_mode = st.sidebar.radio(
    "Pilih Provider",
    ["Auto (Google→Groq)", "Google Gemini", "Groq"]
)

# Manual API key input
if provider_mode in ["Google Gemini", "Auto (Google→Groq)"]:
    GOOGLE_API_KEY = st.sidebar.text_input("🔑 Google API Key", value=GOOGLE_API_KEY, type="password")

if provider_mode in ["Groq", "Auto (Google→Groq)"]:
    GROQ_API_KEY = st.sidebar.text_input("🔑 Groq API Key", value=GROQ_API_KEY, type="password")

mode = st.sidebar.radio("Mode Analisis", ["Python Agent", "SQL Agent"])
uploaded_files = st.sidebar.file_uploader(
    "Upload dataset (CSV/XLSX, multi-file)", type=["csv", "xlsx"], accept_multiple_files=True
)
sql_uri = st.sidebar.text_input("MySQL/SQLite URI", value="sqlite:///sample.db")

# --- Memory ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Provider Selection ---
def get_llm():
    if provider_mode == "Google Gemini":
        if not GOOGLE_API_KEY:
            st.error("Google API Key wajib diisi!")
            return None
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)

    elif provider_mode == "Groq":
        if not GROQ_API_KEY:
            st.error("Groq API Key wajib diisi!")
            return None
        return ChatGroq(model="llama-3.3-70b-versati", api_key=GROQ_API_KEY, temperature=0)

    elif provider_mode == "Auto (Google→Groq)":
        # Try Google first
        if GOOGLE_API_KEY:
            try:
                return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
            except Exception as e:
                if "429" in str(e):
                    st.warning("⚠️ Google quota habis, otomatis switch ke Groq...")
                else:
                    st.error(f"Google API error: {e}")
        # Fallback Groq
        if GROQ_API_KEY:
            return ChatGroq(model="llama-3.3-70b-versati", api_key=GROQ_API_KEY, temperature=0)
        else:
            st.error("Groq API Key belum diisi.")
            return None

llm = get_llm()

# --- Tools setup ---
tools = []

if mode == "Python Agent":
    if uploaded_files:
        try:
            dfs = {}
            for file in uploaded_files:
                if file.name.endswith(".csv"):
                    dfs[file.name] = pd.read_csv(file)
                elif file.name.endswith(".xlsx"):
                    dfs[file.name] = pd.read_excel(file)

            # Jika banyak file, gabungkan ke satu agent
            for name, df in dfs.items():
                pandas_agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    handle_parsing_errors=True,
                    allow_dangerous_code=True,
                )
                tools.append(PythonREPLTool())
                tools.extend(pandas_agent.tools)
            st.success(f"{len(dfs)} file berhasil dimuat untuk Python Agent.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    else:
        st.info("Upload satu atau lebih file CSV/XLSX untuk Python Agent.")

elif mode == "SQL Agent":
    try:
        db = SQLDatabase.from_uri(sql_uri)
        db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
        tools.append(db_chain)
        st.success("Koneksi database berhasil.")
    except Exception as e:
        st.error(f"Gagal konek ke database: {e}")

# --- Agent setup ---
agent = None
if tools and llm:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

# --- Chat UI ---
if agent:
    st.subheader("💬 Chat dengan Data Anda")
    user_input = st.chat_input("Tanyakan sesuatu...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            try:
                response = agent.run(user_input)
                st.markdown(response)
            except Exception as e:
                if "429" in str(e):
                    st.error("⚠️ Quota provider habis. Ganti API Key atau coba provider lain.")
                else:
                    st.error(f"Error: {e}")
