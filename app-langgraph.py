# app-langgraph.py 
import os
import traceback
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# OpenAI official SDK
from openai import OpenAI

# LangChain
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain.agents import Tool
from langchain_experimental.tools.python.tool import PythonREPLTool

# ============ Load ENV ============
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY tidak ditemukan di .env")
    st.stop()

# Handle sk-proj key
if OPENAI_API_KEY.startswith("sk-proj-"):
    if not OPENAI_PROJECT_ID:
        st.error("❌ OPENAI_PROJECT_ID wajib diisi di .env untuk key sk-proj-...")
        st.stop()
    client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ============ Helper ============
def test_openai():
    """Tes koneksi ke OpenAI official SDK"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Tes koneksi berhasil?"}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Gagal test OpenAI: {e}"

# ============ Streamlit UI ============
st.set_page_config(page_title="LangGraph Chatbot", layout="wide")
st.title("🤖 LangGraph Chatbot (Python Multi-file + MySQL + Fallback)")

# Tes koneksi
if st.sidebar.button("🔌 Tes koneksi OpenAI"):
    st.sidebar.info(test_openai())

# Pilih mode
mode = st.sidebar.radio("Pilih Mode Analisis", ["Python Agent", "MySQL Agent", "Auto Detect"])

# Multi-file upload (untuk Python Agent)
uploaded_files = st.sidebar.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"], accept_multiple_files=True)

# Input koneksi MySQL (untuk SQL Agent)
mysql_user = st.sidebar.text_input("MySQL User", value="root")
mysql_pass = st.sidebar.text_input("MySQL Password", type="password")
mysql_host = st.sidebar.text_input("MySQL Host", value="localhost")
mysql_db = st.sidebar.text_input("MySQL Database", value="test")

# Build LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# Build tools
tools = []
df_agents = {}  # simpan beberapa agent dataframe
sql_tool = None
python_tool = PythonREPLTool()

if mode in ["Python Agent", "Auto Detect"]:
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)

                agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    handle_parsing_errors=True,
                    allow_dangerous_code=True,
                )
                df_agents[file.name] = agent
                st.success(f"✅ File **{file.name}** dimuat untuk Python Agent")
            except Exception as e:
                st.error(f"Gagal membaca file {file.name}: {e}")

if mode in ["MySQL Agent", "Auto Detect"]:
    try:
        uri = f"mysql+pymysql://{mysql_user}:{mysql_pass}@{mysql_host}/{mysql_db}"
        db = SQLDatabase.from_uri(uri)
        db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
        sql_tool = Tool(
            name="MySQL Database",
            func=db_chain.run,
            description="Gunakan untuk menjawab pertanyaan tentang database MySQL",
        )
        st.success("✅ MySQL Agent siap")
    except Exception as e:
        st.error(f"Gagal konek MySQL: {e}")

# ============ Chat UI ============
st.subheader("💬 Chat dengan Data/Database Anda")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ketik pertanyaan..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = None
    try:
        if mode == "Python Agent" and df_agents:
            # pilih dataset berdasarkan mention nama file di prompt
            matched = [name for name in df_agents if name.lower() in prompt.lower()]
            if matched:
                agent = df_agents[matched[0]]
                answer = agent.run(prompt)
            else:
                answer = f"⚠️ Sebutkan nama file (misalnya: {list(df_agents.keys())[0]}) agar saya tahu dataset mana yang dipakai."
        elif mode == "MySQL Agent" and sql_tool:
            answer = sql_tool.run(prompt)
        elif mode == "Auto Detect":
            if "sql" in prompt.lower() and sql_tool:
                answer = sql_tool.run(prompt)
            elif "python" in prompt.lower() and df_agents:
                # fallback ke dataset pertama
                first_file = list(df_agents.keys())[0]
                answer = df_agents[first_file].run(prompt)
            else:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = resp.choices[0].message.content
        else:
            answer = "⚠️ Tidak ada agent aktif atau data belum dimuat."
    except Exception as e:
        answer = f"❌ Error: {e}\n\n{traceback.format_exc()}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
