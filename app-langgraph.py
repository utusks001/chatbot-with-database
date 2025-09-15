# app-langgraph.py 
import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import mysql.connector

# LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase

# OpenAI official
from openai import OpenAI

# ------------------------
# 1. Load environment
# ------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID", "")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY belum di-set di .env")
    st.stop()

# Jika key format sk-proj-... maka wajib ada project id
if OPENAI_API_KEY.startswith("sk-proj-"):
    if not OPENAI_PROJECT_ID:
        st.error("❌ OPENAI_PROJECT_ID wajib diisi di .env untuk key sk-proj-...")
        st.stop()
    # patch env supaya SDK/LangChain bisa pakai project
    os.environ["OPENAI_PROJECT_ID"] = OPENAI_PROJECT_ID

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ------------------------
# 2. Build LLM (LangChain)
# ------------------------
try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )
except Exception as e:
    st.error(f"❌ Gagal inisialisasi ChatOpenAI: {e}")
    st.stop()

# ------------------------
# 3. Fallback Official SDK
# ------------------------
client = None
try:
    if OPENAI_API_KEY.startswith("sk-proj-"):
        client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.warning(f"⚠️ OpenAI official client gagal: {e}")

# ------------------------
# 4. Streamlit UI
# ------------------------
st.set_page_config(page_title="LangGraph + SQL + CSV Chatbot", layout="wide")
st.title("🤖 LangGraph Chatbot (MySQL + Python Agent + Multi-CSV/XLSX)")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Dataset", "🗄 SQL DB"])

# ------------------------
# 5. Multi-file Upload
# ------------------------
with tab2:
    uploaded_files = st.file_uploader("Upload CSV/XLSX (bisa banyak file)", type=["csv", "xlsx"], accept_multiple_files=True)

    dfs = {}
    if uploaded_files:
        for f in uploaded_files:
            if f.name.endswith(".csv"):
                dfs[f.name] = pd.read_csv(f)
            elif f.name.endswith(".xlsx"):
                dfs[f.name] = pd.read_excel(f)

        st.success(f"{len(dfs)} file berhasil dimuat ✅")
        for name, df in dfs.items():
            st.subheader(name)
            st.dataframe(df.head())

# ------------------------
# 6. MySQL Connection
# ------------------------
with tab3:
    st.subheader("Koneksi MySQL")
    db_host = st.text_input("Host", value="localhost")
    db_user = st.text_input("User", value="root")
    db_pass = st.text_input("Password", type="password")
    db_name = st.text_input("Database", value="test")

    db = None
    if st.button("Connect DB"):
        try:
            db = SQLDatabase.from_uri(f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}")
            st.success("✅ Koneksi DB berhasil")
        except Exception as e:
            st.error(f"❌ Gagal koneksi DB: {e}")

# ------------------------
# 7. Build Agents
# ------------------------
tools = []

# Python DataFrame Agents untuk tiap file
if dfs:
    for name, df in dfs.items():
        agent_df = create_pandas_dataframe_agent(llm, df, verbose=True)
        tools.append(Tool(
            name=f"Dataframe_{name}",
            func=agent_df.run,
            description=f"Analisis dataset {name}"
        ))

# SQL Agent
if db:
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
    tools.append(Tool(
        name="MySQL",
        func=db_chain.run,
        description="Eksekusi query ke database MySQL"
    ))

# Init Agent
if tools:
    agent = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True)
else:
    agent = None

# ------------------------
# 8. Chat
# ------------------------
with tab1:
    st.subheader("Chat dengan AI")
    user_input = st.text_area("Masukkan pertanyaan:")

    if st.button("Kirim"):
        if agent:
            try:
                answer = agent.run(user_input)
                st.success(answer)
            except Exception as e:
                st.error(f"❌ Error saat agent run: {e}")
        else:
            st.warning("⚠️ Belum ada dataset/DB yang dimuat.")

    if client:
        st.markdown("---")
        st.write("🔹 Test langsung ke OpenAI official SDK (bypass LangChain)")
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello, are you working?"}],
                max_tokens=30,
            )
            st.info(f"SDK response: {resp.choices[0].message.content}")
        except Exception as e:
            st.error(f"❌ SDK error: {e}")
