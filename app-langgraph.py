# app-langgraph.py 
import os
import streamlit as st
from dotenv import load_dotenv

# OpenAI official SDK
from openai import OpenAI

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory

import pandas as pd

# --- Load env ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Patch API KEY untuk sk-proj ---
if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-proj-"):
    if not OPENAI_PROJECT_ID:
        st.error("❌ OPENAI_PROJECT_ID wajib diset di .env jika memakai sk-proj- API key!")
    else:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["OPENAI_PROJECT_ID"] = OPENAI_PROJECT_ID

# --- OpenAI official client test ---
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        if OPENAI_PROJECT_ID:
            client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)
    except Exception as e:
        st.error(f"Gagal inisialisasi OpenAI client: {e}")

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot Data Analysis", layout="wide")
st.title("🤖 Chatbot Data Analysis (Python Agent + MySQL)")

st.sidebar.header("⚙️ Pengaturan")
mode = st.sidebar.radio("Pilih Mode Analisis", ["Python Agent", "MySQL Agent"])

uploaded_files = st.sidebar.file_uploader(
    "Upload Dataset (CSV/XLSX) – bisa multi-file",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
)
sql_uri = st.sidebar.text_input("MySQL URI", value="mysql+pymysql://user:pass@localhost:3306/dbname")

# --- LLM setup ---
llm = None
if OPENAI_API_KEY:
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=MODEL,
        temperature=0,
        streaming=True,
    )

# --- Memory ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Tools ---
tools = []

if mode == "Python Agent":
    if uploaded_files:
        dfs = {}
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                else:
                    raise ValueError("Format file tidak didukung")

                dfs[uploaded_file.name] = df

                # Buat agent dataframe
                agent_df = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    allow_dangerous_code=True,  # patch penting
                )
                tools.append(
                    Tool(
                        name=f"Dataframe_{uploaded_file.name}",
                        func=agent_df.run,
                        description=f"Analisis dataset {uploaded_file.name}",
                    )
                )
                st.success(f"✅ {uploaded_file.name} berhasil dimuat.")
            except Exception as e:
                st.error(f"Gagal membaca {uploaded_file.name}: {e}")
    else:
        st.info("Upload minimal 1 file CSV/XLSX untuk analisis.")

elif mode == "MySQL Agent":
    try:
        db = SQLDatabase.from_uri(sql_uri)
        db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
        tools.append(
            Tool(
                name="MySQL",
                func=db_chain.run,
                description="Jalankan query analisis pada database MySQL",
            )
        )
        st.success("✅ Koneksi MySQL berhasil.")
    except Exception as e:
        st.error(f"Gagal konek MySQL: {e}")

# --- Agent ---
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
    st.subheader("💬 Chat dengan Data / Database")
    user_input = st.chat_input("Ketik pertanyaan...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            try:
                response = agent.run(user_input)
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")
