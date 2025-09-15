# app-langgraph.py 
import os
import streamlit as st
from dotenv import load_dotenv

# LangChain / LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import pandas as pd

# --- Load environment ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    st.warning(
        "OPENAI_API_KEY belum ditemukan di .env — agent tidak akan bekerja tanpa API key."
    )

if not LANGSMITH_API_KEY:
    st.info(
        "LANGSMITH_API_KEY belum ditemukan di .env — logging/observability ke LangSmith akan dinonaktifkan."
    )

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot Data Analysis", layout="wide")
st.title("🤖 Chatbot Data Analysis dengan LangGraph + LangSmith")

st.sidebar.header("Pengaturan")
mode = st.sidebar.radio("Pilih Mode Analisis", ["Python Agent", "SQL Agent"])

uploaded_file = st.sidebar.file_uploader(
    "Upload Data untuk Analisis (CSV/XLSX)", type=["csv", "xlsx"]
)
sql_uri = st.sidebar.text_input("SQLite URI untuk SQL Agent", value="sqlite:///sample.db")

# --- LLM setup ---
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
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Format file tidak didukung.")

            pandas_agent = create_pandas_dataframe_agent(
                llm, df, verbose=True, handle_parsing_errors=True
            )
            tools.append(PythonREPLTool())
            tools.extend(pandas_agent.tools)
            st.success(
                f"File **{uploaded_file.name}** berhasil dimuat, siap untuk analisis dengan Python Agent."
            )
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    else:
        st.warning("Upload file CSV/XLSX untuk mulai analisis dengan Python Agent.")

if mode == "SQL Agent":
    try:
        db = SQLDatabase.from_uri(sql_uri)
        db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
        tools.append(db_chain)
        st.success("Koneksi database SQL berhasil.")
    except Exception as e:
        st.error(f"Gagal konek ke database: {e}")

# --- Agent ---
if tools:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )
else:
    agent = None

# --- Chat UI ---
if agent:
    st.subheader("💬 Chat dengan Data Anda")
    user_input = st.chat_input("Ketik pertanyaan Anda...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            try:
                response = agent.run(user_input)
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")
