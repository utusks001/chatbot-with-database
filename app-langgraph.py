# app-langgraph.py 
import os
import streamlit as st
from dotenv import load_dotenv

# ===== Load Environment =====
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="Chatbot Data Analysis", layout="wide")
st.title("🤖 Chatbot Data Analysis + API Key Checker")

# ===== API Key Check =====
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY belum ditemukan di .env atau environment variable.")
    st.stop()

st.info(f"🔑 API Key terdeteksi: {OPENAI_API_KEY[:10]}... (length: {len(OPENAI_API_KEY)})")

# Diagnostic: versi library
try:
    import openai
    import langchain
    import langchain_openai
    import langchain_core

    st.caption("📦 Library versions:")
    st.text(f"openai: {openai.__version__}")
    st.text(f"langchain: {langchain.__version__}")
    st.text(f"langchain-openai: {getattr(langchain_openai, '__version__', 'unknown')}")
    st.text(f"langchain-core: {getattr(langchain_core, '__version__', 'unknown')}")
except Exception as e:
    st.warning(f"Gagal membaca versi library: {e}")

# Test with official OpenAI SDK
st.subheader("1️⃣ Test dengan OpenAI official SDK")
try:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hi in one short sentence."},
        ],
        max_tokens=50,
    )
    st.success("✅ OpenAI official SDK berhasil:")
    st.write(response.choices[0].message.content)
except Exception as e:
    st.error("❌ OpenAI official SDK gagal.")
    st.exception(e)

# Test with LangChain
st.subheader("2️⃣ Test dengan LangChain")
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    llm_test = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=MODEL,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "Say hi in one short sentence."),
    ])

    chain = prompt | llm_test
    result = chain.invoke({})
    st.success("✅ LangChain berhasil:")
    st.write(result.content)
except Exception as e:
    st.error("❌ LangChain gagal.")
    st.exception(e)

# ===== Chatbot Data Analysis =====
st.header("📊 Chatbot Data Analysis")

if not LANGSMITH_API_KEY:
    st.info("ℹ️ LANGSMITH_API_KEY belum ditemukan — logging/observability ke LangSmith akan dinonaktifkan.")

# LangChain / LangGraph imports
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import pandas as pd

# Sidebar
st.sidebar.header("Pengaturan")
mode = st.sidebar.radio("Pilih Mode Analisis", ["Python Agent", "SQL Agent"])
uploaded_file = st.sidebar.file_uploader("Upload Data (CSV/XLSX)", type=["csv", "xlsx"])
sql_uri = st.sidebar.text_input("SQLite URI untuk SQL Agent", value="sqlite:///sample.db")

# LLM untuk Agent
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model=MODEL,
    temperature=0,
    streaming=True,
)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Tools
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
                llm,
                df,
                verbose=True,
                handle_parsing_errors=True,
                allow_dangerous_code=True,  # penting untuk analisis dataframe
            )
            tools.append(PythonREPLTool())
            tools.extend(pandas_agent.tools)
            st.success(f"File **{uploaded_file.name}** berhasil dimuat, siap analisis dengan Python Agent.")
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

# Agent
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

# Chat UI
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
