# app-langgraph.py 
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import sqlite3
import tempfile
from typing import Optional

from langchain import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.memory import ConversationBufferMemory
from langchain.chains import SQLDatabaseChain
from langchain.sql_database import SQLDatabase

# LangSmith tracer
from langchain.callbacks import LangsmithTracer

# --- Load environment ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
st.warning("OPENAI_API_KEY belum ditemukan di .env — agent tidak akan bekerja tanpa API key.")

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot Data Analysis (LangGraph)", layout="wide")
st.title("🤖 Chatbot Data Analysis — LangGraph (stateful)")

col1, col2 = st.columns([2,1])

with col2:
st.header("Settings")
st.text_input("OpenAI API Key", value=OPENAI_API_KEY or "", key="_openai_key", type="password")
st.text_input("LangSmith API Key (optional)", value=LANGSMITH_API_KEY or "", key="_langsmith_key", type="password")
st.selectbox("Mode (toolset)", ["Python Agent", "SQL Agent"], key="mode_select")
reset = st.button("Reset Conversation")

with col1:
st.header("Upload / Data Source")
uploaded_file = st.file_uploader("Upload CSV / Excel (untuk Python Agent)", type=["csv","xlsx"])
uploaded_db = st.file_uploader("Upload SQLite DB (untuk SQL Agent)", type=["db","sqlite"])

# --- LLM & tracing setup ---
llm_kwargs = {"temperature": 0}

if OPENAI_API_KEY:
llm = OpenAI(api_key=OPENAI_API_KEY, model_name=MODEL, **llm_kwargs)
else:
llm = OpenAI(api_key=None, model_name=MODEL, **llm_kwargs) # will error if called

tracer = None
if LANGSMITH_API_KEY:
tracer = LangsmithTracer(project_name="chatbot-data-analysis", api_key=LANGSMITH_API_KEY)
st.sidebar.success("LangSmith tracing enabled")
else:
st.sidebar.info("LangSmith disabled — set LANGSMITH_API_KEY in .env to enable")

# Attach tracer if available
callbacks = [tracer] if tracer else []

# --- Conversation memory (stateful) ---
if "memory" not in st.session_state:
st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if reset:
st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
st.success("Conversation memory reset")

# --- Tools setup ---
tools = []

st.write([t.name for t in tools])
