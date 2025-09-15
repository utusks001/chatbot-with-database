import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

# LangChain & Agents
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities import SQLDatabase

# Load environment
load_dotenv()

# --- Select LLM provider ---
def get_llm():
    if os.getenv("GOOGLE_API_KEY"):
        st.sidebar.success("✅ Using Google Gemini")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    elif os.getenv("GROQ_API_KEY"):
        st.sidebar.success("✅ Using Groq LLaMA/Mixtral")
        return ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    else:
        st.sidebar.error("❌ No valid Google or Groq API key found")
        return None

# --- UI ---
st.set_page_config(page_title="LangGraph Multi-Agent (No OpenAI)", layout="wide")
st.title("🤖 Chatbot with Python + MySQL Agent (No OpenAI)")

llm = get_llm()
if not llm:
    st.stop()

# --- Multi-file upload (CSV/XLSX) ---
uploaded_files = st.file_uploader(
    "Upload CSV/XLSX files", type=["csv", "xlsx"], accept_multiple_files=True
)

dfs = {}
if uploaded_files:
    for file in uploaded_files:
        try:
            if file.name.endswith(".csv"):
                dfs[file.name] = pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                dfs[file.name] = pd.read_excel(file)
        except Exception as e:
            st.error(f"⚠️ Failed to load {file.name}: {e}")

if dfs:
    st.write("📂 Loaded datasets:", list(dfs.keys()))
    selected_file = st.selectbox("Pick a dataset", list(dfs.keys()))
    df = dfs[selected_file]
    st.dataframe(df.head())

    # Python Agent
    if st.checkbox("Enable Python Agent for Data Analysis"):
        agent_df = create_pandas_dataframe_agent(
            llm, df, verbose=True, allow_dangerous_code=True
        )
        query = st.text_input("Ask about the dataset:")
        if query:
            try:
                response = agent_df.run(query)
                st.success(response)
            except Exception as e:
                st.error(f"⚠️ Agent error: {e}")

# --- MySQL Agent ---
if st.checkbox("Enable MySQL Agent"):
    mysql_uri = st.text_input("MySQL URI (e.g. mysql+pymysql://user:pass@localhost/db)")
    if mysql_uri:
        try:
            db = SQLDatabase.from_uri(mysql_uri)
            db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
            sql_query = st.text_input("Ask about the database:")
            if sql_query:
                try:
                    result = db_chain.run(sql_query)
                    st.success(result)
                except Exception as e:
                    st.error(f"⚠️ SQL Agent error: {e}")
        except Exception as e:
            st.error(f"⚠️ DB connection error: {e}")

# --- General Chatbot ---
st.subheader("💬 General Chat")
user_input = st.text_input("Ask me anything:")
if user_input:
    try:
        response = llm.invoke(user_input)
        st.info(response.content)
    except Exception as e:
        st.error(f"⚠️ Chat error: {e}")
