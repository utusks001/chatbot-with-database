import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.agent_toolkits import create_pandas_dataframe_agent

import pandas as pd
import sqlite3

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Chatbot Data Analysis", layout="wide")
st.title("🤖 Chatbot Otomatisasi Data Analysis")

# Pilih mode
mode = st.radio("Pilih Mode Analisis:", ["Python (CSV/Excel)", "SQL (Database)"])

# Siapkan LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)

if mode == "Python (CSV/Excel)":
    st.subheader("Upload File CSV / Excel")
    file = st.file_uploader("Upload data", type=["csv", "xlsx"])

    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.write("📊 Data Preview:")
        st.dataframe(df.head())

        agent = create_pandas_dataframe_agent(llm, df, verbose=True)

        query = st.text_area("Tanyakan sesuatu tentang data Anda:")
        if st.button("Jalankan Analisis"):
            with st.spinner("Sedang menganalisis..."):
                result = agent.run(query)
                st.success(result)

elif mode == "SQL (Database)":
    st.subheader("Koneksi ke Database SQLite")
    db_file = st.file_uploader("Upload SQLite DB", type=["db", "sqlite"])

    if db_file:
        with open("temp.db", "wb") as f:
            f.write(db_file.getbuffer())

        conn_str = "sqlite:///temp.db"
        db_chain = SQLDatabaseChain.from_uri(conn_str, llm)

        query = st.text_area("Tulis pertanyaan (SQL/Natural Language):")
        if st.button("Jalankan Query"):
            with st.spinner("Sedang query ke database..."):
                result = db_chain.run(query)
                st.success(result)
