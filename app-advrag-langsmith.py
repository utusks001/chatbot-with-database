# app-advrag-langsmith.py
import streamlit as st
from data_loader import load_file
from visualizer import detect_column_types, plot_heatmap, plot_boxplot
from chat_engine import get_llm, run_chat
from rag_engine import build_rag_chain, log_to_langsmith
import config

st.set_page_config(page_title="AI Chatbot Dashboard", layout="wide")
st.title("📊 AI Chatbot Dashboard for Data Analysis")

uploaded_file = st.file_uploader("Upload Excel/CSV", type=["csv", "xls", "xlsx"])
if uploaded_file:
    sheets = load_file(uploaded_file)
    selected_sheet = st.selectbox("Select Sheet", list(sheets.keys()))
    df = sheets[selected_sheet]
    st.write(df.head())

    cat_cols, num_cols = detect_column_types(df)
    if len(num_cols) >= 2:
        st.subheader("🔗 Korelasi Numerik")
        st.pyplot(plot_heatmap(df, num_cols))
    if cat_cols and num_cols:
        st.subheader("📦 Boxplot Kategori vs Numerik")
        cat = st.selectbox("Kategori", cat_cols)
        num = st.selectbox("Numerik", num_cols)
        st.pyplot(plot_boxplot(df, cat, num))

    st.sidebar.title("💬 Chat Settings")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    model = st.sidebar.selectbox("LLM", ["openai", "groq", "gemini"], index=0)
    temp = st.sidebar.slider("Temperature", 0.0, 1.0, config.DEFAULT_TEMPERATURE)
    top_p = st.sidebar.slider("Top-p", 0.0, 1.0, config.DEFAULT_TOP_P)
    max_tokens = st.sidebar.slider("Max Tokens", 128, 2048, config.DEFAULT_MAX_TOKENS)

    llm = get_llm(model, temp, top_p, max_tokens)
    user_input = st.text_input("Ask your data question:")
    if user_input:
        response, history = run_chat(llm, st.session_state.chat_history, user_input)
        st.session_state.chat_history = history
        st.sidebar.markdown(f"🙋‍♂️ You: {user_input}")
        st.sidebar.markdown(f"🧠 AI: {response.content}")
        st.write(response.content)

        rag_chain = build_rag_chain(df, llm)
        rag_answer = rag_chain.run(user_input)
        st.subheader("🔍 RAG Answer")
        st.write(rag_answer)
        log_to_langsmith("DataAnalysisChat", user_input, rag_answer)
