# rag_engine.py
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
from langsmith import Client

def build_rag_chain(df, llm):
    docs = DataFrameLoader(df).load()
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    return RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

def log_to_langsmith(run_name, question, answer):
    client = Client()
    client.create_run(name=run_name, inputs={"question": question}, outputs={"answer": answer})
