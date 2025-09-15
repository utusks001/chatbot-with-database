# app-langgraph.py 
# app-langgraph.py
import os
import sqlite3
import traceback
from dotenv import load_dotenv

from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain_experimental.tools.python.tool import PythonREPLTool

# ============ Load ENV ============
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY tidak ditemukan di .env")

# Kalau key pakai format baru sk-proj-... maka project wajib diisi
if OPENAI_API_KEY.startswith("sk-proj-"):
    if not OPENAI_PROJECT_ID:
        raise ValueError(
            "❌ OPENAI_PROJECT_ID wajib diisi di .env untuk key sk-proj-..."
        )
    client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ============ Tester dengan OpenAI official ============
def test_openai():
    try:
        print("✅ Menguji koneksi OpenAI...")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Halo, apakah koneksi API berhasil?"}],
        )
        print("OpenAI response:", resp.choices[0].message.content)
    except Exception as e:
        print("❌ Gagal test OpenAI:", e)
        traceback.print_exc()

# ============ Setup SQLite Dummy DB ============
def init_db():
    conn = sqlite3.connect("test.db")
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    cur.execute("DELETE FROM users")
    cur.executemany(
        "INSERT INTO users (name, age) VALUES (?, ?)",
        [("Alice", 25), ("Bob", 30), ("Charlie", 35)],
    )
    conn.commit()
    conn.close()

# ============ LangChain Setup ============
def build_agents():
    # DB Agent
    db = SQLDatabase.from_uri("sqlite:///test.db")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

    sql_tool = Tool(
        name="SQL Database",
        func=db_chain.run,
        description="Gunakan untuk menjawab pertanyaan tentang database users",
    )

    # Python REPL Agent
    python_tool = PythonREPLTool()

    return [sql_tool, python_tool]

# ============ Main ============
if __name__ == "__main__":
    # Test koneksi OpenAI langsung
    test_openai()

    # Init DB dummy
    init_db()

    # Build agents
    tools = build_agents()
    print("✅ Tools tersedia:", [t.name for t in tools])

    # Demo SQL Agent
    try:
        print("\n🔎 Query ke SQL Agent...")
        result = tools[0].func("SELECT name FROM users WHERE age > 28;")
        print("Hasil query:", result)
    except Exception as e:
        print("❌ Error SQL Agent:", e)

    # Demo Python Agent
    try:
        print("\n🐍 Tes Python Agent...")
        result = tools[1].run("2 + 3 * 5")
        print("Hasil Python Agent:", result)
    except Exception as e:
        print("❌ Error Python Agent:", e)
