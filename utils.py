import pandas as pd
import numpy as np

def load_excel(file_path):
    """Load multi-sheet Excel / CSV"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return {"Sheet1": df}
    else:
        xls = pd.ExcelFile(file_path)
        return {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

def detect_column_types(df):
    """Deteksi kolom numerik vs kategori"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    return numeric_cols, categorical_cols

def chunk_dataframe(df, chunk_size=5000):
    """Bagi dataframe besar menjadi chunk untuk RAG embedding"""
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i+chunk_size])
    return chunks
