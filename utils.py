import pandas as pd

def load_excel(uploaded_file):
    """
    Load multi-sheet Excel atau CSV dari Streamlit UploadedFile.
    uploaded_file: st.file_uploader result
    Returns: dict {sheet_name: dataframe}
    """
    # Cek tipe file berdasarkan nama
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        return {"Sheet1": df}
    else:
        xls = pd.ExcelFile(uploaded_file)
        return {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

def detect_column_types(df):
    """
    Deteksi kolom numerik vs kategori
    Returns: numeric_cols, categorical_cols
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    return numeric_cols, categorical_cols

def chunk_dataframe(df, chunk_size=5000):
    """
    Bagi dataframe besar menjadi chunk untuk RAG embedding
    Returns: list of dataframes
    """
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i+chunk_size])
    return chunks
