import pandas as pd

def load_excel(uploaded_file):
    """
    Load multi-sheet Excel atau CSV dari Streamlit UploadedFile.

    Args:
        uploaded_file: object dari st.file_uploader

    Returns:
        dict: {sheet_name: dataframe}
    """
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        return {"Sheet1": df}
    else:
        xls = pd.ExcelFile(uploaded_file)
        return {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

def detect_column_types(df):
    """
    Deteksi kolom numerik dan kategori dari dataframe.

    Args:
        df: pd.DataFrame

    Returns:
        tuple: (numeric_cols, categorical_cols)
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    return numeric_cols, categorical_cols

def chunk_dataframe(df, chunk_size=5000):
    """
    Bagi dataframe besar menjadi chunk untuk RAG embedding.

    Args:
        df: pd.DataFrame
        chunk_size: int

    Returns:
        list of pd.DataFrame
    """
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i+chunk_size])
    return chunks
