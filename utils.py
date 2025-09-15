import pandas as pd
from io import BytesIO, StringIO

def load_excel(uploaded_file):
    """
    Load multi-sheet Excel atau CSV dari Streamlit UploadedFile.

    Args:
        uploaded_file: objek hasil st.file_uploader

    Returns:
        dict: {sheet_name: dataframe}
    """
    # Gunakan uploaded_file.name untuk cek ekstensi
    filename = uploaded_file.name.lower()

    if filename.endswith('.csv'):
        # CSV
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
        return {"Sheet1": df}
    elif filename.endswith(('.xls', '.xlsx')):
        # Excel
        bytes_data = BytesIO(uploaded_file.getvalue())
        xls = pd.ExcelFile(bytes_data)
        return {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
    else:
        raise ValueError("File harus berekstensi .csv, .xls, atau .xlsx")

def detect_column_types(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    return numeric_cols, categorical_cols

def chunk_dataframe(df, chunk_size=5000):
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i+chunk_size])
    return chunks
