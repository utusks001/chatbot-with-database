# data_loader.py
import pandas as pd

def load_file(uploaded_file):
    sheet_data = {}
    if uploaded_file.name.endswith(".csv"):
        sheet_data["Sheet1"] = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        for sheet in xls.sheet_names:
            sheet_data[sheet] = xls.parse(sheet)
    return sheet_data
