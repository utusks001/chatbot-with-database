# utils1.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def detect_data_types(df):
    """Mendeteksi tipe kolom kategorikal dan numerik."""
    categorical_cols = []
    numeric_cols = []
    
    for col in df.columns:
        # Menangani nilai kosong atau non-numerik yang bisa diubah
        try:
            # Mencoba mengonversi ke numerik
            pd.to_numeric(df[col])
            numeric_cols.append(col)
        except (ValueError, TypeError):
            # Jika gagal, anggap sebagai kategorikal
            categorical_cols.append(col)
            
    return categorical_cols, numeric_cols

def recommend_and_plot(df, categorical_cols, numeric_cols):
    """Merekomendasikan dan membuat visualisasi berdasarkan tipe data."""
    visualizations = {}

    # Visualisasi untuk data numerik tunggal
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribusi: {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frekuensi')
        plt.tight_layout()
        visualizations[f'Histogram dari {col}'] = fig

    # Visualisasi untuk data kategorikal tunggal
    if len(categorical_cols) > 0:
        col = categorical_cols[0]
        fig, ax = plt.subplots(figsize=(8, 6))
        df[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Frekuensi: {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Jumlah')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        visualizations[f'Bar Chart dari {col}'] = fig

    # Visualisasi untuk kombinasi numerik dan kategorikal
    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        num_col = numeric_cols[0]
        cat_col = categorical_cols[0]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
        ax.set_title(f'Box Plot: {num_col} berdasarkan {cat_col}')
        ax.set_xlabel(cat_col)
        ax.set_ylabel(num_col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        visualizations[f'Box Plot dari {num_col} vs {cat_col}'] = fig

    return visualizations
