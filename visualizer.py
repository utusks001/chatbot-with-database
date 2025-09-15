$ visualizer.py
import seaborn as sns
import matplotlib.pyplot as plt

def detect_column_types(df):
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return categorical, numerical

def plot_heatmap(df, numerical_cols):
    fig, ax = plt.subplots()
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    return fig

def plot_boxplot(df, cat_col, num_col):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
    return fig
