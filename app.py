import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# ======================
# LOAD DATA
# ======================
st.title("Dashboard Analisis Data Responden FISIKA KOMPUTASI")

file_path = "fiskom responden.xlsx"
df = pd.read_excel(file_path)

st.subheader("Preview Data")
st.write(df.head())

# ======================
# INFO DATA
# ======================
st.subheader("Informasi Data")
st.write("Jumlah Baris dan Kolom:", df.shape)
st.write("Tipe Data:")
st.write(df.dtypes)

# ======================
# STATISTIK DESKRIPTIF
# ======================
st.subheader("Statistik Deskriptif")
st.write(df.describe())

# ======================
# PILIH KOLOM NUMERIK
# ======================
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_columns) >= 2:
    st.subheader("Visualisasi Data")

    x_col = st.selectbox("Pilih Variabel X", numeric_columns)
    y_col = st.selectbox("Pilih Variabel Y", numeric_columns)

    # Scatter Plot
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Scatter Plot {x_col} vs {y_col}")
    st.pyplot(fig)

    # ======================
    # REGRESI LINEAR
    # ======================
    st.subheader("Analisis Regresi Linear")

    X = df[[x_col]]
    y = df[y_col]

    model = LinearRegression()
    model.fit(X, y)

    st.write("Koefisien:", model.coef_[0])
    st.write("Intercept:", model.intercept_)

    # Statsmodels (detail statistik)
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()

    st.text(model_sm.summary())

else:
    st.warning("Data numerik kurang dari 2 kolom untuk analisis.")
