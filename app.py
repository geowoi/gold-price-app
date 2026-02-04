import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Gold Price Prediction", layout="wide")

st.title("📈 Aplikasi Prediksi Harga Emas")
st.write("Aplikasi kecerdasan buatan untuk analisis dan prediksi harga emas berbasis data historis.")

# ======================
# Ambil Data
# ======================
@st.cache_data
def load_data():
    data = yf.download("GC=F", start="2010-01-01")
    data.reset_index(inplace=True)
    return data

data = load_data()

# 🔧 FIX ERROR
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

# ======================
# Statistik
# ======================
st.subheader("📊 Statistik Harga Emas")
col1, col2, col3 = st.columns(3)

col1.metric("Harga Tertinggi", f"${data['Close'].max():.2f}")
col2.metric("Harga Terendah", f"${data['Close'].min():.2f}")
col3.metric("Rata-rata Harga", f"${data['Close'].mean():.2f}")

# ======================
# Grafik Historis
# ======================
st.subheader("📉 Grafik Harga Emas Historis")
st.line_chart(data.set_index("Date")["Close"])

# ======================
# Prediksi Harga
# ======================
st.subheader("🤖 Prediksi Harga Emas (Machine Learning)")

data["Date_ordinal"] = data["Date"].map(pd.Timestamp.toordinal)
X = data[["Date_ordinal"]]
y = data["Close"]

model = LinearRegression()
model.fit(X, y)

days = st.slider("Prediksi berapa hari ke depan?", 1, 180, 30)

future_dates = np.array([
    data["Date_ordinal"].max() + i for i in range(1, days + 1)
]).reshape(-1, 1)

predictions = model.predict(future_dates)

future_df = pd.DataFrame({
    "Hari ke-": range(1, days + 1),
    "Prediksi Harga ($)": predictions
})

st.dataframe(future_df.head())

fig, ax = plt.subplots()
ax.plot(data["Date"], data["Close"], label="Data Historis")
ax.plot(
    pd.date_range(data["Date"].iloc[-1], periods=days + 1, freq="D")[1:],
    predictions,
    linestyle="--",
    label="Prediksi"
)
ax.legend()
st.pyplot(fig)

st.caption("Dibangun menggunakan Python, Streamlit, dan Machine Learning")
