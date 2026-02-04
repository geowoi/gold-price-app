import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Gold Price App", layout="wide")

st.title("📈 Aplikasi Analisis & Prediksi Harga Emas")
st.write("Sumber data publik (CSV), diproses secara robust")

# ======================
# Load data (ANTI ERROR)
# ======================
@st.cache_data
def load_data():
    url = "https://stooq.pl/q/d/l/?s=gold&i=d"
    df = pd.read_csv(url)

    # 🔒 Normalisasi nama kolom (super penting)
    df.columns = [c.strip().lower() for c in df.columns]

    # Pastikan kolom inti ada
    required_cols = {"date", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Kolom tidak ditemukan. Kolom tersedia: {df.columns}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date")

    return df

data = load_data()

# ======================
# Statistik
# ======================
st.subheader("📊 Statistik Harga Emas")

col1, col2, col3 = st.columns(3)
col1.metric("Harga Tertinggi", f"${data['close'].max():.2f}")
col2.metric("Harga Terendah", f"${data['close'].min():.2f}")
col3.metric("Rata-rata Harga", f"${data['close'].mean():.2f}")

# ======================
# Grafik Historis
# ======================
st.subheader("📉 Grafik Harga Emas Historis")
st.line_chart(data.set_index("date")["close"])

# ======================
# Prediksi Harga
# ======================
st.subheader("🤖 Prediksi Harga Emas (Machine Learning)")

data["date_ordinal"] = data["date"].map(pd.Timestamp.toordinal)
X = data[["date_ordinal"]]
y = data["close"]

model = LinearRegression()
model.fit(X, y)

days = st.slider("Prediksi berapa hari ke depan?", 1, 180, 30)

future_ordinals = np.array([
    data["date_ordinal"].max() + i for i in range(1, days + 1)
]).reshape(-1, 1)

predictions = model.predict(future_ordinals)

future_dates = pd.date_range(
    start=data["date"].iloc[-1],
    periods=days + 1,
    freq="D"
)[1:]

pred_df = pd.DataFrame({
    "Tanggal": future_dates,
    "Prediksi Harga ($)": predictions
})

st.dataframe(pred_df)

fig, ax = plt.subplots()
ax.plot(data["date"], data["close"], label="Data Historis")
ax.plot(future_dates, predictions, "--", label="Prediksi")
ax.legend()
st.pyplot(fig)

st.caption("Aplikasi AI — Data publik CSV, Streamlit, Machine Learning")
