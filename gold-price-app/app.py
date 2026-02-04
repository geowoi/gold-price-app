import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Gold Price App", layout="wide")

st.title("📈 Aplikasi Analisis & Prediksi Harga Emas")
st.write("Data diambil dari Stooq (sumber publik & stabil)")

# ======================
# Ambil Data dari STOOQ
# ======================
@st.cache_data
def load_data():
    url = "https://stooq.pl/q/d/l/?s=gold&i=d"
    data = pd.read_csv(url)

    # Rename kolom biar konsisten
    data.columns = [c.capitalize() for c in data.columns]

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")

    return data

data = load_data()

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
st.subheader("📉 Grafik Harga Emas")
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

future_ordinals = np.array([
    data["Date_ordinal"].max() + i for i in range(1, days + 1)
]).reshape(-1, 1)

predictions = model.predict(future_ordinals)

future_dates = pd.date_range(
    start=data["Date"].iloc[-1],
    periods=days + 1,
    freq="D"
)[1:]

pred_df = pd.DataFrame({
    "Tanggal": future_dates,
    "Prediksi Harga ($)": predictions
})

st.dataframe(pred_df)

fig, ax = plt.subplots()
ax.plot(data["Date"], data["Close"], label="Data Historis")
ax.plot(future_dates, predictions, "--", label="Prediksi")
ax.legend()
st.pyplot(fig)

st.caption("Sumber data: stooq.pl | Dibangun dengan Streamlit & Machine Learning")
