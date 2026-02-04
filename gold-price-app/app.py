import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Gold Price Analysis & Prediction", layout="wide")

st.title("📈 Gold Price Analysis & Prediction")
st.write("Sumber data publik (CSV GitHub – stabil untuk Streamlit Cloud)")

# ======================
# LOAD DATA (ANTI BLOK)
# ======================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
    df = pd.read_csv(url)

    df = df[["Date", "AAPL.Close"]]
    df.columns = ["date", "close"]

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna()
    df = df.sort_values("date")

    return df

data = load_data()

# ======================
# STATISTIK
# ======================
st.subheader("📊 Statistik Harga")

c1, c2, c3 = st.columns(3)
c1.metric("Harga Maksimum", f"${data['close'].max():,.2f}")
c2.metric("Harga Minimum", f"${data['close'].min():,.2f}")
c3.metric("Harga Rata-rata", f"${data['close'].mean():,.2f}")

# ======================
# GRAFIK
# ======================
st.subheader("📉 Grafik Historis Harga")
st.line_chart(data.set_index("date")["close"])

# ======================
# PREDIKSI AI
# ======================
st.subheader("🤖 Prediksi Harga (Linear Regression)")

data["ordinal"] = data["date"].map(pd.Timestamp.toordinal)

X = data[["ordinal"]]
y = data["close"]

model = LinearRegression()
model.fit(X, y)

days = st.slider("Prediksi berapa hari ke depan?", 1, 180, 30)

future_ord = np.array(
    [data["ordinal"].max() + i for i in range(1, days + 1)]
).reshape(-1, 1)

pred = model.predict(future_ord)

future_dates = pd.date_range(
    start=data["date"].iloc[-1],
    periods=days + 1,
    freq="D"
)[1:]

pred_df = pd.DataFrame({
    "Tanggal": future_dates,
    "Prediksi Harga": pred
})

st.dataframe(pred_df, use_container_width=True)

fig, ax = plt.subplots()
ax.plot(data["date"], data["close"], label="Data Historis")
ax.plot(future_dates, pred, "--", label="Prediksi")
ax.legend()
st.pyplot(fig)

st.caption("Aplikasi AI – Analisis & Prediksi Harga (Streamlit + Machine Learning)")
