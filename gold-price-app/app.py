import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Gold Price App", layout="wide")
st.title("📈 Gold Price Analysis & Prediction")
st.write("Sumber data publik & stabil (LBMA Gold Price)")

# ======================
# LOAD DATA (STABIL)
# ======================
@st.cache_data
def load_data():
    url = "https://data.nasdaq.com/api/v3/datasets/LBMA/GOLD.csv"
    df = pd.read_csv(url)

    # Kolom PASTI ADA
    df = df[["Date", "USD (AM)"]].rename(columns={
        "Date": "date",
        "USD (AM)": "close"
    })

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna()
    df = df.sort_values("date")

    return df

data = load_data()

# ======================
# METRICS
# ======================
st.subheader("📊 Statistik Harga Emas")

c1, c2, c3 = st.columns(3)
c1.metric("Harga Tertinggi", f"${data['close'].max():,.2f}")
c2.metric("Harga Terendah", f"${data['close'].min():,.2f}")
c3.metric("Rata-rata Harga", f"${data['close'].mean():,.2f}")

# ======================
# GRAFIK HISTORIS
# ======================
st.subheader("📉 Grafik Historis Harga Emas")
st.line_chart(data.set_index("date")["close"])

# ======================
# PREDIKSI
# ======================
st.subheader("🤖 Prediksi Harga Emas (Linear Regression)")

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
    "Prediksi Harga ($)": pred
})

st.dataframe(pred_df, use_container_width=True)

fig, ax = plt.subplots()
ax.plot(data["date"], data["close"], label="Historis")
ax.plot(future_dates, pred, "--", label="Prediksi")
ax.legend()
st.pyplot(fig)

st.caption("Data: LBMA Gold Price | Streamlit + ML")
