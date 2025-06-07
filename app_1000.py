import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Configurații generale
st.set_page_config(page_title="Analiza Serii de Timp", layout="wide")
st.title("📈 Proiect: Prognoza numărului de șomeri și analiza relației cu PIB-ul")

# === Încărcare date ===
df = pd.read_csv("serii_multivariate.csv", parse_dates=["Data"])
df = df.rename(columns={"Somaj": "Numar someri"})

st.markdown("### 📄 Descriere date")
with st.expander("ℹ️ Informații despre setul de date", expanded=True):
    st.markdown("- **Perioadă**: 2000–2024  \n- **Frecvență**: Trimestrial  \n- **Serii analizate**: Număr șomeri, PIB")
    st.dataframe(df.head())

# === Cerința 1 ===
with st.expander("1️⃣ Modele cu trend determinist sau stochastic"):
    st.markdown("Se verifică existența unui trend în seria numărului de șomeri.")
    fig, ax = plt.subplots()
    ax.plot(df["Data"], df["Numar someri"], label="Număr șomeri")
    ax.set_title("Evoluția numărului de șomeri")
    st.pyplot(fig)

    result = adfuller(df["Numar someri"])
    st.markdown(f"- **ADF statistic**: {result[0]:.4f}  \n- **p-value**: {result[1]:.4f}")
    st.markdown("Seria **nu este staționară**, deci se aplică un model cu trend determinist.")

# === Cerința 2 ===
with st.expander("2️⃣ Serii staționare"):
    df_diff = df["Numar someri"].diff().dropna()
    result_diff = adfuller(df_diff)
    st.markdown(f"ADF după diferențiere: {result_diff[0]:.4f}, p-value = {result_diff[1]:.4f}")
    st.line_chart(df_diff)

# === Cerința 3 ===
with st.expander("3️⃣ Tehnici de netezire exponențială"):
    model = Holt(df["Numar someri"]).fit()
    forecast = model.forecast(10)
    fig, ax = plt.subplots()
    ax.plot(df["Data"], df["Numar someri"], label="Istoric")
    ax.plot(pd.date_range(df["Data"].iloc[-1], periods=11, freq="Q")[1:], forecast, label="Forecast Holt")
    ax.legend()
    st.pyplot(fig)

# === Cerința 4 ===
with st.expander("4️⃣ Modele ARMA/ARIMA/SARIMA"):
    model_arima = ARIMA(df["Numar someri"], order=(2, 1, 2)).fit()
    st.text(model_arima.summary())

# === Cerința 5 ===
with st.expander("5️⃣ Prognoză ARIMA: seturi de training și test"):
    n = 20
    train = df["Numar someri"][:-n]
    test = df["Numar someri"][-n:]
    model = ARIMA(train, order=(2, 1, 2)).fit()
    forecast = model.forecast(steps=n)
    mae = mean_absolute_error(test, forecast)
    rmse = mean_squared_error(test, forecast) ** 0.5
    st.markdown(f"**MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}")

    fig, ax = plt.subplots()
    ax.plot(df["Data"][:-n], train, label="Train")
    ax.plot(df["Data"][-n:], test, label="Test")
    ax.plot(df["Data"][-n:], forecast, label="Forecast")
    ax.legend()
    st.pyplot(fig)

# === Cerința 6 ===
with st.expander("6️⃣ Compararea modelelor ARIMA, SARIMA, Holt-Winters"):
    sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 4)).fit()
    holt = Holt(train).fit()

    pred_arima = model.forecast(n)
    pred_sarima = sarima.forecast(n)
    pred_holt = holt.forecast(n)

    rezultate = {
        "ARIMA": (mean_absolute_error(test, pred_arima), mean_squared_error(test, pred_arima) ** 0.5),
        "SARIMA": (mean_absolute_error(test, pred_sarima), mean_squared_error(test, pred_sarima) ** 0.5),
        "Holt-Winters": (mean_absolute_error(test, pred_holt), mean_squared_error(test, pred_holt) ** 0.5),
    }

    for model, (mae, rmse) in rezultate.items():
        st.markdown(f"- **{model}**: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

# === Cerința 7 ===
with st.expander("7️⃣ Analiză multivariată: Cointegrare, VAR/VECM, Granger, IRF, FEVD"):
    df_multi = df[["Numar someri", "PIB"]].dropna()
    df_diff = df_multi.diff().dropna()

    st.markdown("**7.1 ADF Test:**")
    adf_somaj = adfuller(df_multi["Numar someri"])
    adf_pib = adfuller(df_multi["PIB"])
    st.write(f"Somaj: ADF={adf_somaj[0]:.3f}, p={adf_somaj[1]:.3f}")
    st.write(f"PIB: ADF={adf_pib[0]:.3f}, p={adf_pib[1]:.3f}")

    st.markdown("**7.2 Test Johansen:**")
    joh = coint_johansen(df_multi, det_order=0, k_ar_diff=1)
    st.write("Eigenvalues:", joh.eig)

    st.markdown("**7.3 VECM și coeficienți:**")
    vecm = VECM(df_multi, k_ar_diff=1, coint_rank=1).fit()
    st.text(vecm.summary())

    st.markdown("**7.4 Granger Causality (maxlag=4):**")
    with st.echo():
        grangercausalitytests(df_multi[["Numar someri", "PIB"]], maxlag=4, verbose=False)

    st.markdown("**7.5 Funcție de răspuns la impuls (IRF):**")
    model = VAR(df_diff)
    results = model.fit(maxlags=2, ic="aic")
    irf = results.irf(10)
    fig = irf.plot(impulse="PIB", response="Numar someri")
    st.pyplot(fig.figure)

    st.markdown("**7.6 Descompunerea variantei (FEVD):**")
    fevd = results.fevd(10)
    st.write("Contribuția PIB la variația Somajului:")
    st.write(fevd.decomp[:, 0, 1])
    fig = fevd.plot()
    st.pyplot(fig)

# === Final ===
st.caption("Proiect realizat în Python / Streamlit — Serii de timp (2025)")
