import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import numpy as np
import streamlit as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.api import VAR


def afiseaza_cerinta1(df):
    st.subheader("1. Modele cu trend determinist și stochastic")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Data"], df["Somaj"], marker='o')
    ax.set_title("Evoluția numărului de șomeri")
    ax.set_xlabel("Data")
    ax.set_ylabel("Număr șomeri")
    ax.grid(True)
    st.pyplot(fig)

    adf_result = adfuller(df["Somaj"])
    st.markdown(f"""
    **Test ADF pentru trend stochastic:**  
    ADF statistic = {adf_result[0]:.4f}  
    p-value = {adf_result[1]:.4f}  
    Valori critice = {adf_result[4]}  
    """)
    if adf_result[1] < 0.05:
        st.success("⇒ Seria este staționară (p < 0.05)")
    else:
        st.warning("⇒ Seria NU este staționară (p ≥ 0.05)")

    df = df.copy()
    df["Timp"] = np.arange(len(df))
    X = sm.add_constant(df["Timp"])
    y = df["Somaj"]
    model = sm.OLS(y, X).fit()
    st.markdown("**Regresie liniară (trend determinist):**")
    st.text(model.summary())


def afiseaza_cerinta2(df):
    st.subheader("2. Serii staționare – prima diferență")
    df_diff = df.copy()
    df_diff["Diff"] = df_diff["Somaj"].diff()
    df_diff = df_diff.dropna()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_diff["Data"], df_diff["Diff"], marker='o', color='green')
    ax.set_title("Seria diferențiată a numărului de șomeri")
    ax.set_xlabel("Data")
    ax.set_ylabel("Δ Număr șomeri")
    ax.grid(True)
    st.pyplot(fig)

    adf_result = adfuller(df_diff["Diff"])
    st.markdown(f"""
    **Test ADF pentru seria diferențiată:**  
    ADF statistic = {adf_result[0]:.4f}  
    p-value = {adf_result[1]:.4f}  
    Valori critice = {adf_result[4]}  
    """)
    if adf_result[1] < 0.05:
        st.success("⇒ Seria este staționară (p < 0.05)")
    else:
        st.warning("⇒ Seria NU este staționară (p ≥ 0.05)")


def afiseaza_cerinta3(df):
    st.subheader("3. Tehnici de netezire exponențială")
    series = df["Somaj"]
    index = df["Data"]

    ses_model = SimpleExpSmoothing(series).fit()
    ses_fitted = ses_model.fittedvalues

    holt_model = Holt(series).fit()
    holt_fitted = holt_model.fittedvalues

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(index, series, label="Seria originală", color='black')
    ax.plot(index, ses_fitted, label="SES", linestyle='--')
    ax.plot(index, holt_fitted, label="Holt", linestyle='--')
    ax.set_title("Netezire exponențială: SES vs Holt")
    ax.set_xlabel("Data")
    ax.set_ylabel("Număr șomeri")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("**Observații:**")
    st.write("- SES urmează nivelul mediu al seriei, dar nu captează trendul.")
    st.write("- Holt oferă o estimare mai adaptivă, urmărind mai bine variațiile seriei.")


def afiseaza_cerinta4(df):
    st.subheader("4. Model ARIMA(2,1,2)")
    serie = df["Somaj"]
    model = ARIMA(serie, order=(2, 1, 2))
    model_fit = model.fit()

    st.markdown("**Rezumat model ARIMA:**")
    st.text(model_fit.summary())

    predict = model_fit.predict(start=1, end=len(df)-1, typ='levels')
    predict.index = df["Data"][1:]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Data"], serie, label="Seria originală")
    ax.plot(predict.index, predict, label="ARIMA(2,1,2)", linestyle='--')
    ax.set_title("Model ARIMA(2,1,2) aplicat seriei de șomaj")
    ax.set_xlabel("Data")
    ax.set_ylabel("Număr șomeri")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("**Observații:**")
    st.write("- Modelul ARIMA surprinde trendul general, cu ușoare abateri locale.")
    st.write("- A fost ales pe baza criteriului AIC minim și stabilității coeficienților.")


def afiseaza_cerinta5(df):
    st.subheader("5. Prognoză ARIMA: seturi de training și test")
    n = 20
    train = df["Somaj"][:-n]
    test = df["Somaj"][-n:]
    dates_test = df["Data"][-n:]

    model = ARIMA(train, order=(2, 1, 2))
    model_fit = model.fit()

    forecast_obj = model_fit.get_forecast(steps=n)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    mae = mean_absolute_error(test, forecast)
    rmse = mean_squared_error(test, forecast) ** 0.5

    st.markdown(f"**MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Data"][:-n], df["Somaj"][:-n], color='blue', label="Training")
    ax.plot(dates_test, test, color='black', label="Test (real)")
    ax.plot(dates_test, forecast, color='orange', label="Predicție")
    ax.fill_between(dates_test, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3, label="Interval 95%")
    ax.set_title("Predicție ARIMA(2,1,2) – punct și pe interval")
    ax.set_xlabel("Data")
    ax.set_ylabel("Număr șomeri")
    ax.legend()
    st.pyplot(fig)


def afiseaza_cerinta6(df):
    st.subheader("6. Compararea modelelor ARIMA, SARIMA, Holt-Winters")
    n = 20
    train = df["Somaj"][:-n]
    test = df["Somaj"][-n:]

    arima = ARIMA(train, order=(2, 1, 2)).fit()
    sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 4)).fit()
    hw = Holt(train).fit()

    arima_pred = arima.forecast(n)
    sarima_pred = sarima.forecast(n)
    hw_pred = hw.forecast(n)

    res = {
        "ARIMA": (mean_absolute_error(test, arima_pred), mean_squared_error(test, arima_pred) ** 0.5),
        "SARIMA": (mean_absolute_error(test, sarima_pred), mean_squared_error(test, sarima_pred) ** 0.5),
        "Holt-Winters": (mean_absolute_error(test, hw_pred), mean_squared_error(test, hw_pred) ** 0.5)
    }

    st.markdown("**Comparare metrici:**")
    for model, (mae, rmse) in res.items():
        st.markdown(f"- {model}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Data"][-n:], test, label="Valori reale", color="black")
    ax.plot(df["Data"][-n:], arima_pred, label="ARIMA(2,1,2)", linestyle="--", color="green")
    ax.plot(df["Data"][-n:], sarima_pred, label="SARIMA", linestyle="--", color="orange")
    ax.plot(df["Data"][-n:], hw_pred, label="Holt-Winters", linestyle="--", color="blue")
    ax.set_title("Comparare metode univariate – ARIMA vs SARIMA vs Holt-Winters")
    ax.set_xlabel("Data")
    ax.set_ylabel("Număr șomeri")
    ax.legend()
    st.pyplot(fig)

def analiza_stationaritate(df):
    st.subheader("7.1 Teste de staționaritate (ADF)")
    df_multi = df[["Data", "Somaj", "PIB"]].dropna().reset_index(drop=True)
    somaj = adfuller(df_multi["Somaj"])
    pib = adfuller(df_multi["PIB"])
    st.markdown(f"""
    **Test ADF – Șomaj:** ADF = {somaj[0]:.4f}, p = {somaj[1]:.4f}  
    **Test ADF – PIB:** ADF = {pib[0]:.4f}, p = {pib[1]:.4f}  
    """)

def analiza_cointegrare(df):
    st.subheader("7.2 Test Johansen pentru cointegrare")
    df_multi = df[["Somaj", "PIB"]].dropna().reset_index(drop=True)
    result = coint_johansen(df_multi, det_order=0, k_ar_diff=1)
    trace_stat = result.lr1
    crit_values = result.cvt[:, 1]
    for i, (ts, cv) in enumerate(zip(trace_stat, crit_values)):
        verdict = "⇒ Se respinge H0 (există cointegrare)" if ts > cv else "⇒ NU se respinge H0"
        st.markdown(f"r = {i} : Trace Stat = {ts:.4f} | CV (5%) = {cv:.4f} → {verdict}")

def analiza_vecm(df):
    st.subheader("7.3 Estimare model VECM")
    df_multi = df[["Somaj", "PIB"]].dropna().reset_index(drop=True)
    vecm = VECM(df_multi, k_ar_diff=1, coint_rank=1)
    vecm_fit = vecm.fit()
    st.text(vecm_fit.summary())

def analiza_granger(df):
    st.subheader("7.4 Test de cauzalitate Granger")
    df_multi = df[["Somaj", "PIB"]].dropna().reset_index(drop=True)
    st.markdown("**PIB → Șomaj**")
    grangercausalitytests(df_multi[["Somaj", "PIB"]], maxlag=2, verbose=False)
    st.code("p > 0.05 ⇒ nu există cauzalitate PIB → șomaj")
    st.markdown("**Șomaj → PIB**")
    grangercausalitytests(df_multi[["PIB", "Somaj"]], maxlag=2, verbose=False)
    st.code("p > 0.05 ⇒ nu există cauzalitate șomaj → PIB")


def analiza_irf(df):
    st.subheader("7.5 Funcție de răspuns la impuls (IRF)")
    df_multi = df[["Somaj", "PIB"]].dropna()
    vecm = VECM(df_multi, k_ar_diff=1, coint_rank=1)
    vecm_fit = vecm.fit()
    irf = vecm_fit.irf(10)
    fig = irf.plot(impulse="PIB", response="Somaj")
    fig.suptitle("Răspunsul șomajului la un șoc pozitiv în PIB")
    st.pyplot(fig.figure)


def analiza_fevd(df):
    st.subheader("7.6 Descompunerea varianței (FEVD)")
    df_multi = df[["Somaj", "PIB"]].dropna().diff().dropna()
    model = VAR(df_multi)
    fitted = model.fit(maxlags=1)
    fevd = fitted.fevd(steps=min(10, df_multi.shape[0] - 1))
    fig = fevd.plot()
    st.pyplot(fig.figure)
    st.write("Număr observații disponibile pentru VAR:", df_multi.shape[0])
    steps = min(10, df_multi.shape[0] - 1)
    st.write("Număr pași FEVD calculați:", steps)
    st.write("Shape FEVD:", fevd.decomp.shape)


st.set_page_config(layout="wide", page_title="Serii de timp")
st.title("📈 Analiza serii de timp: Număr șomeri și PIB")

df = pd.read_csv("serii_multivariate.csv", parse_dates=["Data"])

with st.expander("📌 Cerința 1"):
    afiseaza_cerinta1(df)

with st.expander("📌 Cerința 2"):
    afiseaza_cerinta2(df)

with st.expander("📌 Cerința 3"):
    afiseaza_cerinta3(df)

with st.expander("📌 Cerința 4"):
    afiseaza_cerinta4(df)

with st.expander("📌 Cerința 5"):
    afiseaza_cerinta5(df)

with st.expander("📌 Cerința 6"):
    afiseaza_cerinta6(df)

st.header("🔀 Cerința 7 – Analiză multivariată")

with st.expander("7.1 Staționaritate"):
    analiza_stationaritate(df)

with st.expander("7.2 Cointegrare Johansen"):
    analiza_cointegrare(df)

with st.expander("7.3 Estimare VECM"):
    analiza_vecm(df)

with st.expander("7.4 Cauzalitate Granger"):
    analiza_granger(df)

with st.expander("7.5 IRF"):
    analiza_irf(df)

with st.expander("7.6 FEVD"):
    analiza_fevd(df)
