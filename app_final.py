import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Proiect Serii de Timp", page_icon="📈")

st.title("📊 Proiect Serii de Timp – Prognoza șomajului și influența PIB")
st.markdown("""
Aplicație integrată care urmărește analiza și modelarea seriilor de timp univariate și multivariate 
conform cerințelor proiectului.
""")
st.markdown("---")

# === Încărcare fișiere prelucrate ===
df_somaj = pd.read_csv("cerinta1_output.csv", parse_dates=["Data"])
df_diff = pd.read_csv("cerinta2_output.csv", parse_dates=["Data"])
df_multiv = pd.read_csv("serii_multivariate.csv", parse_dates=["Data"])

# === CERINȚA 1 ===
st.header("Cerința 1 – Trend determinist și stochastic")
st.subheader("Vizualizare serie șomaj")
fig1, ax1 = plt.subplots()
ax1.plot(df_somaj["Data"], df_somaj["Numar someri"], marker='o')
ax1.set_title("Evoluția numărului de șomeri")
ax1.grid(True)
st.pyplot(fig1)

st.subheader("Test ADF pentru trend stochastic")
adf_stat, pval, _, _, crit_vals, _ = adfuller(df_somaj["Numar someri"])
with st.expander("Rezultate test ADF"):
    st.write(f"ADF statistic: {adf_stat:.4f}")
    st.write(f"p-value: {pval:.4f}")
    st.write("Valori critice:")
    st.write(crit_vals)

st.subheader("Trend determinist (regresie pe timp)")
df_somaj["Timp"] = np.arange(len(df_somaj))
X = sm.add_constant(df_somaj["Timp"])
y = df_somaj["Numar someri"]
model = sm.OLS(y, X).fit()
with st.expander("Rezumat regresie"):
    st.text(model.summary())

# === CERINȚA 2 ===
st.header("Cerința 2 – Serii staționare (diferențiere)")
fig2, ax2 = plt.subplots()
ax2.plot(df_diff["Data"], df_diff["Diferenta somaj"], marker='o')
ax2.set_title("Seria diferențiată a numărului de șomeri")
ax2.grid(True)
st.pyplot(fig2)

adf_stat2, pval2, _, _, crit_vals2, _ = adfuller(df_diff["Diferenta somaj"])
with st.expander("Test ADF pe seria diferențiată"):
    st.write(f"ADF statistic: {adf_stat2:.4f}")
    st.write(f"p-value: {pval2:.4f}")
    st.write("Valori critice:")
    st.write(crit_vals2)

# === CERINȚA 3 ===
st.header("Cerința 3 – Tehnici de netezire exponențială")
serie = df_somaj["Numar someri"]
ses_model = SimpleExpSmoothing(serie, initialization_method="legacy-heuristic").fit()
holt_model = Holt(serie, initialization_method="legacy-heuristic").fit()

fig3, ax3 = plt.subplots()
ax3.plot(df_somaj["Data"], serie, label="Originală", color="black")
ax3.plot(df_somaj["Data"], ses_model.fittedvalues, label="SES", linestyle="--")
ax3.plot(df_somaj["Data"], holt_model.fittedvalues, label="Holt", linestyle="--")
ax3.set_title("Aplicare SES și Holt")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)

# Restul cerintelor vor fi adaugate intr-un pas urmator
st.markdown("---")
st.info("Următoarele cerințe (4–7) vor fi integrate în continuare.")

# === CERINȚA 4 ===
st.header("Cerința 4 – Alegerea modelului ARIMA optim (AIC)")
configuratii = [(1,1,1), (2,1,1), (1,1,2), (2,1,2)]
rezultate = {}
for cfg in configuratii:
    model = ARIMA(serie, order=cfg).fit()
    rezultate[str(cfg)] = model.aic
best_cfg = min(rezultate, key=rezultate.get)
st.write("Scoruri AIC pentru configurațiile testate:")
st.json(rezultate)
st.success(f"Modelul ales: ARIMA{best_cfg} (AIC minim)")

# === CERINȚA 5 ===
st.header("Cerința 5 – Predicție punctuală și pe interval cu ARIMA")
n = len(serie)
train_size = int(n * 0.8)
train, test = serie[:train_size], serie[train_size:]
dates_train, dates_test = df_somaj["Data"][:train_size], df_somaj["Data"][train_size:]
model_5 = ARIMA(train, order=(2, 1, 2)).fit()
forecast_5 = model_5.get_forecast(steps=len(test))
predicted_mean = forecast_5.predicted_mean
conf_int = forecast_5.conf_int()

fig5, ax5 = plt.subplots()
ax5.plot(dates_train, train, label="Training")
ax5.plot(dates_test, test, label="Test")
ax5.plot(dates_test, predicted_mean, label="Predicție")
ax5.fill_between(dates_test, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.3, label="Interval 95%")
ax5.set_title("Predicție ARIMA(2,1,2)")
ax5.legend()
ax5.grid(True)
st.pyplot(fig5)

# === CERINȚA 6 ===
st.header("Cerința 6 – Comparare metode univariate de prognoză")
sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,0,1,4)).fit()
sarima_forecast = sarima_model.forecast(steps=len(test))
hw_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=4).fit()
hw_forecast = hw_model.forecast(len(test))

fig6, ax6 = plt.subplots()
ax6.plot(dates_test, test, label="Real", color="black")
ax6.plot(dates_test, model_5.forecast(steps=len(test)), label="ARIMA", linestyle="--")
ax6.plot(dates_test, sarima_forecast, label="SARIMA", linestyle="--")
ax6.plot(dates_test, hw_forecast, label="Holt-Winters", linestyle="--")
ax6.set_title("Comparare metode de prognoză")
ax6.legend()
ax6.grid(True)
st.pyplot(fig6)

# === CERINȚA 7 ===
st.header("Cerința 7 – Analiză multivariată: șomaj și PIB")

st.subheader("Test ADF pentru staționaritate")
def test_adf(serie, nume):
    rezultat = adfuller(serie)
    with st.expander(f"Rezultate ADF pentru {nume}"):
        st.write(f"ADF statistic: {rezultat[0]:.4f}")
        st.write(f"p-value: {rezultat[1]:.4f}")
        st.write("Valori critice:", rezultat[4])
        if rezultat[1] < 0.05:
            st.success(f"{nume} este staționară")
        else:
            st.warning(f"{nume} NU este staționară")

test_adf(df_multiv["Somaj"], "Somaj")
test_adf(df_multiv["PIB"], "PIB")

st.subheader("Test de cointegrare Johansen")
johansen = coint_johansen(df_multiv[["Somaj", "PIB"]], det_order=0, k_ar_diff=1)
with st.expander("Rezultate Johansen"):
    for i, trace in enumerate(johansen.lr1):
        cv = johansen.cvt[i, 1]
        st.write(f"r = {i}: Trace Stat = {trace:.2f} | CV (5%) = {cv:.2f} ⇒ {'Există cointegrare' if trace > cv else 'Nu există'}")

st.subheader("Model VECM + funcția de răspuns la impuls")
vecm_model = VECM(df_multiv[["Somaj", "PIB"]], k_ar_diff=1, deterministic="n").fit()
irf = vecm_model.irf(10)
fig_irf = irf.plot(orth=False, impulse="PIB", response="Somaj")
st.pyplot(fig_irf.figure)

st.subheader("Test de cauzalitate Granger")
with st.expander("Rezultate Granger"):
    st.write("PIB → Șomaj")
    grangercausalitytests(df_multiv[["Somaj", "PIB"]], maxlag=4, verbose=True)
    st.write("Șomaj → PIB")
    grangercausalitytests(df_multiv[["PIB", "Somaj"]], maxlag=4, verbose=True)

st.subheader("FEVD – Descompunerea varianței")
data_diff = df_multiv[["Somaj", "PIB"]].diff().dropna()
var_model = VAR(data_diff).fit(maxlags=2, ic='aic')
fevd = var_model.fevd(10)
fig_fevd = fevd.plot()
st.pyplot(fig_fevd.figure)
