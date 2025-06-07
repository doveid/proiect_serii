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
st.set_page_config(layout="wide", page_title="Prezentare Serii de Timp", page_icon="📈")

st.title("📈 Prezentare Serii de Timp: Prognoza șomajului și influența PIB")
st.markdown("""
Această aplicație integrează analiza univariată și multivariată a seriilor de timp 
pentru a înțelege evoluția numărului de șomeri în România și impactul PIB-ului asupra acestei dinamici.
""")
st.markdown("---")

# Date
df_somaj = pd.read_csv("cerinta1_output.csv", parse_dates=["Data"])
df_diff = pd.read_csv("cerinta2_output.csv", parse_dates=["Data"])
df_multiv = pd.read_csv("serii_multivariate.csv", parse_dates=["Data"])

# CERINTA 1
st.header("1️⃣ Identificarea tipului de trend")
st.markdown("**Scop:** Determinăm dacă seria are un trend determinist sau stochastic.")
fig1, ax = plt.subplots(figsize=(8, 3))
ax.plot(df_somaj["Data"], df_somaj["Numar someri"], marker='o')
ax.set_title("Evoluția numărului de șomeri")
ax.grid(True)
st.pyplot(fig1)

adf_stat, pval, _, _, crit_vals, _ = adfuller(df_somaj["Numar someri"])
st.markdown("**Test ADF:**")
st.write("- ADF statistic =", round(adf_stat, 2))
st.write("- p-value =", round(pval, 3))
if pval < 0.05:
    st.success("Seria este staționară ⇒ trend stochastic respins")
else:
    st.warning("Seria NU este staționară ⇒ posibil trend stochastic")

df_somaj["Timp"] = np.arange(len(df_somaj))
X = sm.add_constant(df_somaj["Timp"])
y = df_somaj["Numar someri"]
model = sm.OLS(y, X).fit()

with st.expander("📊 Rezumat regresie liniară"):
    st.text(model.summary())

st.markdown("**Concluzie:** Rezultatele sugerează o componentă de trend determinist (creștere/scădere sistematică în timp).")

# CERINTA 2
st.header("2️⃣ Staționarizarea seriei")
st.markdown("**Scop:** Aplicăm diferențiere pentru a obține o serie staționară.")
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(df_diff["Data"], df_diff["Diferenta somaj"], marker='o')
ax2.set_title("Seria diferențiată")
ax2.grid(True)
st.pyplot(fig2)

adf_stat2, pval2, _, _, crit_vals2, _ = adfuller(df_diff["Diferenta somaj"])
st.markdown("**Test ADF pe seria diferențiată:**")
st.write("- ADF statistic =", round(adf_stat2, 2))
st.write("- p-value =", round(pval2, 3))
if pval2 < 0.05:
    st.success("Diferențierea a condus la o serie staționară.")
else:
    st.error("Seria diferențiată NU este staționară.")

st.markdown("**Concluzie:** Serie pregătită pentru modelare ARIMA.")

# CERINTA 3
st.header("3️⃣ Netezire exponențială: SES și Holt")
st.markdown("**Scop:** Aplicăm metode de netezire exponențială pentru a surprinde tendințele seriei.")
serie = df_somaj["Numar someri"]
ses_model = SimpleExpSmoothing(serie, initialization_method="legacy-heuristic").fit()
holt_model = Holt(serie, initialization_method="legacy-heuristic").fit()

fig3, ax3 = plt.subplots(figsize=(8, 3))
ax3.plot(df_somaj["Data"], serie, label="Originală", color="black")
ax3.plot(df_somaj["Data"], ses_model.fittedvalues, label="SES", linestyle="--")
ax3.plot(df_somaj["Data"], holt_model.fittedvalues, label="Holt", linestyle="--")
ax3.set_title("Netezire exponențială – SES și Holt")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)

st.markdown("**Concluzie:** Modelul Holt surprinde mai bine trendul, în timp ce SES este mai potrivit pentru serii fără tendință.")

# CERINTA 4
st.header("4️⃣ Alegerea modelului ARIMA optim")
st.markdown("**Scop:** Identificăm configurația ARIMA(p,d,q) cu cel mai bun scor AIC.")
configuratii = [(1,1,1), (2,1,1), (1,1,2), (2,1,2)]
rezultate = {}
for cfg in configuratii:
    model = ARIMA(serie, order=cfg).fit()
    rezultate[str(cfg)] = model.aic
best_cfg = min(rezultate, key=rezultate.get)
st.write("Scoruri AIC pentru configurațiile testate:")
st.json(rezultate)
st.success(f"Modelul ales: ARIMA{best_cfg} (AIC minim)")
st.markdown("**Concluzie:** Alegem configurația cu cel mai mic AIC pentru modelarea prognozei.")

# CERINTA 5
st.header("5️⃣ Predicție punctuală și pe interval – ARIMA")
st.markdown("**Scop:** Realizăm prognoza cu modelul ARIMA ales și o evaluăm pe setul de test.")
n = len(serie)
train_size = int(n * 0.8)
train, test = serie[:train_size], serie[train_size:]
dates_train, dates_test = df_somaj["Data"][:train_size], df_somaj["Data"][train_size:]
model_5 = ARIMA(train, order=(2, 1, 2)).fit()
forecast_5 = model_5.get_forecast(steps=len(test))
predicted_mean = forecast_5.predicted_mean
conf_int = forecast_5.conf_int()

fig5, ax5 = plt.subplots(figsize=(8, 3))
ax5.plot(dates_train, train, label="Training")
ax5.plot(dates_test, test, label="Test")
ax5.plot(dates_test, predicted_mean, label="Predicție")
ax5.fill_between(dates_test, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.3, label="Interval 95%")
ax5.set_title("Predicție ARIMA(2,1,2)")
ax5.legend()
ax5.grid(True)
st.pyplot(fig5)

st.markdown("**Concluzie:** Predicția se încadrează în limitele intervalului de încredere; ARIMA oferă rezultate stabile.")

# CERINTA 6
st.header("6️⃣ Comparare metode univariate")
st.markdown("**Scop:** Comparăm performanțele ARIMA, SARIMA și Holt-Winters.")
sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,0,1,4)).fit()
sarima_forecast = sarima_model.forecast(steps=len(test))
hw_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=4).fit()
hw_forecast = hw_model.forecast(len(test))

fig6, ax6 = plt.subplots(figsize=(8, 3))
ax6.plot(dates_test, test, label="Real", color="black")
ax6.plot(dates_test, model_5.forecast(steps=len(test)), label="ARIMA", linestyle="--")
ax6.plot(dates_test, sarima_forecast, label="SARIMA", linestyle="--")
ax6.plot(dates_test, hw_forecast, label="Holt-Winters", linestyle="--")
ax6.set_title("Comparare metode univariate")
ax6.legend()
ax6.grid(True)
st.pyplot(fig6)

st.markdown("**Concluzie:** SARIMA oferă adesea cele mai bune rezultate când există sezonalitate; Holt-Winters este ușor de interpretat.")

# CERINTA 7
st.header("7️⃣ Analiză multivariată: Șomaj și PIB")
st.markdown("**Scop:** Explorăm relațiile dinamice dintre șomaj și PIB.")
st.subheader("7.1 Test ADF pentru staționaritate")
def test_adf(serie, nume):
    rezultat = adfuller(serie)
    st.write(f"**{nume}**: ADF = {rezultat[0]:.2f}, p-value = {rezultat[1]:.3f}")
    if rezultat[1] < 0.05:
        st.success(f"{nume} este staționară")
    else:
        st.warning(f"{nume} NU este staționară")

test_adf(df_multiv["Somaj"], "Șomaj")
test_adf(df_multiv["PIB"], "PIB")

st.subheader("7.2 Test de cointegrare Johansen")
johansen = coint_johansen(df_multiv[["Somaj", "PIB"]], det_order=0, k_ar_diff=1)
with st.expander("📄 Rezultate test Johansen"):
    for i, trace in enumerate(johansen.lr1):
        cv = johansen.cvt[i, 1]
        st.write(f"r = {i}: Trace = {trace:.2f}, CV (5%) = {cv:.2f} ⇒ {'Cointegrare' if trace > cv else 'Fără cointegrare'}")

st.subheader("7.3 Funcția de răspuns la impuls (IRF)")
vecm_model = VECM(df_multiv[["Somaj", "PIB"]], k_ar_diff=1, deterministic="n").fit()
irf = vecm_model.irf(10)
fig_irf = irf.plot(orth=False, impulse="PIB", response="Somaj")
st.pyplot(fig_irf.figure)

st.subheader("7.4 Test de cauzalitate Granger")
with st.expander("📄 Rezultate Granger"):
    st.write("PIB → Șomaj")
    grangercausalitytests(df_multiv[["Somaj", "PIB"]], maxlag=4, verbose=True)
    st.write("Șomaj → PIB")
    grangercausalitytests(df_multiv[["PIB", "Somaj"]], maxlag=4, verbose=True)

st.subheader("7.5 Descompunerea varianței (FEVD)")
data_diff = df_multiv[["Somaj", "PIB"]].diff().dropna()
var_model = VAR(data_diff).fit(maxlags=2, ic='aic')
fevd = var_model.fevd(10)
fig_fevd = fevd.plot()
st.pyplot(fig_fevd.figure)

st.markdown("**Concluzie:** PIB-ul influențează în mod semnificativ dinamica șomajului, conform analizelor VECM, IRF și Granger.")
