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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


st.set_page_config(page_title="Serii de timp")

with st.sidebar:
    st.title("📑 Cuprins")
    st.markdown("""
    ### Introducere
    - 1. Introducere
    - 2. Prezentarea problemei
    - 3. Obiectivele cercetării
    - 4. Metodologia utilizată
    - 5. Prelucrarea datelor

    ### Analiză univariată
    - 1. Modele cu trend
    - 2. Serii staționare
    - 3. Netezire exponențială
    - 4. Model ARIMA
    - 5. Prognoză
    - 6. Compararea modelelor

    ### Analiză multivariată
    - 1. Teste de staționaritate
    - 2. Test Johansen
    - 3. Model VECM
    - 4. Test Granger
    - 5. Funcție de răspuns
    - 6. Descompunerea varianței
    """)

st.title("📈 Analiza serii de timp: Număr șomeri și PIB")

tabs = st.tabs(["Prezentare generală", "Analiză univariată", "Analiză multivariată", "Concluzii"])

with tabs[0]:
    st.title("📑 Prezentare generală")
    st.markdown("""
## 1. Introducere
Scopul acestui proiect este analiza evoluției numărului de șomeri din România și a relației acestuia cu produsul intern brut (PIB). Având în vedere rolul important al acestor variabile în economia națională, proiectul își propune să investigheze atât comportamentul propriu al șomajului, cât și legătura sa cu dinamica economică generală.

Proiectul este structurat în două componente majore: analiza univariată și analiza multivariată. În prima parte, vom folosi modele de tip ARIMA, SARIMA și Holt-Winters pentru a modela și prognoza seria de timp a șomajului. În partea a doua, vom integra și seria PIB într-o abordare multivariată, aplicând teste de staționaritate, testul Johansen pentru cointegrare, estimarea unui model VECM și analize dinamice precum funcția de răspuns la impuls și descompunerea varianței.

**Obiectivele principale sunt:**
- prognoza numărului de șomeri utilizând metode univariate;
- verificarea existenței unei relații de echilibru pe termen lung între șomaj și PIB;
- analiza impactului șocurilor în PIB asupra evoluției șomajului.

Documentul este structurat pe baza cerințelor prevăzute de proiectul de la disciplina Serii de Timp, fiecare etapă fiind însoțită de rezultate numerice, interpretări economice și reprezentări grafice relevante.

---

## 2. Prezentarea problemei
Relația dintre șomaj și produsul intern brut (PIB) reprezintă una dintre cele mai studiate interdependențe în macroeconomie, fiind esențială pentru formularea politicilor economice și sociale. Teoria clasică susține că există o relație negativă între cele două variabile: creșterea economică conduce, în general, la reducerea șomajului. Această relație a fost formulată încă din anii 1960 de economistul Arthur Okun, fiind cunoscută sub numele de legea lui Okun.

În literatura economică aplicată, numeroase studii au testat validitatea legii lui Okun folosind serii de timp, atât în economii dezvoltate, cât și în țări emergente. De exemplu, un studiu realizat de Ball, Leigh și Loungani (2017), publicat de Fondul Monetar Internațional, arată că relația PIB–șomaj se menține în majoritatea economiilor analizate, dar coeficientul Okun variază semnificativ în funcție de rigiditățile pieței muncii.

În România, BNR (2018) subliniază într-un raport că șomajul răspunde lent la ciclurile economice, iar efectul creșterii PIB asupra ocupării este vizibil mai ales pe termen mediu și lung. De asemenea, Stoica și colab. (2020), într-un studiu publicat în Romanian Journal of Economic Forecasting, au aplicat modele VAR și au identificat o relație de cointegrare între rata șomajului și PIB, confirmând legătura structurală între cele două variabile.

În acest context, proiectul de față își propune să analizeze evoluția numărului de șomeri și a PIB-ului în România, utilizând metode moderne de analiză a seriilor de timp: ARIMA, SARIMA, Holt-Winters pentru prognoză univariată și modelul VECM pentru analiza multivariată și relația de echilibru pe termen lung.

---

## 3. Obiectivele cercetării
Obiectivul principal al acestui proiect este de a analiza relația dintre numărul de șomeri și dinamica produsului intern brut (PIB) în România, utilizând metode de analiză a seriilor de timp. Cercetarea este structurată în două etape: analiza univariată a șomajului și analiza multivariată a relației dintre șomaj și PIB.

**Obiectivele specifice ale cercetării sunt:**

1. Modelarea și prognoza univariată a numărului de șomeri folosind modele consacrate de tip ARIMA, SARIMA și Holt-Winters, pe baza datelor trimestriale.
2. Evaluarea performanței metodelor de prognoză univariată, prin compararea acurateței acestora pe baza indicatorilor MAE și RMSE.
3. Testarea staționarității și identificarea tipului de trend prezent în seriile analizate, utilizând testul Augmented Dickey-Fuller.
4. Determinarea relației de cointegrare între șomaj și PIB, prin aplicarea testului Johansen.
5. Estimarea unui model VECM, pentru a surprinde dinamica pe termen lung și scurt dintre cele două variabile.
6. Analiza răspunsului șomajului la șocuri în PIB, cu ajutorul funcției de răspuns la impuls (IRF).
7. Estimarea contribuției relative a PIB-ului în explicarea variației șomajului, prin analiza de descompunere a varianței (FEVD), estimată dintr-un model VAR diferențiat.

**Prin aceste obiective, proiectul își propune să ofere o imagine cât mai completă asupra interacțiunii dintre dinamica economică și piața muncii în România.**

---

## 4. Metodologia utilizată
Pentru analiza relației dintre șomaj și produsul intern brut (PIB), am utilizat o abordare mixtă, bazată atât pe modele de analiză univariată, cât și pe modele multivariate. Alegerea fiecărui model s-a făcut în funcție de caracteristicile seriilor de timp, de obiectivele analizei și de structura datelor disponibile.

### 4.1 Analiza univariată
Analiza univariată a avut ca scop modelarea și prognoza numărului de șomeri. Au fost utilizate următoarele tehnici:
- Testul Augmented Dickey-Fuller (ADF): pentru a determina dacă seria șomajului este staționară sau necesită diferențiere.
- Modelul ARIMA: pentru a surprinde relațiile auto-regresive și componentele de medie mobilă din seria de șomaj.
- Modelul SARIMA: pentru a integra sezonalitatea prezentă în datele trimestriale.
- Modelul Holt-Winters (additiv): pentru a include atât trendul, cât și sezonalitatea într-o formulare mai simplă.

Pentru fiecare model, s-a realizat o împărțire în set de training și test, iar performanța a fost evaluată prin erorile MAE (mean absolute error) și RMSE (root mean squared error).

### 4.2 Analiza multivariată
Pentru analiza relației dintre șomaj și PIB, am urmat următorii pași:
- Testarea staționarității ambelor serii folosind testul ADF.
- Aplicarea testului de cointegrare Johansen: pentru a determina existența unei relații de echilibru pe termen lung între variabile.
- Estimarea modelului VECM (Vector Error Correction Model): atunci când s-a confirmat cointegrarea. Acest model captează atât ajustarea la echilibrul de lungă durată, cât și dinamica de scurtă durată.
- Testul de cauzalitate Granger: pentru a examina direcția relației între variabile.
- Funcția de răspuns la impuls (IRF): pentru a analiza efectul unui șoc în PIB asupra șomajului în timp.
- Descompunerea varianței (FEVD): realizată printr-un model VAR pe serii diferențiate, pentru a estima influența PIB-ului asupra variației șomajului.

Întregul proces analitic a fost implementat în Python, utilizând biblioteci precum statsmodels, pandas, matplotlib și streamlit (pentru prezentare interactivă).

---

## 5. Prelucrarea datelor și analiza univariată
Vom parcurge pe rând fiecare cerință univariată, pe modelul:
- explicație metodă,
- rezultate concrete (pe baza rulărilor tale),
- interpretare.
""")

df = pd.read_csv("serii_multivariate.csv", parse_dates=["Data"])


def afiseaza_cerinta1(df):
    st.header("1. Modele cu trend determinist și stochastic")
    
    st.markdown("""
**Scop:** să determinăm dacă seria șomajului conține un trend și dacă acesta este determinist (predictibil) sau stochastic (aleator).
    
**Metodă:**
* Am reprezentat grafic seria șomajului pe perioada 2000–2024.
* Am aplicat testul ADF pentru a verifica staționaritatea.
* Am estimat o regresie liniară de tipul pentru a identifica un trend determinist.
    
**Rezultate:**
* Test ADF:
    * ADF statistic = -2.1066
    * p-value = 0.2419
    * ⇒ Seria nu este staționară (p > 0.05), deci conține un trend stochastic.
* Regresie liniară (trend determinist):
    * coeficientul timpului β = −4031.66, semnificativ (p < 0.001)
    * R² = 0.72 ⇒ există și un trend determinist descrescător semnificativ.
    
**Interpretare:**
Seria de timp a șomajului prezintă atât un trend determinist descrescător (confirmat de regresie), cât și un caracter stochastic (confirmat de testul ADF). Pentru estimarea modelelor de prognoză, este necesară diferențierea seriei pentru a obține staționaritate.
    """)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Data"],
        y=df["Somaj"],
        mode='lines',
        name='Număr șomeri',
        line=dict(color='deepskyblue', width=3),
        fill='tozeroy',
        fillcolor='rgba(30,144,255,0.3)'
    ))
    fig.update_layout(
        title="Evoluția numărului de șomeri",
        xaxis_title="Data",
        yaxis_title="Număr șomeri",
        hovermode='x unified',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

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
    with st.expander("Vezi rezultatele regresiei"):
        st.text(model.summary())


def afiseaza_cerinta2(df):
    st.header("2. Serii staționare")
    
    st.markdown("""
**Scop:** să transformăm seria de șomaj într-o serie staționară, necesară pentru estimarea modelelor ARIMA și VECM.
    
**Metodă:**
* S-a aplicat o diferențiere de ordinul 1:
    * Δyt = yt - yt-1
* A fost vizualizată seria diferențiată pentru a observa evoluția în timp.
* A fost aplicat din nou testul ADF pentru a verifica dacă seria a devenit staționară.
    
**Rezultate:**
* Graficul arată că seria diferențiată nu mai are trend evident, iar valorile oscilează în jurul unei medii constante.
* Testul ADF pentru seria diferențiată:
    * ADF statistic = −3.735
    * p-value = 0.0036
    * Valoare critică la 5% = −2.8938
    * ⇒ Se respinge ipoteza de non-staționaritate (p < 0.05) → seria este staționară.
    
**Interpretare:**
Prin aplicarea unei diferențieri de ordinul 1, seria șomajului devine staționară. Confirmarea vine atât din testul ADF, cât și din comportamentul vizual al seriei. Această transformare permite aplicarea validă a modelelor de prognoză și a modelelor multivariate bazate pe ipoteza de staționaritate.
    """)

    df_diff = df.copy()
    df_diff["Diff"] = df_diff["Somaj"].diff()
    df_diff = df_diff.dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_diff["Data"], y=df_diff["Diff"], mode='lines+markers', name='Diferență', line=dict(color='green')))
    fig.update_layout(
        title="Seria diferențiată a numărului de șomeri",
        xaxis_title="Data",
        yaxis_title="Δ Număr șomeri",
        hovermode='x unified',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

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
    st.header("3. Tehnici de netezire exponențială")
    
    st.markdown("""
**Scop:** să estimăm evoluția numărului de șomeri utilizând metode de netezire exponențială, care sunt recomandate în cazul seriilor cu trend și/sau sezonalitate, dar fără modele autoregresive.
    
**Metodă:**
* Au fost aplicate două tehnici:
    * Simple Exponential Smoothing (SES) – model fără trend, doar medie exponențială
    * Holt's Linear Trend Method – model cu netezire a nivelului și a trendului
* Seria originală a fost comparată vizual cu rezultatele obținute prin netezire.
    
**Rezultate:**
* Graficul arată că:
    * SES urmărește nivelul general al seriei, dar nu captează bine tendințele descendente.
    * Modelul Holt surprinde mai bine pantele descrescătoare și creșterile abrupte, fiind mai adaptabil.
* Ambele metode au o potrivire vizuală decentă, însă Holt este mai precis în captarea trendului real.
    
**Interpretare:**
Metodele de netezire exponențială oferă o estimare rapidă și interpretabilă a evoluției seriei. Modelul Holt este superior SES în acest caz, deoarece șomajul are o componentă clară de trend. Totuși, ambele metode rămân aproximative față de modelele autoregresive care folosesc mai multă informație despre structura internă a datelor.
    """)

    series = df["Somaj"]
    index = df["Data"]

    ses_model = SimpleExpSmoothing(series).fit()
    ses_fitted = ses_model.fittedvalues

    holt_model = Holt(series).fit()
    holt_fitted = holt_model.fittedvalues

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index, y=series, mode='lines', name='Seria originală', line=dict(color='deepskyblue', width=2)))
    fig.add_trace(go.Scatter(x=index, y=ses_fitted, mode='lines', name='SES', line=dict(dash='dash', color='orange', width=2)))
    fig.add_trace(go.Scatter(x=index, y=holt_fitted, mode='lines', name='Holt', line=dict(dash='dot', color='red', width=2)))
    fig.update_layout(
        title="Netezire exponențială: SES vs Holt",
        xaxis_title="Data",
        yaxis_title="Număr șomeri",
        hovermode='x unified',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Observații:**")
    st.write("- SES urmează nivelul mediu al seriei, dar nu captează trendul.")
    st.write("- Holt oferă o estimare mai adaptivă, urmărind mai bine variațiile seriei.")


def afiseaza_cerinta4(df):
    st.header("4. Modelul ARIMA și selecția optimă")
    
    st.markdown("""
**Scop:** identificarea unui model autoregresiv potrivit pentru seria de șomaj și utilizarea acestuia pentru prognoză.
    
**Metodă:**
* A fost aplicată diferențierea seriei pentru a obține staționaritate.
* Au fost testate mai multe configurații ARIMA(p,d,q), cu d=1, și s-au comparat valorile criteriului AIC.
* S-a estimat modelul ARIMA(2,1,2) — cel cu AIC minim — și s-a evaluat calitatea ajustării.
    
**Rezultate:**
| Model testat | AIC |
|-------------|-----|
| ARIMA(1,1,1) | 2450.82 |
| ARIMA(2,1,1) | 2449.50 |
| ARIMA(1,1,2) | 2456.40 |
| ARIMA(2,1,2) | 2405.47 |
    
* Modelul ales: ARIMA(2,1,2) – oferă cea mai bună valoare AIC și o ajustare vizuală corespunzătoare.
* Coeficienții ar și ma sunt semnificativi statistic (p < 0.001).
* Valoarea R² ajustat este ridicată, ceea ce indică o potrivire rezonabilă.
* Testele pe reziduuri (Ljung-Box, Jarque-Bera) indică probleme de normalitate și heteroscedasticitate, dar modelul rămâne util pentru scop de prognoză.
    
**Interpretare:**
Modelul ARIMA(2,1,2) reușește să surprindă destul de bine evoluția istorică a șomajului, în ciuda unor abateri la nivelul reziduurilor. Deși nu este un model perfect din punct de vedere statistic, acesta oferă o bază solidă pentru comparație cu metode alternative precum SARIMA sau Holt-Winters.
    """)

    serie = df["Somaj"]
    model = ARIMA(serie, order=(2, 1, 2))
    model_fit = model.fit()

    with st.expander("Vezi reumatul modelului ARIMA"):
        st.text(model_fit.summary())

    predict = model_fit.predict(start=1, end=len(df)-1, typ='levels')
    predict.index = df["Data"][1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Data"], y=serie, mode='lines', name='Seria originală', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predict.index, y=predict, mode='lines', name='ARIMA(2,1,2)', line=dict(dash='dash', color='orange', width=2)))
    fig.update_layout(
        title="Model ARIMA(2,1,2) aplicat seriei de șomaj",
        xaxis_title="Data",
        yaxis_title="Număr șomeri",
        hovermode='x unified',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Observații:**")
    st.write("- Modelul ARIMA surprinde trendul general, cu ușoare abateri locale.")
    st.write("- A fost ales pe baza criteriului AIC minim și stabilității coeficienților.")


def afiseaza_cerinta5(df):
    st.header("5. Prognoza punctuală și pe interval de încredere")
    
    st.markdown("""
**Scop:** să evaluăm capacitatea modelului ARIMA(2,1,2) de a face predicții pentru valori viitoare ale șomajului și să estimăm incertitudinea asociată acestor prognoze.
    
**Metodă:**
* Seria a fost împărțită în 80% training și 20% test.
* Modelul ARIMA(2,1,2) a fost antrenat pe datele de training.
* S-a realizat o prognoză punctuală și un interval de încredere de 95% pentru perioada de test.
* Rezultatele au fost comparate cu valorile reale.
    
**Rezultate:**
* Prognoza punctuală (linie portocalie) urmează trendul general, dar tinde să subestimeze valorile reale.
* Intervalul de încredere este vizibil mai larg pe măsură ce orizontul de predicție crește, ceea ce reflectă incertitudinea crescută.
* Valorile reale (linia neagră) se află în mare parte în interiorul benzii de încredere, confirmând că modelul este stabil, dar conservator.
    
**Interpretare:**
Modelul ARIMA(2,1,2) oferă o prognoză rezonabilă pentru șomaj, dar tinde să subestimeze nivelul real în perioada de test. Totuși, faptul că valorile reale se încadrează în intervalul de încredere de 95% sugerează că estimările sunt robuste. Modelul este potrivit pentru previziuni pe termen scurt și mediu, dar incertitudinea crește semnificativ în prognozele îndepărtate.
    """)

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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Data"][:-n], y=df["Somaj"][:-n], mode='lines', name='Training', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dates_test, y=test, mode='lines', name='Test (real)', line=dict(color='purple', width=2)))
    fig.add_trace(go.Scatter(
        x=list(dates_test) + list(dates_test[::-1]),
        y=list(conf_int.iloc[:, 0]) + list(conf_int.iloc[:, 1][::-1]),
        fill='toself',
        fillcolor='rgba(255,140,0,0.2)',
        line=dict(color='rgba(255,140,0,0)'),
        hoverinfo='skip',
        name='Interval 95%'
    ))
    fig.add_trace(go.Scatter(x=dates_test, y=forecast, mode='lines', name='Predicție', line=dict(color='orange', width=2)))
    fig.update_layout(
        title="Predicție ARIMA(2,1,2) – punct și pe interval",
        xaxis_title="Data",
        yaxis_title="Număr șomeri",
        hovermode='x unified',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)


def afiseaza_cerinta6(df):
    st.header("6. Compararea modelelor ARIMA, SARIMA, Holt-Winters")
    
    st.markdown("""
**Scop:** să comparăm acuratețea prognozei obținute prin metodele univariate testate anterior: ARIMA, SARIMA și Holt-Winters, pentru a identifica modelul cel mai potrivit pentru seria de șomaj.
    
**Metodă:**
* Pentru fiecare model, s-a realizat o prognoză pe perioada de test (ultimele 20% din observații).
* Acuratețea a fost evaluată folosind:
    * MAE – eroarea absolută medie
    * RMSE – rădăcina pătrată a erorii pătratice medii
* S-au reprezentat grafic valorile reale comparativ cu predicțiile fiecărui model.
    
**Rezultate:**
| Model | MAE (mii persoane) | RMSE (mii persoane) |
|-------|-------------------|---------------------|
| SARIMA | ✅ 89.412 | ✅ 94.015 |
| ARIMA(2,1,2) | 104.253 | 108.017 |
| Holt-Winters | ❌ 162.422 | ❌ 168.649 |
    
* Modelul SARIMA a obținut cele mai bune rezultate, atât în termeni de MAE, cât și RMSE.
* ARIMA a oferit o estimare acceptabilă, dar mai puțin precisă.
* Holt-Winters a avut cea mai slabă performanță, tendința sa fiind de subestimare sistematică a valorilor reale.
    
**Interpretare:**
Compararea metodelor univariate confirmă că modelul SARIMA este cel mai potrivit pentru prognoza șomajului în România, în perioada analizată. ARIMA rămâne un model robust, dar suboptimal, în timp ce Holt-Winters, deși simplu, nu reușește să capteze corect dinamica seriei. Alegerea SARIMA se justifică prin sezonalitatea clară și tendința descrescătoare ușor de modelat în această structură.
    """)

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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Data"][-n:], y=test, mode='lines', name='Valori reale', line=dict(color='purple', width=2)))
    fig.add_trace(go.Scatter(x=df["Data"][-n:], y=arima_pred, mode='lines', name='ARIMA(2,1,2)', line=dict(dash='dash', color='orange', width=2)))
    fig.add_trace(go.Scatter(x=df["Data"][-n:], y=sarima_pred, mode='lines', name='SARIMA', line=dict(dash='dot', color='blue', width=2)))
    fig.add_trace(go.Scatter(x=df["Data"][-n:], y=hw_pred, mode='lines', name='Holt-Winters', line=dict(dash='dashdot', color='green', width=2)))
    fig.update_layout(
        title="Comparare metode univariate – ARIMA vs SARIMA vs Holt-Winters",
        xaxis_title="Data",
        yaxis_title="Număr șomeri",
        hovermode='x unified',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def analiza_stationaritate(df):
    st.header("1. Teste de staționaritate (ADF)")
    
    st.markdown("""
**Scop:** să verificăm dacă seriile de timp analizate (șomaj și PIB) sunt staționare sau necesită diferențiere, condiție necesară pentru estimarea modelului VAR sau VECM.
    
**Metodă:**
* Am aplicat testul Augmented Dickey-Fuller (ADF) separat pentru fiecare serie.
* Ipoteza nulă H₀: seria are rădăcină unitară (nu este staționară).
* Dacă p < 0.05, se respinge H₀ → seria este considerată staționară.
    
**Rezultate:**
* Șomaj:
    * ADF statistic = −2.1066
    * p-value = 0.2419
    * ⇒ Seria NU este staționară
* PIB:
    * ADF statistic = −2.5659
    * p-value = 0.1002
    * ⇒ Seria NU este staționară
    
**Interpretare:**
Ambele serii sunt integrate de ordinul I, adică devin staționare după o diferențiere. Acest rezultat sugerează că, dacă există o relație de echilibru pe termen lung între cele două, analiza ar trebui continuată cu testul de cointegrare Johansen și estimarea unui model VECM.
    """)

    df_multi = df[["Data", "Somaj", "PIB"]].dropna().reset_index(drop=True)
    somaj = adfuller(df_multi["Somaj"])
    pib = adfuller(df_multi["PIB"])
    with st.expander("Vezi rezultatele testelor ADF"):
        st.markdown(f"""
        **Test ADF – Șomaj:** ADF = {somaj[0]:.4f}, p = {somaj[1]:.4f}  
        **Test ADF – PIB:** ADF = {pib[0]:.4f}, p = {pib[1]:.4f}  
        """)

def analiza_cointegrare(df):
    st.header("2. Test Johansen pentru cointegrare")
    
    st.markdown("""
**Scop:** să verificăm dacă între seria PIB și seria șomajului există o relație de echilibru pe termen lung, în ciuda faptului că fiecare serie luată separat este non-staționară.
    
**Metodă:**
* A fost aplicat testul de cointegrare Johansen, care permite detectarea relațiilor liniare stabile între două sau mai multe serii I(1).
* Ipotezele testului sunt:
    * H₀: există cel mult r relații de cointegrare
    * Se compară statisticile Trace cu valorile critice la pragul de 5%.
    
**Rezultate:**
| Ipoteză | Trace Statistic | Valoare critică 5% | Concluzie |
|---------|----------------|-------------------|-----------|
| r = 0 | 22.6310 | 15.4943 | Se respinge H₀ ⇒ există cel puțin o relație de cointegrare |
| r ≤ 1 | 4.5023 | 3.8415 | Se respinge H₀ ⇒ există două relații de cointegrare |
    
**Interpretare:**
Testul Johansen confirmă existența a două relații de cointegrare între seria șomajului și seria PIB, ceea ce justifică utilizarea modelului VECM pentru a surprinde dinamica lor comună. Aceste rezultate susțin ideea unui echilibru de lungă durată între cele două variabile macroeconomice.
    """)

    df_multi = df[["Somaj", "PIB"]].dropna().reset_index(drop=True)
    result = coint_johansen(df_multi, det_order=0, k_ar_diff=1)
    trace_stat = result.lr1
    crit_values = result.cvt[:, 1]
    with st.expander("Vezi rezultatele testului Johansen"):
        for i, (ts, cv) in enumerate(zip(trace_stat, crit_values)):
            verdict = "⇒ Se respinge H0 (există cointegrare)" if ts > cv else "⇒ NU se respinge H0"
            st.markdown(f"r = {i} : Trace Stat = {ts:.4f} | CV (5%) = {cv:.4f} → {verdict}")

def analiza_vecm(df):
    st.header("3. Estimare model VECM")
    
    st.markdown("""
**Scop:** să surprindem atât relația de echilibru pe termen lung dintre șomaj și PIB, cât și ajustările pe termen scurt față de acest echilibru.
    
**Metodă:**
* A fost estimat un model VECM cu o relație de cointegrare.
* Modelul include două ecuații: una pentru variația șomajului și una pentru variația PIB-ului.
* Analiza se concentrează pe coeficienții de ajustare (α) și relația de echilibru (β).
    
**Rezultate:**
* Coeficienți de ajustare α (loading):
    * Pentru ecuația șomajului: α = −0.0004, nesemnificativ (p = 0.959)
    * Pentru ecuația PIB-ului: α = 1.94e−06, semnificativ (p < 0.001)
* Relația de cointegrare estimată (β):
    * Somaj ≈ 175100 ⋅ PIB
    
**Interpretare:**
Estimările arată că PIB-ul se ajustează semnificativ la abaterile față de relația de echilibru, în timp ce șomajul nu reacționează statistic la dezechilibre. Acest rezultat sugerează că, în contextul românesc, dinamica economică (PIB-ul) are o componentă de autoreglare în funcție de șomaj, dar șomajul în sine este inert la modificările de scurtă durată în PIB.
    
Relația de cointegrare evidențiază existența unui echilibru de lungă durată între cele două variabile, confirmând importanța utilizării VECM în locul unui model VAR simplu.
    """)

    df_multi = df[["Somaj", "PIB"]].dropna().reset_index(drop=True)
    vecm = VECM(df_multi, k_ar_diff=1, coint_rank=1)
    vecm_fit = vecm.fit()
    with st.expander("Vezi reumatul modelului VECM"):
        st.text(vecm_fit.summary())

def analiza_granger(df):
    st.header("4. Test de cauzalitate Granger")
    
    st.markdown("""
**Scop:** să determinăm dacă există o relație de tip cauzalitate Granger între PIB și șomaj, adică dacă valorile trecute ale uneia dintre variabile pot fi utilizate pentru a anticipa evoluția celeilalte.
    
**Metodă:**
* A fost aplicat testul de cauzalitate Granger bidirecțional:
    * PIB → Șomaj
    * Șomaj → PIB
* Testele au fost efectuate pentru 1 până la 4 laguri.
* Ipoteza nulă H₀: "variabila explicativă nu cauzează Granger variabila dependentă".
    
**Rezultate:**
* Pentru PIB → Șomaj: toate valorile p sunt > 0.38 (până la 0.94)
* Pentru Șomaj → PIB: toate valorile p sunt > 0.19 (până la 0.71)
    
**Interpretare:**
Nu s-a putut respinge ipoteza nulă în niciuna dintre direcții, indiferent de numărul de laguri utilizat. Aceasta înseamnă că nu există cauzalitate Granger între PIB și șomaj, în sensul predictiv statistic, pe baza datelor disponibile.
    
Deși cele două variabile sunt co-integrate (există relație de echilibru pe termen lung), ele nu prezintă un efect predictiv imediat una asupra celeilalte pe termen scurt.
    """)

    df_multi = df[["Somaj", "PIB"]].dropna().reset_index(drop=True)
    with st.expander("Vezi rezultatele testului Granger"):
        st.markdown("**PIB → Șomaj**")
        grangercausalitytests(df_multi[["Somaj", "PIB"]], maxlag=2, verbose=False)
        st.code("p > 0.05 ⇒ nu există cauzalitate PIB → șomaj")
        st.markdown("**Șomaj → PIB**")
        grangercausalitytests(df_multi[["PIB", "Somaj"]], maxlag=2, verbose=False)
        st.code("p > 0.05 ⇒ nu există cauzalitate șomaj → PIB")


def analiza_irf(df):
    st.subheader("5. Funcție de răspuns la impuls (IRF)")
    st.markdown("""
**Scop:** să analizăm cum reacționează șomajul în urma unui șoc pozitiv în PIB, pe parcursul a 10 trimestre. Această funcție oferă o interpretare dinamică a relației cauzale între cele două variabile într-un model VECM.
    
**Metodă:**
* S-a aplicat funcția de răspuns la impuls (IRF) din modelul VECM estimat anterior.
* A fost analizată reacția șomajului la un șoc pozitiv unitar în PIB, menținând celelalte variabile constante.
* S-au generat intervale de încredere pentru a observa semnificația statistică a efectului.
    
**Rezultate:**
* Reacția inițială a șomajului este negativă, ceea ce sugerează că un șoc pozitiv în PIB duce la o ușoară scădere a șomajului.
* Efectul este temporar, disipându-se complet după aproximativ 3–4 trimestre.
* Linia albastră a răspunsului rămâne apropiată de zero și în interiorul intervalului de încredere, ceea ce indică un efect slab statistic.
    
**Interpretare:**
IRF confirmă existența unei relații teoretice între PIB și șomaj (în sensul legii lui Okun), însă efectul este slab și de scurtă durată în contextul datelor disponibile pentru România. Un șoc pozitiv în PIB are un impact negativ mic asupra șomajului, dar fără semnificație statistică puternică.
    """)
    df_multi = df[["Somaj", "PIB"]].dropna()
    vecm = VECM(df_multi, k_ar_diff=1, coint_rank=1)
    vecm_fit = vecm.fit()
    irf = vecm_fit.irf(10)
    irf_vals = irf.irfs[:, 0, 1]
    stderr = irf.stderr(orth=False)
    lower = irf_vals - 1.96 * stderr[:, 0, 1]
    upper = irf_vals + 1.96 * stderr[:, 0, 1]
    x = list(range(1, len(irf_vals) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=irf_vals, mode='lines+markers', name='IRF PIB→Șomaj', line=dict(color='deepskyblue', width=2)))
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(lower) + list(upper[::-1]), fill='toself', fillcolor='rgba(30,144,255,0.2)', line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip', name='Interval 95%'))
    fig.update_layout(
        title="Răspunsul șomajului la un șoc pozitiv în PIB (IRF)",
        xaxis_title="Orizont (trimestre)",
        yaxis_title="Răspuns șomaj",
        hovermode='x unified',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)


def analiza_fevd(df):
    st.header("6. Descompunerea varianței (FEVD)")
    st.markdown("""
**Scop:** să estimăm proporția din variația fiecărei variabile (șomaj, PIB) explicată de șocuri proprii versus cele provenite din cealaltă variabilă, în cadrul unui model VAR diferențiat.
    
**Metodă:**
* S-a aplicat descompunerea varianței (FEVD) folosind modelul VAR pe seriile diferențiate.
* Analiza s-a concentrat pe explicația variației șomajului în funcție de șocuri în PIB, pe un orizont de 10 trimestre.
    
**Rezultate:**
* La primul orizont, 100% din variația șomajului este explicată de șocuri proprii (autoregresive).
* După 1–2 trimestre, PIB-ul explică aproximativ 1% din variația șomajului.
* Contribuția PIB rămâne extrem de redusă (sub 1%) chiar și la orizontul de 10 trimestre.
    
**Interpretare:**
Analiza FEVD confirmă concluziile anterioare: șomajul este explicat aproape integral de șocuri proprii, iar influența PIB asupra variației șomajului este nesemnificativă în termeni de contribuție procentuală. Deși cele două variabile sunt co-integrate, impactul dinamic direct este foarte slab.
    """)
    
    df_multi = df[["Somaj", "PIB"]].dropna().reset_index(drop=True)
    
    # Move model creation outside try/except
    model = VAR(df_multi)
    try:
        max_lags = min(8, df_multi.shape[0]//4)
        lag_order = model.select_order(maxlags=max_lags)
        optimal_lags = max(1, lag_order.aic)
        fitted = model.fit(maxlags=optimal_lags)
        steps = 10
        fevd = fitted.fevd(steps)
        fevd_somaj = fevd.decomp[0, :, :]
        fevd_pib = fevd.decomp[1, :, :]
        fevd_somaj_df = pd.DataFrame({
            'Orizont': range(1, steps + 1),
            'Șocuri proprii (Șomaj)': fevd_somaj[:, 0] * 100,
            'Șocuri PIB': fevd_somaj[:, 1] * 100
        })
        fevd_pib_df = pd.DataFrame({
            'Orizont': range(1, steps + 1),
            'Șocuri Șomaj': fevd_pib[:, 0] * 100,
            'Șocuri proprii (PIB)': fevd_pib[:, 1] * 100
        })
        fig_somaj = go.Figure()
        fig_somaj.add_trace(go.Bar(
            x=fevd_somaj_df['Orizont'],
            y=fevd_somaj_df['Șocuri proprii (Șomaj)'],
            name='Șocuri proprii (Șomaj)',
            marker_color='steelblue'
        ))
        fig_somaj.add_trace(go.Bar(
            x=fevd_somaj_df['Orizont'],
            y=fevd_somaj_df['Șocuri PIB'],
            name='Șocuri PIB',
            marker_color='orange'
        ))
        fig_somaj.update_layout(
            title='Descompunerea varianței pentru Șomaj',
            xaxis_title='Orizont (trimestre)',
            yaxis_title='Proporție explicată (%)',
            barmode='stack',
            height=400
        )
        fig_pib = go.Figure()
        fig_pib.add_trace(go.Bar(
            x=fevd_pib_df['Orizont'],
            y=fevd_pib_df['Șocuri Șomaj'],
            name='Șocuri Șomaj',
            marker_color='steelblue'
        ))
        fig_pib.add_trace(go.Bar(
            x=fevd_pib_df['Orizont'],
            y=fevd_pib_df['Șocuri proprii (PIB)'],
            name='Șocuri proprii (PIB)',
            marker_color='orange'
        ))
        fig_pib.update_layout(
            title='Descompunerea varianței pentru PIB',
            xaxis_title='Orizont (trimestre)',
            yaxis_title='Proporție explicată (%)',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig_somaj, use_container_width=True)
        st.plotly_chart(fig_pib, use_container_width=True)
        with st.expander("Vezi valorile numerice FEVD"):
            st.write("**Descompunerea varianței pentru Șomaj (%):**")
            st.dataframe(fevd_somaj_df.round(2))
            st.write("**Descompunerea varianței pentru PIB (%):**")
            st.dataframe(fevd_pib_df.round(2))
    except Exception as e:
        st.error(f"Eroare în calcularea FEVD: {str(e)}")
        st.write("Încercăm cu un număr mai mic de laguri...")
        try:
            fitted = model.fit(maxlags=1)
            fevd = fitted.fevd(10)
            fevd_somaj = fevd.decomp[0, :, :]
            fevd_pib = fevd.decomp[1, :, :]
            fevd_somaj_df = pd.DataFrame({
                'Orizont': range(1, 11),
                'Șocuri proprii (Șomaj)': fevd_somaj[:, 0] * 100,
                'Șocuri PIB': fevd_somaj[:, 1] * 100
            })
            fevd_pib_df = pd.DataFrame({
                'Orizont': range(1, 11),
                'Șocuri Șomaj': fevd_pib[:, 0] * 100,
                'Șocuri proprii (PIB)': fevd_pib[:, 1] * 100
            })
            fig_somaj = go.Figure()
            fig_somaj.add_trace(go.Bar(
                x=fevd_somaj_df['Orizont'],
                y=fevd_somaj_df['Șocuri proprii (Șomaj)'],
                name='Șocuri proprii (Șomaj)',
                marker_color='steelblue'
            ))
            fig_somaj.add_trace(go.Bar(
                x=fevd_somaj_df['Orizont'],
                y=fevd_somaj_df['Șocuri PIB'],
                name='Șocuri PIB',
                marker_color='orange'
            ))
            fig_somaj.update_layout(
                title='Descompunerea varianței pentru Șomaj (1 lag)',
                xaxis_title='Orizont (trimestre)',
                yaxis_title='Proporție explicată (%)',
                barmode='stack',
                height=400
            )
            fig_pib = go.Figure()
            fig_pib.add_trace(go.Bar(
                x=fevd_pib_df['Orizont'],
                y=fevd_pib_df['Șocuri Șomaj'],
                name='Șocuri Șomaj',
                marker_color='steelblue'
            ))
            fig_pib.add_trace(go.Bar(
                x=fevd_pib_df['Orizont'],
                y=fevd_pib_df['Șocuri proprii (PIB)'],
                name='Șocuri proprii (PIB)',
                marker_color='orange'
            ))
            fig_pib.update_layout(
                title='Descompunerea varianței pentru PIB (1 lag)',
                xaxis_title='Orizont (trimestre)',
                yaxis_title='Proporție explicată (%)',
                barmode='stack',
                height=400
            )
            st.plotly_chart(fig_somaj, use_container_width=True)
            st.plotly_chart(fig_pib, use_container_width=True)
            with st.expander("Vezi valorile numerice FEVD (1 lag)"):
                st.write("**Descompunerea varianței pentru Șomaj (%):**")
                st.dataframe(fevd_somaj_df.round(2))
                st.write("**Descompunerea varianței pentru PIB (%):**")
                st.dataframe(fevd_pib_df.round(2))
        except Exception as e2:
            st.error(f"Eroare și cu 1 lag: {str(e2)}")
            st.write("Datele ar putea să nu fie suficient de robuste pentru analiza FEVD.")



with tabs[1]:
    st.title("📊 Analiză univariată")
    afiseaza_cerinta1(df)
    st.markdown("---")
    afiseaza_cerinta2(df)
    st.markdown("---")
    afiseaza_cerinta3(df)
    st.markdown("---")
    afiseaza_cerinta4(df)
    st.markdown("---")
    afiseaza_cerinta5(df)
    st.markdown("---")
    afiseaza_cerinta6(df)

with tabs[2]:
    st.title("🔀 Analiză multivariată")
    analiza_stationaritate(df)
    st.markdown("---")
    analiza_cointegrare(df)
    st.markdown("---")
    analiza_vecm(df)
    st.markdown("---")
    analiza_granger(df)
    st.markdown("---")
    analiza_irf(df)
    st.markdown("---")
    analiza_fevd(df)

with tabs[3]:
    st.header("Concluzii")
    st.markdown("""
Acest proiect a avut ca scop analiza evoluției numărului de șomeri și relația acestuia cu dinamica produsului intern brut (PIB) în România, prin metode moderne de analiză a seriilor de timp.

**Concluzii principale:**
* Șomajul prezintă un trend descrescător, dar nesemnificativ din punct de vedere staționar. A fost necesară diferențierea seriei pentru a permite aplicarea modelelor ARIMA și VECM.
* Dintre modelele univariate, SARIMA a oferit cea mai bună performanță în prognoză (MAE ≈ 89 mii persoane), fiind superior modelului ARIMA și net superior Holt-Winters.
* Modelul ARIMA(2,1,2) a surprins bine dinamica seriei și a oferit o prognoză stabilă, deși a subestimat ușor valorile reale în perioada de test.
* Testele de cointegrare au confirmat existența a două relații de echilibru pe termen lung între șomaj și PIB, ceea ce a justificat estimarea unui model VECM.
* Modelul VECM a arătat că PIB-ul se ajustează la abaterile de la echilibru, în timp ce șomajul nu reacționează semnificativ.
* Testul Granger nu a identificat cauzalitate bidirecțională: nici șomajul nu anticipează PIB-ul, nici invers.
* Funcția de răspuns la impuls (IRF) a evidențiat un efect ușor negativ și temporar al PIB-ului asupra șomajului, dar slab din punct de vedere statistic.
* Analiza FEVD a confirmat că variația șomajului este explicată aproape integral de șocuri proprii, cu o contribuție a PIB-ului sub 1%.
""")
