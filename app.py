import streamlit as st
import pandas as pd
from Streamlit_Serii_Timp import (
    afiseaza_cerinta1, afiseaza_cerinta2, afiseaza_cerinta3,
    afiseaza_cerinta4, afiseaza_cerinta5, afiseaza_cerinta6,
    analiza_stationaritate, analiza_cointegrare, analiza_vecm,
    analiza_granger, analiza_irf, analiza_fevd
)

st.set_page_config(page_title="Analiza Serii de Timp", layout="wide")
st.title("📊 Proiect Serii de Timp – Șomaj și PIB în România")

# Încarcă datele
df = pd.read_csv("serii_multivariate.csv", parse_dates=["Data"])
df.rename(columns={"Somaj": "Numar someri"}, inplace=True)


st.header("🔹 Cerința 1")
afiseaza_cerinta1(df)

st.header("🔹 Cerința 2")
afiseaza_cerinta2(df)

st.header("🔹 Cerința 3")
afiseaza_cerinta3(df)

st.header("🔹 Cerința 4")
afiseaza_cerinta4(df)

st.header("🔹 Cerința 5")
afiseaza_cerinta5(df)

st.header("🔹 Cerința 6")
afiseaza_cerinta6(df)

st.header("🔹 Cerința 7.1")
analiza_stationaritate(df)

st.header("🔹 Cerința 7.2")
analiza_cointegrare(df)

st.header("🔹 Cerința 7.3")
analiza_vecm(df)

st.header("🔹 Cerința 7.4")
analiza_granger(df)

st.header("🔹 Cerința 7.5")
analiza_irf(df)

st.header("🔹 Cerința 7.6")
analiza_fevd(df)
