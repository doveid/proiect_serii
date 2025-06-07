import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Citim fișierul pregătit
df = pd.read_csv("serii_multivariate.csv", parse_dates=["Data"])

# Pregătim funcție pentru testul ADF
def test_adf(serie, nume):
    rezultat = adfuller(serie)
    print(f"\nTest ADF pentru {nume}:")
    print(f"  ADF statistic: {rezultat[0]:.4f}")
    print(f"  p-value: {rezultat[1]:.4f}")
    for key, value in rezultat[4].items():
        print(f"  Valoare critică {key}: {value:.4f}")
    if rezultat[1] < 0.05:
        print(f"  ⇒ Seria {nume} este STAȚIONARĂ (p < 0.05)")
    else:
        print(f"  ⇒ Seria {nume} NU este staționară (p ≥ 0.05)")

# Aplicăm testul
test_adf(df["Somaj"], "Somaj")
test_adf(df["PIB"], "PIB")
