import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Citim seria originală
df = pd.read_csv("cerinta1_output.csv", parse_dates=["Data"])
serie = df["Numar someri"]

# ----- Testăm manual câteva modele ARIMA(p,1,q) -----
configuratii = [(1,1,1), (2,1,1), (1,1,2), (2,1,2)]
rezultate = {}

for cfg in configuratii:
    model = ARIMA(serie, order=cfg)
    rezultat = model.fit()
    rezultate[cfg] = rezultat.aic  # salvăm scorul AIC
    print(f"ARIMA{cfg} -> AIC: {rezultat.aic}")

# ----- Alegem modelul cu AIC minim -----
best_cfg = min(rezultate, key=rezultate.get)
print(f"\nModelul ales: ARIMA{best_cfg} (cu AIC minim)")

# ----- Estimăm modelul ales și afișăm sumarul -----
model_final = ARIMA(serie, order=best_cfg)
rez_final = model_final.fit()
print(rez_final.summary())

# ----- Plot: fitted vs real -----
plt.figure(figsize=(12, 5))
plt.plot(df["Data"], serie, label="Serie originală", color="black")
plt.plot(df["Data"][1:], rez_final.fittedvalues[1:], label=f"ARIMA{best_cfg} - fitted", linestyle='--')
plt.title(f"Model ARIMA{best_cfg} aplicat seriei de șomaj")
plt.xlabel("Data")
plt.ylabel("Număr șomeri")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
