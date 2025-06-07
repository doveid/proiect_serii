import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import warnings

warnings.filterwarnings("ignore")

# Citire seria originală de șomaj
df = pd.read_csv("cerinta1_output.csv", parse_dates=["Data"])
df = df.sort_values("Data").reset_index(drop=True)
serie = df["Numar someri"]

# ----- 1. Simple Exponential Smoothing -----
ses_model = SimpleExpSmoothing(serie, initialization_method="legacy-heuristic").fit()
ses_pred = ses_model.fittedvalues

# ----- 2. Holt's Linear Trend -----
holt_model = Holt(serie, initialization_method="legacy-heuristic").fit()
holt_pred = holt_model.fittedvalues

# ----- Plot comparativ -----
plt.figure(figsize=(12, 6))
plt.plot(df["Data"], serie, label="Serie originală", color='black')
plt.plot(df["Data"], ses_pred, label="SES", linestyle='--')
plt.plot(df["Data"], holt_pred, label="Holt", linestyle='--')
plt.title("Tehnici de netezire exponențială aplicate seriei de șomaj")
plt.xlabel("Data")
plt.ylabel("Număr șomeri")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- Salvare opțională -----
df["SES"] = ses_pred
df["Holt"] = holt_pred
df.to_csv("cerinta3_output.csv", index=False)
