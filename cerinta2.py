import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Citire și pregătire serie originală (asumăm că ai deja fișierul pregătit)
df = pd.read_csv("cerinta1_output.csv", parse_dates=["Data"])
df = df.sort_values("Data").reset_index(drop=True)

# Aplicăm diferențierea primei ordini
df["Diferenta somaj"] = df["Numar someri"].diff()

# Eliminăm primul rând (NaN)
df_diff = df.dropna()

# Test ADF pe seria diferențiată
adf_result = adfuller(df_diff["Diferenta somaj"])
print("ADF statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Valori critice:", adf_result[4])

# Plot noua serie
plt.figure(figsize=(12, 5))
plt.plot(df_diff["Data"], df_diff["Diferenta somaj"], marker='o')
plt.title("Seria diferențiată a numărului de șomeri (prima diferență)")
plt.xlabel("Data")
plt.ylabel("Diferența între trimestre")
plt.grid(True)
plt.tight_layout()
plt.show()

# Salvăm fișierul pentru cerințele următoare (opțional)
df_diff.to_csv("cerinta2_output.csv", index=False)
