import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import numpy as np

from main import df_somaj_clean

# ----- 1. Vizualizare serie -----
plt.figure(figsize=(12, 5))
plt.plot(df_somaj_clean["Data"], df_somaj_clean["Numar someri"], marker='o')
plt.title("Evoluția numărului de șomeri")
plt.xlabel("Data")
plt.ylabel("Număr șomeri")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- 2. Test ADF pentru trend stochastic -----
adf_result = adfuller(df_somaj_clean["Numar someri"])
print("ADF statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Valori critice:", adf_result[4])

# ----- 3. Trend determinist (regresie pe timp) -----
# Cream o variabilă de timp
df_somaj_clean["Timp"] = np.arange(len(df_somaj_clean))

# Regresie liniară: șomaj ~ timp
X = sm.add_constant(df_somaj_clean["Timp"])
y = df_somaj_clean["Numar someri"]
model = sm.OLS(y, X).fit()
print(model.summary())

df_somaj_clean.to_csv("cerinta1_output.csv", index=False)
