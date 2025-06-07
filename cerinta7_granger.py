import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# Citire date
df = pd.read_csv("serii_multivariate.csv", parse_dates=["Data"])

# Selectăm doar valorile (fără coloana Data)
data = df[["Somaj", "PIB"]]

# Aplicăm testul Granger pe 4 laguri
print("=== Test: PIB → Somaj ===")
grangercausalitytests(data[["Somaj", "PIB"]], maxlag=4, verbose=True)

print("\n=== Test: Somaj → PIB ===")
grangercausalitytests(data[["PIB", "Somaj"]], maxlag=4, verbose=True)
