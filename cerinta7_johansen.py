import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Citire date
df = pd.read_csv("serii_multivariate.csv", parse_dates=["Data"])

# Selectăm doar valorile (fără coloana Data)
data = df[["Somaj", "PIB"]]

# Aplicăm testul Johansen
johansen_result = coint_johansen(data, det_order=0, k_ar_diff=1)  # det_order=0: fără constantă

# Afișăm valorile testului
print("=== Valori test Johansen (Trace Statistic) ===")
for i, trace_stat in enumerate(johansen_result.lr1):
    crit_5 = johansen_result.cvt[i, 1]  # 5% critical value
    print(f"r = {i} : Trace Stat = {trace_stat:.4f} | CV (5%) = {crit_5:.4f}")
    if trace_stat > crit_5:
        print("  ⇒ Se respinge H0 (existență cointegrare)")
    else:
        print("  ⇒ Nu se respinge H0")

# Note:
# r = 0 → nu există cointegrare
# r = 1 → există cel puțin 1 relație de cointegrare
