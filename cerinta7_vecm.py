import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM
import warnings
warnings.filterwarnings("ignore")

# Citire date
df = pd.read_csv("serii_multivariate.csv", parse_dates=["Data"])

# Selectăm doar coloanele relevante
data = df[["Somaj", "PIB"]]

# Estimare model VECM
# k_ar_diff = 1 => lag total = 2 (1 pentru diferențe, 1 pentru relația de echilibru)
model = VECM(data, k_ar_diff=1, deterministic="n")  # fără constantă (det_order=0)
vecm_res = model.fit()

# Afișare rezumat
print(vecm_res.summary())
