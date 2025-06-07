import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import VECM
import warnings
warnings.filterwarnings("ignore")

# Citire date
df = pd.read_csv("serii_multivariate.csv", parse_dates=["Data"])
data = df[["Somaj", "PIB"]]

# Estimare model VECM (același ca înainte)
vecm_model = VECM(data, k_ar_diff=1, deterministic="n").fit()

# Calcul funcție de răspuns la impulsuri
irf = vecm_model.irf(10)  # răspuns pe 10 perioade (trimestre)

# Plot IRF: efectul unui șoc în PIB asupra Somajului
irf.plot(orth=False, impulse="PIB", response="Somaj")
plt.title("Răspunsul șomajului la un șoc pozitiv în PIB")
plt.tight_layout()
plt.show()
