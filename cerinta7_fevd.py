import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings("ignore")

# Citim datele
df = pd.read_csv("serii_multivariate.csv", parse_dates=["Data"])
data = df[["Somaj", "PIB"]]

# Diferențiere pentru a obține serii staționare
data_diff = data.diff().dropna()

# Estimăm model VAR pe serii diferențiate
var_model = VAR(data_diff)
results = var_model.fit(maxlags=2, ic='aic')

# Calculăm FEVD pe 10 perioade
fevd = results.fevd(10)

# Afișăm contribuția PIB la variația Somajului
print("\n=== FEVD: Contribuția PIB la variația Somajului ===")
print(fevd.decomp[:, 0, 1])  # timp, variabilă dependentă=Somaj, sursă de șoc=PIB

# Plot
fevd.plot()
plt.tight_layout()
plt.show()
