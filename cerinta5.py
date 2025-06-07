import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Încarcă seria completă
df = pd.read_csv("cerinta1_output.csv", parse_dates=["Data"])
serie = df["Numar someri"]

# ----- Împărțim în training și test -----
n = len(serie)
train_size = int(n * 0.8)
train, test = serie[:train_size], serie[train_size:]
dates_train, dates_test = df["Data"][:train_size], df["Data"][train_size:]

# ----- Estimăm ARIMA pe training -----
model = ARIMA(train, order=(2, 1, 2))  # modelul ales anterior
fit_model = model.fit()

# ----- Predicție pentru setul de test -----
forecast = fit_model.get_forecast(steps=len(test))
predicted_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# ----- Plot -----
plt.figure(figsize=(12, 6))
plt.plot(dates_train, train, label="Training", color="blue")
plt.plot(dates_test, test, label="Test (real)", color="black")
plt.plot(dates_test, predicted_mean, label="Predicție", color="orange")
plt.fill_between(dates_test, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3, label="Interval 95%")
plt.title("Predicție ARIMA(2,1,2) – punct și pe interval")
plt.xlabel("Data")
plt.ylabel("Număr șomeri")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
