import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Citire date
df = pd.read_csv("cerinta1_output.csv", parse_dates=["Data"])
serie = df["Numar someri"]
dates = df["Data"]

# Împărțim în training/test (80/20)
n = len(serie)
train_size = int(n * 0.8)
train, test = serie[:train_size], serie[train_size:]
dates_test = dates[train_size:]

# ----- ARIMA(2,1,2) -----
arima_model = ARIMA(train, order=(2, 1, 2)).fit()
arima_forecast = arima_model.forecast(steps=len(test))

# ----- SARIMA(1,1,1)(1,0,1)[4] -----
sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,0,1,4)).fit()
sarima_forecast = sarima_model.forecast(steps=len(test))

# ----- Holt-Winters -----
hw_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=4).fit()
hw_forecast = hw_model.forecast(len(test))

# ----- Erori -----
def calc_erori(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred) ** 0.5
    return mae, rmse

mae_arima, rmse_arima = calc_erori(test, arima_forecast)
mae_sarima, rmse_sarima = calc_erori(test, sarima_forecast)
mae_hw, rmse_hw = calc_erori(test, hw_forecast)

print(f"ARIMA        - MAE: {mae_arima:.2f}, RMSE: {rmse_arima:.2f}")
print(f"SARIMA       - MAE: {mae_sarima:.2f}, RMSE: {rmse_sarima:.2f}")
print(f"Holt-Winters - MAE: {mae_hw:.2f}, RMSE: {rmse_hw:.2f}")

# ----- Plot -----
plt.figure(figsize=(12, 6))
plt.plot(dates_test, test, label="Valori reale", color="black")
plt.plot(dates_test, arima_forecast, label="ARIMA(2,1,2)", linestyle="--", color="green")
plt.plot(dates_test, sarima_forecast, label="SARIMA", linestyle="--", color="orange")
plt.plot(dates_test, hw_forecast, label="Holt-Winters", linestyle="--", color="blue")
plt.title("Comparare metode univariate – ARIMA vs SARIMA vs Holt-Winters")
plt.xlabel("Data")
plt.ylabel("Număr șomeri")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
