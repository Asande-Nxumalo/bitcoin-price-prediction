
import pandas as pd
import numpy as np

from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Bitcoin data
df = pd.read_csv("data/bitcoin_4h.csv")

# Calculate 4-hour percentage returns
df["returns"] = df["close"].pct_change() * 100

df = df.dropna().reset_index(drop=True)

# Use the first 80% for training
split_index = int(len(df) * 0.8)

train_returns = df["returns"].iloc[:split_index]
test_returns = df["returns"].iloc[split_index:]

# Fit GARCH(1,1)
model = arch_model(
    train_returns,
    mean="Zero",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal"
)

fitted_model = model.fit(disp="off")

print(fitted_model.summary())

# Forecast volatility for the next 6 candles
forecasts = fitted_model.forecast(
    horizon=6,
    reindex=False
)

predicted_volatility = np.sqrt(
    forecasts.variance.iloc[-1].mean()
)

print("\nGARCH 24-HOUR VOLATILITY FORECAST")
print(predicted_volatility)
