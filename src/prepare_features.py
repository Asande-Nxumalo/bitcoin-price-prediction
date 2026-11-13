
import pandas as pd

# Load Bitcoin data
df = pd.read_csv("data/bitcoin_4h.csv")

# Calculate the previous 4-hour return
df["return_1"] = df["close"].pct_change() * 100

# Calculate the previous 24-hour return
df["return_6"] = df["close"].pct_change(periods=6) * 100

# Calculate 6-candle moving average
df["ma_6"] = df["close"].rolling(window=6).mean()

# Calculate 24-candle moving average
df["ma_24"] = df["close"].rolling(window=24).mean()

# Calculate recent volatility using 6 candles
df["volatility_6"] = df["return_1"].rolling(window=6).std()

# Calculate the return over the NEXT 24 hours
df["target_return_24h"] = (
    df["close"].shift(-6) / df["close"] - 1
) * 100

# Remove rows with missing values
df = df.dropna()

# Save the prepared dataset
df.to_csv("data/bitcoin_features.csv", index=False)

# Display the results
print("Feature engineering completed!")
print("\nDataset shape:", df.shape)
print("\nFeatures created:")
print(df[[
    "return_1",
    "return_6",
    "ma_6",
    "ma_24",
    "volatility_6",
    "target_return_24h"
]].head())

print("\nSaved to data/bitcoin_features.csv")
