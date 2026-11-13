
import pandas as pd
import matplotlib.pyplot as plt

# Load Bitcoin data
df = pd.read_csv("data/bitcoin_4h.csv")

# Calculate 4-hour percentage returns
df["returns"] = df["close"].pct_change() * 100

# Calculate absolute returns
df["absolute_returns"] = df["returns"].abs()

# Calculate rolling volatility using 6 candles
df["volatility_6"] = df["returns"].rolling(6).std()

# Calculate rolling volatility using 24 candles
df["volatility_24"] = df["returns"].rolling(24).std()

# Calculate average volatility
print("VOLATILITY ANALYSIS")
print("-------------------")

print("\nAverage 6-candle volatility:")
print(df["volatility_6"].mean())

print("\nAverage 24-candle volatility:")
print(df["volatility_24"].mean())

# Calculate correlation between current and next-period volatility
df["future_volatility_6"] = df["volatility_6"].shift(-6)

correlation = df[
    ["volatility_6", "future_volatility_6"]
].corr().iloc[0, 1]

print("\nCorrelation between current and future volatility:")
print(correlation)

# Plot Bitcoin returns
plt.figure(figsize=(12, 5))

plt.plot(
    df["returns"],
    linewidth=0.7
)

plt.title("Bitcoin 4-Hour Returns")
plt.xlabel("Time")
plt.ylabel("Return (%)")
plt.grid(True)

plt.tight_layout()
plt.savefig("results/bitcoin_returns.png")
plt.show()

# Plot rolling volatility
plt.figure(figsize=(12, 5))

plt.plot(
    df["volatility_6"],
    label="6-candle volatility"
)

plt.plot(
    df["volatility_24"],
    label="24-candle volatility"
)

plt.title("Bitcoin Rolling Volatility")
plt.xlabel("Time")
plt.ylabel("Volatility (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("results/bitcoin_volatility.png")
plt.show()
