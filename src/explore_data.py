import pandas as pd
import matplotlib.pyplot as plt

# Load Bitcoin data
df = pd.read_csv("data/bitcoin_4h.csv")

# Convert timestamps
df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

# Display basic statistics
print("BITCOIN PRICE SUMMARY")
print(df["close"].describe())

# Calculate percentage price changes
df["returns"] = df["close"].pct_change() * 100

print("\nAVERAGE 4-HOUR RETURN:")
print(df["returns"].mean(), "%")

print("\nVOLATILITY (STANDARD DEVIATION):")
print(df["returns"].std(), "%")

# Plot Bitcoin closing prices
plt.figure(figsize=(12, 6))
plt.plot(df["open_time"], df["close"])

plt.title("Bitcoin (BTC/USDT) 4-Hour Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price (USDT)")
plt.grid(True)

plt.savefig("results/bitcoin_price_history.png")
plt.show()