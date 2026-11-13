
import pandas as pd
import matplotlib.pyplot as plt

# Load prepared Bitcoin data
df = pd.read_csv("data/bitcoin_features.csv")

# Create moving-average distance features
df["ma_6_gap"] = (df["close"] / df["ma_6"] - 1) * 100
df["ma_24_gap"] = (df["close"] / df["ma_24"] - 1) * 100

# Select features and prediction target
columns = [
    "return_1",
    "return_6",
    "ma_6_gap",
    "ma_24_gap",
    "volatility_6",
    "target_return_24h"
]

# Calculate correlation matrix
correlation = df[columns].corr()

# Display correlation with future returns
print("CORRELATION WITH FUTURE 24-HOUR RETURNS")
print("---------------------------------------")

print(
    correlation["target_return_24h"]
    .sort_values(ascending=False)
)

# Plot correlation heatmap
plt.figure(figsize=(10, 7))

plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(columns)),
    columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(columns)),
    columns
)

plt.title("Bitcoin Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig("results/correlation_heatmap.png")

plt.show()
