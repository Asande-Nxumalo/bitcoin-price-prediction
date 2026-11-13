import pandas as pd
import matplotlib.pyplot as plt

# Load the backtest results
df = pd.read_csv("results/garch_backtest.csv")

# Plot actual and predicted volatility
plt.figure(figsize=(14, 6))

plt.plot(df["actual"], label="Actual volatility")
plt.plot(df["garch"], label="GARCH(1,1) prediction")
plt.plot(df["random_forest"], label="Random Forest prediction")
plt.plot(df["baseline"], label="Persistence baseline")

plt.title("Bitcoin 24-Hour Volatility: Model Comparison")
plt.xlabel("Forecast observation")
plt.ylabel("Volatility (%)")

plt.legend()
plt.grid(True)

plt.savefig("results/model_comparison.png", dpi=300)
plt.show()

print("Model comparison graph saved successfully!")