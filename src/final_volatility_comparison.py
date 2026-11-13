
import pandas as pd
import matplotlib.pyplot as plt

# Load our saved predictions
df = pd.read_csv(
    "results/weighted_ensemble_predictions.csv"
)

# Create the comparison graph
plt.figure(figsize=(14, 7))

plt.plot(
    df["actual"],
    label="Actual Volatility",
    linewidth=2
)

plt.plot(
    df["garch"],
    label="GARCH"
)

plt.plot(
    df["random_forest"],
    label="Random Forest"
)

plt.plot(
    df["ensemble"],
    label="Weighted Ensemble"
)

plt.plot(
    df["baseline"],
    label="Baseline"
)

# Graph details
plt.title("Bitcoin 24-Hour Volatility Forecast Comparison")
plt.xlabel("Forecast Observation")
plt.ylabel("Volatility (%)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save graph
plt.savefig(
    "results/final_volatility_comparison.png",
    dpi=300
)

plt.show()

print("Final comparison graph saved successfully!")
