
import pandas as pd
import numpy as np

# Load model predictions
df = pd.read_csv(
    "results/weighted_ensemble_predictions.csv"
)

# Define high-volatility threshold
threshold = 1.0

# Actual high-volatility observations
actual_spikes = df["actual"] > threshold

# Model predictions
models = [
    "garch",
    "random_forest",
    "ensemble",
    "baseline"
]

print("\nBITCOIN VOLATILITY SPIKE DETECTION")
print("===================================")

print("High-volatility threshold:", threshold, "%")
print("Actual high-volatility observations:", actual_spikes.sum())

for model in models:

    predicted_spikes = df[model] > threshold

    true_positives = (
        actual_spikes & predicted_spikes
    ).sum()

    false_positives = (
        ~actual_spikes & predicted_spikes
    ).sum()

    false_negatives = (
        actual_spikes & ~predicted_spikes
    ).sum()

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    print(f"\n{model.upper()}")
    print("True positives:", true_positives)
    print("False positives:", false_positives)
    print("False negatives:", false_negatives)
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
