
import pandas as pd

# Load predictions
df = pd.read_csv(
    "results/weighted_ensemble_predictions.csv"
)

models = [
    "garch",
    "random_forest",
    "ensemble",
    "baseline"
]

thresholds = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

print("\nF1-SCORE ANALYSIS")
print("=================")

for threshold in thresholds:

    actual_spikes = df["actual"] > threshold

    print(f"\nTHRESHOLD: {threshold}%")

    for model in models:

        predicted_spikes = df[model] > threshold

        tp = (actual_spikes & predicted_spikes).sum()
        fp = (~actual_spikes & predicted_spikes).sum()
        fn = (actual_spikes & ~predicted_spikes).sum()

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        print(
            f"{model.upper():15} "
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"F1: {f1:.3f}"
        )

print("\nF1-SCORE ANALYSIS COMPLETE")
