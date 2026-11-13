
import pandas as pd

# Load saved predictions
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

print("\nVOLATILITY THRESHOLD ANALYSIS")
print("==============================")

for threshold in thresholds:

    actual_spikes = df["actual"] > threshold

    print(f"\nTHRESHOLD: {threshold}%")
    print("Actual spikes:", actual_spikes.sum())

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

        print(
            f"{model.upper():15} "
            f"Precision: {precision:.3f} "
            f"Recall: {recall:.3f}"
        )

print("\nTHRESHOLD ANALYSIS COMPLETE")
