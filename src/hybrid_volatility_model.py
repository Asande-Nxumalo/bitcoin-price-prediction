import pandas as pd
import numpy as np

# Load the historical Bitcoin data
df = pd.read_csv("data/bitcoin_4h.csv")

# Calculate percentage returns
df["returns"] = df["close"].pct_change() * 100

# Calculate recent volatility
df["volatility_6"] = df["returns"].rolling(6).std()

# Calculate longer-term volatility
df["volatility_24"] = df["returns"].rolling(24).std()

# Remove missing values
df = df.dropna().reset_index(drop=True)

print("BITCOIN DATA PREPARED")
print("=====================")

print("Number of observations:", len(df))
print("Average recent volatility:", round(df["volatility_6"].mean(), 4))
print("Average longer-term volatility:", round(df["volatility_24"].mean(), 4))
# Compare short-term and longer-term volatility
df["volatility_ratio"] = (
    df["volatility_6"] / df["volatility_24"]
)

# Identify elevated short-term volatility
df["high_volatility_regime"] = (
    df["volatility_6"] > df["volatility_24"]
)

print("\nMARKET REGIME ANALYSIS")
print("======================")

high = df["high_volatility_regime"].sum()
low = len(df) - high

print("Elevated volatility observations:", high)
print("Lower volatility observations:", low)

print("Percentage of elevated volatility:",
      round(high / len(df) * 100, 2), "%")
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the existing backtest predictions
predictions = pd.read_csv("results/garch_backtest.csv")

# Reconstruct the exact forecast origins used in the GARCH backtest
raw = pd.read_csv("data/bitcoin_4h.csv")
raw["returns"] = raw["close"].pct_change() * 100
raw["volatility_6"] = raw["returns"].rolling(6).std()
raw["volatility_24"] = raw["returns"].rolling(24).std()
raw["target_volatility_24h"] = (
    raw["returns"].shift(-1).rolling(6).std().shift(-5)
)
raw["raw_index"] = np.arange(len(raw))

valid = raw.dropna().reset_index(drop=True)

tscv = TimeSeriesSplit(n_splits=5, test_size=500, gap=6)

regimes = []

for train_index, test_index in tscv.split(valid):
    for j in range(0, len(test_index), 6):
        position = test_index[j]
        raw_position = int(valid.loc[position, "raw_index"])

        ratio = (
            raw.loc[raw_position, "volatility_6"]
            / raw.loc[raw_position, "volatility_24"]
        )

        regimes.append(ratio)

# Attach the regime indicators to the predictions
predictions["volatility_ratio"] = regimes

# Select a model using information available at forecast time
predictions["hybrid"] = np.where(
    predictions["volatility_ratio"] > 1,
    predictions["garch"],
    predictions["random_forest"]
)

print("\nHYBRID MODEL RESULTS")
print("====================")

for model in ["garch", "random_forest", "hybrid", "baseline"]:
    mae = mean_absolute_error(
        predictions["actual"], predictions[model]
    )

    rmse = np.sqrt(mean_squared_error(
        predictions["actual"], predictions[model]
    ))

    print(f"\n{model.upper()}")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))

predictions.to_csv(
    "results/hybrid_predictions.csv",
    index=False
)

print("\nHybrid predictions saved successfully!")
# =====================================
# THRESHOLD OPTIMISATION
# =====================================

# Identify the five validation periods
predictions["fold"] = np.repeat(
    [1, 2, 3, 4, 5], 84
)

# Use the first three folds for threshold selection
training = predictions[predictions["fold"] <= 3].copy()

# Keep the last two folds for final testing
testing = predictions[predictions["fold"] >= 4].copy()

thresholds = [0.8, 0.9, 1.0, 1.1, 1.2]

best_threshold = None
best_mae = float("inf")

print("\nTHRESHOLD OPTIMISATION")
print("======================")

for threshold in thresholds:

    hybrid = np.where(
        training["volatility_ratio"] > threshold,
        training["garch"],
        training["random_forest"]
    )

    mae = mean_absolute_error(
        training["actual"], hybrid
    )

    print(
        f"Threshold: {threshold} | "
        f"Training MAE: {mae:.4f}"
    )

    if mae < best_mae:
        best_mae = mae
        best_threshold = threshold

print("\nBEST THRESHOLD:", best_threshold)
print("TRAINING MAE:", round(best_mae, 4))

# Evaluate the selected threshold on unseen periods
testing["hybrid"] = np.where(
    testing["volatility_ratio"] > best_threshold,
    testing["garch"],
    testing["random_forest"]
)

print("\nFINAL TEST RESULTS")
print("==================")

for model in ["garch", "random_forest", "hybrid", "baseline"]:

    if model == "hybrid":
        values = testing["hybrid"]
    else:
        values = testing[model]

    mae = mean_absolute_error(testing["actual"], values)

    rmse = np.sqrt(
        mean_squared_error(testing["actual"], values)
    )

    print(f"\n{model.upper()}")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))

    # =====================================
# WEIGHTED ENSEMBLE EXPERIMENT
# =====================================

weights = np.arange(0.1, 1.0, 0.1)

best_weight = None
best_mae = float("inf")

print("\nWEIGHTED ENSEMBLE OPTIMISATION")
print("==============================")

# Find the best weight using the first three folds
for weight in weights:

    ensemble = (
        weight * training["garch"]
        + (1 - weight) * training["random_forest"]
    )

    mae = mean_absolute_error(
        training["actual"], ensemble
    )

    print(
        f"GARCH weight: {weight:.1f} | "
        f"Training MAE: {mae:.4f}"
    )

    if mae < best_mae:
        best_mae = mae
        best_weight = weight

print("\nBEST GARCH WEIGHT:", round(best_weight, 2))
print("BEST TRAINING MAE:", round(best_mae, 4))

# Test the selected weight on the final two folds
testing["ensemble"] = (
    best_weight * testing["garch"]
    + (1 - best_weight) * testing["random_forest"]
)

print("\nWEIGHTED ENSEMBLE TEST RESULTS")
print("==============================")

for model in ["garch", "random_forest", "ensemble", "baseline"]:

    if model == "ensemble":
        prediction = testing["ensemble"]
    else:
        prediction = testing[model]

    mae = mean_absolute_error(
        testing["actual"], prediction
    )

    rmse = np.sqrt(
        mean_squared_error(testing["actual"], prediction)
    )

    print(f"\n{model.upper()}")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    testing[
    [
        "actual",
        "garch",
        "random_forest",
        "ensemble",
        "baseline"
    ]
].to_csv(
    "results/weighted_ensemble_predictions.csv",
    index=False
)

print("\nWeighted ensemble predictions saved!")