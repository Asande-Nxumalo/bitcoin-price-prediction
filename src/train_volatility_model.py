
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# 1. Load Bitcoin data
df = pd.read_csv("data/bitcoin_4h.csv")

# 2. Calculate historical returns
df["returns"] = df["close"].pct_change() * 100

# 3. Create volatility features
df["volatility_6"] = df["returns"].rolling(6).std()
df["volatility_24"] = df["returns"].rolling(24).std()

df["absolute_returns"] = df["returns"].abs()

df["absolute_return_mean_6"] = (
    df["absolute_returns"].rolling(6).mean()
)

# 4. Calculate future 24-hour volatility
df["target_volatility_24h"] = (
    df["returns"]
    .shift(-1)
    .rolling(6)
    .std()
    .shift(-5)
)

# 5. Remove missing observations
df = df.dropna().reset_index(drop=True)

# 6. Select input features
features = [
    "volatility_6",
    "volatility_24",
    "absolute_return_mean_6"
]

X = df[features]
y = df["target_volatility_24h"]

# 7. Create walk-forward validation
tscv = TimeSeriesSplit(
    n_splits=5,
    test_size=500,
    gap=6
)

results = {
    "actual": [],
    "predicted": [],
    "baseline": []
}

# 8. Train and test across five periods
for fold, (train_index, test_index) in enumerate(
    tscv.split(X), start=1
):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # Create Random Forest model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict future volatility
    predictions = model.predict(X_test)

    # Persistence baseline
    baseline_predictions = X_test["volatility_6"].to_numpy()

    # Store results
    results["actual"].extend(y_test.tolist())
    results["predicted"].extend(predictions.tolist())
    results["baseline"].extend(
        baseline_predictions.tolist()
    )

    # Calculate errors
    model_mae = mean_absolute_error(
        y_test, predictions
    )

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )

    print(f"\nVALIDATION PERIOD {fold}")
    print("----------------------")
    print("Model MAE:", model_mae)
    print("Baseline MAE:", baseline_mae)


# 9. Calculate overall performance
actual = np.array(results["actual"])
predicted = np.array(results["predicted"])
baseline = np.array(results["baseline"])

model_mae = mean_absolute_error(actual, predicted)
model_rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

baseline_mae = mean_absolute_error(actual, baseline)
baseline_rmse = np.sqrt(
    mean_squared_error(actual, baseline)
)

# 10. Display final results
print("\nFINAL VOLATILITY FORECAST RESULTS")
print("=================================")

print("\nRandom Forest:")
print("MAE:", model_mae)
print("RMSE:", model_rmse)

print("\nPersistence baseline:")
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/volatility_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
print("MAE:", baseline_mae)
print("RMSE:", baseline_rmse)
