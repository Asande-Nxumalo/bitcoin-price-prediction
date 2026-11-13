
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


# 1. Load Bitcoin dataset
df = pd.read_csv("data/bitcoin_features.csv")


# 2. Create moving-average distance features
df["ma_6_gap"] = (df["close"] / df["ma_6"] - 1) * 100
df["ma_24_gap"] = (df["close"] / df["ma_24"] - 1) * 100


# 3. Select input features
features = [
    "return_1",
    "return_6",
    "ma_6_gap",
    "ma_24_gap",
    "volatility_6"
]

X = df[features]
y = df["target_return_24h"]


# 4. Create walk-forward validation
tscv = TimeSeriesSplit(
    n_splits=5,
    test_size=500,
    gap=6
)


# 5. Store predictions and actual values
results = {
    "Linear Regression": {"actual": [], "predicted": []},
    "Random Forest": {"actual": [], "predicted": []},
    "XGBoost": {"actual": [], "predicted": []},
    "Zero-return baseline": {"actual": [], "predicted": []}
}


# 6. Run the five validation periods
for fold, (train_index, test_index) in enumerate(
    tscv.split(X), start=1
):

    print(f"\nVALIDATION PERIOD {fold}")
    print("----------------------")

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    # Create fresh models for each period
    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    }

    # Train and evaluate each model
    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results[name]["actual"].extend(y_test.tolist())
        results[name]["predicted"].extend(predictions.tolist())

        mae = mean_absolute_error(y_test, predictions)

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        print(f"{name} MAE: {mae:.4f}")
        print(f"{name} RMSE: {rmse:.4f}")

    # Zero-return baseline
    baseline_predictions = np.zeros(len(y_test))

    results["Zero-return baseline"]["actual"].extend(
        y_test.tolist()
    )

    results["Zero-return baseline"]["predicted"].extend(
        baseline_predictions.tolist()
    )

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")


# 7. Calculate overall out-of-sample performance
print("\n\nFINAL WALK-FORWARD RESULTS")
print("==========================")

summary = []

for name, values in results.items():

    actual = np.array(values["actual"])
    predicted = np.array(values["predicted"])

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    direction_accuracy = (
        np.mean(np.sign(actual) == np.sign(predicted)) * 100
    )

    summary.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "Directional Accuracy (%)": direction_accuracy
    })


summary_df = pd.DataFrame(summary)

print(
    summary_df.sort_values(
        by="MAE"
    ).to_string(index=False)
)
