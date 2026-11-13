
import pandas as pd
import numpy as np

from arch import arch_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Bitcoin data
df = pd.read_csv("data/bitcoin_4h.csv")

# Calculate returns and volatility
df["returns"] = df["close"].pct_change() * 100
df["volatility_6"] = df["returns"].rolling(6).std()

# Target: standard deviation of the NEXT six 4-hour returns
df["target_volatility_24h"] = (
    df["returns"].shift(-1)
    .rolling(6)
    .std()
    .shift(-5)
)

# Preserve original row numbers
df["raw_index"] = np.arange(len(df))

# Remove rows with missing values
valid = df.dropna().reset_index(drop=True)

# Load the Random Forest predictions
rf = pd.read_csv("results/volatility_predictions.csv")

# Same five validation periods
tscv = TimeSeriesSplit(
    n_splits=5,
    test_size=500,
    gap=6
)

results = {
    "actual": [],
    "garch": [],
    "random_forest": [],
    "baseline": []
}

# Evaluate each validation period
for fold, (train_index, test_index) in enumerate(
    tscv.split(valid), start=1
):

    print(f"\nVALIDATION PERIOD {fold}")
    print("----------------------")

    # Forecast every six observations
    for j in range(0, len(test_index), 6):

        position = test_index[j]

        # Original position in the Bitcoin dataset
        raw_position = int(valid.loc[position, "raw_index"])

        # All returns observable at this forecast origin
        history = df["returns"].iloc[:raw_position + 1].dropna()

        # Fit GARCH(1,1)
        model = arch_model(
            history,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal"
        )

        fitted = model.fit(disp="off")

        # Forecast the next six 4-hour variances
        forecast = fitted.forecast(
            horizon=6,
            reindex=False
        )

        predicted_volatility = np.sqrt(
            forecast.variance.iloc[-1].mean()
        )

        # Actual future volatility
        actual = valid.loc[position, "target_volatility_24h"]

        # Random Forest prediction at the same origin
        rf_row = (fold - 1) * 500 + j

        rf_prediction = rf.loc[rf_row, "predicted"]

        # Persistence baseline at the same origin
        baseline = valid.loc[position, "volatility_6"]

        results["actual"].append(actual)
        results["garch"].append(predicted_volatility)
        results["random_forest"].append(rf_prediction)
        results["baseline"].append(baseline)

    # Report progress
    print("Forecasts completed:", len(results["actual"]))

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Calculate performance
print("\nFINAL GARCH BACKTEST RESULTS")
print("============================")

for model_name in ["garch", "random_forest", "baseline"]:

    mae = mean_absolute_error(
        results_df["actual"],
        results_df[model_name]
    )

    rmse = np.sqrt(
        mean_squared_error(
            results_df["actual"],
            results_df[model_name]
        )
    )

    print(f"\n{model_name.upper()}")
    print("MAE:", mae)
    print("RMSE:", rmse)

# Save results
results_df.to_csv(
    "results/garch_backtest.csv",
    index=False
)

print("\nGARCH backtest results saved successfully!")
