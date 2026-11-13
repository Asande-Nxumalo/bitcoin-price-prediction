import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load model comparison results
df = pd.read_csv("results/garch_backtest.csv")

# Define high-volatility observations
high_volatility = df[df["actual"] <= 1.0]

print("LOW VOLATILITY ANALYSIS")
print("========================")

print("Total observations:", len(df))
print("High-volatility observations:", len(high_volatility))

# Compare models during high volatility
for model in ["garch", "random_forest", "baseline"]:

    mae = mean_absolute_error(
        high_volatility["actual"],
        high_volatility[model]
    )

    rmse = np.sqrt(mean_squared_error(
        high_volatility["actual"],
        high_volatility[model]
    ))

    print(f"\n{model.upper()}")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))

# Compare average predicted and actual volatility
print("\nAVERAGE HIGH VOLATILITY")
print("Actual:", round(high_volatility["actual"].mean(), 4))
print("GARCH:", round(high_volatility["garch"].mean(), 4))
print("Random Forest:", round(high_volatility["random_forest"].mean(), 4))
