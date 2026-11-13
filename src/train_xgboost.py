
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load prepared Bitcoin data
df = pd.read_csv("data/bitcoin_features.csv")

# Create moving-average distance features
df["ma_6_gap"] = (df["close"] / df["ma_6"] - 1) * 100
df["ma_24_gap"] = (df["close"] / df["ma_24"] - 1) * 100

# Select input features
features = [
    "return_1",
    "return_6",
    "ma_6_gap",
    "ma_24_gap",
    "volatility_6"
]

# Select prediction target
X = df[features]
y = df["target_return_24h"]

# Split data chronologically
split_index = int(len(df) * 0.8)

# Keep a 6-candle gap to prevent target leakage
X_train = X.iloc[:split_index - 6]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index - 6]
y_test = y.iloc[split_index:]

# Create XGBoost model
model = XGBRegressor(
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

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Create zero-return baseline
baseline_predictions = np.zeros(len(y_test))

# Calculate model performance
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Calculate baseline performance
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

# Calculate directional accuracy
direction_accuracy = (
    np.mean(np.sign(y_test) == np.sign(predictions)) * 100
)

# Display results
print("XGBOOST RESULTS")
print("----------------")
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print("\nModel MAE:", mae)
print("Model RMSE:", rmse)
print("Directional accuracy:", direction_accuracy, "%")

print("\nBASELINE RESULTS")
print("----------------")
print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)

# Display feature importance
print("\nFEATURE IMPORTANCE")
print("------------------")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))
