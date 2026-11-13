
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load prepared Bitcoin data
df = pd.read_csv("data/bitcoin_features.csv")

# Create moving-average distance features
df["ma_6_gap"] = (df["close"] / df["ma_6"] - 1) * 100
df["ma_24_gap"] = (df["close"] / df["ma_24"] - 1) * 100

# Select the input features
features = [
    "return_1",
    "return_6",
    "ma_6_gap",
    "ma_24_gap",
    "volatility_6"
]

# Select the prediction target
X = df[features]
y = df["target_return_24h"]

# Split data chronologically: 80% training, 20% testing
split_index = int(len(df) * 0.8)

# Remove the final 6 training observations to prevent target leakage
X_train = X.iloc[:split_index - 6]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index - 6]
y_test = y.iloc[split_index:]

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on unseen data
predictions = model.predict(X_test)

# Create a simple baseline that predicts zero return
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
print("LINEAR REGRESSION RESULTS")
print("-------------------------")
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print("\nModel MAE:", mae)
print("Model RMSE:", rmse)
print("Directional accuracy:", direction_accuracy, "%")

print("\nBASELINE RESULTS")
print("----------------")
print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)
