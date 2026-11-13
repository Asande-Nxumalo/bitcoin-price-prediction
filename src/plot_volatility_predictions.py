
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/volatility_predictions.csv")

plt.figure(figsize=(14, 6))

plt.plot(df["actual"], label="Actual volatility")
plt.plot(df["predicted"], label="Random Forest prediction")
plt.plot(df["baseline"], label="Persistence baseline")

plt.title("Bitcoin 24-Hour Volatility Forecast")
plt.xlabel("Validation observation")
plt.ylabel("Volatility (%)")

plt.legend()
plt.grid(True)

plt.savefig("results/volatility_predictions.png", dpi=300)
plt.show()

print("Graph saved successfully!")

print("\nFIRST 10 PREDICTIONS")
print(df.head(10).round(3))

print("\nCORRELATION BETWEEN ACTUAL AND PREDICTED")
print(df["actual"].corr(df["predicted"]))

print("\nAVERAGE VOLATILITY")
print("Actual:", df["actual"].mean())
print("Random Forest:", df["predicted"].mean())
print("Baseline:", df["baseline"].mean())
print("\nVOLATILITY DISTRIBUTION")

print("Actual standard deviation:",
      df["actual"].std())

print("Predicted standard deviation:",
      df["predicted"].std())

high_volatility = df[df["actual"] > 1.0]

print("\nHIGH VOLATILITY ANALYSIS")

print("Number of high volatility observations:",
      len(high_volatility))

print("Random Forest high volatility MAE:",
      (high_volatility["actual"] -
       high_volatility["predicted"]).abs().mean())