# bitcoin-price-prediction
Bitcoin price forecasting using historical market data and machine learning.
##Market
Bitcoin/USD-Binance Spot
##Timeframe
4-Hour Candles
##Forecast Horizon
24 hours(6 candles)
##Models
-Linear regression
-Random forest
-XGBoost
-LSTM
##Evaluation
Models will be evaluated using:
-Mean Absolute Error(MAE)
-Root Mean Squared Error(RMSE)
-Directional accuracy
##Objective
To compare machine learning models and to investigate on how accurately they forecast Bitcoin prices
# Bitcoin Volatility Forecasting Using Machine Learning and GARCH Models

## Project Overview

This project investigates whether machine learning and statistical time-series models can forecast Bitcoin's short-term volatility using historical BTC/USDT market data.

The project uses 4-hour Binance Spot candlestick data and forecasts volatility over the next 24 hours.

The main objective is to compare different forecasting approaches, evaluate their performance using time-series validation, and investigate their ability to identify periods of elevated market volatility.

## Objectives

- Collect and analyse historical Bitcoin market data.
- Perform exploratory data analysis and feature engineering.
- Develop machine learning models for return and volatility forecasting.
- Implement a GARCH(1,1) volatility model.
- Compare forecasting performance using walk-forward validation.
- Investigate high-volatility detection and model limitations.

## Dataset

- **Asset:** Bitcoin (BTC/USDT)
- **Exchange:** Binance Spot
- **Frequency:** 4-hour candles
- **Historical period:** Approximately 3 years
- **Observations:** 6,575 candles
- **Forecast horizon:** 24 hours

The dataset includes open, high, low, close and volume information.

## Models Developed

### Return Forecasting
- Linear Regression
- Random Forest
- XGBoost

### Volatility Forecasting
- Random Forest
- GARCH(1,1)
- Hybrid regime-switching model
- Weighted ensemble

A persistence baseline was also used to assess whether the forecasting models provided an improvement over a simple historical-volatility estimate.

## Model Evaluation

Models were evaluated using chronological train/test splits and walk-forward validation to reduce the risk of look-ahead bias.

Evaluation metrics included:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Precision
- Recall
- F1-score

## Key Findings

- Random Forest achieved the best overall continuous volatility forecasting performance in the backtest.
- GARCH performed better than Random Forest during the retrospectively identified high-volatility periods.
- The weighted ensemble performed competitively but did not outperform Random Forest on overall continuous forecasting error.
- Return forecasting models did not consistently outperform the zero-return baseline.
- High-volatility detection involved a trade-off between identifying volatility spikes and generating false alarms.

### Volatility Forecasting Results

| Model | MAE | RMSE |
|---|---:|---:|
| Random Forest | 0.3403 | 0.4316 |
| GARCH(1,1) | 0.3888 | 0.4700 |
| Persistence baseline | 0.4180 | 0.5463 |

These results are from the 420-forecast backtest. The separate held-out comparison used 168 forecasts and produced slightly different results.

## Project Structure

```text
bitcoin-price-prediction/
│
├── data/
│   ├── bitcoin_4h.csv
│   └── bitcoin_features.csv
│
├── models/
│
├── notebooks/
│
├── reports/
│   └── model_evaluation.md
│
├── results/
│   ├── bitcoin_price_history.png
│   ├── bitcoin_returns.png
│   ├── bitcoin_volatility.png
│   ├── correlation_heatmap.png
│   ├── final_volatility_comparison.png
│   ├── model_comparison.png
│   └── volatility_predictions.png
│
├── src/
│   ├── bitcoin_data.py
│   ├── explore_data.py
│   ├── prepare_features.py
│   ├── train_linear_regression.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_garch.py
│   ├── train_volatility_model.py
│   ├── backtest_garch.py
│   ├── hybrid_volatility_model.py
│   └── ...
│
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- ARCH
- Git and GitHub

## Limitations

- Cryptocurrency volatility is highly unpredictable.
- Forecasting performance can change across market regimes.
- Random Forest forecasts can underreact to sudden volatility spikes.
- Model performance does not guarantee future predictive accuracy.
- The high-volatility classification thresholds require further independent validation.

## Conclusion

This project demonstrates the application of statistical modelling, machine learning and time-series validation to cryptocurrency volatility forecasting.

The results suggest that Random Forest is the strongest overall continuous volatility forecaster among the tested models, while GARCH provides useful information during periods of elevated volatility.

Further work could investigate additional features, alternative volatility models and more robust out-of-sample evaluation.

**Disclaimer:** This project is for educational and research purposes only. It does not constitute financial or investment advice.


