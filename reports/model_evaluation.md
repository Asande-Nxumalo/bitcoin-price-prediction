
# Bitcoin Volatility Forecasting Using Statistical and Machine Learning Models

## 1. Project Introduction

## 2. Dataset and Data Preparation

### 2.1 Data Source

The project uses historical BTC/USDT spot market data obtained from Binance.

The dataset contains four-hour candlestick observations covering approximately three years.

Each observation contains:

- Open price
- High price
- Low price
- Closing price
- Trading volume
- Candle timestamp

### 2.2 Dataset Description

The original dataset contains 6,575 observations.

The data was inspected for quality issues before model development.

The following checks were performed:

- Missing-value detection.
- Duplicate timestamp detection.
- Four-hour interval consistency.
- Chronological ordering.

The initial data validation found:

- Zero missing values.
- Zero duplicate timestamps.
- Zero unexpected gaps in the four-hour intervals.

### 2.3 Feature Engineering

Additional variables were created to help the models identify historical market behaviour.

These include:

| Feature | Description |
|---|---|
| return_1 | Previous four-hour percentage return |
| return_6 | Percentage return over six candles |
| ma_6 | Six-candle moving average |
| ma_24 | Twenty-four-candle moving average |
| volatility_6 | Rolling six-candle return volatility |
| volatility_24 | Rolling twenty-four-candle return volatility |

Moving-average gaps and absolute returns were also used in the volatility modelling experiments.

### 2.4 Target Variable

The main volatility target is the standard deviation of the next six four-hour returns.

This represents the volatility over the next 24 hours.

The target is constructed using future observations, while model features use only information available at the forecast origin.

This separation is important to prevent future information from leaking into model training.


### 1.1 Background

Bitcoin is a highly volatile cryptocurrency whose market price can experience substantial fluctuations over short periods.

Volatility is an important measure of financial risk. Understanding and forecasting volatility can help investors, traders and risk analysts assess potential market uncertainty.

This project investigates whether historical Bitcoin market data can be used to forecast future volatility over a 24-hour horizon.

### 1.2 Project Objective

The primary objective is to develop, train and evaluate statistical and machine learning models capable of forecasting Bitcoin's 24-hour future volatility.

The project compares four modelling approaches:

1. GARCH (Generalized Autoregressive Conditional Heteroskedasticity).
2. Random Forest Regression.
3. A hybrid regime-switching model.
4. A weighted ensemble model.

A persistence baseline is also included to determine whether the more complex models provide meaningful improvements.

### 1.3 Research Questions

This project investigates the following questions:

- Can historical Bitcoin returns help forecast future volatility?
- Does Random Forest outperform traditional GARCH modelling?
- Can combining statistical and machine learning models improve forecasting accuracy?
- How effectively can the models identify periods of elevated volatility?
- Which model is most suitable for volatility risk analysis?

### 1.4 Forecasting Horizon

The forecasting horizon is 24 hours.

Bitcoin market data is collected at four-hour intervals. Therefore, each forecast estimates volatility over the next six four-hour observations.

The project focuses on volatility forecasting rather than predicting whether Bitcoin's price will increase or decrease.
## 3. Modelling Methodology

### 3.1 Forecasting Approach

The project uses a time-series forecasting approach to predict Bitcoin's future volatility.

The models are trained using historical observations and evaluated on later observations. This preserves the chronological order of the data and reflects how forecasting would work in practice.

The main objective is to estimate the standard deviation of the next six four-hour Bitcoin returns.

### 3.2 Random Forest Regression

Random Forest is an ensemble machine-learning algorithm that combines predictions from multiple decision trees.

The model uses historical return and volatility features, including:

- Recent returns.
- Rolling volatility measures.
- Absolute returns.
- Moving averages of absolute returns.

The model was configured with:

- 300 decision trees.
- Maximum tree depth of 5.
- Minimum samples required to split a node: 20.
- Minimum samples per leaf: 10.

Random Forest was evaluated against a persistence baseline, which assumes that the most recently observed volatility is a reasonable estimate of future volatility.

### 3.3 GARCH(1,1) Model

The Generalized Autoregressive Conditional Heteroskedasticity model is a statistical approach designed to model changing financial-market volatility.

This project uses a GARCH(1,1) specification with:

- Zero mean.
- Normal conditional error distribution.
- One ARCH term.
- One GARCH term.

The model estimates conditional variance using past squared returns and past conditional variance.

GARCH is included because financial returns often exhibit volatility clustering, where periods of high volatility tend to be followed by further periods of high volatility.

### 3.4 Hybrid Regime-Switching Model

A hybrid model was developed to combine the strengths of GARCH and Random Forest.

The model compares short-term volatility with longer-term volatility using the ratio:

Short-term volatility / Long-term volatility.

It uses this ratio to identify changes in the volatility environment and select between the GARCH and Random Forest forecasts.

The regime-switching threshold was investigated using earlier validation folds before evaluating the selected configuration on later folds.

### 3.5 Weighted Ensemble Model

A weighted ensemble was also developed by combining the GARCH and Random Forest forecasts.

The ensemble prediction is calculated as:

Ensemble Forecast = (GARCH weight × GARCH forecast) + (Random Forest weight × Random Forest forecast).

The weights were tested using earlier validation folds. The selected configuration assigned:

- 10% weight to GARCH.
- 90% weight to Random Forest.

The ensemble was then evaluated on later observations.

### 3.6 Time-Series Validation

Randomly shuffling financial time-series observations can introduce unrealistic training and testing conditions.

To avoid this, the project uses chronological validation through TimeSeriesSplit.

The main validation configuration uses:

- Five chronological folds.
- 500 test observations per fold.
- A six-observation gap between training and testing data.

For the GARCH backtest, forecasts were generated at six-observation intervals. This produced 420 forecasts across the five folds.

The hybrid and weighted ensemble configurations were tuned using the earlier folds and evaluated on the later folds.

### 3.7 Evaluation Metrics

The models are evaluated using the following metrics:

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted volatility. Lower values indicate better forecasting accuracy.

**Root Mean Squared Error (RMSE)**

Measures prediction errors while penalising larger errors more heavily. Lower values indicate better performance.

**Precision**

Measures the proportion of predicted high-volatility observations that were actually high-volatility observations.

**Recall**

Measures the proportion of actual high-volatility observations correctly identified by a model.

**F1-score**

Combines precision and recall into a single measure. It is useful when evaluating the detection of high-volatility periods.

The persistence baseline is included to determine whether the statistical and machine-learning models provide improvements over a simple forecasting approach.
## 4. Model Results and Performance Comparison

### 4.1 Random Forest Volatility Forecasting

The Random Forest model was evaluated using five chronological validation folds.

The model achieved an overall:

- MAE: 0.3384
- RMSE: 0.4342

The persistence baseline achieved:

- MAE: 0.4098
- RMSE: 0.5482

Random Forest outperformed the persistence baseline across all five validation folds.

This suggests that historical market features contain useful information for forecasting future volatility.

However, the correlation between actual and predicted volatility was relatively weak. The model also produced smoother forecasts than the actual volatility series, indicating that it did not fully capture sudden volatility spikes.

### 4.2 GARCH Volatility Forecasting

The GARCH(1,1) model was evaluated through a walk-forward backtest.

The results were:

- MAE: 0.3888
- RMSE: 0.4700

The GARCH model outperformed the persistence baseline but did not outperform Random Forest overall.

The estimated GARCH parameters were:

- Omega: 0.1079
- Alpha: 0.1259
- Beta: 0.7771

The sum of alpha and beta was approximately 0.9030.

This indicates persistent volatility, although the estimated persistence is below one.

### 4.3 Overall Model Comparison

The following table compares the three main forecasting approaches and the persistence baseline across 420 walk-forward forecasts.

| Model | MAE | RMSE |
|---|---:|---:|
| Random Forest | 0.3403 | 0.4316 |
| GARCH(1,1) | 0.3888 | 0.4700 |
| Persistence baseline | 0.4180 | 0.5463 |

Random Forest achieved the lowest MAE and RMSE among these models.

This indicates that Random Forest provided the most accurate overall continuous volatility forecasts in this backtest.

### 4.4 Hybrid Model Results

The hybrid model selected between GARCH and Random Forest based on the relationship between short-term and long-term volatility.

The regime threshold was tuned using earlier validation folds and evaluated on later folds.

On the final evaluation folds, the results were:

| Model | MAE | RMSE |
|---|---:|---:|
| Random Forest | 0.3492 | 0.4391 |
| Hybrid | 0.3644 | 0.4496 |
| GARCH(1,1) | 0.3974 | 0.4755 |
| Persistence baseline | 0.3819 | 0.5183 |

The hybrid model improved on GARCH and the persistence baseline but did not outperform Random Forest.

### 4.5 Weighted Ensemble Results

The weighted ensemble combined 10% GARCH and 90% Random Forest.

The weights were selected using earlier validation folds and evaluated on later folds.

The final evaluation results were:

| Model | MAE | RMSE |
|---|---:|---:|
| Random Forest | 0.3492 | 0.4391 |
| Weighted ensemble | 0.3523 | 0.4401 |
| GARCH(1,1) | 0.3974 | 0.4755 |
| Persistence baseline | 0.3819 | 0.5183 |

The weighted ensemble performed close to Random Forest but did not improve on it.

This suggests that adding GARCH forecasts did not provide enough additional information to improve the overall continuous forecasting accuracy.

### 4.6 Key Findings

The experiments produced several important findings:

1. Random Forest achieved the best overall continuous volatility forecasting performance.
2. GARCH performed better than the persistence baseline in the full walk-forward comparison.
3. The hybrid model improved on GARCH but did not outperform Random Forest.
4. The weighted ensemble performed similarly to Random Forest but offered no meaningful improvement in the final evaluation.
5. All models experienced difficulty capturing sudden volatility spikes.
## 5. High-Volatility Detection

### 5.1 Objective

Accurately forecasting volatility is important, but identifying periods of unusually high volatility is also valuable for financial risk management.

This experiment investigates whether the models can identify high-volatility periods using a classification threshold.

An observation is classified as high volatility when its actual or predicted volatility exceeds 1.0%.

### 5.2 Classification Results

The models were evaluated using 168 later-period forecasts.

There were 36 actual high-volatility observations in this evaluation sample.

The results were:

| Model | True Positives | False Positives | False Negatives | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| GARCH | 16 | 16 | 20 | 0.500 | 0.444 |
| Random Forest | 9 | 8 | 27 | 0.529 | 0.250 |
| Weighted Ensemble | 9 | 9 | 27 | 0.500 | 0.250 |
| Persistence Baseline | 17 | 18 | 19 | 0.486 | 0.472 |

### 5.3 Interpretation

The GARCH model identified more high-volatility observations than Random Forest and the weighted ensemble.

Random Forest achieved slightly higher precision than the other models, meaning that a greater proportion of its high-volatility alerts were correct.

However, its recall was relatively low. It missed many actual high-volatility observations.

The persistence baseline achieved the highest recall in this comparison, identifying 17 of the 36 actual high-volatility observations.

These results demonstrate that strong continuous forecasting performance does not necessarily translate into strong high-volatility detection.

### 5.4 Threshold Sensitivity Analysis

Different classification thresholds were investigated to understand how the models' detection performance changes.

The analysis considered thresholds between 0.7% and 1.2%.

At the 0.7% threshold, the weighted ensemble achieved the highest F1-score, approximately 0.574.

At the 1.0% threshold, the persistence baseline achieved an F1-score of approximately 0.479, slightly higher than GARCH at approximately 0.471.

The results show that model performance depends on the selected volatility threshold.

A lower threshold generally identifies more high-volatility observations but can also produce more false alarms.

### 5.5 Limitations of the Detection Experiment

The high-volatility detection experiment has several limitations:

- The final evaluation sample contains only 168 forecasts.
- The number of actual high-volatility observations is relatively small.
- Results can change considerably when the classification threshold changes.
- Thresholds were explored on the same later-period evaluation sample, so these results should be treated as exploratory rather than independent confirmation of an optimal threshold.

A larger evaluation sample and a separate validation period would be useful before selecting a threshold for practical risk-management use.
## 6. Limitations and Recommendations

### 6.1 Model Limitations

Although the models produced useful forecasting results, several limitations should be considered.

**Limited ability to predict sudden volatility spikes**

Random Forest produced relatively smooth forecasts and struggled to capture sudden increases in actual volatility. This is important because extreme market movements are particularly relevant to financial risk management.

**GARCH model limitations**

GARCH captured volatility persistence but did not achieve the lowest overall forecasting error. Its performance also varied depending on the volatility environment.

**Limited dataset**

The dataset contains approximately three years of four-hour Bitcoin observations. Although this provides thousands of observations, it may not capture every possible market condition or extreme market event.

**Model assumptions**

GARCH assumes a particular structure for conditional variance and uses normally distributed errors in this project. Actual financial returns can exhibit heavy tails and extreme observations that are not well represented by a normal distribution.

**Limited high-volatility observations**

The high-volatility detection evaluation contained only 36 actual high-volatility observations. This limits the confidence that can be placed in the precision and recall estimates.

**Threshold selection**

The classification threshold affects the detection results. The threshold analysis was exploratory and used the same later-period evaluation sample. A separate validation period would be required to select and independently assess an operational threshold.

### 6.2 Recommendations for Further Research

The following improvements could be investigated:

1. **Alternative GARCH specifications:** Test GJR-GARCH and EGARCH to investigate whether asymmetric volatility effects improve forecasting performance.
2. **Alternative error distributions:** Compare normal, Student's t and other suitable distributions to better represent extreme financial returns.
3. **Additional machine-learning models:** Investigate gradient boosting and other algorithms.
4. **Longer historical datasets:** Include more market cycles and extreme volatility events.
5. **Additional explanatory variables:** Investigate trading volume, market liquidity and other relevant market indicators.
6. **Out-of-sample testing:** Evaluate the final selected model on a completely separate period that was not used for model development or tuning.
7. **Risk-management applications:** Investigate whether volatility forecasts can improve position sizing, portfolio risk measurement and risk limits.

### 6.3 Practical Implications

The results suggest that machine-learning models can provide useful information for forecasting cryptocurrency volatility.

Random Forest achieved the best overall continuous forecasting accuracy in this project. GARCH also provided useful information about volatility persistence and performed relatively well in some high-volatility conditions.

However, none of the models should be treated as a reliable system for predicting every extreme market movement.

The forecasts are better viewed as supplementary risk-management information rather than guaranteed predictions of future market behaviour.
## 7. Conclusion

### 7.1 Summary of the Project

This project investigated the use of statistical and machine-learning techniques to forecast Bitcoin volatility over a 24-hour horizon.

Historical BTC/USDT four-hour market data was collected, cleaned and transformed into features suitable for time-series modelling.

Four modelling approaches were investigated:

- GARCH(1,1)
- Random Forest Regression
- Hybrid regime-switching model
- Weighted ensemble model

A persistence baseline was included to assess whether the more advanced models provided improvements over a simple forecasting approach.

### 7.2 Main Findings

The experimental results showed that Random Forest achieved the best overall continuous volatility forecasting performance.

The GARCH model also provided useful forecasts and demonstrated the ability to model volatility persistence.

The hybrid and weighted ensemble approaches did not outperform Random Forest in the final continuous forecasting comparisons.

The high-volatility detection experiment showed that identifying extreme volatility is more challenging than forecasting volatility levels. Model performance also changed depending on the classification threshold.

### 7.3 Final Conclusion

The findings demonstrate that historical Bitcoin market data contains useful information for forecasting future volatility.

However, the results also highlight the limitations of statistical and machine-learning models when dealing with sudden and extreme market movements.

Random Forest was the strongest overall model in this study, while GARCH provided a useful statistical benchmark and additional insight into volatility dynamics.

Further research using longer datasets, alternative volatility models and independent out-of-sample testing could improve the reliability of the findings.

Overall, this project demonstrates the application of financial mathematics, statistical modelling, machine learning and time-series validation to a practical financial risk forecasting problem.