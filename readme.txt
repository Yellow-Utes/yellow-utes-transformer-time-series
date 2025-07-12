# Time-series Forecasting using Transformer architecture

Transformer architecture is generally associated with LLM models and not Time-series (which are generally implemented using some form of regression architecture).

As compute becomes quicker and cheaper, and time-series datasets become increasingly large and complex - transformer architecture used by LLMs may prove to be a superior approach more traditional approaches.

# Predictive Purpose of this Model

Purely an educational project - but using a real-world example and dataset - including useful examples of feature engineering and built-in bias.

## Yellow Utes

Yellow Utes rents a fleet of utes in Brisbane. The distance these utes travel is important for two primary reasons - firstly, the business charges per kilometer driven and secondly the fleet requires servicing every 10,000 kms. Understanding capacity utilitisation may be another interesting metric using this model.

## Historical Data Set (distance-driven.csv)
The dataset is based on real-life distance driven by all utes since November 2023. In the interest of protecting customer privacy - since this a public repository - no customer details have been used to create this model. A more accruate model could be discovered if customer demographics and/or their driving history with Yellow Utes.

# date: YYYY-MM-DD
The primary key and context for timeseries data

# distance_km: float
The total distance driven by all utes on that day. This is what we're predicting.

# utes_driven: [0..n]
The number of utes driven on that day.

# day_of_week: int
representation of day of week (1=Sunday..7=Saturday)

# is_weekend: bool(1/0)
Is this a weekend day

# month: int [1..12]
Numerical representation of month

# seasonal_scalar: float[0..1]
0.5 - 0.5 * Sin(2 * Pi() * ( DAYOFYEAR(date) - 14 ) / 365.0)
As a seasonal business, a feature to engineer would be where the data sits in the current phase of the seasonal cycle.
The formula above plots the date on a sine-wave with January 14 peak (peak of summer).

# days_in_business: int[1..n]
How long has yellow utes been in business in days.

# is_holiday:  bool[1/0]
holidays.csv contains a list of public holidays for Brisbane for the training period. This file is merged with the historical dataset to create the final merge for training.

# midday_temp:  float(n)
Temperature in Brisbane at 12 midday (weather.csv contains weather data for all days in training period)

# weather_category: enum(int)
Weather.csv also contains a weather descriptor which is then assigned an inbuilt bias to score the weather. This is another form of bias - a way to tell the model we think rainy days will impact distance driven.
scores = {
            "fog": -10,
            "rain": -50,
            "clouds": 40,
            "clear": 50
        }
# wind_speed: float
Perhaps the wind speed might impact load-hauling driving behaviors.




## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train and evaluate the model:
```python
python timeseries_forecasting_model.py
```

## Data Set

## Model Architecture

The model uses a transformer encoder architecture with:
- input projections
- Positional encoding (a must for timeseries)
- Multi-head self-attention layers
- Feed-forward networks
- Output projection for multi-step prediction

## Features

- includes various time-based features and examples of feature engineering
- MAE, MSE, and RMSE metrics
- Plots predictions vs actuals for analysis
- Univariate time series forecasting
- Uses standard scaling
- Includes early stopping to prevent overfitting

## Notes

- The model is designed for univariate time series forecasting
- Uses standard scaling for numerical stability
- Includes early stopping to prevent overfitting
- Supports both CPU and GPU training