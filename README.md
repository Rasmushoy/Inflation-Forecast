# Inflation-Forecast

This repository contains code and data for forecasting Danish inflation using a wide range of machine learning models and high-dimensional macroeconomic data. The project includes multi-horizon forecasts, real-time evaluation, and stress-testing during the COVID-19 shock.

##  Project Structure

```
.
├── Data/                     # Cleaned and structured data (monthly time series)
├── Lasso/                    # LASSO regression models
├── PCR/                      # Principal Component Regression
├── Random_Forrest/           # Random Forest forecasting models
├── XGBoost/                  # Gradient boosting models
├── data_handling.ipynb       # Preprocessing and construction of explanatory variables
├── multi_horizon_evaluation.ipynb  # Main evaluation of all models across horizons
├── case_covid19.ipynb        # Special case study: Model performance during COVID-19
├── README.md                 # This file
```

## Motivation

Forecasting inflation is crucial for monetary policy, investment decisions, and economic planning. Traditional models often struggle with large datasets and structural shifts. This project leverages modern machine learning techniques and rich macroeconomic data to improve forecasting accuracy across different time horizons.

## Explanatory Variables

We use 219 monthly time series from January 2012 to December 2024, covering:

* Labour market, macroeconomics, prices
* Consumer behaviour, industry wages, demographics
* Cars, housing, energy, crime, confidence indicators

For more, see the appendix in the notebook: `data_handling.ipynb`.

##  Models

Implemented forecasting models include:

* **LASSO** (regularized linear regression)
* **PCR** (dimensionality reduction)
* **Random Forest** (ensemble of decision trees)
* **XGBoost** (gradient boosting)

## Key Notebooks

| Notebook                         | Description                                                    |
| -------------------------------- | -------------------------------------------------------------- |
| `data_handling.ipynb`            | Loads, cleans, and organizes the explanatory variable dataset  |
| `multi_horizon_evaluation.ipynb` | Evaluates model forecasts across different prediction horizons |
| `case_covid19.ipynb`             | Focused evaluation during COVID-19 period (2020–2021)          |

## Dependencies

Main packages:

* `pandas`, `numpy`, `scikit-learn`
* `statsmodels`, `xgboost`
* `matplotlib`, `seaborn`

Install everything via:

```bash
pip install -r requirements.txt
```

## Example Forecast

*(Optional: add a sample plot of forecast vs. actual inflation)*

## 👤Author

Rasmus & Mattias

M.Sc. Student, University of Copenhagen
