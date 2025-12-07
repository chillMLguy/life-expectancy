# Life Expectancy Analysis & Prediction

This repository contains an exploratory data analysis and a predictive model for life expectancy at birth across countries. The main work is in the Jupyter notebook `report.ipynb` which walks through data cleaning, visualization, feature engineering and model training (including hyperparameter tuning).

## Contents

- `report.ipynb` — Jupyter notebook with the full analysis, charts and model training.
- `data_lexp.xlsx` — dataset used by the notebook (life expectancy and country indicators).
- (Other files in the repo as applicable)

## Project summary

The goal is to predict "Life expectancy at birth, total (years)" using country-level indicators such as GDP per capita, access to electricity, population, population growth and health expenditure.

Key points from the notebook:
- Data cleaning: rows with placeholder ".." values are handled; countries with many missing values were dropped; a few countries had manual per-country imputations.
- Feature engineering: GDP per capita was converted to numeric and transformed with natural logarithm (log(GDP per capita)) because of a strong linear relationship between log(GDP per capita) and life expectancy.
- Train/test split: Time-based split — training on years before 2019 and testing on 2019 and later.
- Models tested: Linear Regression, Decision Tree Regressor, Random Forest Regressor.
- Hyperparameter tuning: GridSearchCV on Random Forest (with a scaler in a Pipeline).
- Best final model: Random Forest with cross-validated tuning.
  - Reported final test metrics (from the notebook):
    - RMSE ≈ 3.145
    - R² ≈ 0.829
  - This corresponds to a relative error of about 4–4.5% given the target mean (≈ 69.8 years).

The notebook also includes exploratory visualizations (time series for select countries, GDP per capita vs life expectancy bubble chart, histograms, correlation heatmap).

## Requirements

The notebook was built using standard Python data science libraries. At minimum you should have:

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- openpyxl 
