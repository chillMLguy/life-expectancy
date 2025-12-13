# Life Expectancy Analysis & Prediction

This repository contains an exploratory data analysis and a predictive model for life expectancy at birth across countries. The main work is in the Jupyter notebook `report.ipynb` which walks through data cleaning, visualization, feature engineering and model training (including hyperparameter tuning).
It has been created blog post, which conclude the analysis. Link : 
https://medium.com/@makskulicki8/wealth-power-and-longevity-what-actually-determines-how-long-we-live-bba60be22b0b

## Contents

- `report.ipynb` — Jupyter notebook with the full analysis, charts and model training.
- `data_lexp.xlsx` — dataset used by the notebook (life expectancy and country indicators).
- (Other files in the repo as applicable)

## Project summary

The goal is to predict "Life expectancy at birth, total (years)" using country-level indicators such as GDP per capita, access to electricity, population, population growth and health expenditure.

Key points from the notebook:
- Data cleaning: rows with placeholder ".." values are handled; countries with many missing values were dropped; a few countries had manual per-country imputations with usage of linear regression.
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
- openpyxl (for reading .xlsx files in pandas)

You can install the typical dependencies with:

pip install numpy pandas matplotlib seaborn scikit-learn openpyxl

## Visualizations

The notebook creates several charts to help understand the data:
- Time series of life expectancy for selected major countries.
- Bubble chart of GDP per capita (log scale) vs life expectancy for 2022, sized by population.
- Histograms of feature distributions.
- Correlation heatmap.

These visualizations are generated inline in the notebook.

## Model details

- Baseline models: Linear Regression and Decision Tree (sklearn).
- Best performing model found: Random Forest with hyperparameter tuning via GridSearchCV (tested n_estimators, max_depth, min_samples_split, min_samples_leaf).
- Best CV R² (during grid search) was reported in the notebook; final test R² ≈ 0.829 with RMSE ≈ 3.145.

## Suggestions & possible improvements

- Expand feature set (health system indicators, education, urbanization, smoking prevalence, etc.).
- Try gradient boosting models (XGBoost, LightGBM, CatBoost) with careful tuning.
- Fix small issues (typos in country names, deprecated pandas usage) to make the pipeline more robust and reproducible.

## Acknowledgments

World Bank Open Data for providing open and accessible global datasets

Scikit‑learn community for machine learning tools

## License

This project is provided for educational and research purposes. Please refer to the World Bank data usage policy when reusing the data.


