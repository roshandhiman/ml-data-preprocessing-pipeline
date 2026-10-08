# Data Preprocessing & Feature Engineering Strategy: Telco Customer Churn

A complete, leakage-free preprocessing and feature engineering pipeline, tested on the public **IBM Telco Customer Churn** dataset (7,043 customers, 21 columns). Built for the *Machine Learning Engineer Internship, Week 2 task*.

## What this project covers
| Stage | What was done |
|---|---|
| Data quality | Found **11 hidden missing values** (blank strings in `TotalCharges`) that `isna()` cannot see; checked duplicates and outliers (IQR + z-score) |
| Cleaning | Domain-rule imputation (`TotalCharges = 0` when `tenure = 0`), compared with mean/median imputation |
| Feature engineering | `tenure_group`, `num_services`, `avg_monthly_paid`, `charge_gap`, `has_streaming`, `has_protection`, `auto_pay`, `log_total` |
| Encoding & scaling | `OrdinalEncoder`, `OneHotEncoder`, `StandardScaler` inside a `ColumnTransformer`, **fitted on the train split only** |
| Feature selection | Correlation filter, mutual information, RFE, random-forest importance |
| Dimensionality reduction | PCA (19 of 49 components keep 95% variance) |
| Validation | Baseline vs full pipeline, 5-fold stratified CV |

## Key results
- Churn rate by contract: month-to-month **42.7%**, one-year **11.3%**, two-year **2.8%**
- Churn rate by tenure group: 0-12 months **47.4%** down to 49-72 months **9.5%**
- Full pipeline: Test ROC-AUC **0.8460** vs baseline **0.8423**; a 15-feature RFE subset reaches **0.8458** (70% fewer features)
- Six "No internet service" dummy columns are perfectly correlated with `InternetService_No` and can be dropped

## Run it
```bash
pip install -r requirements.txt
jupyter notebook Data_Preprocessing_Feature_Engineering.ipynb   # outputs are already saved in the notebook
# or run as a script (saves figures to ./figures)
MPLBACKEND=Agg python preprocessing_pipeline.py
```

## Structure
```
data/Telco-Customer-Churn.csv                 dataset (IBM sample data)
Data_Preprocessing_Feature_Engineering.ipynb   full analysis with outputs and plots
preprocessing_pipeline.py                      same code as a plain script
figures/                                       plots used in the report
```

## Data source
IBM Telco Customer Churn sample dataset: https://github.com/IBM/telco-customer-churn-on-icp4d

## Author
Roshanpreet Singh
# ml-data-preprocessing-pipeline
