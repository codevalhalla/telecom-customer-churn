# Telecom Customer Churn Prediction

## Project Overview

This project uses machine learning to predict telecom customer churn. We employ logistic regression to analyze customer behavior and determine the likelihood of a customer canceling their service. The model identifies key factors that influence customer retention and helps develop targeted retention strategies.

## Dataset

**Source**: [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Dataset Size**: 7,043 customer records

### Features

**Numerical Features**:
- `tenure`: Number of months customer has been with the company
- `monthlycharges`: Monthly service charges
- `totalcharges`: Total charges accumulated

**Categorical Features** (16 features):
- `gender`: Customer gender (male/female)
- `seniorcitizen`: Whether customer is a senior citizen (0/1)
- `partner`: Whether customer has a partner (yes/no)
- `dependents`: Whether customer has dependents (yes/no)
- `phoneservice`: Whether customer has phone service (yes/no)
- `multiplelines`: Multiple phone lines (yes/no/no phone service)
- `internetservice`: Internet service type (DSL/fiber optic/none)
- `onlinesecurity`: Online security add-on (yes/no/no internet service)
- `onlinebackup`: Online backup add-on (yes/no/no internet service)
- `deviceprotection`: Device protection add-on (yes/no/no internet service)
- `techsupport`: Tech support add-on (yes/no/no internet service)
- `streamingtv`: Streaming TV service (yes/no/no internet service)
- `streamingmovies`: Streaming movies service (yes/no/no internet service)
- `contract`: Contract type (month-to-month/one year/two year)
- `paperlessbilling`: Paperless billing (yes/no)
- `paymentmethod`: Payment method (electronic check/mailed check/bank transfer/credit card)

**Target Variable**:
- `churn`: Whether customer churned (yes=1/no=0)

## Data Preparation

### Data Cleaning
- Converted all column names to lowercase with underscores
- Converted `totalcharges` from object to numeric type (some records contained spaces)
- Standardized categorical string values to lowercase with underscores
- Converted target variable `churn` to binary (0/1) format
- **No missing values found** in the processed dataset

### Class Distribution
- **No Churn**: 73.0%
- **Churn**: 27.0%
- Global churn rate: 0.27 (27%)

### Data Splitting
- **Training set**: 60% (4,225 records)
- **Validation set**: 20% (1,409 records)
- **Test set**: 20% (1,409 records)

## Exploratory Data Analysis

### Feature Importance Analysis

#### 1. Mutual Information Score
Measures how informative each feature is for predicting churn:

| Feature | MI Score |
|---------|----------|
| contract | 0.0983 |
| onlinesecurity | 0.0631 |
| techsupport | 0.0610 |
| internetservice | 0.0559 |
| onlinebackup | 0.0469 |

**Top Finding**: `contract` is the most informative feature, followed by add-on services.

#### 2. Risk Ratio and Difference Analysis

**Risk Ratio Formula**: Group Churn Rate / Global Churn Rate
- **> 1**: Group more likely to churn
- **< 1**: Group less likely to churn

**Difference Formula**: Global Churn Rate - Group Churn Rate
- **> 0**: Group less likely to churn
- **< 0**: Group more likely to churn

**Key Insights**:
- **Contract Type**: Month-to-month customers have 1.6x higher churn risk vs. two-year contract customers
- **Partner Status**: Customers without partners have 1.22x higher churn risk
- **Dependents**: Customers without dependents have 1.16x higher churn risk
- **Internet Service**: Fiber optic customers have 1.57x higher churn risk than DSL customers
- **Add-on Services**: Customers without online security have 1.56x higher churn risk
- **Payment Method**: Electronic check payers have 1.69x higher churn risk

#### 3. Numerical Feature Correlation

| Feature | Correlation with Churn |
|---------|----------------------|
| tenure | -0.352 |
| monthlycharges | 0.197 |
| totalcharges | -0.196 |

**Interpretation**: Longer tenure strongly reduces churn likelihood; higher monthly charges increase churn risk.

## Methodology

### 1. Feature Engineering
- Used one-hot encoding via `DictVectorizer` (scikit-learn)
- Converted categorical and numerical features into a feature matrix
- Generated 45 features from the original 19 features

### 2. Model: Logistic Regression
- **Algorithm**: Logistic regression with sigmoid activation function
- **Activation**: sigmoid(z) = 1 / (1 + exp(-z)) → outputs probability between 0 and 1
- **Solver**: lbfgs (default)
- **Max Iterations**: 100 (increased for convergence)
- **Classification Threshold**: 0.5

### 3. Model Interpretation
A smaller model trained on key features reveals feature weights:

| Feature | Weight |
|---------|--------|
| contract=month-to-month | 0.971 |
| contract=two_year | -0.948 |
| tenure | -0.036 |
| monthlycharges | 0.027 |
| contract=one_year | -0.024 |

**Interpretation**: Higher positive weights increase churn probability; negative weights decrease it.

## Results

### Model Performance

**Validation Accuracy**: 80.27%

**Test Accuracy**: 81.33%

### Predictions

The model outputs:
- **Probability scores**: Predicted churn probability for each customer (0-1)
- **Binary prediction**: Churn decision (1 = will churn, 0 = will not churn) at 0.5 threshold

## Usage

### Training the Model

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# Load and prepare data
df = pd.read_csv("./WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Data cleaning and preprocessing
df.columns = df.columns.str.lower().str.replace(' ', '_')
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce').fillna(0)

categorical_cols = ['gender', 'contract', 'internetservice', ...]
numerical_cols = ['tenure', 'monthlycharges', 'totalcharges']

# Split data
df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Feature encoding
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train[categorical_cols + numerical_cols].to_dict(orient='records'))
X_test = dv.transform(df_test[categorical_cols + numerical_cols].to_dict(orient='records'))

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, df_train['churn'])

# Make predictions
y_pred = model.predict_proba(X_test)[:, 1]
churn_decision = (y_pred >= 0.5).astype(int)
```

### Predicting for a Single Customer

```python
customer = {
    'gender': 'female',
    'seniorcitizen': 1,
    'contract': 'month-to-month',
    'tenure': 3,
    'monthlycharges': 70.3,
    # ... other features
}

X_customer = dv.transform([customer])
churn_probability = model.predict_proba(X_customer)[:, 1]
churn_decision = (churn_probability >= 0.5)

print(f"Churn Probability: {churn_probability[0]:.2%}")
print(f"Will Churn: {bool(churn_decision[0])}")
```

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
```

Install via:
```bash
pip install pandas numpy scikit-learn matplotlib
```

## Key Files

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Raw dataset
- `churn_prediction.py`: Main analysis and model training script (or Jupyter notebook)

## Business Insights & Recommendations

1. **Priority Segments for Retention**:
   - Month-to-month contract customers (1.6x churn risk)
   - Customers paying by electronic check (1.69x churn risk)
   - Customers without add-on services

2. **Retention Strategies**:
   - Encourage longer-term contracts (two-year contracts reduce churn 15x)
   - Promote add-on services (online security, tech support reduce churn ~56%)
   - Incentivize automatic payment methods over electronic checks

3. **Customer Segments**:
   - High-risk: New customers (low tenure), high monthly charges, month-to-month contracts
   - Low-risk: Long-tenure customers with multi-year contracts and add-on services

## Model Limitations

- Logistic regression assumes linear relationships between features and log-odds of churn
- Does not capture complex feature interactions
- Class imbalance (27% churn) may affect model performance on minority class
- Threshold of 0.5 may not be optimal for business objectives

## Future Improvements

- Experiment with tree-based models (Random Forest, XGBoost, LightGBM)
- Implement class balancing techniques (SMOTE, class weights)
- Hyperparameter tuning and cross-validation
- Feature engineering and interaction terms
- ROC-AUC optimization with custom decision thresholds
- Ensemble methods for better performance
- Feature importance analysis using SHAP values

## References

- Dataset: [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Scikit-learn Documentation: [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
