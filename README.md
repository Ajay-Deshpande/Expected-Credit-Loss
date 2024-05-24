# Expected Credit Loss

This project focuses on predicting the Current Expected Credit Loss (CECL) by estimating various components such as Probability of Default (PD), Exposure at Default (EAD), and Loss Given Default (LGD). The goal is to create a comprehensive model that can accurately forecast the credit loss for a given loan portfolio.

## Business Problem

CECL is a critical measure for financial institutions to estimate potential credit losses over the life of a loan. Accurately predicting CECL helps in managing risk and ensuring regulatory compliance. This project addresses the need for a robust model that combines historical loan performance data with advanced statistical techniques to predict CECL.

## Approach

The approach taken in this project involves several key steps:
1. **Data Preparation**: Cleaning and preprocessing the loan performance data.
2. **Model Development**:
    - **PD Modeling**: Estimating the probability of default using a risk state transition matrix approach.
    - **EAD Modeling**: Calculating the exposure at default, which is the ratio of unpaid amount to the open balance.
    - **LGD Modeling**: Estimating the loss given default, representing the loss in case of default.
3. **Prediction and Scoring**: Applying the models to generate predictions for PD, EAD, and LGD, and using these predictions to calculate the CECL.
4. **Validation and Evaluation**: Comparing predicted values against actual outcomes and calculating model performance metrics.

## Code Overview

### Data Preparation

Data preparation involves loading the data, handling missing values, clipping values, encoding categorical features, and normalizing numerical features.
Some code for data wrangling

```python
import pandas as pd
import numpy as np

lgd['Actual_CO'] = lgd.Actual_CO.str.replace("[^0-9.]", '', regex=True).astype(float)
lgd['RecoveryAmt'] = lgd.RecoveryAmt.str.replace("[^0-9.]", '', regex=True).astype(float)
lgd['recovery_rate'] = (lgd.default_Balance - lgd[['default_Balance', 'Actual_CO']].min(axis=1) + lgd['RecoveryAmt']) / lgd.default_Balance

# Handling missing values and clipping recovery rate
lgd['Actual_CO'] = lgd['Actual_CO'].fillna(0)
lgd['recovery_rate'] = np.where(np.logical_and(lgd['Actual_CO'] <= 0, lgd['ChargeOff_Event'] == 1), 0, lgd['recovery_rate'])
lgd['recovery_rate'] = np.where(np.logical_and(lgd['Actual_CO'] <= 0, lgd['ChargeOff_Event'] == 0), 1, lgd['recovery_rate'])
lgd['recovery_rate'] = lgd['recovery_rate'].clip(0,1)
```

### Custom Models

#### LGD Model

The LGD model uses a combination of logistic regression and neural network to estimate the loss given default.
We take a probabilistic  multiple level LGD modeling approach for this 
We make three different modeling data - 
1) use entire LGD data for 100% recovery rate modeling; 
2) Exclude 100% recovery rate accounts for 0% recovery rate modeling; and 
3) Exclude 100% and 0% recovery rate accounts for the third component modeling for the group of 0% < recovery rate < 100%.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class LGDRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.models = dict()
        self.models['100_or_not'] = LogisticRegression(random_state=1234, max_iter=800, class_weight={0: .6, 1: .4})
        self.models['0_or_not'] = LogisticRegression(random_state=1234, max_iter=1000, class_weight={0: .9, 1: .1})
        self.models['gt_0_lt_100'] = MLPRegressor(random_state=1234, activation='identity', hidden_layer_sizes=(1,), alpha=1, max_iter=1000, learning_rate='adaptive', warm_start=True)

    def fit(self, X, y):
        self.models['100_or_not'].fit(X, (y == 1).astype(int))
        X, y = X[y != 1], y[y != 1]
        self.models['0_or_not'].fit(X, (y != 0).astype(int))
        X, y = X[y != 0], y[y != 0]
        self.models['gt_0_lt_100'].partial_fit(X, y)
        self.models["gt_0_lt_100"].out_activation_ = "logistic"
        self.models['gt_0_lt_100'].fit(X, y)
        return self

    def predict(self, X):
        y_pred_100 = self.models['100_or_not'].predict_proba(X)
        y_pred_0 = self.models['0_or_not'].predict_proba(X)
        y_pred_inbtw = self.models['gt_0_lt_100'].predict(X).clip(0, 1)
        return y_pred_100[:, 1] + y_pred_100[:, 0] * y_pred_0[:, 1] + y_pred_100[:, 0] * y_pred_0[:, 0] * y_pred_inbtw
```

We try to predict the recovery rate which is 1 - LGD.
```
Recovery Rate formula = (Default_Balance - min(Default_Balance, Charge_off) + Recovery_Amount) / Default_Balance.
```
The recovery rate is a value between 0 and 1 and therefore we make use of the MLPRegressor and apply Logistic Activation to ensure values are between 0 and 1


### Pipeline and Column Transformer

Combining preprocessing steps and the custom LGD model into a pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

# Define the column transformer for preprocessing
column_transformer = ColumnTransformer([
    ('Categorical_Encoder', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='infrequent_if_exist', min_frequency=.02, max_categories=8), ['SECSegmentDe_YrMo', 'SICIndustryMapDe']),
    ('Normalizer', PowerTransformer(), X.select_dtypes(np.number).columns.tolist())
], verbose_feature_names_out=False, verbose=True)

# Create the final pipeline
LGD_model = Pipeline([
    ('column_transformer', column_transformer),
    ('LGDRegressor', LGDRegressor())
])

# Fit the model
LGD_model.fit(X_train, y_train)

# Make predictions
preds = LGD_model.predict(X_test)
```

### Prediction and Scoring

Combining the models to generate final predictions for PD, EAD, LGD, and calculating the CECL.

```python
# Predicting values using the trained models
def get_all_predictions(snapshot_data, models):
    predictions = models['PA']['model'].predict_proba(snapshot_data[models['PA']['required_columns']])
    A_preds = pd.DataFrame(predictions, columns=['A-A', 'A-B', 'A-C', 'A-D', 'A-E'], index=snapshot_data.index)
    predictions = models['PB']['model'].predict_proba(snapshot_data[models['PB']['required_columns']])
    B_preds = pd.DataFrame(predictions, columns=['B-A', 'B-B', 'B-C', 'B-D', 'B-E'], index=snapshot_data.index)
    predictions = models['PC']['model'].predict_proba(snapshot_data[models['PC']['required_columns']])
    C_preds = pd.DataFrame(predictions, columns=['C-A', 'C-B', 'C-C', 'C-D', 'C-E'], index=snapshot_data.index)
    EAD_preds = pd.DataFrame(models['EAD']['model'].predict(snapshot_data[models['EAD']['required_columns']]), columns=['EAD_pred'], index=snapshot_data.index)
    LGD_preds = pd.DataFrame(models['LGD']['model'].predict(snapshot_data[models['LGD']['required_columns']]), columns=['recovery_rate_pred'], index=snapshot_data.index)
    
    return pd.concat([snapshot_data, A_preds, B_preds, C_preds, EAD_preds, LGD_preds], axis=1)

# Calculate the final ECL
forecast_df['ECL_pred'] = forecast_df['PD'] * forecast_df['EAD_pred'] * forecast_df['OpenAmt_YrMo'] * (1 - forecast_df['recovery_rate_pred'])
```

### Output and Evaluation

The final output is saved to a CSV file, and various metrics are calculated to evaluate the model's performance.

```python
# Print performance metrics
print("R2 Score:", r2_score(y_test, preds))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, preds)))
print("Mean Absolute Error:", mean_absolute_error(y_test, preds))
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(y_test, preds))
```

## Results

Key metrics such as R2 score, mean squared error, root mean squared error, mean absolute error, and mean absolute percentage error are used to evaluate the predictions.
The model has poor performance overall with MAE having values as low as 0.3 but the Percentage error (MAPE) is huge. The model was built on highly skewed data with very little data for default cases.

By following this approach, we ensure that our CECL predictions are based on robust statistical models and reflect the true risk associated with the loan portfolio. This helps financial institutions in managing risk more effectively and making informed decisions.