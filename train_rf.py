import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from diffprivlib.mechanisms import Laplace

# Function to add differential privacy noise
def add_dp_noise(data, epsilon, sensitivity):
    mech = Laplace(epsilon=epsilon, sensitivity=sensitivity)
    return [max(0, mech.randomise(float(val))) for val in data]

# Function to perform hyperparameter tuning with GridSearchCV
def tune_random_forest_hyperparameters(X, y):
    # Local hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_percentage_error')
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_estimator_

# Function to train Random Forest model with differential privacy and hyperparameter tuning
def train_random_forest_model(
    df,
    target_column='quantity_sold',
    exog_column='sell_price',
    test_size=0.2,
    epsilon=1.0,
    sensitivity=1.0
):
    # Convert date columns to datetime if needed
    if 'date' in df.columns and not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])
        
    # Define exogenous variables and target
    exog = df[exog_column].values.reshape(-1, 1)
    y = df[target_column].values

    # Split the data into train and test sets
    split_idx = int(len(df) * (1 - test_size))
    train_y, test_y = y[:split_idx], y[split_idx:]
    train_exog, test_exog = exog[:split_idx], exog[split_idx:]

    # Perform hyperparameter tuning
    best_params, rf_model = tune_random_forest_hyperparameters(train_exog, train_y)
    print(f"Best hyperparameters: {best_params}")

    # Forecast using the trained model
    forecast = rf_model.predict(test_exog)

    # Apply differential privacy to the forecast
    dp_forecast = add_dp_noise(forecast, epsilon, sensitivity)

    # Calculate the Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(test_y, dp_forecast)

    print(f"✅ Random Forest MAPE (DP): {mape:.4f}")
    
    return rf_model, mape, dp_forecast, test_y

# Function to forecast using the Random Forest model and add differential privacy noise
def forecast_random_forest(model, future_exog, epsilon=1.0, sensitivity=1.0):
    # Handle scalar int/float input for exogenous variables
    if isinstance(future_exog, (int, float)):
        future_exog = np.array([[future_exog]])
    else:
        future_exog = np.array(future_exog)
        if future_exog.ndim == 1:
            future_exog = future_exog.reshape(-1, 1)

    # Forecast using the Random Forest model
    forecast = model.predict(future_exog)

    # Apply differential privacy noise
    dp_forecast = add_dp_noise(forecast, epsilon, sensitivity)

    # Ensure non-negative, rounded integers
    return np.array(dp_forecast).clip(min=0).round().astype(int)