import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from diffprivlib.mechanisms import Laplace

# Function to add differential privacy noise
def add_dp_noise(data, epsilon, sensitivity):
    mech = Laplace(epsilon=epsilon, sensitivity=sensitivity)
    return [max(0, mech.randomise(float(val))) for val in data]

# Function to perform hyperparameter tuning with GridSearchCV
def tune_sarimax_hyperparameters(X, y):
    # Local hyperparameter grid for SARIMAX
    param_grid = {
        'order': [(1, 0, 0), (1, 1, 0), (2, 1, 1)],  # (p, d, q)
        'seasonal_order': [(1, 0, 0, 12), (1, 1, 1, 12), (0, 1, 1, 12)],  # (P, D, Q, s)
    }

    best_params = None
    best_model = None
    best_score = float('inf')

    for order in param_grid['order']:
        for seasonal_order in param_grid['seasonal_order']:
            try:
                # Fit SARIMAX with current hyperparameters
                model = SARIMAX(y, exog=X, order=order, seasonal_order=seasonal_order)
                model_fit = model.fit(disp=False)

                # Forecast for validation
                forecast = model_fit.forecast(steps=len(y), exog=X)

                # Calculate MAPE as the scoring metric
                mape = mean_absolute_percentage_error(y, forecast)
                if mape < best_score:
                    best_score = mape
                    best_params = {'order': order, 'seasonal_order': seasonal_order}
                    best_model = model_fit
            except Exception as e:
                print(f"Error during fitting SARIMAX with {order}, {seasonal_order}: {e}")

    return best_params, best_model

# Function to train SARIMAX model with differential privacy and hyperparameter tuning
def train_sarimax_model(
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
    best_params, sarimax_model_fit = tune_sarimax_hyperparameters(train_exog, train_y)
    print(f"Best hyperparameters: {best_params}")

    # Forecast using the trained model
    forecast = sarimax_model_fit.forecast(steps=len(test_y), exog=test_exog)

    # Apply differential privacy to the forecast
    dp_forecast = add_dp_noise(forecast, epsilon, sensitivity)

    # Calculate the Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(test_y, dp_forecast)

    print(f"✅ SARIMAX MAPE (DP): {mape:.4f}")
    
    return sarimax_model_fit, mape, dp_forecast, test_y
# Function to forecast using the SARIMAX model and add differential privacy noise
def forecast_sarimax(model, df, n_days, exog_column='sell_price', epsilon=1.0, sensitivity=1.0):
    """
    Forecasts the next `n_days` using a trained SARIMAX model.

    Parameters:
    - model: The trained SARIMAX model
    - df: The dataframe containing the historical data (including the exogenous variable)
    - n_days: Number of days into the future to forecast
    - exog_column: The column in `df` containing the exogenous variable (e.g., 'sell_price')
    - epsilon: The epsilon value for differential privacy
    - sensitivity: The sensitivity value for differential privacy

    Returns:
    - dp_forecast: Forecasted values with differential privacy noise added
    """
    
    # Ensure `df` has the necessary columns
    if exog_column not in df.columns:
        raise ValueError(f"'{exog_column}' column not found in the dataframe")
    
    # Extract the exogenous data for the forecast period (e.g., future sell prices)
    future_exog = df[exog_column].tail(n_days).values.reshape(-1, 1)
    
    # Forecast using the SARIMAX model
    forecast = model.forecast(steps=n_days, exog=future_exog)

    # Apply differential privacy noise
    dp_forecast = add_dp_noise(forecast, epsilon, sensitivity)

    # Ensure non-negative, rounded integers
    dp_forecast = np.array(dp_forecast).clip(min=0).round().astype(int)

    return dp_forecast