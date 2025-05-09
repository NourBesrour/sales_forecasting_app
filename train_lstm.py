import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from diffprivlib.mechanisms import Laplace  # Import Laplace mechanism from diffprivlib

# Data preparation for LSTM
def prepare_lstm_data(df, target_column='quantity_sold', exog_column='sell_price', test_size=0.2, timesteps=1):
    y = df[target_column].values
    exog = df[exog_column].values

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_exog = MinMaxScaler(feature_range=(0, 1))

    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    exog_scaled = scaler_exog.fit_transform(exog.reshape(-1, 1))

    X, y_lstm = [], []
    for i in range(timesteps, len(df)):
        X.append(np.hstack([y_scaled[i - timesteps:i], exog_scaled[i - timesteps:i]]))
        y_lstm.append(y_scaled[i])
    X = np.array(X)
    y_lstm = np.array(y_lstm)

    train_size = int(len(df) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

    return X_train, X_test, y_train, y_test, scaler_y, scaler_exog

# Build LSTM model
def build_lstm_model(input_shape, units=50):
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# Train the model with DP-enhanced predictions
def train_lstm_model(df, target_column='quantity_sold', exog_column='sell_price', test_size=0.2, timesteps=1, epochs=20, batch_size=32, lstm_units=50, epsilon=1.0):
    X_train, X_test, y_train, y_test, scaler_y, scaler_exog = prepare_lstm_data(
        df, target_column, exog_column, test_size, timesteps
    )

    model = build_lstm_model(X_train.shape[1:], units=lstm_units)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    predictions_scaled = model.predict(X_test)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)
    sensitivity = 1.0

    # Use Laplace mechanism for differential privacy
    laplace = Laplace(epsilon=epsilon, sensitivity=sensitivity)
    dp_predictions = [max(0, laplace.randomise(val)) for val in predictions.flatten()]

    mape = mean_absolute_percentage_error(y_test_original, predictions)

    return model, mape, dp_predictions, y_test_original, scaler_y, scaler_exog

def forecast_lstm(model, df, n_days, timesteps=1, scaler_y=None, scaler_exog=None,
                  target_column='quantity_sold', exog_column='sell_price', epsilon=1.0):
    """
    Forecast future values using an LSTM model with differential privacy.

    Parameters:
    - model: Trained Keras LSTM model.
    - df: DataFrame containing historical data.
    - n_days: Number of future days to predict.
    - timesteps: Number of past timesteps used for prediction.
    - scaler_y: Optional pre-fitted scaler for the target variable.
    - scaler_exog: Optional pre-fitted scaler for the exogenous variable.
    - target_column: Column name for the target variable.
    - exog_column: Column name for the exogenous feature.
    - epsilon: Privacy budget for differential privacy.

    Returns:
    - np.ndarray of shape (n_days,) with integer, positive forecasts.
    """

    # Work on a copy to preserve the original DataFrame
    df_copy = df.copy()

    # Fit scalers if not provided
    if scaler_y is None or scaler_exog is None:
        scaler_y = MinMaxScaler()
        scaler_exog = MinMaxScaler()
        scaler_y.fit(df_copy[target_column].values.reshape(-1, 1))
        scaler_exog.fit(df_copy[exog_column].values.reshape(-1, 1))

    # Get the last timesteps of target and exogenous data
    last_y = scaler_y.transform(df_copy[target_column].values[-timesteps:].reshape(-1, 1))
    last_exog = scaler_exog.transform(df_copy[exog_column].values[-timesteps:].reshape(-1, 1))

    forecast_scaled = []

    # Predict n_days ahead
    for _ in range(n_days):
        # Combine target and exogenous for input
        x_input = np.hstack([last_y, last_exog]).reshape(1, timesteps, 2)

        # Predict next value (scaled)
        y_pred_scaled = model.predict(x_input, verbose=0)
        forecast_scaled.append(y_pred_scaled[0, 0])

        # Update sequences
        last_y = np.vstack([last_y[1:], y_pred_scaled])
        last_exog = np.vstack([last_exog[1:], last_exog[-1:]])  # repeat last exog

    # Add differential privacy noise
    laplace = Laplace(epsilon=epsilon, sensitivity=1.0)
    dp_forecast = [laplace.randomise(val) for val in forecast_scaled]

    # Inverse transform and clean
    forecast = scaler_y.inverse_transform(np.array(dp_forecast).reshape(-1, 1)).flatten()

    # Ensure non-negative, rounded integers
    return np.maximum(0, forecast).round().astype(int)