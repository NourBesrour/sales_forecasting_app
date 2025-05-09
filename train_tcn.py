import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from diffprivlib.mechanisms import Laplace  # Differential Privacy Library

# Data preparation for TCN
def prepare_tcn_data(df, target_column='quantity_sold', exog_column='sell_price', test_size=0.2, timesteps=1):
    y = df[target_column].values
    exog = df[exog_column].values

    scaler_y = MinMaxScaler()
    scaler_exog = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    exog_scaled = scaler_exog.fit_transform(exog.reshape(-1, 1))

    X, y_tcn = [], []
    for i in range(timesteps, len(df)):
        X.append(np.hstack([y_scaled[i - timesteps:i], exog_scaled[i - timesteps:i]]))
        y_tcn.append(y_scaled[i])
    X = np.array(X)
    y_tcn = np.array(y_tcn)

    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y_tcn[:split_idx], y_tcn[split_idx:], scaler_y, scaler_exog

# Build TCN model
def build_tcn_model(input_shape, n_filters=64, kernel_size=3, dropout_rate=0.2):
    model = Sequential()
    model.add(Conv1D(n_filters, kernel_size, activation='relu', padding='causal', input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(n_filters, kernel_size, activation='relu', padding='causal'))
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(n_filters, kernel_size, activation='relu', padding='causal'))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# Train TCN model with DP-enhanced predictions
def train_tcn_model(df, target_column='quantity_sold', exog_column='sell_price',
                     test_size=0.2, timesteps=1, epochs=20, batch_size=32, n_filters=64, kernel_size=3, dropout_rate=0.2, epsilon=1.0):
    X_train, X_test, y_train, y_test, scaler_y, scaler_exog = prepare_tcn_data(
        df, target_column, exog_column, test_size, timesteps)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)  # 2 because of target and exog
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 2)

    model = build_tcn_model(X_train.shape[1:], n_filters, kernel_size, dropout_rate)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)

    # Apply Differential Privacy noise using Laplace mechanism
    laplace = Laplace(epsilon=epsilon, sensitivity=1.0)
    dp_predictions = [max(0, laplace.randomise(val)) for val in y_pred.flatten()]

    mape = mean_absolute_percentage_error(y_true, y_pred)
    return model, mape, dp_predictions, y_true.flatten(), scaler_y, scaler_exog

# Forecast using TCN model with Differential Privacy
def forecast_tcn_he(model, df, n_days, timesteps=1, scaler_y=None, scaler_exog=None,
                    target_column='quantity_sold', exog_column='sell_price', epsilon=1.0):
    df_copy = df.copy()

    if scaler_y is None or scaler_exog is None:
        scaler_y = MinMaxScaler()
        scaler_exog = MinMaxScaler()
        scaler_y.fit(df_copy[target_column].values.reshape(-1, 1))
        scaler_exog.fit(df_copy[exog_column].values.reshape(-1, 1))

    last_y = scaler_y.transform(df_copy[target_column].values[-timesteps:].reshape(-1, 1))
    last_exog = scaler_exog.transform(df_copy[exog_column].values[-timesteps:].reshape(-1, 1))

    forecast_scaled = []

    for _ in range(n_days):
        x_input = np.hstack([last_y, last_exog]).reshape(1, timesteps, 2)
        y_pred_scaled = model.predict(x_input, verbose=0)
        forecast_scaled.append(y_pred_scaled[0, 0])

        # Update sequences
        last_y = np.vstack([last_y[1:], y_pred_scaled])
        last_exog = np.vstack([last_exog[1:], last_exog[-1:]])  # repeat last exog

    # Apply Differential Privacy noise using Laplace mechanism
    laplace = Laplace(epsilon=epsilon, sensitivity=1.0)
    dp_forecast = [laplace.randomise(val) for val in forecast_scaled]

    forecast = scaler_y.inverse_transform(np.array(dp_forecast).reshape(-1, 1)).flatten()
    return forecast