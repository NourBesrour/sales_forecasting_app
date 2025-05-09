import os
import pandas as pd
import streamlit as st
import mlflow
import plotly.graph_objects as go
from utils import create_future_dates
from train_sarimax import train_sarimax_model, forecast_sarimax
from train_lstm import train_lstm_model, forecast_lstm
from train_tcn import train_tcn_model, forecast_tcn_he
from train_rf import train_random_forest_model, forecast_random_forest  # Import your custom RF methods

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

def prediction():
    st.subheader("Forecasting Sales using Model Stacking")
    print(f"Current working directory (prediction.py): {os.getcwd()}")

    if st.session_state.get('uploaded_df') is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state['uploaded_df']
    available_ids = df['SKU'].unique()
    selected_id = st.selectbox("Select the SKU to forecast", available_ids)
    st.session_state['selected_id_prediction'] = selected_id

    # Filter data for selected SKU
    df_id = df[df['SKU'] == selected_id].copy()
    if 'date' not in df_id.columns:
        st.error("The 'date' column is missing in the dataset.")
        return

    try:
        df_id['date'] = pd.to_datetime(df_id['date'])
        df_id.sort_values('date', inplace=True)
        df_id.set_index('date', inplace=True)
    except Exception as e:
        st.error(f"Error processing 'date' column: {e}")
        return

    if 'quantity_sold' not in df_id.columns or 'sell_price' not in df_id.columns:
        st.error("The dataset must contain 'quantity_sold' and 'sell_price' columns.")
        return

    st.write("Preview of filtered data:", df_id.tail())
    forecast_range = st.slider("Forecast Range (Days)", min_value=1, max_value=28, value=7)

    if st.button("Predict"):
        with st.spinner(f"Training and evaluating models for SKU: {selected_id}..."):
            models, mapes, scalers = {}, {}, {}
            df_train = df_id.reset_index().copy()

            # Train SARIMAX
            try:
                with mlflow.start_run(run_name=f"SARIMAX_{selected_id}") :
                    sarimax_model, sarimax_mape, *_ = train_sarimax_model(df_train[['quantity_sold', 'sell_price']])
                    models['SARIMAX'] = sarimax_model
                    mapes['SARIMAX'] = sarimax_mape
                    mlflow.log_metric("mape", sarimax_mape)
                    st.info(f"SARIMAX trained. MAPE: {sarimax_mape:.4f}")
            except Exception as e:
                st.error(f"Error training SARIMAX: {e}")

            # Train LSTM
            try:
                with mlflow.start_run(run_name=f"LSTM_{selected_id}") :
                    lstm_model, lstm_mape, _, _, scaler_y, scaler_exog = train_lstm_model(df_train)
                    models['LSTM'] = lstm_model
                    mapes['LSTM'] = lstm_mape
                    scalers['LSTM'] = {'y': scaler_y, 'exog': scaler_exog}
                    mlflow.log_metric("mape", lstm_mape)
                    
                    st.info(f"LSTM trained. MAPE: {lstm_mape:.4f}")
            except Exception as e:
                st.error(f"Error training LSTM: {e}")

            # Train TCN
            try:
                with mlflow.start_run(run_name=f"TCN_{selected_id}") :
                    tcn_model, tcn_mape, _, _, scaler_y, scaler_exog = train_tcn_model(df_train)
                    models['TCN'] = tcn_model
                    mapes['TCN'] = tcn_mape
                    scalers['TCN'] = {'y': scaler_y, 'exog': scaler_exog}
                    mlflow.log_metric("mape", tcn_mape)
                    st.info(f"TCN trained. MAPE: {tcn_mape:.4f}")
            except Exception as e:
                st.error(f"Error training TCN: {e}")

            # Train Random Forest with Differential Privacy
            try:
                with mlflow.start_run(run_name=f"RandomForest_{selected_id}") :
                    rf_model, rf_mape, dp_forecast, test_y = train_random_forest_model(df_train)
                    models['RandomForest'] = rf_model
                    mapes['RandomForest'] = rf_mape
                    mlflow.log_metric("mape", rf_mape)
                    st.info(f"Random Forest trained. MAPE: {rf_mape:.4f}")
            except Exception as e:
                st.error(f"Error training Random Forest: {e}")

            if not mapes:
                st.error("No models were trained successfully.")
                return

            best_model_name = min(mapes, key=mapes.get)
            best_model = models[best_model_name]
            st.subheader(f"Best Model: {best_model_name} (MAPE: {mapes[best_model_name]:.4f})")
            st.write(f"Generating {forecast_range} days forecast using {best_model_name}...")

            try:
                if best_model_name == "SARIMAX":
                     forecast_values = forecast_sarimax(best_model, df_train, forecast_range)
                elif best_model_name == "LSTM":
                    forecast_values = forecast_lstm(
                        best_model,
                        df_train,
                        forecast_range,
                    )
                elif best_model_name == "TCN":
                    forecast_values = forecast_tcn_he(
                        best_model,
                        df_train,
                        forecast_range,
                    )
                elif best_model_name == "RandomForest":
                    forecast_values = forecast_random_forest(
                        best_model,
                        df_train[['sell_price']].tail(forecast_range),  # Assuming 'sell_price' as future input
                        epsilon=1.0,  # Specify privacy parameters as needed
                        sensitivity=1.0
                    )
                else:
                    st.error("Selected model is not supported for forecasting.")
                    return

                future_dates = create_future_dates(df_id.index[-1], forecast_range)
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'forecasted_quantity_sold': forecast_values
                })
                st.write("Forecast:")
                # Plot forecast only
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['forecasted_quantity_sold'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))
                fig.update_layout(
                    title=f"Forecasted Sales for SKU: {selected_id}",
                    xaxis_title='Date',
                    yaxis_title='Forecasted Quantity Sold'
                )
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error during forecasting: {e}")
