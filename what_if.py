import streamlit as st
import pandas as pd
import mlflow
import plotly.graph_objects as go
from utils import create_future_dates
from train_sarimax import train_sarimax_model, forecast_sarimax
from train_lstm import train_lstm_model, forecast_lstm
from train_tcn import train_tcn_model, forecast_tcn_he
from train_rf import train_random_forest_model, forecast_random_forest  # Add the Random Forest import


mlflow.set_tracking_uri("file:./mlruns")

def what_if_analysis():
    st.subheader("What-If Analysis: Impact of Discount")
    st.write("Explore the effect of a discount on predicted sales.")

    if st.session_state.get('uploaded_df') is not None:
        df = st.session_state['uploaded_df']
        available_ids = df['SKU'].unique()
        selected_id = st.selectbox("Select the ID to analyze", available_ids)

        df_id = df[df['SKU'] == selected_id].copy()
        if 'date' not in df_id.columns:
            st.error("The 'date' column is missing in the DataFrame.")
            return
        try:
            df_id['date'] = pd.to_datetime(df_id['date'])
            df_id.sort_values('date', inplace=True)
            df_id.set_index('date', inplace=True)
        except Exception as e:
            st.error(f"Error processing the 'date' column: {e}")
            return

        if 'quantity_sold' not in df_id.columns or 'sell_price' not in df_id.columns:
            st.error("The DataFrame must contain 'quantity_sold' and 'sell_price' columns.")
            return

        st.write("Preview of data for selected ID:", df_id.tail())

        forecast_range = st.slider("Forecast Range (Days)", min_value=1, max_value=28, value=7)
        if 'discount_percentage' not in st.session_state:
            st.session_state.discount_percentage = 0

        discount_percentage = st.slider(
            "Discount Percentage on Sell Price (%)",
            0, 90, st.session_state.discount_percentage,
            key='discount_percentage'
        )

        if st.button("Run What-If Analysis"):
            with st.spinner(f"Running analysis for ID: {selected_id}..."):
                models = {}
                mapes = {}
                scalers = {}
                forecast_df_no_discount = None
                forecast_df_with_discount = None
                best_model_name = None
                best_model = None

                # Prepare training data once
                if 'sell_price_with_discount' not in df_id.columns:
                    df_id['sell_price_with_discount'] = df_id['sell_price'] * (1 - discount_percentage / 100)

                df_train = df_id.reset_index().copy()

                def train_and_evaluate(df_train):
                    models = {}
                    mapes = {}
                    scalers = {}

                    # Train Random Forest model
                    try:
                        rf_model, rf_mape, _, _ = train_random_forest_model(df_train)  # Use Random Forest model
                        models['Random Forest'] = rf_model
                        mapes['Random Forest'] = rf_mape
                    except Exception as e:
                        st.error(f"Error training Random Forest: {e}")

                    # Train SARIMAX model
                    try:
                        sarimax_model, sarimax_mape, _, _ = train_sarimax_model(df_train[['quantity_sold', 'sell_price']])
                        models['SARIMAX'] = sarimax_model
                        mapes['SARIMAX'] = sarimax_mape
                    except Exception as e:
                        st.error(f"Error training SARIMAX: {e}")

                    # Train LSTM model
                    try:
                        dp_lstm_model, dp_lstm_mape, _, _, scaler_y_lstm, scaler_exog_lstm = train_lstm_model(
                            df_train)
                        models['LSTM'] = dp_lstm_model
                        mapes['LSTM'] = dp_lstm_mape
                        scalers['LSTM'] = {'y': scaler_y_lstm, 'exog': scaler_exog_lstm}
                    except Exception as e:
                        st.error(f"Error training LSTM: {e}")

                    # Train TCN model
                    try:
                        dp_tcn_model, dp_tcn_mape, _, _, scaler_y_tcn, scaler_exog_tcn = train_tcn_model(
                            df_train)
                        models['TCN'] = dp_tcn_model
                        mapes['TCN'] = dp_tcn_mape
                        scalers['TCN'] = {'y': scaler_y_tcn, 'exog': scaler_exog_tcn}
                    except Exception as e:
                        st.error(f"Error training TCN: {e}")

                    if mapes:
                        best_model_name = min(mapes, key=mapes.get)
                        best_model = models[best_model_name]
                        best_scalers = scalers.get(best_model_name)
                        return best_model, best_model_name, best_scalers
                    return None, None, None

                best_model, best_model_name, best_scalers = train_and_evaluate(df_train)
                if best_model:
                    # --- Forecast WITHOUT Discount --- #
                    y_pred_no_discount = None  # ✅ Initialize to avoid error
                    exog_future_no_discount = [df_id['sell_price'].iloc[-1]] * forecast_range

                    if best_model_name == "Random Forest":
                        try:
                            y_pred_no_discount = forecast_random_forest(best_model, exog_future_no_discount)  # Use Random Forest for forecast
                        except Exception as e:
                            st.error(f"Error during Random Forest forecast (no discount): {e}")
                    elif best_model_name == "SARIMAX":
                        try:
                            # forecast_sarimax(best_model, df_train, forecast_range)
                            y_pred_no_discount = forecast_sarimax(best_model, df_train, forecast_range)  # Pass the model and training data
                        except Exception as e:
                            st.error(f"Error during SARIMAX forecast (no discount): {e}")
                    elif best_model_name == "LSTM":
                        try:
                            y_pred_no_discount = forecast_lstm(
                                best_model, df_train, n_days=forecast_range,
                                scaler_y=best_scalers['y'], scaler_exog=best_scalers['exog'],
                                target_column='quantity_sold', exog_column='sell_price'
                            )
                        except Exception as e:
                            st.error(f"Error during LSTM forecast (no discount): {e}")
                    elif best_model_name == "TCN":
                        try:
                            y_pred_no_discount = forecast_tcn_he(
                                best_model, df_train, n_days=forecast_range,
                                scaler_y=best_scalers['y'], scaler_exog=best_scalers['exog'],
                                target_column='quantity_sold', exog_column='sell_price'
                            )
                        except Exception as e:
                            st.error(f"Error during TCN forecast (no discount): {e}")

                    # --- Forecast WITH Discount --- #
                    y_pred_with_discount = None
                    exog_future_with_discount = [df_id['sell_price_with_discount'].iloc[-1]] * forecast_range

                    if best_model_name == "Random Forest":
                        try:
                            y_pred_with_discount = forecast_random_forest(best_model, exog_future_with_discount)  # Use Random Forest for forecast
                        except Exception as e:
                            st.error(f"Error during Random Forest forecast (with discount): {e}")
                    elif best_model_name == "SARIMAX":
                        try:
                            # Forecast with SARIMAX using the appropriate method
                            y_pred_with_discount = forecast_sarimax(best_model, df_train, forecast_range)  # Pass the model and training data
                        except Exception as e:
                            st.error(f"Error during SARIMAX forecast (with discount): {e}")
                    elif best_model_name == "LSTM":
                        try:
                            y_pred_with_discount = forecast_lstm(
                                best_model, df_train, n_days=forecast_range,
                                scaler_y=best_scalers['y'], scaler_exog=best_scalers['exog'],
                                target_column='quantity_sold', exog_column='sell_price_with_discount'
                            )
                        except Exception as e:
                            st.error(f"Error during LSTM forecast (with discount): {e}")
                    elif best_model_name == "TCN":
                        try:
                            y_pred_with_discount = forecast_tcn_he(
                                best_model, df_train, n_days=forecast_range,
                                scaler_y=best_scalers['y'], scaler_exog=best_scalers['exog'],
                                target_column='quantity_sold', exog_column='sell_price_with_discount'
                            )
                        except Exception as e:
                            st.error(f"Error during TCN forecast (with discount): {e}")

                    # --- Display Comparison Chart --- #
                    if y_pred_no_discount is not None and y_pred_with_discount is not None:
                        forecast_dates = create_future_dates(df_id.index[-1], forecast_range)
                        forecast_df_no_discount = pd.DataFrame({
                            'Date': forecast_dates,
                            'No Discount': y_pred_no_discount
                        }).set_index('Date')

                        forecast_df_with_discount = pd.DataFrame({
                            'Date': forecast_dates,
                            f'{discount_percentage}% Discount': y_pred_with_discount
                        }).set_index('Date')

                        comparison_df = forecast_df_no_discount.join(forecast_df_with_discount, how='outer')

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=comparison_df.index,
                            y=comparison_df['No Discount'],
                            mode='lines+markers',
                            name='No Discount',
                            line=dict(color='blue', width=2),
                            marker=dict(size=6)
                        ))
                        fig.add_trace(go.Scatter(
                            x=comparison_df.index,
                            y=comparison_df[f'{discount_percentage}% Discount'],
                            mode='lines+markers',
                            name=f'{discount_percentage}% Discount',
                            line=dict(color='red', width=2),
                            marker=dict(size=6)
                        ))
                        fig.update_layout(
                            title=f"Comparison of Forecasted Sales for SKU: {selected_id} with and without Discount",
                            xaxis_title='Date',
                            yaxis_title='Forecasted Quantity Sold'
                        )
                        st.plotly_chart(fig)
                else:
                    st.warning("Could not find the best model for forecasting.")
