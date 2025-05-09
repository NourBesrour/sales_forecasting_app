import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def show_data_preview():
    st.subheader("Data Preview")

    if st.session_state.get('uploaded_df') is not None:
        df = st.session_state['uploaded_df']

        # Convert 'date' to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # Drop rows where date could not be parsed

        # Let user select a Product ID
        product_ids = df['SKU'].unique()
        selected_id = st.selectbox("Select a Product ID", product_ids)

        # Let user select the number of recent days
        num_days = st.number_input("Show data for the past N days", min_value=1, value=10, step=1)

        if selected_id:
            # Filter for selected product ID
            df_id = df[df['SKU'] == selected_id].copy()
            df_id.sort_values('date', inplace=True)

            # Determine latest date and date window
            latest_date = df_id['date'].max()
            start_date = latest_date - pd.Timedelta(days=num_days - 1)  # To include today as day 1

            df_filtered = df_id[df_id['date'].between(start_date, latest_date)]

            st.subheader(f"Sales & Price History for Product ID: {selected_id}")
            st.caption(f"Showing data from {start_date.date()} to {latest_date.date()}")

            if df_filtered.empty:
                st.warning("No sales data found for this ID in the selected time range.")
            else:
                st.dataframe(df_filtered[['date', 'quantity_sold', 'sell_price']])

                # Create dual-axis chart
                fig_combined = go.Figure()

                # Quantity Sold trace (left y-axis)
                fig_combined.add_trace(go.Scatter(
                    x=df_filtered['date'],
                    y=df_filtered['quantity_sold'],
                    mode='lines+markers',
                    name='Quantity Sold',
                    yaxis='y1',
                    line=dict(color='royalblue')
                ))

                # Sell Price trace (right y-axis)
                fig_combined.add_trace(go.Scatter(
                    x=df_filtered['date'],
                    y=df_filtered['sell_price'],
                    mode='lines+markers',
                    name='Sell Price',
                    yaxis='y2',
                    line=dict(color='darkorange')
                ))

                # Layout
                fig_combined.update_layout(
                    title='Quantity Sold and Sell Price Over Time',
                    xaxis=dict(title='Date'),
                    yaxis=dict(
                        title='Quantity Sold',
                        titlefont=dict(color='royalblue', size=14),
                        tickfont=dict(color='royalblue')
                    ),
                    yaxis2=dict(
                        title='Sell Price',
                        titlefont=dict(color='darkorange', size=14),
                        tickfont=dict(color='darkorange'),
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(x=0.01, y=1.1, orientation="h")
                )

                st.plotly_chart(fig_combined, use_container_width=True)

    else:
        st.info("Please upload your data on the 'Upload Data' page.")
