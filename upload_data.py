import pandas as pd
import zipfile
import io
import streamlit as st

def load_and_process_data():
    uploaded_file = st.file_uploader("Upload a ZIP file containing a single CSV", type=["zip"])
    if uploaded_file is not None:
        try:
            with zipfile.ZipFile(uploaded_file) as zf:
                csv_name = [name for name in zf.namelist() if name.endswith(".csv")]
                if not csv_name:
                    st.error("No CSV file found in the ZIP archive.")
                    return None
                if len(csv_name) > 1:
                    st.error("More than one CSV file found in the ZIP archive. Please ensure only one CSV is present.")
                    return None
                with zf.open(csv_name[0]) as f:
                    df = pd.read_csv(io.TextIOWrapper(f, encoding='utf-8'))
                    if 'date' not in df.columns:
                        st.error("The CSV file must contain a 'date' column.")
                        return None
                    if 'quantity_sold' not in df.columns:
                        st.error("The CSV file must contain a 'quantity_sold' column.")
                        return None
                    df['date'] = pd.to_datetime(df['date'])
                    return df
        except zipfile.BadZipFile:
            st.error("Invalid ZIP file.")
            return None
    return None

def filter_data_by_id(df, selected_id):
    filtered_df = df[df['SKU'] == selected_id].copy()
    if not filtered_df.empty:
        if 'date' in filtered_df.columns and 'quantity_sold' in filtered_df.columns:
            df_grouped = filtered_df.groupby('date')['quantity_sold'].sum().reset_index()
            df_grouped.set_index('date', inplace=True)
            return df_grouped
        else:
            st.error("The DataFrame must have 'date' and 'quantity_sold' columns for filtering and grouping.")
            return None
    else:
        return None