# utils.py
import pandas as pd

def create_future_dates(last_date, n_days):
    return [last_date + pd.Timedelta(days=i) for i in range(1, n_days + 1)]