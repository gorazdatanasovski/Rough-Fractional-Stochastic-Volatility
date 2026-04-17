import polars as pl
import pandas as pd
from xbbg import blp

def extract_spx_intraday(start_date: str, end_date: str) -> pl.DataFrame:
    """
    Iterates over business days. Input dates exceeding Bloomberg's 140-day 
    intraday limit will organically return empty frames and be discarded.
    """
    business_days = pd.bdate_range(start=start_date, end=end_date)
    intraday_arrays = []

    for current_date in business_days:
        dt_str = current_date.strftime('%Y-%m-%d')
        
        # Exact syntax: bdib exclusively accepts a singular 'dt' parameter.
        df_pd = blp.bdib(ticker="SPX Index", dt=dt_str, interval=5, typ='TRADE')
        
        if not df_pd.empty:
            # Flatten multi-index and strictly isolate the price array
            df_pd.columns = df_pd.columns.droplevel(0)
            df_pd = df_pd[['close']].reset_index()
            
            # Cast to Polars to eliminate column mismatches and enforce computational speed
            df_pl = pl.from_pandas(df_pd).rename({"index": "timestamp", "close": "price"})
            intraday_arrays.append(df_pl)

    if not intraday_arrays:
        raise ValueError("Data stream void. Horizon exceeds the 140-day Bloomberg intraday limit.")

    # Concatenate the isolated daily arrays into the continuous time series matrix
    return pl.concat(intraday_arrays)