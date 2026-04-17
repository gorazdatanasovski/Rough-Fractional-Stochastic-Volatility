import polars as pl
import numpy as np

def compute_log_returns(df: pl.DataFrame) -> pl.DataFrame:
    # Compute differential: ln(S_t) - ln(S_{t-1})
    return df.with_columns(
        log_return = (pl.col("price") / pl.col("price").shift(1)).log()
    ).drop_nulls()
    