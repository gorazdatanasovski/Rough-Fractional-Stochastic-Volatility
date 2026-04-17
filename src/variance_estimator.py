import polars as pl

def compute_realized_variance(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            trading_date = pl.col("timestamp").dt.date(),
            squared_return = pl.col("log_return").pow(2)
        )
        .group_by("trading_date")
        .agg(
            realized_variance = pl.col("squared_return").sum()
        )
        .with_columns(
            log_volatility = pl.col("realized_variance").log()
        )
        .sort("trading_date")
    )