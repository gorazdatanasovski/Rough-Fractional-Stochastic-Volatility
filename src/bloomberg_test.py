import sys
import logging
from datetime import date, timedelta
import pathlib

try:
    from xbbg import blp
except ImportError as e:
    sys.exit(
        f"[FATAL] xbbg not found. Install via: pip install xbbg\n  → {e}"
    )

try:
    import polars as pl
except ImportError as e:
    sys.exit(
        f"[FATAL] Polars not found. Install via: pip install polars\n  → {e}"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION & PIPELINE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

TICKER: str = "SPX Index"          
BAR_SIZE: int = 5                  
LOOKBACK_DAYS: int = 140      
SESSION_START: str = "09:30:00"   
SESSION_END: str = "16:00:00"      

_END_DATE: date = date.today()
_START_DATE: date = _END_DATE - timedelta(days=LOOKBACK_DAYS)

START_DATE: str = _START_DATE.strftime("%Y-%m-%d")
END_DATE: str = _END_DATE.strftime("%Y-%m-%d")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — BLOOMBERG EXTRACTION ENGINE
#  Interface : xbbg.blp.bdib
#  Output    : Raw OHLCV 5-minute bar DataFrame (initially pandas, cast to Polars)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_intraday_bars(
    ticker: str = TICKER,
    bar_size: int = BAR_SIZE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pl.DataFrame:
    
    log.info(
        "Extracting Bloomberg BDIB | Ticker: %s | Bars: %dmin | Window: %s → %s",
        ticker, bar_size, start_date, end_date,
    )

    try:
        raw_pd = blp.bdib(
            ticker=ticker,
            start_datetime=start_date,
            end_datetime=end_date,
            interval=bar_size,
        )
    except Exception as exc:
        raise ConnectionError(
            f"Bloomberg BDIB extraction failed for '{ticker}'. "
            f"Ensure the Bloomberg Terminal is running and authenticated.\n"
            f"  Underlying error: {exc}"
        ) from exc

    if raw_pd is None or raw_pd.is_empty():
        raise ValueError(
            f"Bloomberg returned an empty dataset for ticker '{ticker}'. "
            f"Verify session hours and entitlements."
        )

    raw: pl.DataFrame = pl.from_arrow(raw_pd.to_native())
    log.info("Raw extraction complete | Shape: %s", raw.shape)
  
    raw = raw.rename({c: c.lower().strip() for c in raw.columns})

    required_cols = {"time", "close"}
   
    if "time" in raw.columns:
        raw = raw.rename({"time": "datetime"})

    missing = required_cols - {"time", "close"} - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns after extraction: {missing}")

    if raw["datetime"].dtype != pl.Datetime:
        raw = raw.with_columns(
            pl.col("datetime").cast(pl.Datetime("us"))
        )

    raw = (
        raw
        .sort("datetime", descending=False)
        .unique(subset=["datetime"], keep="first", maintain_order=True)
    )

    log.info("Extraction validated | Bars retained: %d", len(raw))
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — MATHEMATICAL TRANSFORMATION PIPELINE
#  All operations are strictly Polars-vectorized (no Python-level loops).
# ═══════════════════════════════════════════════════════════════════════════════

def compute_log_realized_variance(bars: pl.DataFrame) -> pl.DataFrame:

    log.info("Stage 1 | Computing intraday log returns  Y_{t_i}")

    bars_with_returns: pl.DataFrame = (
        bars
        .with_columns([
            pl.col("close")
              .log(base=2.718281828459045)   
              .diff()                       
              .alias("log_return"),       
            pl.col("datetime").dt.date().alias("trade_date"),
            pl.col("datetime").dt.time().alias("bar_time"),
        ])
        .with_columns(
            pl.when(
                pl.col("bar_time").cast(pl.Utf8).str.starts_with(
                    SESSION_START[:5]         
                )
            )
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col("log_return"))
            .alias("log_return")              
        )
        .drop_nulls(subset=["log_return"])
    )

    log.info(
        "Stage 1 complete | Valid intraday returns: %d",
        len(bars_with_returns),
    )

    log.info("Stage 2 | Aggregating realized variance  v_hat_t = Σ Y_{t_i}²")

    realized_variance: pl.DataFrame = (
        bars_with_returns
        .group_by("trade_date")
        .agg(
            (pl.col("log_return") ** 2)
              .sum()
              .alias("v_hat_t"),           
            pl.col("log_return")
              .count()
              .alias("n_bars"),
        )
        .sort("trade_date", descending=False)
    )

    log.info(
        "Stage 2 complete | Trading days aggregated: %d",
        len(realized_variance),
    )

    log.info("Stage 3 | Applying log compression  X_t = ln(v_hat_t)")

    log_realized_variance: pl.DataFrame = (
        realized_variance
        .with_columns(
            pl.col("v_hat_t")
              .log(base=2.718281828459045)    
              .alias("X_t")                   
        )
        .filter(pl.col("X_t").is_finite())
        .select([
            pl.col("trade_date").alias("date"),
            pl.col("n_bars"),
            pl.col("v_hat_t"),
            pl.col("X_t"),
        ])
    )

    log.info(
        "Stage 3 complete | Final X_t series length: %d days",
        len(log_realized_variance),
    )

    return log_realized_variance


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — OUTPUT & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def print_diagnostics(result: pl.DataFrame) -> None:
    x_t = result["X_t"]
    v_hat = result["v_hat_t"]

    separator = "─" * 70
    print(f"\n{separator}")
    print(f"  LOG-REALIZED VARIANCE  |  X_t = ln(v_hat_t)  |  SPX Index")
    print(separator)
    print(f"  Trading days (N)     : {len(result)}")
    print(f"  Date range           : {result['date'][0]}  →  {result['date'][-1]}")
    print(separator)
    print(f"  X_t  (log-RV)        mean = {x_t.mean():+.6f}")
    print(f"                        std  = {x_t.std():.6f}")
    print(f"                        min  = {x_t.min():+.6f}")
    print(f"                        max  = {x_t.max():+.6f}")
    print(separator)
    print(f"  v_hat_t (RV × 10⁶)   mean = {(v_hat.mean() * 1e6):.4f}")
    print(f"                        max  = {(v_hat.max() * 1e6):.4f}")
    print(separator)

    print("\n  LAST 10 OBSERVATIONS:\n")
    print(
        result
        .tail(10)
        .with_columns(
            pl.col("v_hat_t").map_elements(lambda x: f"{x:.8e}", return_dtype=pl.Utf8),
            pl.col("X_t").map_elements(lambda x: f"{x:+.6f}", return_dtype=pl.Utf8),
        )
        .__str__()
    )
    print(separator + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — MAIN EXECUTION ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline() -> pl.DataFrame:
    log.info("=" * 70)
    log.info("  SPX LOG-REALIZED VARIANCE PIPELINE  |  INITIALISING")
    log.info("=" * 70)

    raw_bars: pl.DataFrame = extract_intraday_bars()
    X_t_series: pl.DataFrame = compute_log_realized_variance(raw_bars)

    print_diagnostics(X_t_series)

    log.info("Pipeline execution complete. X_t series returned.")
    return X_t_series

if __name__ == "__main__":
    X_t = run_pipeline()

    project_root = pathlib.Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = data_dir / "spx_log_realized_variance.parquet"
    
    X_t.write_parquet(output_path, compression="zstd", compression_level=9)
    log.info("X_t series written to: %s", output_path)