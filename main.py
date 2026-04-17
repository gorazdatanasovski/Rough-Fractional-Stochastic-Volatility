import os
import polars as pl
from src.bloomberg_extraction import extract_spx_intraday
from src.return_transformation import compute_log_returns
from src.variance_estimator import compute_realized_variance

def execute_superior_pipeline():
    # 1. Extraction: Ingest 5-minute SPX grid
    raw_data = extract_spx_intraday(start_date="2010-01-01", end_date="2026-04-16")
    
    # 2. Transformation: Map to continuous natural log returns
    returns_data = compute_log_returns(raw_data)
    
    # 3. Estimation: Aggregate realized variance
    variance_data = compute_realized_variance(returns_data)
    
    # 4. Directory Verification
    if not os.path.exists("data"):
        os.makedirs("data")
        
    # 5. Serialization: Enforce date-hiding arguments to eliminate column mismatches
    # Polars write_excel does not write a row index by default, fulfilling the constraint.
    export_path = "data/SPX_Realized_Variance.xlsx"
    variance_data.write_excel(export_path)

if __name__ == "__main__":
    execute_superior_pipeline()