import numpy as np

print(np.log(15))

import polars as pl
import numpy as np
import statsmodels.api as sm
import os

def execute_isolated_econometric_diagnostic():
    file_path = "data/SPX_Realized_Variance_Oxford.xlsx"
    if not os.path.exists(file_path):
        print("Error: Target matrix not found. Run your extraction script first.")
        return

    df = pl.read_excel(file_path)
    log_variance_array = df.get_column("Log_Variance").to_numpy()

    lags = np.arange(1, 22)
    q = 1.0
    empirical_moments = []

    for delta in lags:
        increments = np.abs(log_variance_array[delta:] - log_variance_array[:-delta])
        valid_increments = increments[~np.isnan(increments)]
        m_q = np.mean(valid_increments ** q)
        empirical_moments.append(m_q)

    X = np.log(lags)
    Y = np.log(empirical_moments)

    X_with_intercept = sm.add_constant(X)
    
    model = sm.OLS(Y, X_with_intercept)
    results = model.fit()

    print("\n==============================================================================")
    print("                      ISOLATED ECONOMETRIC DIAGNOSTIC: q=1.0                  ")
    print("==============================================================================")
    print(results.summary())
    
    h_extracted = results.params[1] / q
    print("\n------------------------------------------------------------------------------")
    print(f"Mathematical Extraction Complete.")
    print(f"Extracted Hurst Parameter (H) from OLS Slope: {h_extracted:.4f}")
    print("------------------------------------------------------------------------------\n")

if __name__ == "__main__":
    execute_isolated_econometric_diagnostic()