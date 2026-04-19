import polars as pl
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

def execute_complete_har_matrix():
    file_path = "data/SPX_Realized_Variance_Oxford.xlsx"
    if not os.path.exists(file_path):
        raise FileNotFoundError("Empirical dataset void. Execute oxford extraction first.")
        
    df = pl.read_excel(file_path)
    X = df.get_column("Log_Variance").to_numpy()
    N = len(X)
    
    rolling_window = 500
    horizons = [1, 5, 20]
    global_mean = np.mean(X)
    results_archive = {}

    print("\n" + "="*70)
    print("PHASE 1: STATIC ECONOMETRICS (IN-SAMPLE HAR(3) FOR HORIZON 1)")
    print("="*70)
    
    delta_static = 1
    max_lag = 19  # A 20-day moving average requires t and the trailing 19 days
    
    # Mathematical Alignment: Targets (t+delta) mapped against features at state (t)
    valid_t = np.arange(max_lag, N - delta_static)
    Y_static = X[valid_t + delta_static]
    X_features_static = np.zeros((len(valid_t), 3))
    
    for idx, t in enumerate(valid_t):
        X_features_static[idx, 0] = X[t]                                  # Daily Component
        X_features_static[idx, 1] = np.mean(X[t - 4 : t + 1])             # Weekly Component
        X_features_static[idx, 2] = np.mean(X[t - 19 : t + 1])            # Monthly Component
        
    X_design_static = sm.add_constant(X_features_static)
    static_model = sm.OLS(Y_static, X_design_static).fit()
    
    print(static_model.summary(
        yname=f"Log-Variance (t+{delta_static})", 
        xname=["Intercept (K_0)", "Daily (C_1)", "Weekly (C_5)", "Monthly (C_20)"]
    ))
    
    print("\n" + "="*70)
    print("PHASE 2: ROLLING FORECAST (OUT-OF-SAMPLE P-RATIOS & RESIDUALS)")
    print("="*70)
    
    for delta in horizons:
        forecasts = []
        actuals = []
        start_idx = rolling_window + max_lag + delta
        
        for t in range(start_idx, N - delta):
            Y_train = X[t - rolling_window : t]
            X_train = np.zeros((rolling_window, 3))
            
            # Formulation of the trailing 500-day matrix without lookahead bias
            for j in range(rolling_window):
                target_idx = t - rolling_window + j
                feat_idx = target_idx - delta
                
                X_train[j, 0] = X[feat_idx]
                X_train[j, 1] = np.mean(X[feat_idx - 4 : feat_idx + 1])
                X_train[j, 2] = np.mean(X[feat_idx - 19 : feat_idx + 1])
                
            X_train_design = np.column_stack((np.ones(rolling_window), X_train))
            coefficients, _, _, _ = np.linalg.lstsq(X_train_design, Y_train, rcond=None)
            
            # Formulate the present state exactly at time t
            present_state = np.array([
                1.0, 
                X[t], 
                np.mean(X[t - 4 : t + 1]), 
                np.mean(X[t - 19 : t + 1])
            ])
            
            forecasts.append(np.dot(coefficients, present_state))
            actuals.append(X[t + delta])
            
        forecasts = np.array(forecasts)
        actuals = np.array(actuals)
        residuals = actuals - forecasts
        
        mse = np.sum(residuals**2)
        variance_total = np.sum((actuals - global_mean)**2)
        p_ratio = mse / variance_total
        
        print(f"Rolling HAR(3) | Delta: {delta:02d} Day | Out-of-Sample P-Ratio: {p_ratio:.3f}")
        
        identifier = f"HAR_3_D_{delta}"
        results_archive[identifier] = {
            'actuals': actuals,
            'forecasts': forecasts,
            'residuals': residuals
        }

    # Serialization of the benchmark vectors
    np.savez("data/HAR_Out_Of_Sample_Results.npz", **results_archive)
    print("\nOut-of-sample residuals serialized to data/HAR_Out_Of_Sample_Results.npz")
    print("="*70 + "\n")

    # PHASE 3: SUPERIOR VISUAL ARCHITECTURE
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 0.8
    })

    plt.figure(figsize=(12, 5))
    
    plot_window = 150
    display_actual = results_archive['HAR_3_D_1']['actuals'][-plot_window:]
    display_forecast = results_archive['HAR_3_D_1']['forecasts'][-plot_window:]
    time_axis = np.arange(plot_window)

    plt.plot(time_axis, display_actual, color='#000000', label='Empirical Log-Variance', alpha=0.9)
    plt.plot(time_axis, display_forecast, color='#0072B2', label=r'$HAR(3)$ Forecast ($\Delta=1$)', linestyle=(0, (2, 2)), alpha=0.9)

    plt.title(r"Heterogeneous Autoregressive Benchmark: $HAR(3)$", fontsize=14, pad=15)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel(r"Log-Variance $X_t$", fontsize=12)
    plt.legend(fontsize=10, frameon=True, edgecolor='black', fancybox=False, loc='upper left')
    plt.grid(False)
    plt.tight_layout()
    
    plt.savefig("data/SPX_HAR_Forecast.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    execute_complete_har_matrix()