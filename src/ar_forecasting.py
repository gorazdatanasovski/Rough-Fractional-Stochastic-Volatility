import polars as pl
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

def execute_complete_ar_matrix():
    file_path = "data/SPX_Realized_Variance_Oxford.xlsx"
    if not os.path.exists(file_path):
        raise FileNotFoundError("Empirical dataset void. Execute oxford extraction first.")
        
    df = pl.read_excel(file_path)
    X = df.get_column("Log_Variance").to_numpy()
    N = len(X)
    
    rolling_window = 500
    horizons = [1, 5, 20]
    p_lags = [5, 10]
    global_mean = np.mean(X)
    
    # Storage for future comparative residual analysis
    results_archive = {}

    print("\n" + "="*70)
    print("PHASE 1: STATIC ECONOMETRICS (IN-SAMPLE AR(5) FOR HORIZON 1)")
    print("="*70)
    
    delta_static = 1
    p_static = 5
    
    Y_static = X[p_static - 1 + delta_static : N]
    X_features_static = np.zeros((len(Y_static), p_static))
    for i in range(p_static):
        X_features_static[:, i] = X[p_static - 1 - i : N - delta_static - i]
        
    X_design_static = sm.add_constant(X_features_static)
    static_model = sm.OLS(Y_static, X_design_static).fit()
    
    print(static_model.summary(
        yname=f"Log-Variance (t+{delta_static})", 
        xname=["Intercept (K_0)"] + [f"Lag {i+1} (C_{i+1})" for i in range(p_static)]
    ))
    
    print("\n" + "="*70)
    print("PHASE 2: ROLLING FORECAST (OUT-OF-SAMPLE P-RATIOS & RESIDUALS)")
    print("="*70)
    
    for p in p_lags:
        for delta in horizons:
            forecasts = []
            actuals = []
            start_idx = rolling_window + delta + p
            
            for t in range(start_idx, N - delta):
                Y_train = X[t - rolling_window : t]
                X_train = np.zeros((rolling_window, p))
                for i in range(p):
                    X_train[:, i] = X[t - rolling_window - delta - i : t - delta - i]
                
                X_train_design = np.column_stack((np.ones(rolling_window), X_train))
                coefficients, _, _, _ = np.linalg.lstsq(X_train_design, Y_train, rcond=None)
                
                present_state = np.array([1.0] + [X[t - i] for i in range(p)])
                forecasts.append(np.dot(coefficients, present_state))
                actuals.append(X[t + delta])
                
            forecasts = np.array(forecasts)
            actuals = np.array(actuals)
            residuals = actuals - forecasts
            
            mse = np.sum(residuals**2)
            variance_total = np.sum((actuals - global_mean)**2)
            p_ratio = mse / variance_total
            
            print(f"Rolling AR({p:02d}) | Delta: {delta:02d} Day | Out-of-Sample P-Ratio: {p_ratio:.3f}")
            
            # Archive parameters for visualization and later RFSV comparison
            identifier = f"AR_{p}_D_{delta}"
            results_archive[identifier] = {
                'actuals': actuals,
                'forecasts': forecasts,
                'residuals': residuals
            }

    # Serialize residuals to disk for the final comparison matrix
    np.savez("data/AR_Out_Of_Sample_Results.npz", **results_archive)
    print("\nOut-of-sample residuals serialized to data/AR_Out_Of_Sample_Results.npz")
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
    display_actual = results_archive['AR_5_D_1']['actuals'][-plot_window:]
    display_forecast = results_archive['AR_5_D_1']['forecasts'][-plot_window:]
    time_axis = np.arange(plot_window)

    plt.plot(time_axis, display_actual, color='#000000', label='Empirical Log-Variance', alpha=0.9)
    plt.plot(time_axis, display_forecast, color='#D55E00', label=r'$AR(5)$ Forecast ($\Delta=1$)', linestyle=(0, (2, 2)), alpha=0.9)

    plt.title(r"Autoregressive Forecast Failure: $AR(5)$ Phase Lag", fontsize=14, pad=15)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel(r"Log-Variance $X_t$", fontsize=12)
    plt.legend(fontsize=10, frameon=True, edgecolor='black', fancybox=False, loc='upper left')
    plt.grid(False)
    plt.tight_layout()
    
    plt.savefig("data/SPX_AR_Forecast.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    execute_complete_ar_matrix()