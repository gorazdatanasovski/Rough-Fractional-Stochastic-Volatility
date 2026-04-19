import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os

def execute_apex_rfsv_matrix():
    # 1. Ingestion of Empirical Data and Baselines
    file_path = "data/SPX_Realized_Variance_Oxford.xlsx"
    ar_path = "data/AR_Out_Of_Sample_Results.npz"
    har_path = "data/HAR_Out_Of_Sample_Results.npz"
    
    if not all(os.path.exists(p) for p in [file_path, ar_path, har_path]):
        raise FileNotFoundError("Missing empirical data or serialized benchmark residuals. Execute AR and HAR first.")
        
    df = pl.read_excel(file_path)
    X = df.get_column("Log_Variance").to_numpy()
    N = len(X)
    
    # UNLOCKED: allow_pickle=True permits loading custom Python dictionaries from the archive
    ar_archive = np.load(ar_path, allow_pickle=True)
    har_archive = np.load(har_path, allow_pickle=True)
    
    # 2. Fractional Parameterization
    H = 0.14
    rolling_window = 500
    horizons = [1, 5, 20]
    global_mean = np.mean(X)
    
    rfsv_archive = {}

    print("\n" + "="*80)
    print("PHASE 3: RFSV RIEMANN INTEGRAL FORECAST (OUT-OF-SAMPLE)")
    print("="*80)
    
    # 3. Deterministic Fractional Integration
    for delta in horizons:
        forecasts = []
        actuals = []
        
        # Strict alignment with the HAR maximum lag index to ensure 1:1 residual comparison
        max_lag_har = 19
        start_idx = rolling_window + max_lag_har + delta
        
        for t in range(start_idx, N - delta):
            # Extract the trailing 500-day history relative to present time t
            history = np.zeros(rolling_window)
            for k_idx in range(rolling_window):
                history[k_idx] = X[t - k_idx] 
                
            # Riemann sum approximation via midpoint rule to neutralize the u=0 singularity
            k = np.arange(1, rolling_window + 1)
            u = (k - 0.5) / delta
            
            # The core RFSV weighting equation
            weights = (np.cos(H * np.pi) / np.pi) * (1.0 / ((u + 1.0) * (u ** (H + 0.5)))) * (1.0 / delta)
            
            # Strict normalization to account for the finite 500-day truncation
            weights /= np.sum(weights)
            
            # Deterministic forecast execution (No OLS required)
            rfsv_forecast = np.sum(weights * history)
            
            forecasts.append(rfsv_forecast)
            actuals.append(X[t + delta])
            
        forecasts = np.array(forecasts)
        actuals = np.array(actuals)
        residuals = actuals - forecasts
        
        mse = np.sum(residuals**2)
        variance_total = np.sum((actuals - global_mean)**2)
        p_ratio = mse / variance_total
        
        identifier = f"RFSV_D_{delta}"
        rfsv_archive[identifier] = p_ratio
        
        print(f"RFSV Integral | Horizon: {delta:02d} Day | Out-of-Sample P-Ratio: {p_ratio:.3f}")

    print("\n" + "="*80)
    print("FINAL BENCHMARK COMPARISON MATRIX (OUT-OF-SAMPLE P-RATIOS)")
    print("="*80)
    print(f"{'Horizon':<12} | {'AR(5)':<12} | {'AR(10)':<12} | {'HAR(3)':<12} | {'RFSV (Apex)':<12}")
    print("-" * 80)
    
    for delta in horizons:
        # UNLOCKED: .item() unwraps the 0-d object array back into a readable Python dictionary
        ar5_res = ar_archive[f'AR_5_D_{delta}'].item()
        ar5_p = np.sum((ar5_res['actuals'] - ar5_res['forecasts'])**2) / np.sum((ar5_res['actuals'] - global_mean)**2)
        
        ar10_res = ar_archive[f'AR_10_D_{delta}'].item()
        ar10_p = np.sum((ar10_res['actuals'] - ar10_res['forecasts'])**2) / np.sum((ar10_res['actuals'] - global_mean)**2)
        
        har_res = har_archive[f'HAR_3_D_{delta}'].item()
        har_p = np.sum((har_res['actuals'] - har_res['forecasts'])**2) / np.sum((har_res['actuals'] - global_mean)**2)
        
        rfsv_p = rfsv_archive[f'RFSV_D_{delta}']
        
        print(f"Delta = {delta:<5d} | {ar5_p:<12.3f} | {ar10_p:<12.3f} | {har_p:<12.3f} | {rfsv_p:<12.3f}")
    
    print("="*80 + "\n")

    # 4. Superior Visual Architecture
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
    # Aligning the RFSV actuals with the HAR array length for identical visualization mapping
    # UNLOCKED: Applied .item() here as well
    display_actual = har_archive['HAR_3_D_1'].item()['actuals'][-plot_window:]
    
    # We must quickly recalculate D=1 for the graph specifically to extract the array, as the loop overwrote 'forecasts'
    temp_forecasts = []
    delta_vis = 1
    start_idx_vis = rolling_window + max_lag_har + delta_vis
    for t in range(start_idx_vis, N - delta_vis):
        history = np.array([X[t - k_idx] for k_idx in range(rolling_window)])
        k = np.arange(1, rolling_window + 1)
        u = (k - 0.5) / delta_vis
        weights = (np.cos(H * np.pi) / np.pi) * (1.0 / ((u + 1.0) * (u ** (H + 0.5)))) * (1.0 / delta_vis)
        weights /= np.sum(weights)
        temp_forecasts.append(np.sum(weights * history))
        
    display_rfsv = temp_forecasts[-plot_window:]
    time_axis = np.arange(plot_window)

    plt.plot(time_axis, display_actual, color='#000000', label='Empirical Log-Variance', alpha=0.9)
    plt.plot(time_axis, display_rfsv, color='#009E73', label=r'$RFSV$ Forecast ($\Delta=1$)', linestyle=(0, (2, 2)), alpha=0.9)

    plt.title(r"Fractional Superiority: $RFSV$ Continuous Integration ($H=0.14$)", fontsize=14, pad=15)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel(r"Log-Variance $X_t$", fontsize=12)
    plt.legend(fontsize=10, frameon=True, edgecolor='black', fancybox=False, loc='upper left')
    plt.grid(False)
    plt.tight_layout()
    
    plt.savefig("data/SPX_RFSV_Forecast.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    execute_apex_rfsv_matrix()