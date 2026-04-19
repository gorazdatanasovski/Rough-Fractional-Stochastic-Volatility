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
    
    ar_archive = np.load(ar_path, allow_pickle=True)
    har_archive = np.load(har_path, allow_pickle=True)
    
    # 2. Fractional Parameterization
    H = 0.14
    rolling_window = 500
    horizons = [1, 5, 20]
    global_mean = np.mean(X)
    
    rfsv_archive = {}
    table_data = []

    print("\n" + "="*80)
    print("PHASE 3: RFSV RIEMANN INTEGRAL FORECAST")
    print("="*80)
    
    # 3. Deterministic Fractional Integration
    for delta in horizons:
        forecasts = []
        actuals = []
        
        max_lag_har = 19
        start_idx = rolling_window + max_lag_har + delta
        
        for t in range(start_idx, N - delta):
            history = np.zeros(rolling_window)
            for k_idx in range(rolling_window):
                history[k_idx] = X[t - k_idx] 
                
            k = np.arange(1, rolling_window + 1)
            u = (k - 0.5) / delta
            
            # The core RFSV weighting equation
            weights = (np.cos(H * np.pi) / np.pi) * (1.0 / ((u + 1.0) * (u ** (H + 0.5)))) * (1.0 / delta)
            weights /= np.sum(weights)
            
            rfsv_forecast = np.sum(weights * history)
            
            forecasts.append(rfsv_forecast)
            actuals.append(X[t + delta])
            
        forecasts = np.array(forecasts)
        actuals = np.array(actuals)
        
        mse = np.sum((actuals - forecasts)**2)
        variance_total = np.sum((actuals - global_mean)**2)
        p_ratio = mse / variance_total
        
        rfsv_archive[f"RFSV_D_{delta}"] = p_ratio
        
        print(f"RFSV Integral | Horizon: {delta:02d} | P-Ratio: {p_ratio:.3f}")

        # Extract matching baseline P-ratios
        ar5_res = ar_archive[f'AR_5_D_{delta}'].item()
        ar5_p = np.sum((ar5_res['actuals'] - ar5_res['forecasts'])**2) / np.sum((ar5_res['actuals'] - global_mean)**2)
        
        ar10_res = ar_archive[f'AR_10_D_{delta}'].item()
        ar10_p = np.sum((ar10_res['actuals'] - ar10_res['forecasts'])**2) / np.sum((ar10_res['actuals'] - global_mean)**2)
        
        har_res = har_archive[f'HAR_3_D_{delta}'].item()
        har_p = np.sum((har_res['actuals'] - har_res['forecasts'])**2) / np.sum((har_res['actuals'] - global_mean)**2)
        
        table_data.append([
            f"$\\Delta = {delta}$", 
            f"{ar5_p:.3f}", 
            f"{ar10_p:.3f}", 
            f"{har_p:.3f}", 
            f"{p_ratio:.3f}"
        ])

    print("\n" + "="*80)
    print("P-RATIO COMPARISON MATRIX")
    print("="*80)
    print(f"{'Horizon':<12} | {'AR(5)':<12} | {'AR(10)':<12} | {'HAR(3)':<12} | {'RFSV':<12}")
    print("-" * 80)
    for row in table_data:
        term_delta = row[0].replace('$\\Delta = ', '').replace('$', '')
        print(f"Delta = {term_delta:<5} | {row[1]:<12} | {row[2]:<12} | {row[3]:<12} | {row[4]:<12}")
    print("="*80 + "\n")

    # 4. Superior Visual Architecture
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 0.8,
        "font.size": 11
    })

    # --- Subplot 1: The P-Ratio Benchmark Table ---
    fig1, ax1 = plt.subplots(figsize=(8, 2.5))
    ax1.axis('tight')
    ax1.axis('off')
    
    col_labels = ['Horizon', 'AR(5)', 'AR(10)', 'HAR(3)', 'RFSV']
    
    table = ax1.table(
        cellText=table_data, 
        colLabels=col_labels, 
        loc='center', 
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(0)
        cell.set_facecolor('white')
        cell.set_text_props(weight='normal', color='black')
        
        # Horizontal rules for structure
        if row == 0 or row == len(table_data):
            cell.visible_edges = 'B'
            cell.set_linewidth(1.0)
            
        # Strict typographic isolation of the RFSV numerical advantage
        if col == 4 and row > 0:
            cell.set_text_props(weight='bold', color='#009E73')

    plt.title(r"P-Ratio Evaluation", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("data/SPX_Benchmark_Comparison_Table.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # --- Subplot 2: The Forecasting Trajectory Graph ---
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    
    plot_window = 150
    display_actual = har_archive['HAR_3_D_1'].item()['actuals'][-plot_window:]
    
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

    ax2.plot(time_axis, display_actual, color='#000000', label='Empirical', alpha=0.9)
    ax2.plot(time_axis, display_rfsv, color='#009E73', label=r'RFSV ($\Delta=1$)', linestyle=(0, (2, 2)), alpha=0.9)

    ax2.set_title(r"RFSV Forecast Trajectory", fontsize=14, pad=15)
    ax2.set_xlabel("Time (Days)", fontsize=12)
    ax2.set_ylabel(r"Log-Variance $X_t$", fontsize=12)
    ax2.legend(fontsize=10, frameon=True, edgecolor='black', fancybox=False, loc='upper left')
    ax2.grid(False)
    fig2.tight_layout()
    
    plt.savefig("data/SPX_RFSV_Forecast.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    execute_apex_rfsv_matrix()