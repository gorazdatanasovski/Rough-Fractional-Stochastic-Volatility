import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os

def execute_superior_fractional_regression():
    # 1. Data Ingestion Protocol
    file_path = "data/SPX_Realized_Variance_Oxford.xlsx"
    if not os.path.exists(file_path):
        raise FileNotFoundError("Empirical dataset void. Run oxford_extraction.py first.")
    
    df = pl.read_excel(file_path)
    log_variance_array = df.get_column("Log_Variance").to_numpy()

    # 2. Parameterization
    lags = np.arange(1, 22)  
    q_moments = [0.5, 1.0, 1.5, 2.0]
    
    empirical_moments = {q: [] for q in q_moments}

    for delta in lags:
        increments = np.abs(log_variance_array[delta:] - log_variance_array[:-delta])
        valid_increments = increments[~np.isnan(increments)]
        for q in q_moments:
            m_q = np.mean(valid_increments ** q)
            empirical_moments[q].append(m_q)

    # 3. Absolute Mathematical Aesthetic Architecture
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

    plt.figure(figsize=(10, 6))
    
    # Perceptually orthogonal palette maximizing CIELAB distance
    colors = ['#000000', '#D55E00', '#0072B2', '#009E73']
    markers = ['o', 's', '^', 'D']
    micro_dash = (0, (2, 2)) 
    
    h_estimates = []
    log_lags = np.log(lags)

    # 4. Strict OLS Regression and Plotting
    for i, q in enumerate(q_moments):
        log_m_q = np.log(empirical_moments[q])
        slope, intercept = np.polyfit(log_lags, log_m_q, 1)
        h_extracted = slope / q
        h_estimates.append(h_extracted)
        
        # Empirical coordinates: Distinct geometric markers with 0.5pt boundaries
        plt.plot(log_lags, log_m_q, marker=markers[i], color=colors[i], 
                 markeredgecolor='black', markeredgewidth=0.5, markersize=5, 
                 linestyle='', label=rf'$q={q}$ $(H={h_extracted:.4f})$')
        
        # Regression trajectory: High-density micro-dash
        plt.plot(log_lags, slope * log_lags + intercept, color=colors[i], 
                 linestyle=micro_dash, alpha=0.85)

    # 5. Synthesis
    final_h = np.mean(h_estimates)
    
    plt.title(rf"Empirical Scaling of SPX Log-Volatility: $H \approx {final_h:.4f}$", fontsize=14, pad=15)
    plt.xlabel(r"$\log(\Delta)$", fontsize=12)
    plt.ylabel(r"$\log m(q, \Delta)$", fontsize=12)
    
    plt.legend(fontsize=10, frameon=True, edgecolor='black', fancybox=False)
    plt.grid(False)
    plt.tight_layout()
    
    # 6. Serialization
    plt.savefig("data/SPX_Rough_Scaling_Superior.png", dpi=300, bbox_inches='tight')
    print(f"Regression Execution Complete. Universal Hurst Estimate: {final_h:.4f}")
    plt.show()

if __name__ == "__main__":
    execute_superior_fractional_regression()