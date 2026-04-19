

# Continuous Fractional Integration for Rough Volatility

### Theoretical Architecture
Empirical market variance exhibits structural roughness, defined by a Hurst parameter of $H \approx 0.14$. The volatility surface does not follow a smooth Brownian diffusion. The distant past continuously ripples into the present through a slow, fractional power-law decay.

Incumbent econometric structures ($AR$, $HAR$) enforce artificial amnesia. They map volatility through discrete lag operators, artificially truncating the market's memory into rigid, sequentially arbitrary blocks. This results in mathematically guaranteed lookahead bias in-sample and structural failure during out-of-sample execution.

This architecture discards discrete lag coefficients. It utilizes a continuous Riemann integral to map the exact long-range fractal dependence of the underlying asset. 

### Core Mathematical Engine
The predictive expectation of future log-variance is derived through the continuous weighting of all historical micro-intervals:

$$\mathbb{E}[X_{t+\Delta} | \mathcal{F}_t] \approx \frac{\cos(H\pi)}{\pi} \int_0^r \frac{X_{t-\Delta u}}{(u+1)u^{H+0.5}} du$$

* $X_{t-\Delta u}$: The continuous historical log-variance data array.
* $u+1$: The mathematical anchor neutralizing origin singularity.
* $u^{H+0.5}$: The fractal decay engine, bending the memory curve strictly according to the asset's persistence parameter.

### Execution Framework
This matrix operates exclusively on 1-step ahead, out-of-sample rolling forecasts. The algorithm is structurally blindfolded to the prediction target. All econometric inference is abandoned in favor of raw, physical residual measurement. The mathematical weights are dictated entirely by the single fractal dimension $H$, eliminating coefficient overfitting.
