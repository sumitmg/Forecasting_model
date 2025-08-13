# Alerts Forecast — ARIMA/SARIMA/SARIMAX (Notebook)

This project is an **old practice notebook** that forecasts daily *Alerts* using classic time‑series models. It demonstrates a full workflow from quick EDA and stationarity checks to fitting **ARIMA**, **SARIMA**, and **SARIMAX** models with `pmdarima` and visualizing forecasts with confidence bands.

> **File:** `Alerts_forecast.ipynb`  
> **Input expected:** a CSV with at least two columns: `Date` (parseable dates) and `Value` (numeric target).  
> **Frequency used in the notebook:** Daily. The seasonal period is tried as `m=30` (proxy for monthly seasonality over daily data).

---

## 1) What the notebook covers

1. **Imports & setup**
   - `pandas`, `numpy`, `matplotlib`
   - `statsmodels.tsa.stattools.adfuller` for stationarity testing
   - `pmdarima` for `auto_arima` (model selection for ARIMA/SARIMA/SARIMAX)

2. **Data loading & preparation**
   - Reads a CSV (example placeholder: `Forecasting/data.csv`)
   - Parses `Date`, sets it as index
   - Keeps a single target column `Value`

3. **Quick EDA**
   - Plot of the time series
   - Rolling mean & rolling std (window=12 in the notebook)
   - **ADF test** (Augmented Dickey–Fuller) to check (non)‑stationarity

4. **Baseline forecasting (ARIMA)**
   - `auto_arima` (non‑seasonal) automatically picks `(p, d, q)`
   - Residual diagnostics
   - Function `forecast(...)` plots the forecast + 95% CI

5. **Seasonal forecasting (SARIMA)**
   - `auto_arima` with `seasonal=True` and `m=30` (approx. monthly seasonality on daily data)
   - Diagnostics + forecast plot

6. **Exogenous regressors (SARIMAX)**
   - Adds a simple `month_index` feature (1–12) as an **exogenous** driver
   - `auto_arima` with `exogenous=...`
   - Forecast with exogenous path constructed for the horizon

---

## 2) Notable concepts explained

- **Stationarity & ADF test**: Many classical models assume the series is stationary; ADF tests for a unit root. Differencing (the `d` in ARIMA) helps remove trends.
- **ARIMA (p, d, q)**: Autoregressive terms (p), differencing (d), and moving average terms (q). No explicit seasonality.
- **SARIMA (P, D, Q, m)**: Adds seasonal components to ARIMA with period `m` (e.g., 7 for weekly seasonality in daily data; 12 for monthly data if the frequency is monthly).
- **SARIMAX**: SARIMA + **eXogenous** regressors. In the notebook, a simple calendar feature (`month_index`) is used.
- **`auto_arima`**: Automates order selection via information criteria (AIC/BIC) and can difference the series for you.
- **Forecast uncertainty**: Confidence intervals are derived from the model’s residual variance; they are **not** quantiles of an empirical distribution.

---

## 3) Libraries used

- **pandas** — I/O, data manipulation
- **numpy** — numerics
- **matplotlib** — plotting
- **statsmodels** — ADF test (and general TS toolkit)
- **pmdarima** — `auto_arima` for ARIMA/SARIMA/SARIMAX model selection

> Tip: In a script/package, avoid Jupyter magics like `%matplotlib inline` and notebook‑only calls like `!pip install ...`.

---

## 4) How to run

1. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib statsmodels pmdarima
   ```

2. **Prepare your data**
   Ensure a CSV with columns:
   - `Date` — parseable to a datetime
   - `Value` — numeric target to forecast

   Example:
   ```csv
   Date,Value
   2024-01-01,123
   2024-01-02,118
   ...
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook Alerts_forecast.ipynb
   ```

4. **Edit the data path**
   In the cell that reads the CSV, point to your file (e.g., `data/alerts.csv`).

5. **Run cells top → bottom**
   You should see EDA plots, diagnostics, and forecasts with intervals.

---

## 5) Common pitfalls & quick improvements

- **Correct frequency & continuity**
  - Make sure the `Date` index is continuous (fill/handle missing dates). Use `asfreq('D')` + imputation if needed.
  - Choose the **right seasonal period**: for daily data, common `m` values are 7 (weekly), 30/31 (monthly proxy), 365 (yearly). If the data is monthly, use `m=12`.

- **Train/validation split & backtesting**
  - The notebook fits on **all** data. For a real use case, create a **time‑based split** and **backtest** using rolling or expanding windows.
  - Track metrics like **MAE**, **RMSE**, **sMAPE**, and **MASE**.

- **Feature engineering for SARIMAX**
  - Add richer calendar effects: **dow**, **dom**, **month**, **holiday flags**, **payday/closing day**, **events**, **marketing**, etc.
  - Use **lagged** versions of the target and regressors (e.g., `Value_lag7`, `Value_lag14`, `Value_lag28`).

- **Diagnostics**
  - Check residual **autocorrelation** (ACF/PACF), **normality**, and **heteroskedasticity**.
  - Watch for **non‑stationarity** (apply differencing or transformations).

- **Reproducibility & packaging**
  - Replace notebook magics with Python modules and a CLI (e.g., `src/` + `scripts/`).
  - Parameterize `m`, horizon, and features in a config file.

---

## 6) Newer & stronger approaches you can consider (2025)

While ARIMA‑family models are strong baselines, the ecosystem has evolved substantially. Depending on data size and constraints, consider:

### A) Faster/broader classical baselines
- **Nixtla’s `statsforecast`**: Fast, parallel **AutoARIMA**, **AutoETS**, **TBATS/BATS** (good for **multiple seasonalities**), **Theta**, etc. Excellent for large‑scale forecasting and backtesting.
- **`prophet` / `neuralprophet`**: Additive models with built‑in piecewise trends, holidays, and multiple seasonalities. `neuralprophet` supports AR‑lags and quantile forecasts.

### B) Tree‑based ML with time features
- **LightGBM/XGBoost/CatBoost** with engineered features (lags, rolling means, calendar/holiday features, recent aggregates). Often very competitive and easy to explain/deploy. Use **walk‑forward validation** and **grouped cross‑validation** if you have multiple series.

### C) Modern deep learning
- **N‑BEATS / N‑HiTS** (robust, univariate/limited‑exog)
- **Temporal Fusion Transformer (TFT)** (handles static/covariate features, interpretable attention/feature importances)
- **PatchTST** (strong on long context windows for univariate daily/hourly data)
- Libraries: **`darts`**, **`pytorch‑forecasting`**, **`gluonts`**/**`torch‑ts`**, **`nixtla/nbeatsx`**

### D) Probabilistic & quantile forecasting
- Instead of just 95% CI from ARIMA, fit **quantile models** (e.g., LightGBM quantile loss, TFT quantiles) or **distributional** forecasts to get P10/P50/P90, etc.

### E) Multiple seasonality & anomalies
- For daily business data, **weekly + yearly** patterns are common. Use **TBATS/MSTL/Prophet** to capture both.
- Apply **anomaly detection** and **special‑days** handling (outlier capping, intervention variables).

### F) Evaluation & governance (esp. banking/risk)
- Use **backtesting with gaps** (to mimic data latency).
- Track **data drift**; retrain based on triggers (MAPE degradation, concept drift, regime change).
- Keep a **model card**: data lineage, assumptions, limitations, and monitoring plan.

---

## 7) Minimal example: choosing `m` for daily data

- Weekly seasonality: `m=7`
- Monthly proxy: `m=30` or `m=31`
- Yearly: `m=365` (requires enough history; can be slow for SARIMA)

> For multiple seasonalities, prefer **TBATS** or **Prophet/NeuralProphet** over plain SARIMA.

---

## 8) Suggested next steps for this notebook

- Add a **train/validation split** and a **backtest** function (rolling origin) with metrics.
- Auto‑detect the seasonal period via **STL** or spectral methods when unsure.
- Parameterize `m`, horizon, exogenous features, and file paths.
- Export forecasts to CSV/Excel along with intervals and diagnostics.
- Optionally refactor into a small package with `src/`, `configs/`, and `scripts/`.

---

## 9) Project structure (suggested)

```
alerts-forecast/
├─ data/
│  └─ alerts.csv
├─ notebooks/
│  └─ Alerts_forecast.ipynb
├─ src/
│  ├─ data.py          # load/clean, ensure frequency
│  ├─ features.py      # calendar, lags, rolling stats
│  ├─ models.py        # ARIMA/SARIMA/SARIMAX wrappers
│  ├─ evaluate.py      # backtesting, metrics
│  └─ plot.py          # common plotting
├─ scripts/
│  ├─ fit_arima.py
│  └─ forecast.py
├─ configs/
│  └─ base.yaml        # horizon, m, features, paths
└─ README.md
```

---

## 10) Repro tips

- Fix the random seeds where applicable.
- Avoid using `!pip install` inside the notebook; track deps in `requirements.txt` or `pyproject.toml`.
- Use **UTC** or a consistent timezone if timestamps carry time‑of‑day.
- Log model params and metrics (e.g., with MLflow) for auditability.

---

**License:** MIT (or your choice)  
**Author:** (fill in your name)

