# Crypto Index Analyzer

Python-based toolkit that analyzes and ranks the leading cryptocurrencies each week. The system combines two layers:

1. A fundamentals/liquidity dashboard (market-cap growth, volume trend, volatility, developer activity, on-chain fundamentals).
2. A predictive 30-day outlook that estimates the probability and magnitude of positive returns using machine learning.

Both layers run automatically through GitHub Actions and publish to a GitHub Pages dashboard.

## Key Features
- Tracks the top 40 cryptocurrencies by market capitalisation using the CoinGecko API (configurable in `config.yaml`).
- Builds fundamentals metrics (market-cap growth, volume trend, volatility, developer telemetry, on-chain strength) and renders them in `reports/crypto_index.md`.
- Trains LightGBM models on daily OHLCV history (momentum, volatility, drawdown, RSI, volume acceleration) to predict 30-day returns.
- Publishes predictive outputs (`reports/predictive_index.md` / `.json`) with probability of upside, expected return, risk, and composite score for ~50 coins.
- Generates a “Top 10 1-Month Outlook” basket, calibration metrics, equity curve, and feature-importance charts under `docs/assets/`.
- Ships with a scheduled GitHub Actions workflow (`.github/workflows/update.yml`) that refreshes fundamentals and predictive outputs every Monday and commits the updated reports/dashboard.

## Project Layout
- `config.yaml` - metric weights, coin count, base currency, trend window.
- `requirements.txt` - Python dependencies (requests, pandas, scikit-learn, lightgbm, etc.).
- `reports/crypto_index.md` - fundamentals snapshot (auto-generated weekly).
- `reports/predictive_index.md` / `.json` - ML-based 30-day outlook.
- `reports/predictive_backtest.*` - training/backtest summary artefacts.
- `models/` - persisted LightGBM models (`probability_model.pkl`, `return_model.pkl`, `metadata.json`).
- `docs/` - GitHub Pages dashboard (`index.html` + charts/assets/data).
- `src/crypto_index_analyzer/` - source package (data fetchers, scoring, predictive pipeline, dashboard renderer, trainers).

## Usage & Configuration

### Live Dashboard
After enabling GitHub Pages (branch `master`, folder `/docs`), the dashboard publishes at `https://sjb554.github.io/crypto-index-analyzer/`. The page now has two panels: a weekly fundamentals table and a predictive 1-month outlook with charts.

### Automated Updates (GitHub Actions)
The workflow runs every Monday at 12:00 UTC (`.github/workflows/update.yml`). Steps:
1. Install dependencies from `requirements.txt`.
2. Run the fundamentals/predictive inference pipeline (`python -m crypto_index_analyzer.run --config config.yaml --output reports/crypto_index.md`).
3. Ensure `docs/.nojekyll` exists so GitHub Pages serves static files.
4. Commit and push any changed reports, JSON, history CSVs, models, or dashboard assets.

Provide repository secrets before enabling the workflow:
- `COINGECKO_API_KEY` – **required for training** and recommended for inference (CoinGecko now requires authenticated access to historical OHLCV).
- `GITHUB_TOKEN` – use the default GitHub Actions token or your PAT for larger API quotas.

### Manual Training & Refresh
The predictive models need an initial training run (and retraining when you want to refresh the fit). Run locally:
```powershell
# 1. Export your CoinGecko API key (required for market_chart requests)
$env:COINGECKO_API_KEY = 'your-key'

# 2. (Optional) activate the project virtualenv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Train models + backtests (takes several minutes for ~50 coins, 24 months history)
$env:PYTHONPATH = 'src'
python -m crypto_index_analyzer.train_predictive --repo-root .
```
Outputs:
- `models/` directory with LightGBM classifier/regressor artefacts and metadata.
- `reports/predictive_backtest.json` / `.md` summarising AUC, PR-AUC, hit rate, excess return, turnover.
- Charts in `docs/assets/` (`predictive_equity.png`, `predictive_calibration.png`, `predictive_feature_importance.png`).

Once models exist, the weekly GitHub Action only performs inference – no retraining required.

To run a local fundamentals + predictive refresh (without retraining):
```powershell
$env:PYTHONPATH = 'src'
python -m crypto_index_analyzer.run --force
```
Use `--force` if you want to regenerate predictive artefacts even when rankings are unchanged.

### Customising `config.yaml`
- `weights`: fundamentals weighting for the weekly composite score (must sum to 1.0).
- `settings.top_n`: number of assets surfaced in the fundamentals Markdown report (5-10 recommended).
- `settings.coins_limit`: number of coins fetched for both fundamentals and predictive inference (keep <=40 for free-tier API limits unless you have paid access).
- `settings.days_for_trend`: window (days) used for fundamentals trend metrics (predictive training always fetches 24 months of daily data).

### Predictive Modelling Notes
- Features: 5/10/20/30-day momentum, rolling volatility, volume acceleration, 30-day drawdown, RSI, and coin identifiers.
- Models: LightGBM binary classifier (probability of positive 30-day return) + LightGBM regressor (expected 30-day return magnitude).
- Walk-forward backtest: weekly expanding-window evaluation across the last 24 months.
- Composite score: `probability × expected_return ÷ risk` (risk = 30-day volatility).
- History logging: predictions appended to `reports/predictive_index_history.csv`, used for dashboard charts.

### Troubleshooting
- **401/429 from CoinGecko**: set `COINGECKO_API_KEY` as an environment variable or repository secret. Without it, historical OHLCV endpoints reject requests and the predictive pipeline skips training/inference.
- **Models missing**: `crypto_index_analyzer.run` skips predictive outputs if `models/` is absent; run the training script first.
- **Large dependency install**: LightGBM & scikit-learn are heavy; enable pip caching in your CI if run times become an issue.
- **Slow training**: fetching 24 months for ~50 coins requires ~5–8 minutes with free-tier rate limiting.

## How It Works
1. **Fundamentals layer**
   - Fetches live metrics from CoinGecko + GitHub for the top N market-cap assets.
   - Normalises metrics, applies weights from `config.yaml`, and renders the weekly Markdown report and dashboard table.
2. **Predictive layer**
   - Collects two years of daily OHLCV per coin, engineers momentum/volatility features, and labels forward 30-day returns.
   - Runs walk-forward LightGBM backtests, logs metrics, and persists models.
   - During weekly inference, loads models, scores the latest features, ranks ~50 assets by composite outlook, and updates Markdown/JSON/history/HTML.
3. **Dashboard**
   - Served via GitHub Pages (`docs/index.html`). Shows fundamentals table, predictive outlook table, top-10 basket metrics, equity curve, calibration plot, feature importance, and composite trend chart.

## License
No licence provided; add one if redistribution is required.
