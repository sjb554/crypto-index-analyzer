from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics

from .data_sources import CoinGeckoClient

HISTORY_DAYS = 730
TOP_COIN_LIMIT = 50
FEATURE_WINDOWS = (5, 10, 20, 30)
VOL_WINDOWS = (7, 14, 30)
RNG_SEED = 42


@dataclass
class PredictiveConfig:
    base_currency: str
    coins_limit: int = TOP_COIN_LIMIT
    history_days: int = HISTORY_DAYS


@dataclass
class BacktestMetrics:
    auc: float
    pr_auc: float
    hit_rate_top10: float
    basket_vs_btc: float
    turnover: float
    calibration: List[Tuple[float, float]] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)


class HistoricalDataFetcher:
    def __init__(self, config: PredictiveConfig, client: Optional[CoinGeckoClient] = None) -> None:
        self.config = config
        self.client = client or CoinGeckoClient()

    def get_top_coins(self) -> List[Dict[str, str]]:
        coins = self.client.get_top_coins(
            limit=self.config.coins_limit,
            vs_currency=self.config.base_currency,
        )
        results: List[Dict[str, str]] = []
        for entry in coins:
            coin_id = entry.get("id")
            if not coin_id:
                continue
            results.append(
                {
                    "id": coin_id,
                    "symbol": entry.get("symbol", "").upper(),
                    "name": entry.get("name", coin_id),
                    "market_cap_rank": entry.get("market_cap_rank") or 0,
                }
            )
        return results

    def fetch_ohlcv(self, coin_id: str) -> pd.DataFrame:
        data = self.client.get_market_chart(
            coin_id,
            vs_currency=self.config.base_currency,
            days=self.config.history_days,
        )
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if not prices:
            return pd.DataFrame()

        df = pd.DataFrame(prices, columns=["timestamp", "price"]).set_index("timestamp")
        df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert(None)
        volume_df = pd.DataFrame(volumes, columns=["timestamp", "volume"]).set_index("timestamp")
        volume_df.index = pd.to_datetime(volume_df.index, unit="ms", utc=True).tz_convert(None)
        df = df.join(volume_df, how="left")
        df = df.resample("1D").last().ffill()
        df.dropna(inplace=True)
        df["return_1d"] = df["price"].pct_change()
        return df


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1 / window, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain_ema / (loss_ema + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _rolling_max_drawdown(prices: pd.Series, window: int = 30) -> pd.Series:
    roll_max = prices.rolling(window).max()
    drawdown = prices / (roll_max + 1e-9) - 1
    return drawdown


def engineer_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    for window in FEATURE_WINDOWS:
        df[f"momentum_{window}"] = df["price"].pct_change(window)
        df[f"return_{window}"] = df["price"].pct_change(window)
    for window in VOL_WINDOWS:
        df[f"volatility_{window}"] = df["return_1d"].rolling(window).std()
    df["volume_trend_7_30"] = (
        df["volume"].rolling(7).mean() / (df["volume"].rolling(30).mean() + 1e-9) - 1
    )
    df["drawdown_30"] = _rolling_max_drawdown(df["price"], 30)
    df["rsi_14"] = _compute_rsi(df["price"], 14)
    df.dropna(inplace=True)
    return df


def build_dataset(
    fetcher: HistoricalDataFetcher, coins: Sequence[Dict[str, str]]
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for coin in coins:
        coin_id = coin["id"]
        try:
            raw = fetcher.fetch_ohlcv(coin_id)
        except Exception:
            time.sleep(1)
            continue
        if raw.empty or len(raw) < 120:
            continue
        feats = engineer_features(raw)
        feats["coin_id"] = coin_id
        feats["symbol"] = coin["symbol"]
        feats["name"] = coin["name"]
        feats["market_cap_rank"] = coin["market_cap_rank"]
        feats["forward_return_30"] = raw["price"].pct_change(-30).reindex(feats.index)
        feats["target_up"] = (feats["forward_return_30"] > 0).astype(int)
        feats.dropna(inplace=True)
        frames.append(feats)
        time.sleep(0.3)
    if not frames:
        raise RuntimeError("No historical data available for predictive training")
    dataset = pd.concat(frames, axis=0)
    dataset.sort_index(inplace=True)
    return dataset


def make_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    feature_cols = [
        col
        for col in df.columns
        if col
        not in {
            "coin_id",
            "symbol",
            "name",
            "market_cap_rank",
            "target_up",
            "forward_return_30",
            "price",
            "volume",
            "return_1d",
        }
    ]
    features = df[feature_cols].copy()
    features["coin_id"] = df["coin_id"].astype("category")
    y_class = df["target_up"].astype(int)
    y_reg = df["forward_return_30"].astype(float)
    return features, y_class, y_reg


def walk_forward_split(
    df: pd.DataFrame,
    step_days: int = 7,
    min_history_days: int = 180,
) -> Iterable[Tuple[pd.Index, pd.Index]]:
    unique_dates = sorted(df.index.unique())
    for idx in range(min_history_days, len(unique_dates) - 30, step_days):
        train_dates = unique_dates[: idx - 30]
        test_date = unique_dates[idx]
        if not train_dates:
            continue
        train_idx = df.index.isin(train_dates)
        test_idx = df.index == test_date
        if test_idx.sum() == 0:
            continue
        yield df.index[train_idx], df.index[test_idx]


def _fit_classifier(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=RNG_SEED,
    )
    model.fit(X, y)
    return model


def _fit_regressor(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=RNG_SEED,
    )
    model.fit(X, y)
    return model


def backtest_models(X: pd.DataFrame, y_class: pd.Series, y_reg: pd.Series) -> Tuple[BacktestMetrics, pd.DataFrame]:
    predictions: List[Dict[str, float]] = []
    aucs: List[float] = []
    pr_aucs: List[float] = []
    top_hits: List[float] = []
    basket_excess: List[float] = []
    turnover_counts: List[float] = []

    prev_selection: Optional[set[str]] = None

    for train_idx, test_idx in walk_forward_split(X):
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train_class, y_test_class = y_class.loc[train_idx], y_class.loc[test_idx]
        y_train_reg, y_test_reg = y_reg.loc[train_idx], y_reg.loc[test_idx]

        clf = _fit_classifier(X_train, y_train_class)
        reg = _fit_regressor(X_train, y_train_reg)

        proba = clf.predict_proba(X_test)[:, 1]
        expected = reg.predict(X_test)
        realized = y_test_reg.values
        risk = X_test.get("volatility_30", pd.Series(np.nan, index=X_test.index)).values
        coin_ids = X_test["coin_id"].astype(str).values

        aucs.append(metrics.roc_auc_score(y_test_class, proba))
        pr_aucs.append(metrics.average_precision_score(y_test_class, proba))

        date = X_test.index[0]
        rows = []
        for idx_value, coin, prob, exp_ret, act_ret, risk_val in zip(
            test_idx, coin_ids, proba, expected, realized, risk
        ):
            row = {
                "index": str(idx_value),
                "coin_id": coin,
                "probability": float(prob),
                "expected_return": float(exp_ret),
                "actual_return": float(act_ret),
                "risk": float(risk_val) if np.isfinite(risk_val) else float("nan"),
                "up": int(act_ret > 0),
            }
            rows.append(row)
            predictions.append(row)

        frame = pd.DataFrame(rows)
        ranked = frame.sort_values("probability", ascending=False)
        top10 = ranked.head(10)
        top_hits.append(top10["up"].mean())
        basket_ret = top10["actual_return"].mean()
        btc_row = frame[frame["coin_id"] == "bitcoin"]
        btc_ret = btc_row["actual_return"].mean() if not btc_row.empty else frame["actual_return"].mean()
        basket_excess.append(basket_ret - btc_ret)
        selection = set(top10["coin_id"].tolist())
        if prev_selection is not None:
            turnover_counts.append(len(selection.symmetric_difference(prev_selection)))
        prev_selection = selection

    results = pd.DataFrame(predictions)

    calibration: List[Tuple[float, float]] = []
    if not results.empty:
        bins = np.linspace(0, 1, 11)
        results["prob_bin"] = pd.cut(results["probability"], bins)
        for _, subset in results.groupby("prob_bin"):
            if not subset.empty:
                calibration.append((float(subset["probability"].mean()), float(subset["up"].mean())))

    feature_importance: Dict[str, float] = {}
    if not X.empty:
        clf_full = _fit_classifier(X, y_class)
        feature_importance = {
            feature: float(importance)
            for feature, importance in zip(X.columns, clf_full.feature_importances_)
        }

    metrics_summary = BacktestMetrics(
        auc=float(np.mean(aucs)) if aucs else float("nan"),
        pr_auc=float(np.mean(pr_aucs)) if pr_aucs else float("nan"),
        hit_rate_top10=float(np.mean(top_hits)) if top_hits else float("nan"),
        basket_vs_btc=float(np.mean(basket_excess)) if basket_excess else float("nan"),
        turnover=float(np.mean(turnover_counts)) if turnover_counts else float("nan"),
        calibration=calibration,
        feature_importance=feature_importance,
    )
    return metrics_summary, results


def fit_final_models(X: pd.DataFrame, y_class: pd.Series, y_reg: pd.Series) -> Tuple[lgb.LGBMClassifier, lgb.LGBMRegressor]:
    classifier = _fit_classifier(X, y_class)
    regressor = _fit_regressor(X, y_reg)
    return classifier, regressor


def save_models(
    models_dir: Path,
    classifier: lgb.LGBMClassifier,
    regressor: lgb.LGBMRegressor,
    feature_columns: List[str],
    metadata: Dict[str, object],
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, models_dir / "probability_model.pkl")
    joblib.dump(regressor, models_dir / "return_model.pkl")
    metadata_payload = {**metadata, "feature_columns": feature_columns}
    (models_dir / "metadata.json").write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")


def load_models(models_dir: Path) -> Tuple[lgb.LGBMClassifier, lgb.LGBMRegressor, Dict[str, object]]:
    classifier = joblib.load(models_dir / "probability_model.pkl")
    regressor = joblib.load(models_dir / "return_model.pkl")
    metadata = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    return classifier, regressor, metadata


def latest_feature_matrix(
    fetcher: HistoricalDataFetcher, coins: Sequence[Dict[str, str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for coin in coins:
        try:
            raw = fetcher.fetch_ohlcv(coin["id"])
        except Exception:
            continue
        feats = engineer_features(raw)
        if feats.empty:
            continue
        latest = feats.tail(1).copy()
        latest["coin_id"] = coin["id"]
        latest["symbol"] = coin["symbol"]
        latest["name"] = coin["name"]
        latest["market_cap_rank"] = coin["market_cap_rank"]
        frames.append(latest)
        time.sleep(0.1)
    if not frames:
        raise RuntimeError("Unable to assemble latest feature matrix")
    matrix = pd.concat(frames)
    feature_cols = [
        col
        for col in matrix.columns
        if col
        not in {
            "coin_id",
            "symbol",
            "name",
            "market_cap_rank",
            "price",
            "volume",
            "return_1d",
        }
    ]
    X = matrix[feature_cols].copy()
    X["coin_id"] = matrix["coin_id"].astype("category")
    context = matrix[["coin_id", "symbol", "name", "market_cap_rank"]].copy()
    context["volatility_30"] = matrix.get("volatility_30", np.nan)
    return X, context
