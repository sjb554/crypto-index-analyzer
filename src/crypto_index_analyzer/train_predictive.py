from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config_loader import load_config
from .predictive import (
    HISTORY_DAYS,
    BacktestMetrics,
    HistoricalDataFetcher,
    PredictiveConfig,
    backtest_models,
    build_dataset,
    fit_final_models,
    make_feature_matrix,
    save_models,
)

ASSETS_OUTPUTS = {
    "predictive_equity.png": "Top 10 1-Month Outlook vs BTC",
    "predictive_calibration.png": "Calibration Curve",
    "predictive_feature_importance.png": "Feature Importance",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _render_placeholder(output: Path, message: str) -> None:
    _ensure_dir(output.parent)
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _render_equity_curve(results: pd.DataFrame, output: Path) -> None:
    if results.empty:
        _render_placeholder(output, "No backtest data")
        return
    results["date"] = pd.to_datetime(results["index"])
    grouped = results.groupby("date")
    basket_eq = 1.0
    btc_eq = 1.0
    equity_rows: List[tuple[datetime, float, float]] = []
    for date, subset in grouped:
        ranked = subset.sort_values("probability", ascending=False)
        top10 = ranked.head(10)
        basket_ret = top10["actual_return"].mean()
        basket_eq *= 1 + basket_ret
        btc_row = subset[subset["coin_id"] == "bitcoin"]
        btc_ret = btc_row["actual_return"].mean() if not btc_row.empty else subset["actual_return"].mean()
        btc_eq *= 1 + btc_ret
        equity_rows.append((date, basket_eq, btc_eq))
    frame = pd.DataFrame(equity_rows, columns=["date", "basket", "btc"])
    plt.figure(figsize=(9, 5))
    plt.plot(frame["date"], frame["basket"], label="Top 10 basket")
    plt.plot(frame["date"], frame["btc"], label="BTC benchmark")
    plt.xlabel("Date")
    plt.ylabel("Equity (rebased)")
    plt.title("Top 10 1-Month Outlook vs BTC")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _ensure_dir(output.parent)
    plt.savefig(output, dpi=160)
    plt.close()


def _render_calibration(results: pd.DataFrame, output: Path) -> None:
    if results.empty:
        _render_placeholder(output, "No calibration data")
        return
    bins = np.linspace(0, 1, 10)
    results["prob_bin"] = pd.cut(results["probability"], bins)
    preds = []
    obs = []
    for _, group in results.groupby("prob_bin"):
        if group.empty:
            continue
        preds.append(group["probability"].mean())
        obs.append(group["up"].mean())
    if not preds:
        _render_placeholder(output, "No calibration data")
        return
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(preds, obs, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _ensure_dir(output.parent)
    plt.savefig(output, dpi=160)
    plt.close()


def _render_feature_importance(importances: Dict[str, float], output: Path) -> None:
    if not importances:
        _render_placeholder(output, "No importance data")
        return
    items = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:15]
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    plt.figure(figsize=(8, 6))
    plt.barh(labels, values)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Feature importance")
    plt.tight_layout()
    _ensure_dir(output.parent)
    plt.savefig(output, dpi=160)
    plt.close()


def _summarise_backtest(results: pd.DataFrame) -> Dict[str, float]:
    if results.empty:
        return {"hit_rate": float("nan"), "basket_vs_btc": float("nan"), "turnover": float("nan")}
    results["date"] = pd.to_datetime(results["index"])
    grouped = results.groupby("date")
    hits = []
    excess = []
    turnover = []
    prev_selection: Optional[set[str]] = None
    for _, subset in grouped:
        ranked = subset.sort_values("probability", ascending=False)
        top10 = ranked.head(10)
        hits.append(top10["up"].mean())
        basket_ret = top10["actual_return"].mean()
        btc_row = subset[subset["coin_id"] == "bitcoin"]
        btc_ret = btc_row["actual_return"].mean() if not btc_row.empty else subset["actual_return"].mean()
        excess.append(basket_ret - btc_ret)
        selection = set(top10["coin_id"].tolist())
        if prev_selection is not None:
            turnover.append(len(selection.symmetric_difference(prev_selection)))
        prev_selection = selection
    return {
        "hit_rate": float(np.mean(hits)) if hits else float("nan"),
        "basket_vs_btc": float(np.mean(excess)) if excess else float("nan"),
        "turnover": float(np.mean(turnover)) if turnover else float("nan"),
    }


def run_training(repo_root: Path) -> Dict[str, object]:
    cfg = load_config()
    predictive_cfg = PredictiveConfig(
        base_currency=cfg.settings.base_currency,
        coins_limit=cfg.settings.coins_limit,
        history_days=HISTORY_DAYS,
    )
    fetcher = HistoricalDataFetcher(predictive_cfg)
    coins = fetcher.get_top_coins()
    dataset = build_dataset(fetcher, coins)
    X, y_class, y_reg = make_feature_matrix(dataset)
    metrics_summary, backtest_results = backtest_models(X, y_class, y_reg)
    classifier, regressor = fit_final_models(X, y_class, y_reg)

    models_dir = repo_root / "models"
    save_models(
        models_dir,
        classifier,
        regressor,
        list(X.columns),
        metadata={
            "trained_at": datetime.utcnow().isoformat(),
            "base_currency": predictive_cfg.base_currency,
            "coins_limit": predictive_cfg.coins_limit,
            "history_days": predictive_cfg.history_days,
        },
    )

    reports_dir = repo_root / "reports"
    _ensure_dir(reports_dir)
    backtest_results.to_csv(reports_dir / "predictive_backtest_results.csv", index=False)

    aggregate = _summarise_backtest(backtest_results.copy())
    summary = {
        "auc": metrics_summary.auc,
        "pr_auc": metrics_summary.pr_auc,
        "hit_rate_top10": metrics_summary.hit_rate_top10,
        "basket_vs_btc": aggregate["basket_vs_btc"],
        "turnover": aggregate["turnover"],
    }
    (reports_dir / "predictive_backtest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Predictive Backtest Summary",
        "",
        f"- AUC: {summary['auc']:.3f}",
        f"- PR-AUC: {summary['pr_auc']:.3f}",
        f"- Hit rate (top 10 basket): {summary['hit_rate_top10']:.3f}",
        f"- Average excess return vs BTC: {summary['basket_vs_btc']:.3%}",
        f"- Average turnover (weekly): {summary['turnover']:.2f} names",  # type: ignore[index]
    ]
    (reports_dir / "predictive_backtest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    assets_dir = repo_root / "docs" / "assets"
    _render_equity_curve(backtest_results.copy(), assets_dir / "predictive_equity.png")
    _render_calibration(backtest_results.copy(), assets_dir / "predictive_calibration.png")
    _render_feature_importance(metrics_summary.feature_importance, assets_dir / "predictive_feature_importance.png")

    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train predictive crypto models")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    summary = run_training(args.repo_root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
