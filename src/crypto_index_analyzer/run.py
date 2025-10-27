from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config_loader import ConfigError, load_config
from .data_sources import DataSourceError, collect_metrics
from .dashboard import render_dashboard
from .predictive import (
    HISTORY_DAYS,
    PredictiveConfig,
    HistoricalDataFetcher,
    latest_feature_matrix,
    load_models,
)
from .reporting import write_markdown_report
from .scoring import score_coins

PREDICTIVE_REPORT_MD = "predictive_index.md"
PREDICTIVE_REPORT_JSON = "predictive_index.json"
PREDICTIVE_HISTORY_CSV = "predictive_index_history.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the latest crypto index report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to configuration YAML (defaults to repo root config.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the generated Markdown report (defaults to reports/crypto_index.md).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force predictive artifact regeneration even if rankings are unchanged.",
    )
    return parser


def _write_predictive_history(history_path: Path, timestamp: datetime, frame: pd.DataFrame) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    record_date = timestamp.isoformat()
    columns = ["timestamp", "coin_id", "symbol", "rank", "probability", "expected_return", "risk"]
    rows = [
        [
            record_date,
            row["coin_id"],
            row["symbol"],
            int(row["rank"]),
            float(row["probability"]),
            float(row["expected_return"]),
            float(row["risk"]),
        ]
        for _, row in frame.iterrows()
    ]
    df = pd.DataFrame(rows, columns=columns)
    if history_path.exists():
        df.to_csv(history_path, mode="a", index=False, header=False)
    else:
        df.to_csv(history_path, mode="w", index=False, header=True)


def _write_predictive_reports(repo_root: Path, timestamp: datetime, predictive_df: pd.DataFrame, currency: str) -> Dict[str, object]:
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_payload = predictive_df.to_dict(orient="records")
    (reports_dir / PREDICTIVE_REPORT_JSON).write_text(
        json.dumps({"generated_at": timestamp.isoformat(), "currency": currency, "assets": json_payload}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Predictive Crypto Index (30-Day Outlook)",
        "",
        f"Generated on **{timestamp.strftime('%Y-%m-%d %H:%M UTC')}** using models trained on recent history in `{currency.upper()}`.",
        "",
        "| Rank | Asset | p(up) | Expected Return | Risk (vol) | Composite Score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in predictive_df.iterrows():
        asset = f"{row['name']} ({row['symbol']})"
        prob = f"{row['probability'] * 100:.1f}%"
        exp_ret = f"{row['expected_return'] * 100:.1f}%"
        risk = f"{row['risk'] * 100:.1f}%" if np.isfinite(row["risk"]) else "N/A"
        score = f"{row['score']:.3f}"
        lines.append(
            f"| {int(row['rank'])} | {asset} | {prob} | {exp_ret} | {risk} | {score} |"
        )

    top10 = predictive_df.head(10)
    if not top10.empty:
        lines.extend(
            [
                "",
                "## Top 10 1-Month Outlook Basket",
                "",
                "| Metric | Value |",
                "| --- | --- |",
                f"| Average expected return | {top10['expected_return'].mean() * 100:.2f}% |",
                f"| Average probability of positive return | {top10['probability'].mean() * 100:.2f}% |",
                f"| Average volatility (30d) | {top10['risk'].mean() * 100:.2f}% |",
            ]
        )
    (reports_dir / PREDICTIVE_REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "top10_expected": float(top10["expected_return"].mean()) if not top10.empty else float("nan"),
        "top10_probability": float(top10["probability"].mean()) if not top10.empty else float("nan"),
    }


def generate_predictive_outputs(
    config, repo_root: Path, timestamp: datetime, *, force: bool = False
) -> Optional[Dict[str, object]]:
    models_dir = repo_root / "models"
    if not models_dir.exists():
        print("Predictive models directory not found; skipping predictive outputs.")
        return None

    classifier, regressor, metadata = load_models(models_dir)
    fetcher = HistoricalDataFetcher(
        PredictiveConfig(
            base_currency=config.settings.base_currency,
            coins_limit=config.settings.coins_limit,
            history_days=HISTORY_DAYS,
        )
    )
    coins = fetcher.get_top_coins()
    if not coins:
        print("No coins retrieved for predictive inference.")
        return None

    X_latest, context = latest_feature_matrix(fetcher, coins)
    proba = classifier.predict_proba(X_latest)[:, 1]
    expected = regressor.predict(X_latest)
    risk = context.get("volatility_30")
    if risk is None:
        risk = pd.Series(np.nan, index=context.index)
    risk = risk.abs().replace(0, np.nan)

    context = context.copy()
    context['coin_id'] = context['coin_id'].astype(str)
    context["probability"] = proba
    context["expected_return"] = expected
    context["risk"] = risk
    context["score"] = context["probability"] * context["expected_return"] / (context["risk"].fillna(context["risk"].median()) + 1e-6)
    context.sort_values("score", ascending=False, inplace=True)
    context["rank"] = np.arange(1, len(context) + 1)

    history_path = repo_root / "reports" / PREDICTIVE_HISTORY_CSV
    _write_predictive_history(history_path, timestamp, context)
    summary = _write_predictive_reports(repo_root, timestamp, context, config.settings.base_currency)

    return {
        "records": context,
        "summary": summary,
        "generated_at": timestamp,
        "currency": config.settings.base_currency,
    }


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))
        return 2

    try:
        metrics = collect_metrics(config)
    except DataSourceError as exc:
        parser.error(str(exc))
        return 3

    if not metrics:
        parser.error("No metrics were collected. Check API availability and configuration.")
        return 4

    scored = score_coins(metrics, config.weights)
    if not scored:
        parser.error("Unable to score assets. Check metric calculations.")
        return 5

    top_n = scored[: config.settings.top_n]
    repo_root = Path(__file__).resolve().parents[2]
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or reports_dir / "crypto_index.md"
    timestamp = datetime.now(timezone.utc)
    write_markdown_report(
        top_n,
        output_path=output_path,
        currency=config.settings.base_currency,
        generated_at=timestamp,
    )

    predictive_payload = None
    try:
        predictive_payload = generate_predictive_outputs(
            config, repo_root, timestamp, force=args.force
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Predictive pipeline failed: {exc}")

    chart_rel_path = Path("assets/score_trend.png")
    dashboard_path = repo_root / "docs" / "index.html"
    render_dashboard(
        snapshot={
            "generated_at": timestamp,
            "currency": config.settings.base_currency,
            "top_assets": [
                {
                    "rank": coin.rank,
                    "name": coin.name,
                    "symbol": coin.symbol,
                    "score": coin.composite_score * 100,
                    "market_cap": coin.metadata.get("market_cap"),
                    "market_cap_growth": coin.raw_metrics.get("market_cap_growth"),
                    "volume_trend": coin.raw_metrics.get("volume_trend"),
                    "volatility": coin.raw_metrics.get("volatility"),
                    "developer_activity": coin.normalized_metrics.get("developer_activity"),
                    "on_chain": coin.normalized_metrics.get("on_chain_metric"),
                    "github_repo": coin.metadata.get("github_repo"),
                }
                for coin in top_n
            ],
        },
        chart_path=chart_rel_path,
        predictive=predictive_payload,
        output_path=dashboard_path,
    )

    print(f"Report written to {output_path.resolve()}")
    if predictive_payload:
        print("Predictive outlook generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())



