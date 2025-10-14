from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config_loader import ConfigError, load_config
from .dashboard import (
    append_history,
    build_snapshot,
    load_snapshot,
    rankings_changed,
    render_dashboard,
    render_score_chart,
    write_snapshot,
)
from .data_sources import DataSourceError, collect_metrics
from .reporting import write_markdown_report
from .scoring import score_coins


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
        help="Force artifact regeneration even if rankings have not changed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
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
    docs_dir = repo_root / "docs"
    docs_data_dir = docs_dir / "data"
    chart_rel_path = Path("assets/score_trend.png")
    chart_path = docs_dir / chart_rel_path

    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    snapshot = build_snapshot(top_n, config.settings.base_currency, timestamp)

    snapshot_path = reports_dir / "crypto_index.json"
    previous_snapshot = load_snapshot(snapshot_path)
    if not args.force and not rankings_changed(previous_snapshot, snapshot):
        print("No ranking changes detected; skipping artifact updates.")
        return 0

    output_path = args.output or reports_dir / "crypto_index.md"
    write_markdown_report(
        top_n,
        output_path=output_path,
        currency=config.settings.base_currency,
        generated_at=timestamp,
    )

    write_snapshot(snapshot_path, snapshot)
    write_snapshot(docs_data_dir / "crypto_index.json", snapshot)
    history_path = reports_dir / "crypto_index_history.csv"
    append_history(history_path, snapshot)
    render_score_chart(history_path, chart_path)
    dashboard_path = docs_dir / "index.html"
    render_dashboard(snapshot, chart_rel_path, dashboard_path)

    print(f"Report written to {output_path.resolve()}")
    print(f"Snapshot saved to {snapshot_path.resolve()}")
    print(f"Dashboard updated at {dashboard_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
