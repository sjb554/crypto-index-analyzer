from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config_loader import ConfigError, load_config
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
    output_path = args.output or Path(__file__).resolve().parents[2] / "reports/crypto_index.md"
    timestamp = datetime.now(timezone.utc)
    write_markdown_report(
        top_n,
        output_path=output_path,
        currency=config.settings.base_currency,
        generated_at=timestamp,
    )

    rel_output = output_path.resolve()
    print(f"Report written to {rel_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
