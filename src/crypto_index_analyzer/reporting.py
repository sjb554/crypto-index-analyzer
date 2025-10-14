from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from .scoring import ScoredCoin


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _fmt_sigma(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _fmt_score(value: float) -> str:
    return f"{value * 100:.1f}"


def build_markdown_report(
    scores: Iterable[ScoredCoin],
    *,
    currency: str,
    generated_at: datetime,
) -> str:
    items: List[ScoredCoin] = list(scores)
    timestamp = generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Weekly Crypto Index",
        "",
        f"Generated on **{timestamp}** using market data in {currency.upper()}.",
        "",
        "Scores are weighted composites (0-100) derived from market cap growth, trading volume trends, volatility, developer activity, and on-chain fundamentals.",
        "",
        f"## Top {len(items)} Assets",
        "",
        "| Rank | Name (Symbol) | Score | 30d Market Cap Change | Volume Trend | Volatility (sigma) | Dev Activity | On-Chain | GitHub |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for entry in items:
        growth_raw = entry.raw_metrics.get("market_cap_growth")
        volume_raw = entry.raw_metrics.get("volume_trend")
        volatility_raw = entry.raw_metrics.get("volatility")
        dev_norm = entry.normalized_metrics.get("developer_activity", 0.0)
        on_chain_norm = entry.normalized_metrics.get("on_chain_metric", 0.0)
        repo = entry.metadata.get("github_repo") or "-"

        lines.append(
            "| {rank} | {name} ({symbol}) | {score} | {cap_growth} | {volume} | {volatility} | {dev} | {on_chain} | {repo} |".format(
                rank=entry.rank,
                name=entry.name,
                symbol=entry.symbol,
                score=_fmt_score(entry.composite_score),
                cap_growth=_fmt_percent(growth_raw),
                volume=_fmt_percent(volume_raw),
                volatility=_fmt_sigma(volatility_raw),
                dev=f"{dev_norm * 100:.1f}",
                on_chain=f"{on_chain_norm * 100:.1f}",
                repo=repo,
            )
        )

    if items:
        lines.extend([
            "",
            "## Notes",
            "",
            "- Developer activity values blend recent commit velocity with community telemetry from GitHub.",
            "- On-chain scores combine circulating/maximum supply ratios and total value locked when provided by CoinGecko.",
            "- Volatility is the 30-day standard deviation of daily returns (lower is better in the composite score).",
        ])

    lines.extend([
        "",
        "## Data Sources",
        "",
        "- [CoinGecko API](https://www.coingecko.com/en/api) for market, volume, and on-chain data.",
        "- [GitHub REST API](https://docs.github.com/en/rest) for repository activity telemetry.",
    ])

    return "\n".join(lines).strip() + "\n"


def write_markdown_report(
    scores: Iterable[ScoredCoin],
    output_path: Path,
    *,
    currency: str,
    generated_at: datetime,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = build_markdown_report(scores, currency=currency, generated_at=generated_at)
    output_path.write_text(content, encoding="utf-8")
