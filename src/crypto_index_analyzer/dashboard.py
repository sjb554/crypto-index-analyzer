from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .scoring import ScoredCoin


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coin_label(coin: ScoredCoin) -> str:
    return f"{coin.name} ({coin.symbol})"


def _percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _dev_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}"


def _volatility_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _format_currency(value: Optional[float], currency: str) -> str:
    if value is None or value == 0:
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        scaled = value / 1_000_000_000_000
        unit = "T"
    elif abs_value >= 1_000_000_000:
        scaled = value / 1_000_000_000
        unit = "B"
    elif abs_value >= 1_000_000:
        scaled = value / 1_000_000
        unit = "M"
    else:
        scaled = value
        unit = ""
    return f"{currency.upper()} {scaled:,.2f}{unit}"


def build_snapshot(coins: Sequence[ScoredCoin], currency: str, generated_at: datetime) -> dict:
    summary_lines: List[str] = []

    by_growth = [
        (coin, coin.raw_metrics.get("market_cap_growth"))
        for coin in coins
        if isinstance(coin.raw_metrics.get("market_cap_growth"), (int, float))
    ]
    if by_growth:
        top_growth, growth_value = max(by_growth, key=lambda item: item[1])
        summary_lines.append(
            f"{_coin_label(top_growth)} leads 30-day market-cap growth at {_percent(growth_value)}."
        )
        bottom_growth, bottom_value = min(by_growth, key=lambda item: item[1])
        if bottom_value is not None and bottom_value < 0:
            summary_lines.append(
                f"{_coin_label(bottom_growth)} is still under pressure with {_percent(bottom_value)} market-cap change."
            )

    by_volume = [
        (coin, coin.raw_metrics.get("volume_trend"))
        for coin in coins
        if isinstance(coin.raw_metrics.get("volume_trend"), (int, float))
    ]
    if by_volume:
        top_volume, volume_value = max(by_volume, key=lambda item: item[1])
        summary_lines.append(
            f"Trading interest is strongest in {_coin_label(top_volume)} with {_percent(volume_value)} volume acceleration."
        )

    by_volatility = [
        (coin, coin.raw_metrics.get("volatility"))
        for coin in coins
        if isinstance(coin.raw_metrics.get("volatility"), (int, float))
    ]
    if by_volatility:
        lowest_vol, vol_value = min(by_volatility, key=lambda item: item[1])
        summary_lines.append(
            f"{_coin_label(lowest_vol)} shows the lowest 30-day volatility at {_volatility_percent(vol_value)}."
        )

    by_dev = [
        (coin, coin.normalized_metrics.get("developer_activity"))
        for coin in coins
        if isinstance(coin.normalized_metrics.get("developer_activity"), (int, float))
    ]
    if by_dev:
        top_dev, dev_value = max(by_dev, key=lambda item: item[1])
        summary_lines.append(
            f"Developer momentum favours {_coin_label(top_dev)} with {_dev_percent(dev_value)} activity score."
        )

    snapshot = {
        "generated_at": generated_at.astimezone().isoformat(),
        "currency": currency.upper(),
        "summary": {
            "lines": summary_lines,
        },
        "top_assets": [],
    }

    for coin in coins:
        snapshot["top_assets"].append(
            {
                "rank": coin.rank,
                "id": coin.id,
                "name": coin.name,
                "symbol": coin.symbol,
                "composite_score": round(coin.composite_score * 100, 2),
                "raw_metrics": {
                    "market_cap_growth": _safe_float(coin.raw_metrics.get("market_cap_growth")),
                    "volume_trend": _safe_float(coin.raw_metrics.get("volume_trend")),
                    "volatility": _safe_float(coin.raw_metrics.get("volatility")),
                },
                "normalized_metrics": {
                    "developer_activity": _safe_float(coin.normalized_metrics.get("developer_activity")),
                    "on_chain": _safe_float(coin.normalized_metrics.get("on_chain_metric")),
                },
                "metadata": {
                    "market_cap": _safe_float(coin.metadata.get("market_cap")),
                    "total_volume": _safe_float(coin.metadata.get("total_volume")),
                    "price_change_percentage_30d": _safe_float(
                        coin.metadata.get("price_change_percentage_30d")
                    ),
                    "last_updated": coin.metadata.get("last_updated"),
                    "github_repo": coin.metadata.get("github_repo"),
                    "latest_price": _safe_float(coin.metadata.get("latest_price")),
                    "market_cap_rank": coin.metadata.get("market_cap_rank"),
                    "reference_currency": coin.metadata.get("reference_currency"),
                },
            }
        )

    return snapshot


def load_snapshot(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def rankings_changed(previous: Optional[dict], current: dict) -> bool:
    if not previous:
        return True
    prev_ids = [item.get("id") for item in previous.get("top_assets", [])]
    curr_ids = [item.get("id") for item in current.get("top_assets", [])]
    return prev_ids != curr_ids


def write_snapshot(path: Path, snapshot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=False), encoding="utf-8")


def append_history(history_path: Path, snapshot: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = snapshot["generated_at"]
    rows = [
        {
            "generated_at": timestamp,
            "rank": asset["rank"],
            "id": asset["id"],
            "symbol": asset["symbol"],
            "composite_score": asset["composite_score"],
        }
        for asset in snapshot.get("top_assets", [])
    ]
    header = ["generated_at", "rank", "id", "symbol", "composite_score"]

    write_header = not history_path.exists()
    if history_path.exists():
        with history_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            last_timestamp = None
            for row in reader:
                last_timestamp = row.get("generated_at")
        if last_timestamp == timestamp:
            return

    with history_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def render_score_chart(history_path: Path, chart_path: Path, max_assets: int = 5) -> None:
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    if not history_path.exists():
        _render_placeholder_chart(chart_path, "No history yet")
        return

    with history_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        entries = list(reader)

    if not entries:
        _render_placeholder_chart(chart_path, "No history yet")
        return

    for entry in entries:
        entry["generated_at"] = datetime.fromisoformat(entry["generated_at"])
        entry["composite_score"] = float(entry["composite_score"])
        entry["rank"] = int(entry["rank"])

    entries.sort(key=lambda row: row["generated_at"])

    latest_timestamp = max(row["generated_at"] for row in entries)
    latest_assets = sorted(
        [row for row in entries if row["generated_at"] == latest_timestamp],
        key=lambda row: row["rank"]
    )[:max_assets]
    tracked_ids = {row["id"] for row in latest_assets}

    series: dict[str, List[tuple[datetime, float]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in entries:
        if row["id"] not in tracked_ids:
            continue
        series[row["id"]].append((row["generated_at"], row["composite_score"]))
        labels[row["id"]] = row["symbol"]

    plt.figure(figsize=(10, 6))
    for asset_id, points in series.items():
        points.sort(key=lambda item: item[0])
        dates, scores = zip(*points)
        plt.plot(dates, scores, marker="o", label=labels.get(asset_id, asset_id))

    plt.title("Composite score trend (top assets)")
    plt.xlabel("Run date")
    plt.ylabel("Score (0-100)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()


def _render_placeholder_chart(chart_path: Path, message: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()


def render_dashboard(snapshot: dict, chart_rel_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: Iterable[str] = snapshot.get("summary", {}).get("lines", [])
    top_assets = snapshot.get("top_assets", [])
    currency = snapshot.get("currency", "USD")
    generated_at = snapshot.get("generated_at", "")
    chart_src = chart_rel_path.as_posix()

    def _format_row(asset: dict) -> str:
        metrics = asset.get("raw_metrics", {})
        normalized = asset.get("normalized_metrics", {})
        metadata = asset.get("metadata", {})
        market_cap = _format_currency(metadata.get("market_cap"), currency)
        total_volume = _format_currency(metadata.get("total_volume"), currency)
        price_change = metadata.get("price_change_percentage_30d")
        price_change_str = _percent(price_change) if price_change is not None else "N/A"
        github_repo = metadata.get("github_repo")
        github_cell = (
            f'<a href="{html.escape(github_repo)}" target="_blank" rel="noopener">Repo</a>'
            if github_repo
            else "-"
        )
        return (
            f"<tr><td>{asset['rank']}</td>"
            f"<td>{html.escape(asset['name'])} <span class='muted'>({html.escape(asset['symbol'])})</span></td>"
            f"<td>{asset['composite_score']:.1f}</td>"
            f"<td>{market_cap}</td>"
            f"<td>{_percent(metrics.get('market_cap_growth'))}</td>"
            f"<td>{_percent(metrics.get('volume_trend'))}</td>"
            f"<td>{_volatility_percent(metrics.get('volatility'))}</td>"
            f"<td>{_dev_percent(normalized.get('developer_activity'))}</td>"
            f"<td>{total_volume}</td>"
            f"<td>{price_change_str}</td>"
            f"<td>{github_cell}</td></tr>"
        )

    rows_html = "\n" + "\n".join(_format_row(asset) for asset in top_assets) if top_assets else ""
    summary_html = "\n".join(f"<li>{html.escape(line)}</li>" for line in lines) or "<li>No highlights available.</li>"

    html_content = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Crypto Index Analyzer</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; }}
    header {{ background: #1e293b; padding: 2rem 1.5rem; text-align: center; }}
    header h1 {{ margin: 0 0 0.5rem 0; font-size: 2rem; }}
    main {{ padding: 1.5rem; max-width: 1100px; margin: 0 auto; }}
    section {{ margin-bottom: 2rem; background: #1e293b; border-radius: 12px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.4); overflow: hidden; }}
    section h2 {{ margin: 0; padding: 1rem 1.5rem; background: rgba(148, 163, 184, 0.1); font-size: 1.25rem; border-bottom: 1px solid rgba(148, 163, 184, 0.2); }}
    section .content {{ padding: 1.5rem; }}
    ul {{ padding-left: 1.25rem; }}
    table {{ width: 100%; border-collapse: collapse; color: #e2e8f0; }}
    th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid rgba(148, 163, 184, 0.2); }}
    th {{ background: rgba(148, 163, 184, 0.12); font-weight: 600; }}
    tr:hover {{ background: rgba(148, 163, 184, 0.08); }}
    .muted {{ color: #94a3b8; font-size: 0.85rem; }}
    img.chart {{ display: block; max-width: 100%; height: auto; border-radius: 0 0 12px 12px; }}
    a {{ color: #38bdf8; }}
    footer {{ text-align: center; padding: 2rem 1rem; color: #94a3b8; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Crypto Index Analyzer</h1>
    <p>Updated {html.escape(generated_at)} &bull; Market data in {html.escape(currency)}</p>
  </header>
  <main>
    <section>
      <h2>Highlights</h2>
      <div class=\"content\">
        <ul>
          {summary_html}
        </ul>
      </div>
    </section>
    <section>
      <h2>Top {len(top_assets)} Assets</h2>
      <div class=\"content\">
        <div style=\"overflow-x:auto;\">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Asset</th>
                <th>Score</th>
                <th>Market Cap</th>
                <th>30d Cap ?</th>
                <th>Volume Trend</th>
                <th>Volatility</th>
                <th>Dev Activity</th>
                <th>Total Volume</th>
                <th>30d Price ?</th>
                <th>GitHub</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
        <p style=\"margin-top:1rem;\"><a href=\"data/crypto_index.json\">Download the latest snapshot (JSON)</a></p>
      </div>
    </section>
    <section>
      <h2>Composite Score Trend</h2>
      <div class=\"content\">
        <img class=\"chart\" src=\"{chart_src}\" alt=\"Composite score trend chart\" />
        <p class=\"muted\">Top assets are tracked run-to-run; chart updates after each scheduled refresh.</p>
      </div>
    </section>
  </main>
  <footer>
    <p>Data sources: CoinGecko (market/volume/on-chain) and GitHub REST API (developer activity). Automation runs weekly via GitHub Actions.</p>
  </footer>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")

