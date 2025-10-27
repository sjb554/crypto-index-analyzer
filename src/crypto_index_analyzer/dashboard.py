from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _format_percent(value: Optional[float], digits: int = 1) -> str:
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def _render_top_assets(snapshot: Dict[str, object]) -> str:
    rows = []
    for asset in snapshot.get("top_assets", []):
        rows.append(
            "<tr>"
            f"<td>{asset.get('rank')}</td>"
            f"<td>{asset.get('name')} <span class='muted'>({asset.get('symbol')})</span></td>"
            f"<td>{asset.get('score'):.1f}</td>"
            f"<td>{_format_percent(asset.get('market_cap_growth'))}</td>"
            f"<td>{_format_percent(asset.get('volume_trend'))}</td>"
            f"<td>{_format_percent(asset.get('volatility'), digits=2)}</td>"
            f"<td>{_format_percent(asset.get('developer_activity'), digits=1)}</td>"
            f"<td>{_format_percent(asset.get('on_chain'), digits=1)}</td>"
            f"<td>{asset.get('github_repo') or '-'}" "</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _render_predictive_table(predictive: Dict[str, object]) -> str:
    frame = predictive.get("records")
    if frame is None or frame.empty:
        return "<tr><td colspan='6'>Predictive data unavailable.</td></tr>"
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            "<tr>"
            f"<td>{int(row['rank'])}</td>"
            f"<td>{row['name']} <span class='muted'>({row['symbol']})</span></td>"
            f"<td>{row['probability']*100:.1f}%</td>"
            f"<td>{row['expected_return']*100:.1f}%</td>"
            f"<td>{row['risk']*100:.1f}%</td>"
            f"<td>{row['score']:.3f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _render_summary_list(lines: Iterable[str]) -> str:
    return "\n".join(f"<li>{line}</li>" for line in lines) or "<li>No highlights available.</li>"


def render_dashboard(
    snapshot: Dict[str, object],
    chart_path: Path,
    output_path: Path,
    predictive: Optional[Dict[str, object]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_ts: datetime = snapshot.get("generated_at", datetime.utcnow())
    predictive_ts = predictive.get("generated_at") if predictive else None

    predictive_table = ""
    predictive_summary = ""
    predictive_charts = ""
    if predictive and predictive.get("records") is not None:
        predictive_table = _render_predictive_table(predictive)
        summary = predictive.get("summary", {})
        predictive_summary = "\n".join(
            [
                f"Average expected return (top 10): {summary.get('top10_expected', float('nan')) * 100:.2f}%",
                f"Average probability (top 10): {summary.get('top10_probability', float('nan')) * 100:.2f}%",
            ]
        )
        charts = [
            ("assets/predictive_equity.png", "Top 10 vs BTC"),
            ("assets/predictive_calibration.png", "Calibration"),
            ("assets/predictive_feature_importance.png", "Feature Importance"),
        ]
        chart_tags = []
        for rel_path, title in charts:
            chart_tags.append(
                f"<figure><img src='{rel_path}' alt='{title}'></img><figcaption>{title}</figcaption></figure>"
            )
        predictive_charts = "".join(chart_tags)
    else:
        predictive_table = "<tr><td colspan='6'>Predictive models not available. Run training first.</td></tr>"
        predictive_summary = "Predictive models not available."

    top_assets_rows = _render_top_assets(snapshot)

    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Crypto Index Analyzer</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; }}
    header {{ background: #1e293b; padding: 2rem 1.5rem; text-align: center; }}
    main {{ padding: 1.5rem; max-width: 1100px; margin: 0 auto; }}
    section {{ background: #1e293b; margin-bottom: 2rem; border-radius: 12px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.4); overflow: hidden; }}
    section h2 {{ margin: 0; padding: 1rem 1.5rem; background: rgba(148,163,184,0.1); border-bottom: 1px solid rgba(148,163,184,0.2); }}
    .content {{ padding: 1.5rem; }}
    table {{ width: 100%; border-collapse: collapse; color: #e2e8f0; }}
    th, td {{ padding: 0.75rem; border-bottom: 1px solid rgba(148,163,184,0.2); text-align: left; }}
    th {{ background: rgba(148,163,184,0.12); font-weight: 600; }}
    tr:hover {{ background: rgba(148,163,184,0.08); }}
    .muted {{ color: #94a3b8; font-size: 0.85rem; }}
    figure {{ margin: 1rem 0; }}
    figure img {{ max-width: 100%; border-radius: 10px; }}
    footer {{ text-align: center; padding: 2rem 1rem; color: #94a3b8; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Crypto Index Analyzer</h1>
    <p>Market data snapshot: {snapshot_ts.strftime('%Y-%m-%d %H:%M UTC')} &bull; Currency: {snapshot.get('currency', 'USD').upper()}</p>
  </header>
  <main>
    <section>
      <h2>Weekly Market Leaders</h2>
      <div class='content'>
        <div style='overflow-x:auto;'>
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Asset</th>
                <th>Score</th>
                <th>30d Cap ?</th>
                <th>Volume Trend</th>
                <th>Volatility</th>
                <th>Dev Activity</th>
                <th>On-Chain</th>
                <th>GitHub</th>
              </tr>
            </thead>
            <tbody>
              {top_assets_rows}
            </tbody>
          </table>
        </div>
      </div>
    </section>
    <section>
      <h2>1-Month Outlook (Predictive)</h2>
      <div class='content'>
        <p class='muted'>Latest predictive run: {predictive_ts.strftime('%Y-%m-%d %H:%M UTC') if predictive_ts else 'N/A'}</p>
        <div style='overflow-x:auto;'>
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Asset</th>
                <th>p(up)</th>
                <th>Expected Return</th>
                <th>Risk (vol)</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {predictive_table}
            </tbody>
          </table>
        </div>
        <p>{predictive_summary}</p>
        <div class='chart-grid'>
          {predictive_charts}
        </div>
      </div>
    </section>
    <section>
      <h2>Market Score Trend</h2>
      <div class='content'>
        <img src='{chart_path.as_posix()}' alt='Composite score trend' style='max-width:100%; border-radius: 10px;'>
      </div>
    </section>
  </main>
  <footer>
    <p>Data sources: CoinGecko &middot; GitHub REST API &middot; Models retrained as needed.</p>
  </footer>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
