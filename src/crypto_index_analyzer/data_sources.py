from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .config_loader import AnalyzerConfig


class DataSourceError(RuntimeError):
    """Raised when external data could not be retrieved."""


@dataclass
class CoinMetrics:
    id: str
    symbol: str
    name: str
    market_cap_rank: int
    latest_price: Optional[float]
    market_cap_growth: Optional[float]
    volume_trend: Optional[float]
    volatility: Optional[float]
    developer_activity: Optional[float]
    on_chain_metric: Optional[float]
    github_repo: Optional[str]
    reference_currency: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoinGeckoClient:
    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(
        self,
        *,
        session: Optional[requests.Session] = None,
        retries: int = 3,
        backoff_seconds: float = 5.0,
    ) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "crypto-index-analyzer/1.0",
            }
        )
        self.api_key = os.environ.get("COINGECKO_API_KEY")
        if self.api_key:
            self.session.headers["x-cg-pro-api-key"] = self.api_key
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def _request(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        url = f"{self.BASE_URL}{path}"
        last_error: Optional[requests.HTTPError] = None
        for attempt in range(self.retries):
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 429 and attempt < self.retries - 1:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else self.backoff_seconds * (attempt + 1)
                time.sleep(min(wait_seconds, 60))
                continue
            try:
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                last_error = exc
                if 500 <= response.status_code < 600 and attempt < self.retries - 1:
                    time.sleep(self.backoff_seconds * (attempt + 1))
                    continue
                break
        if last_error:
            raise last_error
        raise requests.HTTPError("Failed to fetch CoinGecko data")

    def get_top_coins(self, *, limit: int, vs_currency: str) -> List[Dict[str, Any]]:
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "price_change_percentage": "1h,24h,7d,30d",
            "sparkline": "false",
        }
        response = self._request("/coins/markets", params=params)
        return response.json()

    def get_market_chart(self, coin_id: str, *, vs_currency: str, days: int) -> Dict[str, Any]:
        params = {
            "vs_currency": vs_currency,
            "days": days,
            "interval": "daily",
        }
        response = self._request(f"/coins/{coin_id}/market_chart", params=params)
        return response.json()

    def get_coin_details(self, coin_id: str) -> Dict[str, Any]:
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "true",
            "sparkline": "false",
        }
        response = self._request(f"/coins/{coin_id}", params=params)
        return response.json()


class GitHubClient:
    BASE_URL = "https://api.github.com"

    def __init__(self, *, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "crypto-index-analyzer/1.0",
        }
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)

    def get_repo_snapshot(self, full_name: str) -> Optional[Dict[str, Any]]:
        repo_url = f"{self.BASE_URL}/repos/{full_name}"
        repo_resp = self.session.get(repo_url, timeout=30)
        if repo_resp.status_code == 404:
            return None
        repo_resp.raise_for_status()
        repo_data = repo_resp.json()

        commits_url = f"{repo_url}/stats/commit_activity"
        commit_resp = self.session.get(commits_url, timeout=30)
        recent_commits = None
        if commit_resp.status_code == 200:
            activity = commit_resp.json()
            if isinstance(activity, list) and activity:
                recent_commits = sum(week.get("total", 0) for week in activity[-4:])
        # The stats endpoint may return 202 (processing). In that case we leave commits as None.

        return {
            "full_name": repo_data.get("full_name", full_name),
            "stargazers_count": repo_data.get("stargazers_count", 0),
            "forks_count": repo_data.get("forks_count", 0),
            "open_issues_count": repo_data.get("open_issues_count", 0),
            "subscribers_count": repo_data.get("subscribers_count", 0),
            "pushed_at": repo_data.get("pushed_at"),
            "recent_commits": recent_commits,
        }


def _compute_market_cap_growth(market_caps: List[List[float]]) -> Optional[float]:
    if len(market_caps) < 2:
        return None
    start = market_caps[0][1]
    end = market_caps[-1][1]
    if not start:
        return None
    return (end - start) / start


def _compute_volume_trend(total_volumes: List[List[float]]) -> Optional[float]:
    if len(total_volumes) < 4:
        return None
    midpoint = len(total_volumes) // 2
    first_window = [point[1] for point in total_volumes[:midpoint] if point[1] is not None]
    second_window = [point[1] for point in total_volumes[midpoint:] if point[1] is not None]
    if not first_window or not second_window:
        return None
    first_avg = sum(first_window) / len(first_window)
    second_avg = sum(second_window) / len(second_window)
    if not first_avg:
        return None
    return (second_avg - first_avg) / first_avg


def _compute_volatility(prices: List[List[float]]) -> Optional[float]:
    if len(prices) < 3:
        return None
    returns: List[float] = []
    for idx in range(1, len(prices)):
        prev = prices[idx - 1][1]
        curr = prices[idx][1]
        if prev:
            returns.append((curr - prev) / prev)
    if len(returns) < 2:
        return None
    mean_val = sum(returns) / len(returns)
    variance = sum((value - mean_val) ** 2 for value in returns) / (len(returns) - 1)
    return math.sqrt(variance)


def _compute_developer_activity(snapshot: Optional[Dict[str, Any]]) -> Optional[float]:
    if snapshot is None:
        return None
    commits = snapshot.get("recent_commits")
    stars = snapshot.get("stargazers_count", 0)
    forks = snapshot.get("forks_count", 0)
    watchers = snapshot.get("subscribers_count", 0)
    open_issues = snapshot.get("open_issues_count", 0)

    def _log_scale(value: Optional[int]) -> float:
        if value is None:
            return 0.0
        return math.log1p(max(value, 0))

    components = {
        "commits": _log_scale(commits),
        "stars": _log_scale(stars),
        "forks": _log_scale(forks),
        "watchers": _log_scale(watchers),
        "open_issues": _log_scale(open_issues),
    }

    return (
        components["commits"] * 0.6
        + components["stars"] * 0.2
        + components["forks"] * 0.1
        + components["watchers"] * 0.05
        + components["open_issues"] * 0.05
    )


def _compute_on_chain(details: Dict[str, Any]) -> Optional[float]:
    market_data = details.get("market_data") or {}
    circulating = market_data.get("circulating_supply")
    total_supply = market_data.get("total_supply")
    max_supply = market_data.get("max_supply")
    supply_base = max_supply or total_supply
    ratio = None
    if circulating and supply_base:
        if supply_base == 0:
            ratio = None
        else:
            ratio = min(max(circulating / supply_base, 0.0), 1.0)

    tvl_raw = market_data.get("total_value_locked")
    tvl = None
    if isinstance(tvl_raw, dict):
        numeric_values = [value for value in tvl_raw.values() if isinstance(value, (int, float))]
        if numeric_values:
            tvl = numeric_values[0]
    elif isinstance(tvl_raw, (int, float)):
        tvl = tvl_raw

    tvl_component = None
    if isinstance(tvl, (int, float)) and tvl > 0:
        tvl_component = min(math.log10(tvl + 1) / 6, 1.0)

    components: List[float] = []
    if ratio is not None:
        components.append(ratio)
    if tvl_component is not None:
        components.append(tvl_component)

    if not components:
        return None
    return sum(components) / len(components)


def collect_metrics(config: AnalyzerConfig) -> List[CoinMetrics]:
    gecko = CoinGeckoClient()
    github = GitHubClient()

    try:
        top_coins = gecko.get_top_coins(
            limit=config.settings.coins_limit,
            vs_currency=config.settings.base_currency,
        )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            raise DataSourceError(
                "CoinGecko rate limit reached. Provide COINGECKO_API_KEY or rerun after the cooldown."
            ) from exc
        raise DataSourceError("Failed to retrieve top coins from CoinGecko") from exc
    except requests.RequestException as exc:
        raise DataSourceError("Failed to retrieve top coins from CoinGecko") from exc

    metrics: List[CoinMetrics] = []
    for coin in top_coins:
        coin_id = coin.get("id")
        if not coin_id:
            continue
        try:
            chart = gecko.get_market_chart(
                coin_id,
                vs_currency=config.settings.base_currency,
                days=config.settings.days_for_trend,
            )
            details = gecko.get_coin_details(coin_id)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                time.sleep(gecko.backoff_seconds)
            continue
        except requests.RequestException:
            continue

        market_cap_growth = _compute_market_cap_growth(chart.get("market_caps", []))
        volume_trend = _compute_volume_trend(chart.get("total_volumes", []))
        volatility = _compute_volatility(chart.get("prices", []))

        repos = ((details.get("links") or {}).get("repos_url") or {}).get("github") or []
        repo_full_name = None
        developer_activity = None
        if repos:
            repo_full_name = repos[0]
            try:
                snapshot = github.get_repo_snapshot(repo_full_name)
            except requests.RequestException:
                snapshot = None
            developer_activity = _compute_developer_activity(snapshot)
        on_chain_metric = _compute_on_chain(details)

        metrics.append(
            CoinMetrics(
                id=coin_id,
                symbol=coin.get("symbol", "").upper(),
                name=coin.get("name", coin_id),
                market_cap_rank=coin.get("market_cap_rank") or 0,
                latest_price=coin.get("current_price"),
                market_cap_growth=market_cap_growth,
                volume_trend=volume_trend,
                volatility=volatility,
                developer_activity=developer_activity,
                on_chain_metric=on_chain_metric,
                github_repo=repo_full_name,
                reference_currency=config.settings.base_currency,
                metadata={
                    "market_cap": coin.get("market_cap"),
                    "total_volume": coin.get("total_volume"),
                    "price_change_percentage_30d": coin.get("price_change_percentage_30d_in_currency"),
                    "last_updated": coin.get("last_updated"),
                },
            )
        )

        time.sleep(random.uniform(2.0, 3.0))

    return metrics



