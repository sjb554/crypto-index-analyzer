from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .config_loader import WeightConfig
from .data_sources import CoinMetrics


@dataclass
class ScoredCoin:
    id: str
    name: str
    symbol: str
    composite_score: float
    normalized_metrics: Dict[str, float]
    raw_metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    rank: int


def _normalize(values: List[Any], *, invert: bool = False) -> List[float]:
    numeric_values = [value for value in values if isinstance(value, (int, float))]
    if not numeric_values:
        return [0.0 for _ in values]

    min_val = min(numeric_values)
    max_val = max(numeric_values)
    if abs(max_val - min_val) < 1e-12:
        base = 0.5
        return [base if isinstance(value, (int, float)) else 0.0 for value in values]

    normalized: List[float] = []
    for value in values:
        if not isinstance(value, (int, float)):
            normalized.append(0.0)
            continue
        norm = (value - min_val) / (max_val - min_val)
        if invert:
            norm = 1 - norm
        normalized.append(max(0.0, min(norm, 1.0)))
    return normalized


def score_coins(coins: List[CoinMetrics], weights: WeightConfig) -> List[ScoredCoin]:
    if not coins:
        return []

    metric_definitions = [
        ("market_cap_growth", "market_cap_growth"),
        ("volume_trend", "volume_trend"),
        ("volatility", "volatility"),
        ("developer_activity", "developer_activity"),
        ("on_chain_metric", "on_chain"),
    ]

    value_matrix: Dict[str, List[Any]] = {
        metric: [getattr(coin, metric) for coin in coins]
        for metric, _ in metric_definitions
    }

    normalized_matrix: Dict[str, List[float]] = {}
    availability: Dict[str, bool] = {}
    for metric, _ in metric_definitions:
        invert = metric == "volatility"
        normalized_matrix[metric] = _normalize(value_matrix[metric], invert=invert)
        availability[metric] = any(isinstance(value, (int, float)) for value in value_matrix[metric])

    weight_map = weights.as_dict
    available_weight = sum(weight_map[key] for metric, key in metric_definitions if availability[metric])
    if available_weight <= 0:
        available_weight = 1.0
    scored: List[ScoredCoin] = []
    for idx, coin in enumerate(coins):
        metric_scores = {
            metric: normalized_matrix[metric][idx]
            for metric, _ in metric_definitions
        }
        composite = 0.0
        for metric, weight_key in metric_definitions:
            if not availability[metric]:
                continue
            weight = weight_map[weight_key] / available_weight
            composite += metric_scores[metric] * weight

        scored.append(
            ScoredCoin(
                id=coin.id,
                name=coin.name,
                symbol=coin.symbol,
                composite_score=composite,
                normalized_metrics=metric_scores,
                raw_metrics={metric: getattr(coin, metric) for metric, _ in metric_definitions},
                metadata={**coin.metadata, "github_repo": coin.github_repo, "latest_price": coin.latest_price, "market_cap_rank": coin.market_cap_rank, "reference_currency": coin.reference_currency},
                rank=0,
            )
        )

    scored.sort(key=lambda item: item.composite_score, reverse=True)
    for position, entry in enumerate(scored, start=1):
        entry.rank = position

    return scored

