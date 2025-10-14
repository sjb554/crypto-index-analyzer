from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import yaml


class ConfigError(Exception):
    """Raised when the configuration file cannot be parsed or validated."""


@dataclass
class WeightConfig:
    market_cap_growth: float
    volume_trend: float
    volatility: float
    developer_activity: float
    on_chain: float

    @property
    def as_dict(self) -> Dict[str, float]:
        return {
            "market_cap_growth": self.market_cap_growth,
            "volume_trend": self.volume_trend,
            "volatility": self.volatility,
            "developer_activity": self.developer_activity,
            "on_chain": self.on_chain,
        }


@dataclass
class Settings:
    top_n: int
    coins_limit: int
    base_currency: str
    days_for_trend: int


@dataclass
class AnalyzerConfig:
    weights: WeightConfig
    settings: Settings


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def _validate_weights(weights: Dict[str, Any]) -> WeightConfig:
    required_keys = {
        "market_cap_growth",
        "volume_trend",
        "volatility",
        "developer_activity",
        "on_chain",
    }
    missing = required_keys - weights.keys()
    if missing:
        raise ConfigError(f"Missing weights for: {', '.join(sorted(missing))}")

    values = {key: float(weights[key]) for key in required_keys}
    total = sum(values.values())
    if not 0.99 <= total <= 1.01:
        raise ConfigError(
            "Weights must sum to approximately 1. Found total={:.3f}".format(total)
        )

    return WeightConfig(**values)


def _validate_settings(settings: Dict[str, Any]) -> Settings:
    required_keys = {"top_n", "coins_limit", "base_currency", "days_for_trend"}
    missing = required_keys - settings.keys()
    if missing:
        raise ConfigError(f"Missing settings for: {', '.join(sorted(missing))}")

    top_n = int(settings["top_n"])
    coins_limit = int(settings["coins_limit"])
    if top_n <= 0 or top_n > coins_limit:
        raise ConfigError("'top_n' must be between 1 and 'coins_limit'.")

    days_for_trend = int(settings["days_for_trend"])
    if days_for_trend <= 0:
        raise ConfigError("'days_for_trend' must be positive.")

    base_currency = str(settings["base_currency"]).lower()

    return Settings(
        top_n=top_n,
        coins_limit=coins_limit,
        base_currency=base_currency,
        days_for_trend=days_for_trend,
    )


def load_config(path: Path | None = None) -> AnalyzerConfig:
    """Load analyzer configuration from YAML file."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    try:
        weights = _validate_weights(data["weights"])
        settings = _validate_settings(data["settings"])
    except KeyError as exc:
        raise ConfigError(f"Missing configuration section: {exc}") from exc

    return AnalyzerConfig(weights=weights, settings=settings)
