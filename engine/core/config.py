from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ----------------------------
# Helpers
# ----------------------------

def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        # treat naive as UTC to avoid hidden local-time nondeterminism
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _canonical_json_bytes(obj: Any) -> bytes:
    # Stable serialization:
    # - sorted keys
    # - no whitespace
    # - ensure_ascii for stable bytes across environments
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return s.encode("utf-8")


# ----------------------------
# Config models (Pydantic v2)
# ----------------------------

class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1)
    start_ts: datetime
    end_ts: datetime
    seed: int = Field(42, ge=0)
    initial_cash: float = Field(100_000.0, gt=0)
    quote_asset: str = Field("USDT", min_length=2)

    @field_validator("start_ts", "end_ts", mode="before")
    @classmethod
    def _parse_dt(cls, v: Any) -> datetime:
        # Allow YAML strings like "2026-01-01T00:00:00Z"
        if isinstance(v, datetime):
            return _ensure_utc(v)
        if isinstance(v, str):
            s = v.strip()
            # Normalize Z suffix for fromisoformat
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            return _ensure_utc(dt)
        raise TypeError(f"Expected datetime or ISO string, got {type(v)}")

    @field_validator("end_ts")
    @classmethod
    def _validate_range(cls, v: datetime, info) -> datetime:
        start = info.data.get("start_ts")
        if isinstance(start, datetime) and v <= start:
            raise ValueError("end_ts must be strictly greater than start_ts")
        return v


class InstrumentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instrument_id: int = Field(..., gt=0)
    symbol: str = Field(..., min_length=3)
    exchange_id: int = Field(..., gt=0)
    instrument_type: Literal["SPOT", "PERP", "FUT", "OPT"] = "SPOT"

    contract_size: float = Field(1.0, gt=0)
    price_tick: float = Field(..., gt=0)
    qty_step: float = Field(..., gt=0)
    min_notional: float = Field(10.0, ge=0)
    quote_asset: str = Field("USDT", min_length=2)


class UniverseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instruments: List[InstrumentConfig] = Field(..., min_length=1)


class EventSourceConfig(BaseModel):
    """
    Minimal, flexible event source config.
    You can extend this later (Timescale, L2, trades, etc.) without breaking existing configs.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(..., min_length=1)   # e.g. timer, parquet_ohlcv
    name: str = Field(..., min_length=1)

    # Timer fields
    freq: Optional[str] = None

    # Parquet OHLCV fields
    path: Optional[str] = None
    timeframe: Optional[str] = None
    instrument_ids: Optional[List[int]] = None


class FeatureStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["parquet", "noop"] = "parquet"
    root: str = "features/"
    cache_instruments: int = Field(16, ge=1)


class FeatureSetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_set: str = Field(..., min_length=1)
    timeframe: Optional[str] = None
    required_columns: List[str] = Field(default_factory=list)


class FeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    store: FeatureStoreConfig
    sets: Dict[str, FeatureSetConfig] = Field(default_factory=dict)


class StrategyScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    on: List[str] = Field(default_factory=list)  # e.g., ["TIMER"], ["BAR"]


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., min_length=1)
    class_: str = Field(..., alias="class", min_length=3)  # dotted path
    cost_profile: str = Field("default", min_length=1)
    universe: List[int] = Field(default_factory=list)  # instrument_ids
    feature_set: Optional[str] = None
    schedule: StrategyScheduleConfig = Field(default_factory=StrategyScheduleConfig)
    params: Dict[str, Any] = Field(default_factory=dict)


class ExecutionLatencyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    submit: int = Field(50, ge=0)  # ms
    cancel: int = Field(50, ge=0)  # ms


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["sim", "live"] = "sim"
    order_latency_ms: ExecutionLatencyConfig = Field(default_factory=ExecutionLatencyConfig)


class CostProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 6.0
    k_impact: float = 0.25
    k_adverse: float = 0.8
    latency_sec: float = 0.25
    adv_quote: float = 500_000_000.0
    vol_daily: float = 0.04


class RiskPortfolioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_gross_leverage: float = 2.0
    max_drawdown: float = 0.20
    max_position_weight: float = 0.20
    max_turnover_per_day: float = 1.0


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    portfolio: RiskPortfolioConfig = Field(default_factory=RiskPortfolioConfig)


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run: RunConfig
    universe: UniverseConfig
    event_sources: List[EventSourceConfig] = Field(default_factory=list)
    features: Optional[FeaturesConfig] = None
    strategies: List[StrategyConfig] = Field(default_factory=list)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    cost_profiles: Dict[str, CostProfileConfig] = Field(default_factory=dict)
    risk: RiskConfig = Field(default_factory=RiskConfig)


# ----------------------------
# Loader + checksum
# ----------------------------

def load_config(path: str | Path) -> BacktestConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config file must parse to a YAML mapping/object at the top level.")
    return BacktestConfig.model_validate(raw)


def config_checksum(cfg: BacktestConfig) -> str:
    """
    Stable checksum for run reproducibility.

    We hash the canonical JSON representation of the full validated config.
    (No runtime timestamps are added here; run_id can incorporate seed separately.)
    """
    obj = cfg.model_dump(mode="json", by_alias=True, exclude_none=True)
    digest = hashlib.sha256(_canonical_json_bytes(obj)).hexdigest()
    return digest