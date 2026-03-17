from __future__ import annotations

import argparse
import importlib
import logging
import math
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.core.config import BacktestConfig, config_checksum, load_config
from engine.core.logging import setup_logging


# ----------------------------
# Minimal smoke “context”
# ----------------------------

@dataclass(frozen=True)
class SmokeInstrument:
    instrument_id: int
    symbol: str


class SmokeFeatureStore:
    """
    Minimal feature store for E0 smoke:
    - stores per instrument a DataFrame indexed by ts (UTC)
    - get() returns the last known value at or before ts
    """
    def __init__(self, features_by_iid: Dict[int, pd.DataFrame]):
        self.features_by_iid = features_by_iid

    def get(self, instrument_id: int, ts: pd.Timestamp, feature_name: str) -> float:
        df = self.features_by_iid[instrument_id]
        # last row at or before ts
        idx = df.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return float("nan")
        val = df.iloc[idx].get(feature_name, float("nan"))
        try:
            return float(val)
        except Exception:
            return float("nan")


class SmokeContext:
    def __init__(self, instruments: List[SmokeInstrument], feature_store: SmokeFeatureStore):
        self.instruments = {i.instrument_id: i for i in instruments}
        self.feature_store = feature_store
        self._now: Optional[pd.Timestamp] = None

    def set_now(self, ts: pd.Timestamp) -> None:
        self._now = ts

    def now(self) -> pd.Timestamp:
        assert self._now is not None
        return self._now

    def get_feature(self, instrument_id: int, name: str, feature_set: Optional[str] = None) -> float:
        return self.feature_store.get(instrument_id, self.now(), name)


# ----------------------------
# Utilities
# ----------------------------

def make_run_id(cfg_name: str, checksum: str, seed: int) -> str:
    # deterministic, short, traceable
    return f"{cfg_name}-{checksum[:8]}-{seed}"


def import_from_path(dotted_path: str):
    """
    dotted_path: "strategies.library.s05_trend.MacdAdxTrend"
    """
    mod_path, cls_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def generate_synthetic_ohlcv_1h(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    instrument_ids: List[int],
    seed: int,
) -> pd.DataFrame:
    """
    Generate deterministic hourly OHLCV for each instrument_id using a random walk.
    """
    idx = pd.date_range(start=start_ts, end=end_ts, freq="1h", tz="UTC", inclusive="left")
    rows = []

    for iid in instrument_ids:
        rng = np.random.default_rng(seed + int(iid) * 101)
        # deterministic initial price per iid
        base_price = 100.0 + (iid % 1000) * 0.5

        price = base_price
        for ts in idx:
            # hourly log return
            lr = rng.normal(loc=0.0, scale=0.003)  # ~0.3% hourly vol
            close = price * math.exp(lr)
            open_ = price

            # crude high/low around open/close
            hi = max(open_, close) * (1.0 + abs(rng.normal(0, 0.001)))
            lo = min(open_, close) * (1.0 - abs(rng.normal(0, 0.001)))

            volq = float(abs(rng.normal(0, 1.0)) * 1_000_000.0)

            rows.append(
                {
                    "ts_event": ts,
                    "instrument_id": iid,
                    "timeframe": "1h",
                    "open": float(open_),
                    "high": float(hi),
                    "low": float(lo),
                    "close": float(close),
                    "volume_quote": volq,
                    "is_final": True,
                }
            )
            price = close

    df = pd.DataFrame(rows)
    df = df.sort_values(["instrument_id", "ts_event"]).reset_index(drop=True)
    return df


def compute_bar_features_1h(ohlcv: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Compute a subset of bar features per instrument.
    Output DataFrame is indexed by ts_event (UTC) with feature columns.
    """
    out: Dict[int, pd.DataFrame] = {}

    def rsi(series: pd.Series, n: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = (-delta.clip(upper=0)).rolling(n).mean()
        rs = up / (down + 1e-12)
        return 100 - 100 / (1 + rs)

    for iid, sub in ohlcv.groupby("instrument_id"):
        s = sub.set_index("ts_event").sort_index()
        close = s["close"]

        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (s["high"] - s["low"]),
                (s["high"] - prev_close).abs(),
                (s["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14).mean()

        rsi14 = rsi(close, 14)

        # Donchian
        donchian_hh20 = s["high"].rolling(20).max()
        donchian_ll20 = s["low"].rolling(20).min()

        # Residual zscore vs ema50
        resid = close - ema50
        resid_mean = resid.rolling(100).mean()
        resid_std = resid.rolling(100).std()
        resid_z100 = (resid - resid_mean) / (resid_std + 1e-12)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()

        # ADX placeholder (real ADX is more involved); for E0 smoke we provide a stable proxy:
        # Use normalized EMA separation as a weak "trend strength" proxy
        adx14 = (ema50 - ema200).abs() / (close + 1e-12) * 100.0

        feats = pd.DataFrame(
            {
                "ema50": ema50,
                "ema200": ema200,
                "atr14": atr14,
                "rsi14": rsi14,
                "donchian_hh20": donchian_hh20,
                "donchian_ll20": donchian_ll20,
                "resid_z100": resid_z100,
                "macd": macd,
                "macd_sig": macd_sig,
                "adx14": adx14,
            }
        )
        out[int(iid)] = feats

    return out


def load_strategies(cfg: BacktestConfig) -> List[Any]:
    strategies = []
    for s_cfg in cfg.strategies:
        cls = import_from_path(s_cfg.class_)
        # instantiate with params; allow constructor signatures to vary
        try:
            strat = cls(
                universe=s_cfg.universe,
                feature_set=s_cfg.feature_set,
                cost_profile=s_cfg.cost_profile,
                **(s_cfg.params or {}),
            )
        except TypeError:
            # fallback: try params only
            strat = cls(**(s_cfg.params or {}))
            # set common attributes if they exist
            if hasattr(strat, "universe"):
                setattr(strat, "universe", s_cfg.universe)
            if hasattr(strat, "feature_set"):
                setattr(strat, "feature_set", s_cfg.feature_set)
            if hasattr(strat, "cost_profile"):
                setattr(strat, "cost_profile", s_cfg.cost_profile)
        strategies.append(strat)
    return strategies


# ----------------------------
# Main smoke runner
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to backtest YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    checksum = config_checksum(cfg)
    run_id = make_run_id(cfg.run.name, checksum, cfg.run.seed)

    setup_logging(run_id, level="INFO")
    log = logging.getLogger("smoke")

    log.info("Loaded config", extra={"extra": {"config": str(args.config), "checksum": checksum, "run_id": run_id}})

    # Build synthetic data (no external services needed)
    start_ts = cfg.run.start_ts.astimezone(timezone.utc)
    end_ts = cfg.run.end_ts.astimezone(timezone.utc)
    instrument_ids = [i.instrument_id for i in cfg.universe.instruments]

    ohlcv = generate_synthetic_ohlcv_1h(
        start_ts=pd.Timestamp(start_ts),
        end_ts=pd.Timestamp(end_ts),
        instrument_ids=instrument_ids,
        seed=cfg.run.seed,
    )
    feature_frames = compute_bar_features_1h(ohlcv)

    instruments = [SmokeInstrument(i.instrument_id, i.symbol) for i in cfg.universe.instruments]
    fs = SmokeFeatureStore(feature_frames)
    ctx = SmokeContext(instruments=instruments, feature_store=fs)

    # Strategy wiring check
    strategies = load_strategies(cfg)
    log.info("Strategies instantiated", extra={"extra": {"count": len(strategies), "ids": [s.id for s in cfg.strategies]}})

    # Run a minimal time loop (hourly)
    times = pd.date_range(start=start_ts, end=end_ts, freq="1h", tz="UTC", inclusive="left")
    # Determinism: seed RNG used for any optional randomness here
    rng = np.random.default_rng(cfg.run.seed)

    call_counts = {s_cfg.id: 0 for s_cfg in cfg.strategies}

    for ts in times:
        ctx.set_now(ts)

        # Smoke “tick”: call generate_signal if implemented; ignore NotImplementedError
        for strat, s_cfg in zip(strategies, cfg.strategies):
            try:
                # If should_run exists, respect it
                if hasattr(strat, "should_run"):
                    if not strat.should_run(ctx, {"TIMER"}):
                        continue
                if hasattr(strat, "generate_signal"):
                    _ = strat.generate_signal(ctx)
                call_counts[s_cfg.id] += 1
            except NotImplementedError:
                # acceptable for E0 scaffolding
                continue
            except Exception as e:
                log.exception("Strategy call failed", extra={"extra": {"strategy_id": s_cfg.id, "err": str(e)}})
                return 2

        # optional tiny deterministic workload to prove loop is “doing something”
        _ = float(rng.normal(0, 1))

    # Deterministic summary line (stable given config+seed)
    summary = {
        "run_id": run_id,
        "checksum": checksum[:16],
        "instruments": len(instrument_ids),
        "bars": int(len(ohlcv)),
        "ticks": int(len(times)),
        "strategy_calls": call_counts,
    }
    log.info("SMOKE_OK", extra={"extra": summary})

    # Also print one stable non-JSON line for quick eyeballing (kept deterministic)
    print(f"SMOKE_OK run_id={run_id} checksum={checksum[:16]} ticks={len(times)} bars={len(ohlcv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())