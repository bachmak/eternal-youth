"""
load_forecast_profile.py

Robuste Last-Prognose für MPC-Horizon.

Ziel:
- pipeline.py ruft nur predict_load_horizon(...) auf
- Dieses Modul ist robust gegen:
  - RangeIndex (kein DatetimeIndex)
  - unterschiedliche Zeitspaltennamen (time/timestamp/...)
  - String-Zeitstempel mit gemischten Zeitzonen (utc=True)
- Forecast-Logik:
  (1) Warmup/Fallback: einfacher Wochenprofil-Forecast (weekday + time-of-day)
  (2) Full: Baseline-Profil (recency weighted) + PV-informiertes Residualmodell (Ridge)

Benötigt:
- numpy, pandas
- scikit-learn (Ridge)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception as e:
    Ridge = None


# =========================
# Helpers: Time handling
# =========================

_CANDIDATE_TIME_COLS = [
    "time", "timestamp", "datetime", "date", "ts", "Time", "Zeit", "Zeitstempel"
]

def _try_parse_datetime_series(s: pd.Series) -> pd.Series:
    """Parse a series to datetime (UTC). Returns dt Series with NaT where invalid."""
    return pd.to_datetime(s, utc=True, errors="coerce")

def infer_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to infer a time column by checking common names and parse success rate.
    Returns column name or None.
    """
    # direct name hits first
    for c in _CANDIDATE_TIME_COLS:
        if c in df.columns:
            dt = _try_parse_datetime_series(df[c])
            if dt.notna().mean() > 0.8:
                return c

    # otherwise, try any object-like column and see if it parses well
    for c in df.columns:
        if c == df.index.name:
            continue
        if df[c].dtype == "O" or "datetime" in str(df[c].dtype).lower():
            dt = _try_parse_datetime_series(df[c])
            if dt.notna().mean() > 0.8:
                return c
    return None

def ensure_datetime_index(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    default_step_minutes: int = 5,
) -> pd.DataFrame:
    """
    Ensure df has a DatetimeIndex.
    If already DatetimeIndex -> ok.
    Else:
      - parse time_col (or infer it) and set as index
      - if no time column is usable: create synthetic datetime index at 5-min steps
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    df = df.copy()

    col = time_col or infer_time_column(df)
    if col is not None:
        dt = _try_parse_datetime_series(df[col])
        if dt.notna().any():
            df.index = dt
            return df

    # Hard fallback: synthetic timeline so the model still runs
    # (works for MPC indexing; weekday semantics are meaningless then)
    start = pd.Timestamp("1970-01-01", tz="UTC")
    df.index = pd.date_range(
        start=start,
        periods=len(df),
        freq=pd.Timedelta(minutes=default_step_minutes),
    )
    return df


# =========================
# Forecast model
# =========================

@dataclass
class ForecastConfig:
    # data resolution assumption (pipeline uses 5min)
    step_minutes: int = 5

    # training window control (samples)
    min_history_samples: int = 2 * 288  # at least 2 days of 5-min samples
    lookback_days: int = 70
    min_train_days: int = 25

    # day-class proxy (vacation/low/high)
    class_prev_days: int = 3
    q_low: float = 0.25
    q_high: float = 0.75

    # recency weighting
    recency_half_life_days: float = 21.0

    # time-of-day weighting
    morning: Tuple[int, int] = (6, 10)
    evening: Tuple[int, int] = (17, 22)
    w_morning: float = 2.0
    w_evening: float = 2.2
    w_default: float = 1.0

    # ridge
    ridge_alpha: float = 6.0


def time_of_day_key(ts: pd.Timestamp) -> str:
    return ts.strftime("%H:%M")

def sample_weight(ts: pd.Timestamp, cfg: ForecastConfig) -> float:
    h = ts.hour
    if cfg.morning[0] <= h < cfg.morning[1]:
        return cfg.w_morning
    if cfg.evening[0] <= h < cfg.evening[1]:
        return cfg.w_evening
    return cfg.w_default

def recency_weight(day_dist: int, cfg: ForecastConfig) -> float:
    hl = max(cfg.recency_half_life_days, 1e-6)
    lam = np.log(2.0) / hl
    return float(np.exp(-lam * day_dist))

def daily_energy_kwh(day_df: pd.DataFrame, load_col: str, cfg: ForecastConfig) -> float:
    dt_h = (cfg.step_minutes / 60.0)
    return float(np.nansum(day_df[load_col].values) * dt_h)


def build_daily_table(df: pd.DataFrame, load_col: str, cfg: ForecastConfig) -> pd.DataFrame:
    out = []
    # df MUST have DatetimeIndex here
    day_series = df.index.strftime("%Y-%m-%d")
    tmp = df.copy()
    tmp["_day"] = day_series

    for day, ddf in tmp.groupby("_day", sort=True):
        if ddf.empty:
            continue
        date = pd.to_datetime(day).date()
        wd = pd.to_datetime(day).weekday()
        out.append({
            "day": day,
            "date": date,
            "weekday": wd,
            "total_kwh": daily_energy_kwh(ddf, load_col, cfg),
        })
    return pd.DataFrame(out).sort_values("day").reset_index(drop=True)


def weekday_quantiles(daily: pd.DataFrame, cfg: ForecastConfig) -> Dict[int, Tuple[float, float]]:
    qmap: Dict[int, Tuple[float, float]] = {}
    for wd in range(7):
        s = daily.loc[daily["weekday"] == wd, "total_kwh"].dropna()
        if len(s) < 10:
            s = daily["total_kwh"].dropna()
        qmap[wd] = (float(s.quantile(cfg.q_low)), float(s.quantile(cfg.q_high)))
    return qmap


def classify_day_from_past(
    daily: pd.DataFrame,
    day: str,
    cfg: ForecastConfig,
    qmap: Dict[int, Tuple[float, float]],
) -> str:
    row = daily.loc[daily["day"] == day]
    if row.empty:
        return "mid"
    wd = int(row.iloc[0]["weekday"])
    q_low, q_high = qmap[wd]

    idx = int(row.index[0])
    if idx <= 0:
        return "mid"

    prev = daily.iloc[max(0, idx - cfg.class_prev_days): idx]
    if len(prev) < max(2, cfg.class_prev_days):
        return "mid"

    proxy = float(prev["total_kwh"].mean())
    if proxy <= q_low:
        return "low"
    if proxy >= q_high:
        return "high"
    return "mid"


def build_training_days(
    daily: pd.DataFrame,
    target_day: str,
    cfg: ForecastConfig,
    qmap: Dict[int, Tuple[float, float]],
) -> List[str]:
    row = daily.loc[daily["day"] == target_day]
    if row.empty:
        return []
    wd = int(row.iloc[0]["weekday"])
    idx = int(row.index[0])

    start = max(0, idx - cfg.lookback_days)
    hist = daily.iloc[start:idx].copy()
    if hist.empty:
        return []

    target_cls = classify_day_from_past(daily, target_day, cfg, qmap)

    hist["cls"] = [
        classify_day_from_past(daily, d, cfg, qmap)
        for d in hist["day"].tolist()
    ]

    subset = hist[(hist["weekday"] == wd) & (hist["cls"] == target_cls)]
    if len(subset) < cfg.min_train_days:
        subset = hist[hist["weekday"] == wd]

    return subset["day"].tolist()


def weighted_baseline_profile(
    df: pd.DataFrame,
    train_days: List[str],
    target_day: str,
    load_col: str,
    cfg: ForecastConfig,
) -> pd.Series:
    """
    Returns baseline profile: mean load per time-of-day (HH:MM).
    Weighted by recency (newer days > older days).
    """
    if train_days:
        hist = df[df["_day"].isin(train_days)].copy()
    else:
        hist = df[df["_day"] < target_day].copy()

    if hist.empty:
        return pd.Series(dtype=float)

    hist["_tod"] = hist.index.map(time_of_day_key)

    unique_days = sorted(hist["_day"].unique())
    day2dist = {d: (len(unique_days) - 1 - i) for i, d in enumerate(unique_days)}
    hist["_w"] = hist["_day"].map(lambda d: recency_weight(day2dist[d], cfg))

    # weighted mean: sum(w*y)/sum(w) per tod
    g = hist.groupby("_tod", sort=False)
    num = g.apply(lambda x: np.nansum(x["_w"].values * x[load_col].values))
    den = g["_w"].sum()
    prof = (num / den).astype(float)
    prof.name = "base_kw"
    return prof


def build_residual_ridge(
    df: pd.DataFrame,
    train_days: List[str],
    base: pd.Series,
    load_col: str,
    pv_col: str,
    cfg: ForecastConfig,
):
    """
    Train ridge on residual = load - base(tod).
    Features: pv_now, pv_l1, pv_l2, y_l1, y_l2, hour_sin, hour_cos
    Weighted by time-of-day importance * recency.
    """
    if Ridge is None:
        return None

    hist = df[df["_day"].isin(train_days)].copy()
    if hist.empty:
        return None

    hist = hist.dropna(subset=[load_col, pv_col]).copy()
    hist["_tod"] = hist.index.map(time_of_day_key)
    hist["_base"] = hist["_tod"].map(base).astype(float)
    hist["_resid"] = hist[load_col] - hist["_base"]

    # lags inside each day
    hist["_pv_l1"] = hist.groupby("_day")[pv_col].shift(1)
    hist["_pv_l2"] = hist.groupby("_day")[pv_col].shift(2)
    hist["_y_l1"] = hist.groupby("_day")[load_col].shift(1)
    hist["_y_l2"] = hist.groupby("_day")[load_col].shift(2)

    minutes = (hist.index.hour * 60 + hist.index.minute).astype(float).to_numpy()
    ang = 2.0 * np.pi * (minutes / (24.0 * 60.0))
    hist["_hsin"] = np.sin(ang)
    hist["_hcos"] = np.cos(ang)

    # weights
    hist["_w_tod"] = hist.index.map(lambda ts: sample_weight(ts, cfg)).astype(float)

    uniq_days = sorted(hist["_day"].unique())
    day2dist = {d: (len(uniq_days) - 1 - i) for i, d in enumerate(uniq_days)}
    hist["_w_rec"] = hist["_day"].map(lambda d: recency_weight(day2dist[d], cfg)).astype(float)

    hist["_w"] = hist["_w_tod"] * hist["_w_rec"]

    cols = [pv_col, "_pv_l1", "_pv_l2", "_y_l1", "_y_l2", "_hsin", "_hcos"]
    hist = hist.dropna(subset=cols + ["_resid"]).copy()
    if len(hist) < 500:
        return None

    X = hist[cols].to_numpy(dtype=float)
    y = hist["_resid"].to_numpy(dtype=float)
    w = hist["_w"].to_numpy(dtype=float)

    model = Ridge(alpha=cfg.ridge_alpha, fit_intercept=True, random_state=0)
    model.fit(X, y, sample_weight=w)
    return model


def predict_load_horizon(
    df: pd.DataFrame,
    curr_idx: int,
    horizon_size: int,
    history_size: int,
    load_col: str,
    pv_col: str,
    mode: str = "auto",     # "fallback" or "auto"
    time_col: Optional[str] = None,
    cfg: Optional[ForecastConfig] = None,
) -> np.ndarray:
    """
    Main API for pipeline:
      returns np.array(horizon_size,) forecast of load [kW]
    """
    cfg = cfg or ForecastConfig()

    if load_col not in df.columns or pv_col not in df.columns:
        raise KeyError(f"df must contain columns '{load_col}' and '{pv_col}'")

    # 1) Ensure datetime index (FIXES your error)
    df2 = ensure_datetime_index(df, time_col=time_col, default_step_minutes=cfg.step_minutes)

    # 2) Slice history window and build helper columns
    start_idx = max(0, curr_idx - history_size)
    df_hist = df2.iloc[start_idx:curr_idx].copy()
    df_fut = df2.iloc[curr_idx:curr_idx + horizon_size].copy()

    # if horizon exceeds df length, pad with last known timestamp spacing
    if len(df_fut) < horizon_size:
        # create a placeholder future index continuation
        if len(df2) >= 2:
            step = df2.index[-1] - df2.index[-2]
        else:
            step = pd.Timedelta(minutes=cfg.step_minutes)

        last_ts = df2.index[min(curr_idx, len(df2)-1)]
        missing = horizon_size - len(df_fut)
        extra_idx = pd.date_range(start=last_ts + step, periods=missing, freq=step)
        extra = pd.DataFrame(index=extra_idx, columns=df2.columns)
        df_fut = pd.concat([df_fut, extra], axis=0)

    # add day key
    df_hist["_day"] = df_hist.index.strftime("%Y-%m-%d")
    df_fut["_day"] = df_fut.index.strftime("%Y-%m-%d")

    # 3) Warmup fallback conditions
    if len(df_hist) < cfg.min_history_samples or mode == "fallback":
        # weekday profile only (simple + robust)
        return _weekday_profile_forecast(df_hist, df_fut, load_col, cfg)

    # 4) Full mode: daily table and training days selection
    daily = build_daily_table(df_hist, load_col, cfg)
    if daily.empty:
        return _weekday_profile_forecast(df_hist, df_fut, load_col, cfg)

    qmap = weekday_quantiles(daily, cfg)

    # We forecast the first day of the horizon based on that "target day"
    target_day = df_fut["_day"].iloc[0]
    train_days = build_training_days(daily, target_day, cfg, qmap)

    base = weighted_baseline_profile(df_hist, train_days, target_day, load_col, cfg)
    if base.empty:
        return _weekday_profile_forecast(df_hist, df_fut, load_col, cfg)

    # baseline prediction for horizon
    df_fut["_tod"] = df_fut.index.map(time_of_day_key)
    base_pred = df_fut["_tod"].map(base).to_numpy(dtype=float)

    # ridge residual correction (PV-informed)
    model = build_residual_ridge(df_hist, train_days, base, load_col, pv_col, cfg)
    if model is None:
        return np.clip(base_pred, 0.0, None)

    # Build horizon features
    fut = df_fut.copy()
    fut["_pv_l1"] = fut.groupby("_day")[pv_col].shift(1)
    fut["_pv_l2"] = fut.groupby("_day")[pv_col].shift(2)

    # IMPORTANT: In MPC you DON'T know future load -> y_lags must come from history
    # We approximate by using last observed loads and then feeding back predictions.
    # Simple approach: use history last two loads for all horizon steps as constant.
    last_vals = df_hist[load_col].dropna()
    y_l1 = float(last_vals.iloc[-1]) if len(last_vals) >= 1 else 0.0
    y_l2 = float(last_vals.iloc[-2]) if len(last_vals) >= 2 else y_l1

    fut["_y_l1"] = y_l1
    fut["_y_l2"] = y_l2

    minutes = (fut.index.hour * 60 + fut.index.minute).astype(float).to_numpy()
    ang = 2.0 * np.pi * (minutes / (24.0 * 60.0))
    fut["_hsin"] = np.sin(ang)
    fut["_hcos"] = np.cos(ang)

    cols = [pv_col, "_pv_l1", "_pv_l2", "_y_l1", "_y_l2", "_hsin", "_hcos"]
    Xf = fut[cols].to_numpy(dtype=float)

    resid = np.zeros(horizon_size, dtype=float)
    valid = np.all(np.isfinite(Xf), axis=1)
    if valid.any():
        resid[valid] = model.predict(Xf[valid])

    pred = base_pred + resid
    pred = np.clip(pred, 0.0, None)
    return pred[:horizon_size]


def _weekday_profile_forecast(
    df_hist: pd.DataFrame,
    df_fut: pd.DataFrame,
    load_col: str,
    cfg: ForecastConfig
) -> np.ndarray:
    """
    Simple fallback:
      forecast(t) = mean load of same weekday & time-of-day in history,
      else global time-of-day mean,
      else global mean.
    """
    if df_hist.empty:
        return np.zeros(len(df_fut), dtype=float)

    hist = df_hist.dropna(subset=[load_col]).copy()
    if hist.empty:
        return np.zeros(len(df_fut), dtype=float)

    hist["_tod"] = hist.index.map(time_of_day_key)
    hist["_wd"] = hist.index.weekday

    fut = df_fut.copy()
    fut["_tod"] = fut.index.map(time_of_day_key)
    fut["_wd"] = fut.index.weekday

    # mean by (wd, tod)
    prof_wd = hist.groupby(["_wd", "_tod"])[load_col].mean()
    # fallback mean by tod
    prof_tod = hist.groupby("_tod")[load_col].mean()
    global_mean = float(hist[load_col].mean())

    out = np.empty(len(fut), dtype=float)
    for i, (wd, tod) in enumerate(zip(fut["_wd"].to_numpy(), fut["_tod"].to_numpy())):
        v = prof_wd.get((wd, tod), np.nan)
        if not np.isfinite(v):
            v = prof_tod.get(tod, np.nan)
        if not np.isfinite(v):
            v = global_mean
        out[i] = float(v)

    return np.clip(out, 0.0, None)
