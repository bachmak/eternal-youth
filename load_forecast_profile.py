"""
load_forecast_profile.py (FAST CACHED VERSION)

Drop-in replacement for your current module.

Key goals:
- Same public API: predict_load_horizon(df, curr_idx, horizon_size, history_size, load_col, pv_col, ...)
- NEVER returns NaN/inf
- Much faster for long simulations by caching:
  (1) Parsed datetime index + day/tod encodings per DataFrame
  (2) Baseline profile + ridge model per target day (trained once per day)

Important design choice for speed/stability:
- We DO NOT drop rows due to time parsing issues. If time parsing is too poor,
  we fall back to a synthetic timeline, keeping row count stable for curr_idx semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception:
    Ridge = None


# =============================================================================
# Config
# =============================================================================

@dataclass
class ForecastConfig:
    step_minutes: int = 5

    # training selection
    lookback_days: int = 70
    min_train_days: int = 25

    # fallback threshold
    min_history_samples: int = 2 * 288

    # day-class proxy
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

    # safety
    clip_nonnegative: bool = True


# =============================================================================
# Internal caches
# =============================================================================

# Cache for prepared DF (keyed by object id + shape signature)
_PREP_CACHE: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

# Cache for trained day-models (keyed by (prep_key, target_day_str))
_DAY_MODEL_CACHE: Dict[Tuple[Tuple[int, int, int], str], Dict[str, Any]] = {}


# =============================================================================
# Helpers
# =============================================================================

_CANDIDATE_TIME_COLS = [
    "time", "timestamp", "datetime", "date", "ts",
    "Time", "Zeit", "Zeitstempel"
]


def _try_parse_datetime_series_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _infer_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in _CANDIDATE_TIME_COLS:
        if c in df.columns:
            dt = _try_parse_datetime_series_utc(df[c])
            if dt.notna().mean() > 0.8:
                return c
    # try any object-like
    for c in df.columns:
        if df[c].dtype == "O":
            dt = _try_parse_datetime_series_utc(df[c])
            if dt.notna().mean() > 0.8:
                return c
    return None


def _tod_grid(step_minutes: int) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Canonical tod keys and mapping to [0..n-1]
    """
    n = int(round(24 * 60 / step_minutes))
    base = pd.Timestamp("2000-01-01 00:00:00")
    idx = pd.date_range(base, periods=n, freq=pd.Timedelta(minutes=step_minutes))
    keys = np.array([t.strftime("%H:%M") for t in idx], dtype=object)
    mapping = {k: i for i, k in enumerate(keys.tolist())}
    return keys, mapping


def _safe_fill_profile_full_grid(profile_vec: np.ndarray) -> np.ndarray:
    """
    profile_vec length n, may contain NaNs; fill smoothly.
    """
    v = np.asarray(profile_vec, dtype=float).copy()
    if np.all(np.isfinite(v)):
        return v

    # interpolate over index
    x = np.arange(len(v), dtype=float)
    mask = np.isfinite(v)
    if mask.any():
        v[~mask] = np.interp(x[~mask], x[mask], v[mask])
    else:
        v[:] = 0.0

    # still guard
    if np.any(~np.isfinite(v)):
        m = float(np.nanmean(v))
        if not np.isfinite(m):
            m = 0.0
        v = np.nan_to_num(v, nan=m, posinf=m, neginf=0.0)

    return v


def _sample_weight_hours(hours: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    """
    Vectorized time-of-day weight by hour.
    """
    w = np.full_like(hours, cfg.w_default, dtype=float)
    w[(hours >= cfg.morning[0]) & (hours < cfg.morning[1])] = cfg.w_morning
    w[(hours >= cfg.evening[0]) & (hours < cfg.evening[1])] = cfg.w_evening
    return w


def _recency_weights_for_days(unique_days: np.ndarray, cfg: ForecastConfig) -> Dict[str, float]:
    """
    unique_days: sorted list/array of day strings, oldest->newest.
    Return dict day->weight (newer bigger) using half-life.
    """
    hl = max(cfg.recency_half_life_days, 1e-6)
    lam = np.log(2.0) / hl
    # newest dist=0
    n = len(unique_days)
    out = {}
    for i, d in enumerate(unique_days):
        dist = (n - 1 - i)
        out[str(d)] = float(np.exp(-lam * dist))
    return out


def _finalize_pred(pred: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    arr = np.asarray(pred, dtype=float)

    if np.any(~np.isfinite(arr)):
        m = float(np.nanmean(arr))
        if not np.isfinite(m):
            m = 0.0
        arr = np.nan_to_num(arr, nan=m, posinf=m, neginf=0.0)

    if cfg.clip_nonnegative:
        arr = np.clip(arr, 0.0, None)

    return arr


# =============================================================================
# Preparation (cached per df)
# =============================================================================

def _prepare_df(
    df: pd.DataFrame,
    load_col: str,
    pv_col: str,
    cfg: ForecastConfig,
    time_col: Optional[str] = None,
) -> Tuple[Tuple[int, int, int], Dict[str, Any]]:
    """
    Build cached arrays for fast forecasting:
    - dt_index (local naive) OR synthetic index
    - day_str per row
    - weekday per row
    - tod_idx per row (0..287)
    """
    # cache key based on object id + shape signature
    prep_key = (id(df), int(df.shape[0]), int(df.shape[1]))
    if prep_key in _PREP_CACHE:
        prep = _PREP_CACHE[prep_key]
        # basic sanity: required columns still present
        if load_col in prep["df"].columns and pv_col in prep["df"].columns:
            return prep_key, prep
        # else: fallthrough to rebuild

    # Build working df2 (try to avoid heavy copies, but ensure we can add arrays safely)
    df2 = df.copy()

    # Ensure numeric columns (cheap, and prevents hidden object issues)
    df2[load_col] = pd.to_numeric(df2[load_col], errors="coerce")
    df2[pv_col] = pd.to_numeric(df2[pv_col], errors="coerce")

    # Build datetime index (do NOT drop rows)
    if isinstance(df2.index, pd.DatetimeIndex):
        dt_index = df2.index
        if not dt_index.is_monotonic_increasing:
            dt_index = dt_index.sort_values()
            df2 = df2.loc[dt_index].copy()
    else:
        col = time_col or _infer_time_col(df2)
        if col is not None:
            dt_utc = _try_parse_datetime_series_utc(df2[col])
            parse_rate = float(dt_utc.notna().mean())
            if parse_rate > 0.8:
                # Convert to local naive, but keep length stable:
                # fill NaT with synthetic continuation
                dt_local = dt_utc.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)

                # For any NaT -> fill via synthetic stepping
                if dt_local.isna().any():
                    # build synthetic baseline, then overwrite valid entries
                    start = pd.Timestamp("1970-01-01 00:00:00")
                    syn = pd.date_range(
                        start=start,
                        periods=len(df2),
                        freq=pd.Timedelta(minutes=cfg.step_minutes),
                        name="time"
                    )
                    dt_filled = pd.Series(syn, index=df2.index, dtype="datetime64[ns]")
                    dt_filled.loc[dt_local.notna()] = dt_local.loc[dt_local.notna()].values
                    dt_index = pd.DatetimeIndex(dt_filled.values, name="time")
                else:
                    dt_index = pd.DatetimeIndex(dt_local.values, name="time")
            else:
                # parsing too poor -> synthetic
                start = pd.Timestamp("1970-01-01 00:00:00")
                dt_index = pd.date_range(
                    start=start,
                    periods=len(df2),
                    freq=pd.Timedelta(minutes=cfg.step_minutes),
                    name="time"
                )
        else:
            start = pd.Timestamp("1970-01-01 00:00:00")
            dt_index = pd.date_range(
                start=start,
                periods=len(df2),
                freq=pd.Timedelta(minutes=cfg.step_minutes),
                name="time"
            )

    # Ensure monotonic index for consistent day order
    # (If dt_index not monotonic due to filled NaTs, we sort but keep iloc semantics stable? Sorting would break curr_idx.)
    # So we DO NOT sort here. We preserve row order = iloc order.
    df2.index = dt_index

    # Precompute tod mapping
    tod_keys, tod_map = _tod_grid(cfg.step_minutes)
    tod_str = df2.index.strftime("%H:%M").to_numpy(dtype=object)
    tod_idx = np.fromiter((tod_map.get(t, 0) for t in tod_str), count=len(tod_str), dtype=int)

    day_str = df2.index.strftime("%Y-%m-%d").to_numpy(dtype=object)
    weekday = df2.index.weekday.to_numpy(dtype=int)
    hours = df2.index.hour.to_numpy(dtype=int)
    minutes = df2.index.minute.to_numpy(dtype=int)

    prep = {
        "df": df2,
        "tod_keys": tod_keys,
        "tod_idx": tod_idx,
        "day_str": day_str,
        "weekday": weekday,
        "hours": hours,
        "minutes": minutes,
    }
    _PREP_CACHE[prep_key] = prep
    return prep_key, prep


# =============================================================================
# Day model building (cached per target day)
# =============================================================================

def _daily_kwh_from_history(
    hist_load: np.ndarray,
    hist_day: np.ndarray,
    cfg: ForecastConfig,
) -> pd.DataFrame:
    """
    Build daily table: day, weekday, total_kwh
    hist_day is array of day strings aligned with hist_load
    """
    dt_h = cfg.step_minutes / 60.0
    # group by day using pandas for simplicity (days count is small)
    tmp = pd.DataFrame({"day": hist_day, "load": hist_load})
    g = tmp.groupby("day", sort=True)["load"].sum(min_count=1)
    total_kwh = (g * dt_h).to_numpy(dtype=float)
    days = g.index.to_numpy(dtype=object)
    weekday = pd.to_datetime(days).weekday
    out = pd.DataFrame({"day": days, "weekday": weekday.astype(int), "total_kwh": total_kwh})
    return out.sort_values("day").reset_index(drop=True)


def _weekday_quantiles(daily: pd.DataFrame, cfg: ForecastConfig) -> Dict[int, Tuple[float, float]]:
    qmap: Dict[int, Tuple[float, float]] = {}
    all_s = daily["total_kwh"].dropna()
    for wd in range(7):
        s = daily.loc[daily["weekday"] == wd, "total_kwh"].dropna()
        if len(s) < 10:
            s = all_s
        if len(s) == 0:
            qmap[wd] = (0.0, 0.0)
        else:
            qmap[wd] = (float(s.quantile(cfg.q_low)), float(s.quantile(cfg.q_high)))
    return qmap


def _classify_day_from_past(daily: pd.DataFrame, day: str, cfg: ForecastConfig, qmap: Dict[int, Tuple[float, float]]) -> str:
    row = daily.loc[daily["day"] == day]
    if row.empty:
        return "mid"
    wd = int(row.iloc[0]["weekday"])
    q_low, q_high = qmap.get(wd, (np.nan, np.nan))

    pos = int(row.index[0])
    if pos <= 0:
        return "mid"

    prev = daily.iloc[max(0, pos - cfg.class_prev_days):pos]
    if len(prev) < max(2, cfg.class_prev_days):
        return "mid"

    proxy = float(prev["total_kwh"].mean())
    if np.isfinite(q_low) and proxy <= q_low:
        return "low"
    if np.isfinite(q_high) and proxy >= q_high:
        return "high"
    return "mid"


def _build_training_days(daily: pd.DataFrame, target_day: str, cfg: ForecastConfig, qmap: Dict[int, Tuple[float, float]]) -> np.ndarray:
    row = daily.loc[daily["day"] == target_day]
    if row.empty:
        return np.array([], dtype=object)
    wd = int(row.iloc[0]["weekday"])
    pos = int(row.index[0])

    start = max(0, pos - cfg.lookback_days)
    hist = daily.iloc[start:pos].copy()
    if hist.empty:
        return np.array([], dtype=object)

    target_cls = _classify_day_from_past(daily, target_day, cfg, qmap)

    # classify hist days (small, ok to loop)
    cls = []
    for d in hist["day"].to_numpy(dtype=object):
        cls.append(_classify_day_from_past(daily, str(d), cfg, qmap))
    hist["cls"] = cls

    subset = hist[(hist["weekday"] == wd) & (hist["cls"] == target_cls)]
    if len(subset) < cfg.min_train_days:
        subset = hist[hist["weekday"] == wd]

    return subset["day"].to_numpy(dtype=object)


def _build_baseline_profile(
    hist_load: np.ndarray,
    hist_day: np.ndarray,
    hist_tod_idx: np.ndarray,
    train_days: np.ndarray,
    cfg: ForecastConfig,
) -> np.ndarray:
    """
    Weighted mean per time-of-day index using recency weights per day.
    Returns profile of length n_tod (e.g. 288)
    """
    n_tod = int(round(24 * 60 / cfg.step_minutes))
    prof = np.full(n_tod, np.nan, dtype=float)

    if len(hist_load) == 0:
        return np.zeros(n_tod, dtype=float)

    if len(train_days) > 0:
        mask = np.isin(hist_day, train_days)
    else:
        mask = np.ones_like(hist_day, dtype=bool)

    y = hist_load[mask].astype(float, copy=False)
    d = hist_day[mask]
    t = hist_tod_idx[mask]

    # recency weights by day
    unique_days = np.unique(d)
    unique_days_sorted = np.array(sorted(unique_days.tolist()), dtype=object)
    day_w = _recency_weights_for_days(unique_days_sorted, cfg)
    w = np.fromiter((day_w[str(dd)] for dd in d), count=len(d), dtype=float)

    # aggregate sum(w*y) and sum(w) by tod
    wy = w * y
    sum_wy = np.zeros(n_tod, dtype=float)
    sum_w = np.zeros(n_tod, dtype=float)

    # fast bincount
    sum_wy += np.bincount(t, weights=wy, minlength=n_tod)
    sum_w += np.bincount(t, weights=w, minlength=n_tod)

    with np.errstate(divide="ignore", invalid="ignore"):
        prof = sum_wy / sum_w

    prof = _safe_fill_profile_full_grid(prof)
    return prof


def _train_ridge_residual(
    hist_load: np.ndarray,
    hist_pv: np.ndarray,
    hist_day: np.ndarray,
    hist_tod_idx: np.ndarray,
    hist_hours: np.ndarray,
    hist_minutes: np.ndarray,
    train_days: np.ndarray,
    base_profile: np.ndarray,
    cfg: ForecastConfig,
):
    if Ridge is None:
        return None

    if len(train_days) == 0:
        return None

    mask = np.isin(hist_day, train_days)
    if not mask.any():
        return None

    y = hist_load[mask].astype(float, copy=False)
    pv = hist_pv[mask].astype(float, copy=False)
    tod = hist_tod_idx[mask]
    hours = hist_hours[mask]
    minutes = hist_minutes[mask]
    days = hist_day[mask]

    # drop NaNs
    good = np.isfinite(y) & np.isfinite(pv)
    if good.sum() < 500:
        return None

    y = y[good]
    pv = pv[good]
    tod = tod[good]
    hours = hours[good]
    minutes = minutes[good]
    days = days[good]

    base = base_profile[tod]
    resid = y - base

    # lags within same day (simple: reset lag at day change)
    pv_l1 = np.roll(pv, 1)
    pv_l2 = np.roll(pv, 2)
    y_l1 = np.roll(y, 1)
    y_l2 = np.roll(y, 2)

    day_change = np.r_[True, days[1:] != days[:-1]]
    pv_l1[day_change] = np.nan
    pv_l2[day_change] = np.nan
    y_l1[day_change] = np.nan
    y_l2[day_change] = np.nan

    # hour sin/cos
    minutes_of_day = (hours * 60 + minutes).astype(float)
    ang = 2.0 * np.pi * (minutes_of_day / (24.0 * 60.0))
    hsin = np.sin(ang)
    hcos = np.cos(ang)

    X = np.column_stack([pv, pv_l1, pv_l2, y_l1, y_l2, hsin, hcos])

    # weights: tod importance * recency
    w_tod = _sample_weight_hours(hours.astype(int), cfg)
    unique_days = np.unique(days)
    unique_days_sorted = np.array(sorted(unique_days.tolist()), dtype=object)
    day_w = _recency_weights_for_days(unique_days_sorted, cfg)
    w_rec = np.fromiter((day_w[str(dd)] for dd in days), count=len(days), dtype=float)
    w = w_tod * w_rec

    # final drop rows with NaNs in X or resid
    good2 = np.all(np.isfinite(X), axis=1) & np.isfinite(resid) & np.isfinite(w)
    if good2.sum() < 500:
        return None

    X2 = X[good2]
    r2 = resid[good2]
    w2 = w[good2]

    model = Ridge(alpha=cfg.ridge_alpha, fit_intercept=True, random_state=0)
    model.fit(X2, r2, sample_weight=w2)
    return model


def _get_day_model(
    prep_key: Tuple[int, int, int],
    prep: Dict[str, Any],
    target_day: str,
    curr_idx: int,
    load_col: str,
    pv_col: str,
    cfg: ForecastConfig,
) -> Dict[str, Any]:
    """
    Build or fetch (cached) model artifacts for a target day:
    - baseline profile (np array length 288)
    - ridge model (optional)
    - fallback weekday/tod profiles (optional)
    """
    cache_key = (prep_key, target_day)
    if cache_key in _DAY_MODEL_CACHE:
        return _DAY_MODEL_CACHE[cache_key]

    df2 = prep["df"]

    # Use history strictly before target_day to train once per day (stable + fast)
    day_str = prep["day_str"]
    hist_mask = (day_str < target_day)

    # Also ensure we only use data that is actually before curr_idx (causality)
    # If curr_idx is inside target_day, hist_mask already excludes target_day.
    # For safety if target_day could be <= current history day, we enforce iloc<curr_idx.
    iloc_mask = np.zeros(len(df2), dtype=bool)
    iloc_mask[:max(0, curr_idx)] = True
    hist_mask = hist_mask & iloc_mask

    hist_load = df2[load_col].to_numpy(dtype=float)[hist_mask]
    hist_pv = df2[pv_col].to_numpy(dtype=float)[hist_mask]
    hist_day = prep["day_str"][hist_mask]
    hist_tod_idx = prep["tod_idx"][hist_mask]
    hist_hours = prep["hours"][hist_mask]
    hist_minutes = prep["minutes"][hist_mask]
    hist_weekday = prep["weekday"][hist_mask]

    # If not enough history -> store minimal fallback artifacts
    if len(hist_load) < cfg.min_history_samples:
        artifacts = {
            "base_profile": None,
            "ridge": None,
            "fallback_ready": True,
            "hist_load": hist_load,
            "hist_day": hist_day,
            "hist_tod_idx": hist_tod_idx,
            "hist_weekday": hist_weekday,
        }
        _DAY_MODEL_CACHE[cache_key] = artifacts
        return artifacts

    # daily table for training-day selection
    daily = _daily_kwh_from_history(hist_load, hist_day, cfg)
    if daily.empty:
        artifacts = {
            "base_profile": None,
            "ridge": None,
            "fallback_ready": True,
            "hist_load": hist_load,
            "hist_day": hist_day,
            "hist_tod_idx": hist_tod_idx,
            "hist_weekday": hist_weekday,
        }
        _DAY_MODEL_CACHE[cache_key] = artifacts
        return artifacts

    qmap = _weekday_quantiles(daily, cfg)
    train_days = _build_training_days(daily, target_day, cfg, qmap)

    base_profile = _build_baseline_profile(
        hist_load=hist_load,
        hist_day=hist_day,
        hist_tod_idx=hist_tod_idx,
        train_days=train_days,
        cfg=cfg,
    )

    ridge_model = _train_ridge_residual(
        hist_load=hist_load,
        hist_pv=hist_pv,
        hist_day=hist_day,
        hist_tod_idx=hist_tod_idx,
        hist_hours=hist_hours,
        hist_minutes=hist_minutes,
        train_days=train_days,
        base_profile=base_profile,
        cfg=cfg,
    )

    artifacts = {
        "base_profile": base_profile,
        "ridge": ridge_model,
        "fallback_ready": False,
        # keep minimal fallback buffers (optional)
        "hist_load": hist_load,
        "hist_day": hist_day,
        "hist_tod_idx": hist_tod_idx,
        "hist_weekday": hist_weekday,
    }
    _DAY_MODEL_CACHE[cache_key] = artifacts
    return artifacts


# =============================================================================
# Fallback forecast (fast)
# =============================================================================

def _weekday_profile_forecast_fast(
    hist_load: np.ndarray,
    hist_weekday: np.ndarray,
    hist_tod_idx: np.ndarray,
    fut_weekday: np.ndarray,
    fut_tod_idx: np.ndarray,
    cfg: ForecastConfig,
) -> np.ndarray:
    """
    Fast fallback:
    - mean by (weekday, tod)
    - else mean by tod
    - else global mean
    """
    n_tod = int(round(24 * 60 / cfg.step_minutes))
    out = np.zeros(len(fut_tod_idx), dtype=float)

    if len(hist_load) == 0 or not np.isfinite(hist_load).any():
        return out

    good = np.isfinite(hist_load)
    y = hist_load[good]
    wd = hist_weekday[good]
    tod = hist_tod_idx[good]

    global_mean = float(np.nanmean(y))
    if not np.isfinite(global_mean):
        global_mean = 0.0

    # mean by tod
    sum_y_t = np.bincount(tod, weights=y, minlength=n_tod).astype(float)
    cnt_t = np.bincount(tod, minlength=n_tod).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_t = sum_y_t / cnt_t
    mean_t = _safe_fill_profile_full_grid(mean_t)
    mean_t = np.where(np.isfinite(mean_t), mean_t, global_mean)

    # mean by (wd,tod): build 7 x n_tod table
    sum_wd = np.zeros((7, n_tod), dtype=float)
    cnt_wd = np.zeros((7, n_tod), dtype=float)

    # accumulate
    for wdi, todi, yi in zip(wd, tod, y):
        if 0 <= wdi <= 6:
            sum_wd[wdi, todi] += yi
            cnt_wd[wdi, todi] += 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_wd = sum_wd / cnt_wd

    # predict
    for i, (wdi, todi) in enumerate(zip(fut_weekday, fut_tod_idx)):
        v = mean_wd[wdi, todi] if 0 <= wdi <= 6 else np.nan
        if not np.isfinite(v):
            v = mean_t[todi] if 0 <= todi < n_tod else global_mean
        if not np.isfinite(v):
            v = global_mean
        out[i] = float(v)

    return out


# =============================================================================
# Public API (same signature)
# =============================================================================

def predict_load_horizon(
    df: pd.DataFrame,
    curr_idx: int,
    horizon_size: int,
    history_size: int,
    load_col: str,
    pv_col: str,
    mode: str = "auto",
    time_col: Optional[str] = None,
    cfg: Optional[ForecastConfig] = None,
) -> np.ndarray:
    cfg = cfg or ForecastConfig()

    if load_col not in df.columns or pv_col not in df.columns:
        raise KeyError(f"df must contain columns '{load_col}' and '{pv_col}'")

    # prepare/cached arrays
    prep_key, prep = _prepare_df(df, load_col=load_col, pv_col=pv_col, cfg=cfg, time_col=time_col)
    df2 = prep["df"]

    # curr_idx positional semantics (keep stable)
    n = len(df2)
    if curr_idx < 0:
        curr_idx = 0
    if curr_idx > n:
        curr_idx = n

    # history window (for fallback / y_lags)
    start_idx = max(0, curr_idx - int(history_size))
    hist_len = curr_idx - start_idx

    # future indices
    end_idx = min(n, curr_idx + int(horizon_size))
    fut_len = end_idx - curr_idx

    # Build future tod/weekdays; if horizon exceeds df length -> extend synthetic
    fut_tod_idx = prep["tod_idx"][curr_idx:end_idx].copy()
    fut_weekday = prep["weekday"][curr_idx:end_idx].copy()
    fut_hours = prep["hours"][curr_idx:end_idx].copy()
    fut_minutes = prep["minutes"][curr_idx:end_idx].copy()

    if fut_len < horizon_size:
        missing = horizon_size - fut_len
        # extend using periodic day-time stepping
        # We assume constant 5-min steps; build continued tod indices.
        last_tod = int(fut_tod_idx[-1]) if fut_len > 0 else int(prep["tod_idx"][max(0, curr_idx - 1)])
        ext_tod = (last_tod + np.arange(1, missing + 1)) % int(round(24 * 60 / cfg.step_minutes))
        fut_tod_idx = np.concatenate([fut_tod_idx, ext_tod.astype(int)], axis=0)

        # weekday extension: roll forward when wrapping day boundary
        # Approx: infer from current timestamp progression; good enough for MPC tail.
        last_wd = int(fut_weekday[-1]) if fut_len > 0 else int(prep["weekday"][max(0, curr_idx - 1)])
        wd = []
        cur = last_wd
        # if ext_tod wraps from high->low, increment weekday
        prev = last_tod
        for t in ext_tod:
            if t < prev:  # day wrapped
                cur = (cur + 1) % 7
            wd.append(cur)
            prev = t
        fut_weekday = np.concatenate([fut_weekday, np.array(wd, dtype=int)], axis=0)

        # hours/minutes are only used for sin/cos; derive from tod index
        # tod_idx * step_minutes => minutes of day
        ext_mod = ext_tod.astype(int) * cfg.step_minutes
        ext_h = (ext_mod // 60).astype(int)
        ext_m = (ext_mod % 60).astype(int)
        fut_hours = np.concatenate([fut_hours, ext_h], axis=0)
        fut_minutes = np.concatenate([fut_minutes, ext_m], axis=0)

    # Warmup fallback (fast)
    if mode == "fallback" or hist_len < cfg.min_history_samples:
        hist_load = df2[load_col].to_numpy(dtype=float)[start_idx:curr_idx]
        hist_weekday = prep["weekday"][start_idx:curr_idx]
        hist_tod = prep["tod_idx"][start_idx:curr_idx]
        pred = _weekday_profile_forecast_fast(
            hist_load=hist_load,
            hist_weekday=hist_weekday,
            hist_tod_idx=hist_tod,
            fut_weekday=fut_weekday,
            fut_tod_idx=fut_tod_idx,
            cfg=cfg,
        )
        return _finalize_pred(pred[:horizon_size], cfg)

    # Full mode: build/get model for target_day
    target_day = str(prep["day_str"][min(curr_idx, n - 1)]) if n > 0 else "1970-01-01"
    artifacts = _get_day_model(
        prep_key=prep_key,
        prep=prep,
        target_day=target_day,
        curr_idx=curr_idx,
        load_col=load_col,
        pv_col=pv_col,
        cfg=cfg,
    )

    # If model couldn't be built -> fallback
    if artifacts.get("fallback_ready", False) or artifacts.get("base_profile") is None:
        hist_load = df2[load_col].to_numpy(dtype=float)[start_idx:curr_idx]
        hist_weekday = prep["weekday"][start_idx:curr_idx]
        hist_tod = prep["tod_idx"][start_idx:curr_idx]
        pred = _weekday_profile_forecast_fast(
            hist_load=hist_load,
            hist_weekday=hist_weekday,
            hist_tod_idx=hist_tod,
            fut_weekday=fut_weekday,
            fut_tod_idx=fut_tod_idx,
            cfg=cfg,
        )
        return _finalize_pred(pred[:horizon_size], cfg)

    base_profile = artifacts["base_profile"]
    base_pred = base_profile[fut_tod_idx[:horizon_size]].astype(float, copy=False)

    ridge_model = artifacts.get("ridge", None)
    if ridge_model is None:
        return _finalize_pred(base_pred[:horizon_size], cfg)

    # Build horizon features (vectorized)
    # pv in horizon: if df ends, pad with last known pv or zeros
    pv_arr_full = df2[pv_col].to_numpy(dtype=float)
    pv_fut = pv_arr_full[curr_idx:min(n, curr_idx + horizon_size)]
    if len(pv_fut) < horizon_size:
        pad = horizon_size - len(pv_fut)
        last = float(pv_fut[-1]) if len(pv_fut) > 0 and np.isfinite(pv_fut[-1]) else 0.0
        pv_fut = np.concatenate([pv_fut, np.full(pad, last, dtype=float)], axis=0)

    pv_l1 = np.roll(pv_fut, 1); pv_l1[0] = np.nan
    pv_l2 = np.roll(pv_fut, 2); pv_l2[:2] = np.nan

    # load lags: use last observed values (cheap)
    load_arr_full = df2[load_col].to_numpy(dtype=float)
    y_l1 = float(load_arr_full[curr_idx - 1]) if curr_idx - 1 >= 0 and np.isfinite(load_arr_full[curr_idx - 1]) else 0.0
    y_l2 = float(load_arr_full[curr_idx - 2]) if curr_idx - 2 >= 0 and np.isfinite(load_arr_full[curr_idx - 2]) else y_l1
    y_l1_vec = np.full(horizon_size, y_l1, dtype=float)
    y_l2_vec = np.full(horizon_size, y_l2, dtype=float)

    minutes_of_day = (fut_hours[:horizon_size].astype(float) * 60.0 + fut_minutes[:horizon_size].astype(float))
    ang = 2.0 * np.pi * (minutes_of_day / (24.0 * 60.0))
    hsin = np.sin(ang)
    hcos = np.cos(ang)

    Xf = np.column_stack([pv_fut, pv_l1, pv_l2, y_l1_vec, y_l2_vec, hsin, hcos])

    resid = np.zeros(horizon_size, dtype=float)
    valid = np.all(np.isfinite(Xf), axis=1)
    if valid.any():
        resid[valid] = ridge_model.predict(Xf[valid])

    pred = base_pred[:horizon_size] + resid
    return _finalize_pred(pred, cfg)
