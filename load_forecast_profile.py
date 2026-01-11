"""
load_forecast_profile.py

Robuste Last-Prognose für MPC-Horizon.

Ziel:
- pipeline.py ruft nur predict_load_horizon(...) auf
- Dieses Modul ist robust gegen:
  - RangeIndex (kein DatetimeIndex)
  - unterschiedliche Zeitspaltennamen (time/timestamp/...)
  - String-Zeitstempel mit gemischten Zeitzonen (+02:00/+01:00)  -> parse with utc=True
  - DST / Sommerzeit-Winterzeit Wechsel -> wird in lokale Zeit (Europe/Berlin) umgerechnet
  - fehlende Time-of-Day Keys -> Forecast darf NIE NaN enthalten

Forecast-Logik (bewusst pragmatisch für MPC):
(1) Warmup/Fallback: Wochenprofil (weekday + time-of-day)
(2) Full: Baseline-Profil (recency weighted) + PV-informiertes Residualmodell (Ridge)

Benötigt:
- numpy, pandas
- scikit-learn (optional; Ridge), sonst läuft Fallback/Baseline-only

Wichtig:
- pipeline gibt curr_idx als Zeilenposition (iloc-Index) rein -> wir behalten das!
- Wir setzen intern nur einen DatetimeIndex, damit weekday/tod sauber funktionieren.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except Exception:
    Ridge = None


# =============================================================================
# Helpers: Time handling
# =============================================================================

_CANDIDATE_TIME_COLS = [
    "time", "timestamp", "datetime", "date", "ts",
    "Time", "Zeit", "Zeitstempel"
]


def _try_parse_datetime_series_utc(s: pd.Series) -> pd.Series:
    """
    Parse a series to datetime64[ns, UTC].
    Handles mixed offsets (+02:00/+01:00) robustly via utc=True.
    """
    return pd.to_datetime(s, utc=True, errors="coerce")


def infer_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to infer a time column by:
    1) checking common names
    2) then scanning object-like columns and using parse success rate
    """
    # direct name hits first
    for c in _CANDIDATE_TIME_COLS:
        if c in df.columns:
            dt = _try_parse_datetime_series_utc(df[c])
            if dt.notna().mean() > 0.8:
                return c

    # otherwise, try any object-like column
    for c in df.columns:
        if c == df.index.name:
            continue
        if df[c].dtype == "O" or "datetime" in str(df[c].dtype).lower():
            dt = _try_parse_datetime_series_utc(df[c])
            if dt.notna().mean() > 0.8:
                return c

    return None


def ensure_datetime_index(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    default_step_minutes: int = 5,
    tz_local: str = "Europe/Berlin",
) -> pd.DataFrame:
    """
    Ensure df has a *monotonic* DatetimeIndex in LOCAL time (naive).

    - If already DatetimeIndex: just sort.
    - Else parse time_col (or infer) using utc=True, convert to tz_local,
      then drop tz info (naive local timestamps).
    - If no time column is usable: create synthetic 5-min index.

    Why local naive?
    - because weekday/time-of-day logic should follow local clock time
      and stay stable across DST transitions.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        out = df
        if not out.index.is_monotonic_increasing:
            out = out.sort_index()
        return out

    out = df.copy()

    col = time_col or infer_time_column(out)
    if col is not None:
        dt_utc = _try_parse_datetime_series_utc(out[col])
        if dt_utc.notna().any():
            # drop rows with unparseable timestamps
            out = out.loc[dt_utc.notna()].copy()
            dt_utc = dt_utc.loc[dt_utc.notna()]

            # convert UTC -> local tz -> remove tz (naive local timestamps)
            dt_local = dt_utc.dt.tz_convert(tz_local).dt.tz_localize(None)

            out.index = pd.DatetimeIndex(dt_local.values, name="time")
            out = out.sort_index()
            return out

    # Hard fallback: synthetic timeline so the model still runs
    start = pd.Timestamp("1970-01-01 00:00:00")
    out.index = pd.date_range(
        start=start,
        periods=len(out),
        freq=pd.Timedelta(minutes=default_step_minutes),
        name="time",
    )
    return out


def _full_tod_grid(cfg_step_minutes: int) -> List[str]:
    """
    Create the canonical list of time-of-day keys for one day:
    ["00:00", "00:05", ..., "23:55"] for 5-min.
    """
    # 24h * 60 / step
    n = int(round(24 * 60 / cfg_step_minutes))
    base = pd.Timestamp("2000-01-01 00:00:00")
    idx = pd.date_range(base, periods=n, freq=pd.Timedelta(minutes=cfg_step_minutes))
    return [t.strftime("%H:%M") for t in idx]


# =============================================================================
# Forecast config
# =============================================================================

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

    # safety
    clip_nonnegative: bool = True


def time_of_day_key(ts: pd.Timestamp) -> str:
    return ts.strftime("%H:%M")


def sample_weight(ts: pd.Timestamp, cfg: ForecastConfig) -> float:
    """
    Give some time-of-day more importance in training.
    """
    h = ts.hour
    if cfg.morning[0] <= h < cfg.morning[1]:
        return cfg.w_morning
    if cfg.evening[0] <= h < cfg.evening[1]:
        return cfg.w_evening
    return cfg.w_default


def recency_weight(day_dist: int, cfg: ForecastConfig) -> float:
    """
    Exponential half-life weighting: dist=0 -> 1.0, dist=hl -> 0.5
    """
    hl = max(cfg.recency_half_life_days, 1e-6)
    lam = np.log(2.0) / hl
    return float(np.exp(-lam * day_dist))


def daily_energy_kwh(day_df: pd.DataFrame, load_col: str, cfg: ForecastConfig) -> float:
    """
    Convert power samples [kW] to energy [kWh] for the day.
    """
    dt_h = (cfg.step_minutes / 60.0)
    return float(np.nansum(day_df[load_col].to_numpy(dtype=float)) * dt_h)


def build_daily_table(df: pd.DataFrame, load_col: str, cfg: ForecastConfig) -> pd.DataFrame:
    """
    Build one row per day from df (must have DatetimeIndex).
    """
    tmp = df.copy()
    tmp["_day"] = tmp.index.strftime("%Y-%m-%d")

    rows = []
    for day, ddf in tmp.groupby("_day", sort=True):
        if ddf.empty:
            continue
        rows.append({
            "day": day,
            "weekday": int(pd.to_datetime(day).weekday()),
            "total_kwh": daily_energy_kwh(ddf, load_col, cfg),
        })
    if not rows:
        return pd.DataFrame(columns=["day", "weekday", "total_kwh"])
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def weekday_quantiles(daily: pd.DataFrame, cfg: ForecastConfig) -> Dict[int, Tuple[float, float]]:
    """
    Compute per-weekday energy quantiles to classify days (low/mid/high proxy).
    """
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


def classify_day_from_past(
    daily: pd.DataFrame,
    day: str,
    cfg: ForecastConfig,
    qmap: Dict[int, Tuple[float, float]],
) -> str:
    """
    Classify a day as low/mid/high based on mean of previous days' energy.
    Uses only past days relative to 'day' within the 'daily' table (sorted).
    """
    row = daily.loc[daily["day"] == day]
    if row.empty:
        return "mid"
    wd = int(row.iloc[0]["weekday"])
    q_low, q_high = qmap.get(wd, (np.nan, np.nan))

    # position in the daily table
    idx = int(row.index[0])
    if idx <= 0:
        return "mid"

    prev = daily.iloc[max(0, idx - cfg.class_prev_days): idx]
    if len(prev) < max(2, cfg.class_prev_days):
        return "mid"

    proxy = float(prev["total_kwh"].mean())
    if np.isfinite(q_low) and proxy <= q_low:
        return "low"
    if np.isfinite(q_high) and proxy >= q_high:
        return "high"
    return "mid"


def build_training_days(
    daily: pd.DataFrame,
    target_day: str,
    cfg: ForecastConfig,
    qmap: Dict[int, Tuple[float, float]],
) -> List[str]:
    """
    Choose training days in lookback window:
    - same weekday
    - similar class (low/mid/high) based on past-days proxy
    - if too few: use same weekday only
    """
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


# =============================================================================
# Profiles / Models
# =============================================================================

def _safe_fill_profile_to_full_grid(profile: pd.Series, cfg: ForecastConfig) -> pd.Series:
    """
    Ensure the profile has all HH:MM keys for the day. Fill missing values robustly.

    This is the key fix for your observed behavior: missing keys caused NaNs in forecast.
    """
    grid = _full_tod_grid(cfg.step_minutes)

    # Reindex to full grid
    prof = profile.reindex(grid)

    # Fill strategy:
    # 1) interpolate in time order (works well for smooth diurnal curves)
    # 2) ffill/bfill for edges
    # 3) fallback to global mean or 0
    prof = prof.astype(float)
    prof = prof.interpolate(limit_direction="both")
    prof = prof.ffill().bfill()

    m = float(np.nanmean(prof.to_numpy(dtype=float))) if len(prof) else 0.0
    if not np.isfinite(m):
        m = 0.0
    prof = prof.fillna(m)

    prof.name = profile.name if profile.name else "profile"
    return prof


def weighted_baseline_profile(
    df: pd.DataFrame,
    train_days: List[str],
    target_day: str,
    load_col: str,
    cfg: ForecastConfig,
) -> pd.Series:
    """
    Baseline profile: weighted mean load per time-of-day (HH:MM).
    Weighted by recency (newer days > older days).

    Avoids groupby.apply to prevent pandas FutureWarning and to be faster.
    """
    if train_days:
        hist = df[df["_day"].isin(train_days)].copy()
    else:
        hist = df[df["_day"] < target_day].copy()

    if hist.empty:
        return pd.Series(dtype=float, name="base_kw")

    hist["_tod"] = hist.index.map(time_of_day_key)

    # day distances (recency)
    unique_days = sorted(hist["_day"].unique())
    day2dist = {d: (len(unique_days) - 1 - i) for i, d in enumerate(unique_days)}
    w = hist["_day"].map(lambda d: recency_weight(day2dist[d], cfg)).to_numpy(dtype=float)
    y = hist[load_col].to_numpy(dtype=float)

    # weighted mean per tod: sum(w*y) / sum(w)
    tmp = pd.DataFrame({
        "_tod": hist["_tod"].to_numpy(),
        "_wy": w * y,
        "_w": w,
    })
    agg = tmp.groupby("_tod", sort=False).sum(numeric_only=True)
    prof = (agg["_wy"] / agg["_w"]).astype(float)
    prof.name = "base_kw"

    # Ensure full grid (no missing keys -> no NaNs later)
    prof = _safe_fill_profile_to_full_grid(prof, cfg)
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
    Features:
      pv_now, pv_l1, pv_l2, y_l1, y_l2, hour_sin, hour_cos
    Weighted by time-of-day importance * recency.

    If sklearn isn't available or data insufficient -> returns None.
    """
    if Ridge is None:
        return None

    hist = df[df["_day"].isin(train_days)].copy()
    if hist.empty:
        return None

    hist = hist.dropna(subset=[load_col, pv_col]).copy()
    if hist.empty:
        return None

    hist["_tod"] = hist.index.map(time_of_day_key)

    # base is already full-grid filled, so mapping should not yield NaN
    hist["_base"] = hist["_tod"].map(base).to_numpy(dtype=float)
    hist["_resid"] = hist[load_col].to_numpy(dtype=float) - hist["_base"]

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
    w_tod = np.array([sample_weight(ts, cfg) for ts in hist.index], dtype=float)

    uniq_days = sorted(hist["_day"].unique())
    day2dist = {d: (len(uniq_days) - 1 - i) for i, d in enumerate(uniq_days)}
    w_rec = hist["_day"].map(lambda d: recency_weight(day2dist[d], cfg)).to_numpy(dtype=float)
    w = w_tod * w_rec

    cols = [pv_col, "_pv_l1", "_pv_l2", "_y_l1", "_y_l2", "_hsin", "_hcos"]
    hist2 = hist.dropna(subset=cols + ["_resid"]).copy()
    if len(hist2) < 500:
        return None

    X = hist2[cols].to_numpy(dtype=float)
    y = hist2["_resid"].to_numpy(dtype=float)
    ww = w[hist2.index.get_indexer(hist2.index)]  # aligned by row order

    # safety: if ww shape mismatch, just use ones
    if ww.shape[0] != X.shape[0]:
        ww = np.ones(X.shape[0], dtype=float)

    model = Ridge(alpha=cfg.ridge_alpha, fit_intercept=True, random_state=0)
    model.fit(X, y, sample_weight=ww)
    return model


# =============================================================================
# Public API
# =============================================================================

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

    Behavior summary:
    - Always returns finite numbers (no NaN/inf).
    - If not enough history: weekday profile fallback
    - Otherwise: baseline profile + (optional) ridge residual correction
    """
    cfg = cfg or ForecastConfig()

    if load_col not in df.columns or pv_col not in df.columns:
        raise KeyError(f"df must contain columns '{load_col}' and '{pv_col}'")

    # (1) Ensure datetime index in local time (naive)
    df2 = ensure_datetime_index(
        df,
        time_col=time_col,
        default_step_minutes=cfg.step_minutes,
        tz_local="Europe/Berlin",
    )

    # (2) Keep curr_idx as positional index (pipeline semantics)
    if curr_idx < 0:
        curr_idx = 0
    if curr_idx > len(df2):
        curr_idx = len(df2)

    start_idx = max(0, curr_idx - history_size)

    df_hist = df2.iloc[start_idx:curr_idx].copy()
    df_fut = df2.iloc[curr_idx:curr_idx + horizon_size].copy()

    # If horizon exceeds df length -> pad future index continuation (for MPC end)
    if len(df_fut) < horizon_size:
        if len(df2) >= 2:
            step = df2.index[-1] - df2.index[-2]
            if not isinstance(step, pd.Timedelta) or step <= pd.Timedelta(0):
                step = pd.Timedelta(minutes=cfg.step_minutes)
        else:
            step = pd.Timedelta(minutes=cfg.step_minutes)

        last_ts = df2.index[min(curr_idx, len(df2) - 1)] if len(df2) else pd.Timestamp("2000-01-01")
        missing = horizon_size - len(df_fut)
        extra_idx = pd.date_range(start=last_ts + step, periods=missing, freq=step, name="time")
        extra = pd.DataFrame(index=extra_idx, columns=df2.columns)
        df_fut = pd.concat([df_fut, extra], axis=0)

    # helper day keys
    df_hist["_day"] = df_hist.index.strftime("%Y-%m-%d")
    df_fut["_day"] = df_fut.index.strftime("%Y-%m-%d")

    # (3) Warmup fallback
    if mode == "fallback" or len(df_hist) < cfg.min_history_samples:
        pred = _weekday_profile_forecast(df_hist, df_fut, load_col, cfg)
        return _finalize_pred(pred, cfg)

    # (4) Full mode: select training days
    daily = build_daily_table(df_hist, load_col, cfg)
    if daily.empty:
        pred = _weekday_profile_forecast(df_hist, df_fut, load_col, cfg)
        return _finalize_pred(pred, cfg)

    qmap = weekday_quantiles(daily, cfg)
    target_day = str(df_fut["_day"].iloc[0])
    train_days = build_training_days(daily, target_day, cfg, qmap)

    # baseline profile
    base = weighted_baseline_profile(df_hist, train_days, target_day, load_col, cfg)
    if base.empty:
        pred = _weekday_profile_forecast(df_hist, df_fut, load_col, cfg)
        return _finalize_pred(pred, cfg)

    # baseline prediction for horizon
    df_fut["_tod"] = df_fut.index.map(time_of_day_key)
    base_pred = df_fut["_tod"].map(base).to_numpy(dtype=float)

    # safety: base is full grid, but still guard
    if np.any(~np.isfinite(base_pred)):
        # fill NaNs with finite mean
        m = float(np.nanmean(base_pred))
        if not np.isfinite(m):
            m = 0.0
        base_pred = np.nan_to_num(base_pred, nan=m, posinf=m, neginf=0.0)

    # ridge residual correction (PV-informed)
    model = build_residual_ridge(df_hist, train_days, base, load_col, pv_col, cfg)
    if model is None:
        return _finalize_pred(base_pred[:horizon_size], cfg)

    # Build horizon features
    fut = df_fut.copy()

    # PV lags in future: only makes sense if you have future PV series available
    fut["_pv_l1"] = fut.groupby("_day")[pv_col].shift(1)
    fut["_pv_l2"] = fut.groupby("_day")[pv_col].shift(2)

    # For load lags (future), we DON'T know it. Use last observed values as constants.
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

    pred = base_pred[:horizon_size] + resid[:horizon_size]
    return _finalize_pred(pred, cfg)


# =============================================================================
# Fallback forecast
# =============================================================================

def _weekday_profile_forecast(
    df_hist: pd.DataFrame,
    df_fut: pd.DataFrame,
    load_col: str,
    cfg: ForecastConfig
) -> np.ndarray:
    """
    Simple fallback:
      forecast(t) = mean load of same weekday & time-of-day in history,
      else mean by time-of-day,
      else global mean.

    Always produces finite values.
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

    # ensure tod fallback is full grid to avoid holes
    prof_tod = _safe_fill_profile_to_full_grid(prof_tod, cfg)

    global_mean = float(hist[load_col].mean())
    if not np.isfinite(global_mean):
        global_mean = 0.0

    out = np.empty(len(fut), dtype=float)
    for i, (wd, tod) in enumerate(zip(fut["_wd"].to_numpy(), fut["_tod"].to_numpy())):
        v = prof_wd.get((wd, tod), np.nan)
        if not np.isfinite(v):
            v = prof_tod.get(tod, np.nan)
        if not np.isfinite(v):
            v = global_mean
        out[i] = float(v)

    return out


# =============================================================================
# Safety finalize
# =============================================================================

def _finalize_pred(pred: np.ndarray, cfg: ForecastConfig) -> np.ndarray:
    """
    Make absolutely sure we return a clean vector for MPC:
    - dtype float
    - finite values only
    - optionally clip to nonnegative
    """
    arr = np.asarray(pred, dtype=float)

    if np.any(~np.isfinite(arr)):
        m = float(np.nanmean(arr))
        if not np.isfinite(m):
            m = 0.0
        arr = np.nan_to_num(arr, nan=m, posinf=m, neginf=0.0)

    if cfg.clip_nonnegative:
        arr = np.clip(arr, 0.0, None)

    return arr
