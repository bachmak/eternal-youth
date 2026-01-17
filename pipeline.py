# pipeline.py
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# Set the style globally.
# 'science' = base scientific style
# 'ieee' = specific overrides for IEEE (font sizes, column widths)
plt.style.use(['science', 'ieee'])

from load_forecast_profile import predict_load_horizon
from data_parser import batch_collect

import optimizer as optimizer_mod
from optimizer import MPCOptimizer


# ============================================================
# Config
# ============================================================
class Config:
    DT = 5.0 / 60.0
    SAMPLES_PER_DAY = 288
    HISTORY_SAMPLES = 60 * 288
    HORIZON_SAMPLES = 288

    # Re-solve interval (in steps). We apply an N-step plan block of this length.
    MPC_INTERVAL_STEPS = 6

    CAPACITY = 10.0
    P_CH_MAX = 5.0
    P_DIS_MAX = 5.0
    ETA_CH = 0.95
    ETA_DIS = 0.95
    COST_WEAR = 0.02

    # Default penalty weights (will be overridden by preset)
    COST_SOC_HOLDING = 0.25
    W_BELOW_TARGET = 0.10
    W_ABOVE_TARGET = 0.15
    W_HIGH_85 = 2.0
    W_HIGH_95 = 8.0

    SOC_MIN_HARD = 0.05
    SOC_MAX_HARD = 1.00
    SOC_TARGET = 0.20

    PRICE_BUY = 0.3258
    PRICE_SELL = 0.0794

    SOH_EOL = 0.80
    BATTERY_REPLACEMENT_EUR = 4000.0
    SOH_BUDGET = 1.0 - SOH_EOL
    EUR_PER_SOH_PERCENT = BATTERY_REPLACEMENT_EUR / (SOH_BUDGET * 100.0)

    CAL_DAYS = 160.0
    CAL_SOC = 1.00
    CAL_LOSS_PCT = 9.5


# ============================================================
# Penalty presets
# ============================================================
PENALTY_PRESETS = {
    "P0_energy_only": {
        "SOC_TARGET": 0.20,
        "COST_SOC_HOLDING": 0.0,
        "W_BELOW_TARGET": 0.0,
        "W_ABOVE_TARGET": 0.0,
        "W_HIGH_85": 0.0,
        "W_HIGH_95": 0.0,
    },
    "P1_high_ultra_low": {
        "SOC_TARGET": 0.20,
        "COST_SOC_HOLDING": 0.10,
        "W_BELOW_TARGET": 0.0,
        "W_ABOVE_TARGET": 0.0,
        "W_HIGH_85": 0.5,
        "W_HIGH_95": 2.0,
    },
    "P2_high_low": {
        "SOC_TARGET": 0.20,
        "COST_SOC_HOLDING": 0.15,
        "W_BELOW_TARGET": 0.0,
        "W_ABOVE_TARGET": 0.0,
        "W_HIGH_85": 1.0,
        "W_HIGH_95": 4.0,
    },
    "P3_high_mid": {
        "SOC_TARGET": 0.20,
        "COST_SOC_HOLDING": 0.20,
        "W_BELOW_TARGET": 0.0,
        "W_ABOVE_TARGET": 0.0,
        "W_HIGH_85": 1.5,
        "W_HIGH_95": 6.0,
    },
    "P4_high_balanced": {
        "SOC_TARGET": 0.20,
        "COST_SOC_HOLDING": 0.22,
        "W_BELOW_TARGET": 0.05,
        "W_ABOVE_TARGET": 0.08,
        "W_HIGH_85": 1.8,
        "W_HIGH_95": 7.0,
    },
    "P5_high_strong": {
        "SOC_TARGET": 0.20,
        "COST_SOC_HOLDING": 0.30,
        "W_BELOW_TARGET": 0.10,
        "W_ABOVE_TARGET": 0.12,
        "W_HIGH_85": 2.5,
        "W_HIGH_95": 9.0,
    },
    "P6_target20_balanced": {
        "SOC_TARGET": 0.20,
        "COST_SOC_HOLDING": 0.20,
        "W_BELOW_TARGET": 0.10,
        "W_ABOVE_TARGET": 0.12,
        "W_HIGH_85": 1.5,
        "W_HIGH_95": 6.0,
    },
    "P7_target20_maxlife": {
        "SOC_TARGET": 0.20,
        "COST_SOC_HOLDING": 0.25,
        "W_BELOW_TARGET": 0.10,
        "W_ABOVE_TARGET": 0.15,
        "W_HIGH_85": 2.0,
        "W_HIGH_95": 8.0,
    },
    "P8_target50_balanced": {
        "SOC_TARGET": 0.50,
        "COST_SOC_HOLDING": 0.20,
        "W_BELOW_TARGET": 0.10,
        "W_ABOVE_TARGET": 0.12,
        "W_HIGH_85": 1.5,
        "W_HIGH_95": 6.0,
    },
    "P9_target50_strong": {
        "SOC_TARGET": 0.50,
        "COST_SOC_HOLDING": 0.25,
        "W_BELOW_TARGET": 0.12,
        "W_ABOVE_TARGET": 0.15,
        "W_HIGH_85": 2.0,
        "W_HIGH_95": 8.0,
    },
    "P10_target50_maxlife": {
        "SOC_TARGET": 0.50,
        "COST_SOC_HOLDING": 0.30,
        "W_BELOW_TARGET": 0.15,
        "W_ABOVE_TARGET": 0.18,
        "W_HIGH_85": 2.5,
        "W_HIGH_95": 9.0,
    },
}


def apply_penalty_preset(cfg: Config, preset_id: str) -> None:
    if preset_id not in PENALTY_PRESETS:
        raise KeyError(f"Unknown preset_id '{preset_id}'. Available: {list(PENALTY_PRESETS.keys())}")

    p = PENALTY_PRESETS[preset_id]
    cfg.SOC_TARGET = float(p["SOC_TARGET"])
    cfg.COST_SOC_HOLDING = float(p["COST_SOC_HOLDING"])
    cfg.W_BELOW_TARGET = float(p["W_BELOW_TARGET"])
    cfg.W_ABOVE_TARGET = float(p["W_ABOVE_TARGET"])
    cfg.W_HIGH_85 = float(p["W_HIGH_85"])
    cfg.W_HIGH_95 = float(p["W_HIGH_95"])

    print(f"[MPC] Using penalty preset: {preset_id} -> {p}")


# ============================================================
# IO helpers
# ============================================================
def ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
        if t.notna().mean() >= 0.8:
            df["time"] = t.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
            return df

    candidates = ["timestamp", "datetime", "date", "ts", "Time", "Zeit", "Zeitstempel"]
    for c in candidates:
        if c in df.columns:
            t = pd.to_datetime(df[c], utc=True, errors="coerce")
            if t.notna().mean() >= 0.8:
                df["time"] = t.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
                return df

    start = pd.Timestamp("1970-01-01 00:00:00")
    df["time"] = pd.date_range(start, periods=len(df), freq=pd.Timedelta(minutes=5))
    return df


def get_table(df_cache: str, data_folder: str) -> pd.DataFrame:
    if os.path.exists(df_cache):
        df = pd.read_csv(df_cache)
    else:
        df = batch_collect(data_folder)
        if df is None or len(df) == 0:
            raise FileNotFoundError(f"No data found in {data_folder}.")
        df.to_csv(df_cache, index=False)

    if "soc" in df.columns and df["soc"].max() > 1.5:
        df["soc"] = df["soc"] / 100.0

    # Baseline definition: dataset-recorded operation.
    # If import/export not provided, derive from dataset columns if possible.
    if "import" not in df.columns:
        if all(c in df.columns for c in ["consumption", "pv_consumption", "charge", "discharge"]):
            df["import"] = (
                df["consumption"]
                - df["pv_consumption"]
                + df["charge"]
                - df["discharge"]
            ).clip(lower=0)
        else:
            df["import"] = 0.0

    if "export" not in df.columns:
        if all(c in df.columns for c in ["pv", "pv_consumption"]):
            df["export"] = (df["pv"] - df["pv_consumption"]).clip(lower=0)
        else:
            df["export"] = 0.0

    df = ensure_time_column(df)
    return df


# ============================================================
# Calendar aging proxy (√t)
# ============================================================
def soc_multiplier_25c(soc: float) -> float:
    x = np.array([0.00, 0.20, 0.50, 0.70, 0.85, 0.95, 1.00], dtype=float)
    y = np.array([1.10, 1.00, 1.10, 1.25, 1.80, 3.20, 4.00], dtype=float)
    return float(np.interp(float(soc), x, y))


def compute_soh_sqrt_calendar(soc_series: np.ndarray, cfg: Config) -> np.ndarray:
    soc_series = np.asarray(soc_series, dtype=float)
    n = len(soc_series)
    dt_days = cfg.DT / 24.0

    mult_cal = soc_multiplier_25c(cfg.CAL_SOC)
    K_frac = (cfg.CAL_LOSS_PCT / 100.0) / (mult_cal * np.sqrt(cfg.CAL_DAYS))

    soh = np.zeros(n, dtype=float)
    soh[0] = 1.0

    t = 0.0
    eps = 1e-12
    for k in range(1, n):
        m = soc_multiplier_25c(soc_series[k - 1])
        delta_sqrt = np.sqrt(t + dt_days + eps) - np.sqrt(t + eps)
        delta_loss = K_frac * m * delta_sqrt
        soh[k] = max(0.0, soh[k - 1] - delta_loss)
        t += dt_days

    return soh


def estimate_days_to_eol_from_window(soh_series: np.ndarray, cfg: Config) -> float:
    n = len(soh_series)
    if n < 2:
        return np.nan

    dt_days = cfg.DT / 24.0
    t_obs = dt_days * (n - 1)
    if t_obs <= 0:
        return np.nan

    mult_cal = soc_multiplier_25c(cfg.CAL_SOC)
    K_frac = (cfg.CAL_LOSS_PCT / 100.0) / (mult_cal * np.sqrt(cfg.CAL_DAYS))

    loss_obs = max(0.0, 1.0 - float(soh_series[-1]))
    if loss_obs <= 1e-12:
        return np.inf

    m_eff = loss_obs / (K_frac * np.sqrt(t_obs))
    loss_target = cfg.SOH_BUDGET
    sqrt_t = loss_target / (K_frac * m_eff)
    return float(sqrt_t ** 2)


# ============================================================
# KPI helpers
# ============================================================
def energy_cost_eur(import_series, export_series, cfg: Config) -> float:
    dt = cfg.DT
    imp = np.asarray(import_series, dtype=float)
    exp = np.asarray(export_series, dtype=float)
    return float((cfg.PRICE_BUY * imp - cfg.PRICE_SELL * exp).sum() * dt)


def dwell_hours(soc_series, threshold, cfg: Config) -> float:
    soc = np.asarray(soc_series, dtype=float)
    return float((soc >= threshold).sum() * cfg.DT)


def safe_div(a, b):
    a = float(a)
    b = float(b)
    return a / b if b > 1e-12 else np.nan


def calc_scr(pv, pv_consumption) -> float:
    return safe_div(np.sum(pv_consumption), np.sum(pv))


def calc_ss(load, imp) -> float:
    return 1.0 - safe_div(np.sum(imp), np.sum(load))


# ============================================================
# Plot helpers (PDF)
# ============================================================
def _downsample(x, y, max_points=25000):
    n = len(y)
    if n <= max_points:
        return x, y
    step = max(1, n // max_points)
    return x[::step], y[::step]


def save_lineplot_pdf(df, xcol, ycols, title, outfile, ylabel=None):
    x = df[xcol].to_numpy()

    # Standard IEEE double-column width is ~7 inches.
    # Height of 2.5 inches gives a good aspect ratio for time series.
    plt.figure(figsize=(7, 2.5))

    for c in ycols:
        y = df[c].to_numpy(dtype=float)
        xs, ys = _downsample(x, y)
        plt.plot(xs, ys, label=c, linewidth=1.0)
    plt.title(title)
    plt.xlabel(xcol)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend(loc="best", frameon=True)

    plt.savefig(outfile, format='pdf')
    plt.close()


def save_hist_pdf(df, col_a, col_b, title, outfile, bins=60):
    a = df[col_a].to_numpy(dtype=float)
    b = df[col_b].to_numpy(dtype=float)

    # Standard IEEE single-column width is ~3.5 inches
    plt.figure(figsize=(3.5, 2.5))

    plt.hist(a, bins=bins, alpha=0.6, label=col_a)
    plt.hist(b, bins=bins, alpha=0.6, label=col_b)
    plt.title(title)
    plt.xlabel("SoC")
    plt.ylabel("Count")
    plt.legend(loc="best")

    plt.savefig(outfile, format='pdf')
    plt.close()


def save_bar_pdf(labels, values_a, values_b, title, outfile, label_a="Baseline",
                 label_b="MPC"):
    x = np.arange(len(labels))
    width = 0.35

    # Single column width for bar charts
    plt.figure(figsize=(3.5, 2.5))

    plt.bar(x - width / 2, values_a, width, label=label_a)
    plt.bar(x + width / 2, values_b, width, label=label_b)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel("Hours")
    plt.legend(loc="best")

    plt.savefig(outfile, format='pdf')
    plt.close()


def save_step_series_pdf(df, xcol, ycol, title, outfile, ylabel=None):
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy(dtype=float)
    xs, ys = _downsample(x, y)

    # Wide format for audit trails
    plt.figure(figsize=(7, 2.0))

    plt.plot(xs, ys, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time")
    if ylabel:
        plt.ylabel(ylabel)

    plt.savefig(outfile, format='pdf')
    plt.close()


# ============================================================
# MPC simulation core
# ============================================================
def predict_load(df, curr_idx, cfg: Config) -> np.ndarray:
    hist = min(cfg.HISTORY_SAMPLES, curr_idx)
    pred = predict_load_horizon(
        df=df,
        curr_idx=int(curr_idx),
        horizon_size=int(cfg.HORIZON_SAMPLES),
        history_size=int(hist),
        load_col="consumption",
        pv_col="pv",
        mode="auto",
    )
    pred = np.asarray(pred, dtype=float)
    pred[0] = float(df["consumption"].iloc[curr_idx])
    return pred


def run_simulation(df: pd.DataFrame, cfg: Config):
    """
    Baseline = dataset-recorded operation (soc/import/export/charge/discharge/pv_consumption columns).
    MPC      = simulated with:
      - QP plan layer (convex)
      - Hard actuation layer (no grid charge; no simult ch/dis)
      - Applied in plan blocks of length MPC_INTERVAL_STEPS
    """
    n = len(df)
    horizon = int(cfg.HORIZON_SAMPLES)
    if n <= horizon + 2:
        raise ValueError("Not enough data rows for the selected horizon.")

    sim_end = n - horizon

    pv = df["pv"].to_numpy(dtype=float)
    load = df["consumption"].to_numpy(dtype=float)

    # Baseline series from dataset
    soc_base = df["soc"].to_numpy(dtype=float)

    # MPC series arrays
    soc_mpc = np.full(n, np.nan, dtype=float)
    imp_mpc = np.full(n, np.nan, dtype=float)
    exp_mpc = np.full(n, np.nan, dtype=float)
    ch_mpc = np.full(n, np.nan, dtype=float)
    dis_mpc = np.full(n, np.nan, dtype=float)
    pv_cons_mpc = np.full(n, np.nan, dtype=float)

    # Raw QP actions (before actuation hardening)
    ch_mpc_raw = np.full(n, np.nan, dtype=float)
    dis_mpc_raw = np.full(n, np.nan, dtype=float)

    # Actuation audit series
    clamp_delta_ch = np.full(n, np.nan, dtype=float)       # how much charging got clamped by measured surplus
    proj_delta_thr = np.full(n, np.nan, dtype=float)       # |(ch+dis)_after - (ch+dis)_raw|
    simult_raw_flag = np.full(n, 0, dtype=int)             # raw simult indicator
    clamp_active_flag = np.full(n, 0, dtype=int)           # clamp activation indicator

    soc_mpc[0] = float(soc_base[0])

    mpc = MPCOptimizer(cfg)
    print("MPC Optimizer initialized and compiled.")
    print(f"[DEBUG] Imported optimizer from: {getattr(optimizer_mod, '__file__', 'UNKNOWN')}")
    print(f"[DEBUG] MPCOptimizer class module: {MPCOptimizer.__module__}")

    plan_ch = None
    plan_dis = None
    plan_ptr = 0

    eps = 1e-3

    for i, idx in enumerate(range(0, sim_end)):
        do_solve = (idx % cfg.MPC_INTERVAL_STEPS == 0) or (plan_ch is None)

        if do_solve:
            pv_forecast = pv[idx:idx + horizon]
            load_forecast = predict_load(df, idx, cfg)
            curr_soc = float(soc_mpc[idx])

            plan_ch, plan_dis = mpc.solve_plan(
                pv_forecast, load_forecast, curr_soc, n_apply=int(cfg.MPC_INTERVAL_STEPS)
            )
            plan_ptr = 0

        if plan_ch is not None and plan_ptr < len(plan_ch):
            p_ch_raw = float(plan_ch[plan_ptr])
            p_dis_raw = float(plan_dis[plan_ptr])
            plan_ptr += 1
        else:
            p_ch_raw = 0.0
            p_dis_raw = 0.0

        # Store raw QP output
        ch_mpc_raw[idx] = p_ch_raw
        dis_mpc_raw[idx] = p_dis_raw

        # Raw simult audit
        if (p_ch_raw > eps) and (p_dis_raw > eps):
            simult_raw_flag[idx] = 1

        # ============================================================
        # HARD ACTUATION LAYER (realistic feasibility)
        # ============================================================

        # (1) HARD: Never charge from grid (cap by measured instantaneous PV surplus)
        measured_surplus = max(pv[idx] - load[idx], 0.0)
        p_ch = min(p_ch_raw, measured_surplus)
        clamp_delta = max(p_ch_raw - p_ch, 0.0)
        clamp_delta_ch[idx] = clamp_delta
        if clamp_delta > eps:
            clamp_active_flag[idx] = 1

        # (2) HARD: No simultaneous charge & discharge
        # Projection that preserves net battery power: p_net = dis - ch
        p_net = p_dis_raw - p_ch
        p_dis = max(p_net, 0.0)
        p_ch = max(-p_net, 0.0)

        # Projection throughput impact (audit)
        thr_raw = p_ch_raw + p_dis_raw
        thr_act = p_ch + p_dis
        proj_delta_thr[idx] = abs(thr_act - thr_raw)

        # Clip again to physical bounds (safety)
        p_ch = float(np.clip(p_ch, 0.0, cfg.P_CH_MAX))
        p_dis = float(np.clip(p_dis, 0.0, cfg.P_DIS_MAX))

        # Grid slack using ACTUAL pv/load
        grid = float(load[idx] + p_ch - p_dis - pv[idx])
        p_imp = max(grid, 0.0)
        p_exp = max(-grid, 0.0)

        ch_mpc[idx] = p_ch
        dis_mpc[idx] = p_dis
        imp_mpc[idx] = p_imp
        exp_mpc[idx] = p_exp
        pv_cons_mpc[idx] = max(0.0, pv[idx] - p_exp)

        # State propagation
        if idx + 1 < n:
            next_soc = (
                soc_mpc[idx]
                + (p_ch * cfg.ETA_CH - p_dis / cfg.ETA_DIS) * cfg.DT / cfg.CAPACITY
            )
            soc_mpc[idx + 1] = float(np.clip(next_soc, cfg.SOC_MIN_HARD, cfg.SOC_MAX_HARD))

        if i % 2000 == 0:
            print(f"Iteration {i}/{sim_end}")

    # Forward-fill SoC tail if needed
    for k in range(1, n):
        if np.isnan(soc_mpc[k]) and not np.isnan(soc_mpc[k - 1]):
            soc_mpc[k] = soc_mpc[k - 1]

    # Attach MPC columns
    df["soc_mpc"] = soc_mpc
    df["import_mpc"] = imp_mpc
    df["export_mpc"] = exp_mpc
    df["charge_mpc"] = ch_mpc
    df["discharge_mpc"] = dis_mpc
    df["pv_consumption_mpc"] = pv_cons_mpc

    # Attach audit columns
    df["charge_mpc_raw"] = ch_mpc_raw
    df["discharge_mpc_raw"] = dis_mpc_raw
    df["audit_clamp_delta_charge_kw"] = clamp_delta_ch
    df["audit_projection_delta_throughput_kw"] = proj_delta_thr
    df["audit_simult_raw_flag"] = simult_raw_flag
    df["audit_clamp_active_flag"] = clamp_active_flag

    return df, sim_end


# ============================================================
# Main runner for one preset
# ============================================================
def run_one_preset(df_raw: pd.DataFrame, preset_id: str):
    cfg = Config()
    apply_penalty_preset(cfg, preset_id)

    out_dir = os.path.join("out", preset_id)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    df = df_raw.copy()
    df, sim_end = run_simulation(df, cfg)
    df_sim = df.iloc[:sim_end].copy()

    # SoH proxy
    soc_base = df_sim["soc"].to_numpy(dtype=float)
    soc_mpc = df_sim["soc_mpc"].to_numpy(dtype=float)
    soh_base = compute_soh_sqrt_calendar(soc_base, cfg)
    soh_mpc = compute_soh_sqrt_calendar(soc_mpc, cfg)
    df_sim["soh"] = soh_base
    df_sim["soh_mpc"] = soh_mpc

    # Core KPIs
    energy_base = energy_cost_eur(df_sim["import"], df_sim["export"], cfg)
    energy_mpc = energy_cost_eur(df_sim["import_mpc"], df_sim["export_mpc"], cfg)

    scr_base = calc_scr(df_sim["pv"].to_numpy(dtype=float), df_sim["pv_consumption"].to_numpy(dtype=float))
    scr_mpc = calc_scr(df_sim["pv"].to_numpy(dtype=float), df_sim["pv_consumption_mpc"].to_numpy(dtype=float))

    ss_base = calc_ss(df_sim["consumption"].to_numpy(dtype=float), df_sim["import"].to_numpy(dtype=float))
    ss_mpc = calc_ss(df_sim["consumption"].to_numpy(dtype=float), df_sim["import_mpc"].to_numpy(dtype=float))

    soh_drop_base_pct = (1.0 - float(soh_base[-1])) * 100.0
    soh_drop_mpc_pct = (1.0 - float(soh_mpc[-1])) * 100.0

    aging_cost_base = soh_drop_base_pct * cfg.EUR_PER_SOH_PERCENT
    aging_cost_mpc = soh_drop_mpc_pct * cfg.EUR_PER_SOH_PERCENT

    days_to_80_base = estimate_days_to_eol_from_window(soh_base, cfg)
    days_to_80_mpc = estimate_days_to_eol_from_window(soh_mpc, cfg)

    dwell95_base = dwell_hours(soc_base, 0.95, cfg)
    dwell95_mpc = dwell_hours(soc_mpc, 0.95, cfg)
    dwell90_base = dwell_hours(soc_base, 0.90, cfg)
    dwell90_mpc = dwell_hours(soc_mpc, 0.90, cfg)
    dwell85_base = dwell_hours(soc_base, 0.85, cfg)
    dwell85_mpc = dwell_hours(soc_mpc, 0.85, cfg)

    # Energy balance residual sanity check (MPC)
    grid_mpc = df_sim["import_mpc"].to_numpy(dtype=float) - df_sim["export_mpc"].to_numpy(dtype=float)
    batt_mpc = df_sim["discharge_mpc"].to_numpy(dtype=float) - df_sim["charge_mpc"].to_numpy(dtype=float)
    resid_mpc = df_sim["consumption"].to_numpy(dtype=float) - df_sim["pv"].to_numpy(dtype=float) - batt_mpc - grid_mpc

    # Baseline residual (dataset consistency)
    grid_base = df_sim["import"].to_numpy(dtype=float) - df_sim["export"].to_numpy(dtype=float)
    batt_base = df_sim["discharge"].to_numpy(dtype=float) - df_sim["charge"].to_numpy(dtype=float)
    resid_base = df_sim["consumption"].to_numpy(dtype=float) - df_sim["pv"].to_numpy(dtype=float) - batt_base - grid_base

    # Audits
    eps = 1e-3
    sim_ch_raw = df_sim["charge_mpc_raw"].to_numpy(dtype=float)
    sim_dis_raw = df_sim["discharge_mpc_raw"].to_numpy(dtype=float)
    simult_steps_raw = int(np.sum((sim_ch_raw > eps) & (sim_dis_raw > eps)))
    simult_rate_raw_pct = 100.0 * simult_steps_raw / max(1, len(df_sim))

    sim_ch = df_sim["charge_mpc"].to_numpy(dtype=float)
    sim_dis = df_sim["discharge_mpc"].to_numpy(dtype=float)
    simult_steps_after = int(np.sum((sim_ch > eps) & (sim_dis > eps)))
    simult_rate_after_pct = 100.0 * simult_steps_after / max(1, len(df_sim))

    # Clamp audit
    clamp_delta_kw = df_sim["audit_clamp_delta_charge_kw"].to_numpy(dtype=float)
    grid_charge_prevented_kwh_equiv = float(np.nansum(clamp_delta_kw) * cfg.DT)
    clamp_active_steps = int(np.sum(df_sim["audit_clamp_active_flag"].to_numpy(dtype=int)))
    clamp_active_rate_pct = 100.0 * clamp_active_steps / max(1, len(df_sim))

    # Projection audit (throughput delta)
    proj_delta_thr_kw = df_sim["audit_projection_delta_throughput_kw"].to_numpy(dtype=float)
    projection_delta_throughput_kwh_equiv = float(np.nansum(proj_delta_thr_kw) * cfg.DT)

    # Cost deltas
    delta_energy_eur = energy_mpc - energy_base
    delta_energy_pct = 100.0 * delta_energy_eur / max(1e-12, energy_base)
    health_improvement_pct = 100.0 * (soh_drop_base_pct - soh_drop_mpc_pct) / max(1e-12, soh_drop_base_pct)

    # Console summary (human)
    print("\n=== KPI SUMMARY (SIM WINDOW) ===")
    print(f"Preset:               {preset_id}")
    print(f"Energy cost baseline: {energy_base:,.2f} €")
    print(f"Energy cost MPC:      {energy_mpc:,.2f} €")
    print(f"Δ Energy cost:        {delta_energy_eur:,.2f} € ({delta_energy_pct:.2f}%)")
    print(f"SCR baseline:         {scr_base:.4f}")
    print(f"SCR MPC:              {scr_mpc:.4f}")
    print(f"SS baseline:          {ss_base:.4f}")
    print(f"SS MPC:               {ss_mpc:.4f}")
    print(f"SoH drop baseline:    {soh_drop_base_pct:,.3f} %")
    print(f"SoH drop MPC:         {soh_drop_mpc_pct:,.3f} %")
    print(f"Health improvement:   {health_improvement_pct:,.2f} %")
    print(f"Aging € proxy base:   {aging_cost_base:,.2f} €")
    print(f"Aging € proxy MPC:    {aging_cost_mpc:,.2f} €")
    print(f"Days to 80% base:     {days_to_80_base:,.1f} days")
    print(f"Days to 80% MPC:      {days_to_80_mpc:,.1f} days")
    print(f"Dwell SoC>=0.95 [h]:  base={dwell95_base:,.2f}, mpc={dwell95_mpc:,.2f}")
    print(f"Dwell SoC>=0.90 [h]:  base={dwell90_base:,.2f}, mpc={dwell90_mpc:,.2f}")
    print(f"Dwell SoC>=0.85 [h]:  base={dwell85_base:,.2f}, mpc={dwell85_mpc:,.2f}")
    print(f"QP simult ch/dis:     {simult_steps_raw} steps ({simult_rate_raw_pct:.3f}%)")
    print(f"After actuation:      {simult_steps_after} steps ({simult_rate_after_pct:.6f}%)")
    print(f"Clamp active:         {clamp_active_steps} steps ({clamp_active_rate_pct:.3f}%)")
    print(f"Grid-charge prevented (kWh eq): {grid_charge_prevented_kwh_equiv:.6f}")
    print(f"Projection Δthroughput (kWh eq): {projection_delta_throughput_kwh_equiv:.6f}")
    print(f"Residual max|.| BASE: {np.max(np.abs(resid_base)):.6f} kW")
    print(f"Residual RMS  BASE:   {np.sqrt(np.mean(resid_base**2)):.6f} kW")
    print(f"Residual max|.| MPC:  {np.max(np.abs(resid_mpc)):.6f} kW")
    print(f"Residual RMS  MPC:    {np.sqrt(np.mean(resid_mpc**2)):.6f} kW")

    # Save outputs for this preset
    df.to_csv(os.path.join(out_dir, "df_result.csv"), index=False)
    df_sim.to_csv(os.path.join(out_dir, "df_result_sim.csv"), index=False)

    # KPI tables (machine-readable)
    kpi_rows = [
        ("energy_cost_eur", energy_base, energy_mpc),
        ("delta_energy_eur", np.nan, delta_energy_eur),
        ("delta_energy_pct", np.nan, delta_energy_pct),
        ("scr", scr_base, scr_mpc),
        ("ss", ss_base, ss_mpc),
        ("soh_drop_pct", soh_drop_base_pct, soh_drop_mpc_pct),
        ("health_improvement_pct", np.nan, health_improvement_pct),
        ("aging_cost_proxy_eur", aging_cost_base, aging_cost_mpc),
        ("days_to_80pct_soh_proxy", days_to_80_base, days_to_80_mpc),
        ("dwell_hours_soc_ge_0p95", dwell95_base, dwell95_mpc),
        ("dwell_hours_soc_ge_0p90", dwell90_base, dwell90_mpc),
        ("dwell_hours_soc_ge_0p85", dwell85_base, dwell85_mpc),

        # Audits
        ("mpc_simultaneous_rate_raw_pct", np.nan, simult_rate_raw_pct),
        ("mpc_simultaneous_rate_after_pct", np.nan, simult_rate_after_pct),
        ("mpc_clamp_active_rate_pct", np.nan, clamp_active_rate_pct),
        ("mpc_grid_charge_prevented_kwh_equiv", np.nan, grid_charge_prevented_kwh_equiv),
        ("mpc_projection_delta_throughput_kwh_equiv", np.nan, projection_delta_throughput_kwh_equiv),

        ("baseline_balance_residual_max_abs_kw", float(np.max(np.abs(resid_base))), np.nan),
        ("baseline_balance_residual_rms_kw", float(np.sqrt(np.mean(resid_base**2))), np.nan),
        ("mpc_balance_residual_max_abs_kw", np.nan, float(np.max(np.abs(resid_mpc)))),
        ("mpc_balance_residual_rms_kw", np.nan, float(np.sqrt(np.mean(resid_mpc**2)))),
    ]
    pd.DataFrame(kpi_rows, columns=["metric", "baseline", "mpc"]).to_csv(
        os.path.join(out_dir, "kpis.csv"), index=False
    )

    # NEW: single-row KPI summary for easy report-table filling (one file per preset)
    kpi_summary = {
        "preset_id": preset_id,
        "energy_cost_base_eur": energy_base,
        "energy_cost_mpc_eur": energy_mpc,
        "delta_energy_eur": delta_energy_eur,
        "delta_energy_pct": delta_energy_pct,
        "scr_base": scr_base,
        "scr_mpc": scr_mpc,
        "ss_base": ss_base,
        "ss_mpc": ss_mpc,
        "soh_drop_base_pct": soh_drop_base_pct,
        "soh_drop_mpc_pct": soh_drop_mpc_pct,
        "health_improvement_pct": health_improvement_pct,
        "aging_cost_base_proxy_eur": aging_cost_base,
        "aging_cost_mpc_proxy_eur": aging_cost_mpc,
        "days_to_80_base_proxy": days_to_80_base,
        "days_to_80_mpc_proxy": days_to_80_mpc,
        "dwell_h_ge_0p85_base": dwell85_base,
        "dwell_h_ge_0p85_mpc": dwell85_mpc,
        "dwell_h_ge_0p90_base": dwell90_base,
        "dwell_h_ge_0p90_mpc": dwell90_mpc,
        "dwell_h_ge_0p95_base": dwell95_base,
        "dwell_h_ge_0p95_mpc": dwell95_mpc,
        "simult_rate_raw_pct": simult_rate_raw_pct,
        "simult_rate_act_pct": simult_rate_after_pct,
        "clamp_active_rate_pct": clamp_active_rate_pct,
        "grid_charge_prevented_kwh_equiv": grid_charge_prevented_kwh_equiv,
        "projection_delta_throughput_kwh_equiv": projection_delta_throughput_kwh_equiv,
        "baseline_residual_rms_kw": float(np.sqrt(np.mean(resid_base ** 2))),
        "mpc_residual_rms_kw": float(np.sqrt(np.mean(resid_mpc ** 2))),
    }
    pd.DataFrame([kpi_summary]).to_csv(os.path.join(out_dir, "kpi_summary.csv"), index=False)

    # ============================================================
    # Plots
    # ============================================================
    save_lineplot_pdf(
        df_sim, xcol="time", ycols=["pv", "consumption"],
        title="PV and Consumption",
        outfile=os.path.join(plot_dir, "01_pv_vs_load.pdf"), ylabel="kW"
    )
    save_lineplot_pdf(
        df_sim, xcol="time", ycols=["soc", "soc_mpc"],
        title=f"SoC: Baseline vs MPC ({preset_id})",
        outfile=os.path.join(plot_dir, "02_soc_compare.pdf"), ylabel="SoC"
    )
    save_lineplot_pdf(
        df_sim, xcol="time",
        ycols=["import", "import_mpc", "export", "export_mpc"],
        title="Grid flows", outfile=os.path.join(plot_dir, "03_grid_flows.pdf"),
        ylabel="kW"
    )
    save_lineplot_pdf(
        df_sim, xcol="time",
        ycols=["charge", "charge_mpc", "discharge", "discharge_mpc"],
        title="Battery Power",
        outfile=os.path.join(plot_dir, "04_batt_power.pdf"), ylabel="kW"
    )
    save_lineplot_pdf(
        df_sim, xcol="time", ycols=["soh", "soh_mpc"],
        title="SoH Proxy (sqrt calendar)",
        outfile=os.path.join(plot_dir, "05_soh_proxy.pdf"), ylabel="SoH"
    )
    save_hist_pdf(
        df_sim, col_a="soc", col_b="soc_mpc",
        title="SoC Distribution",
        outfile=os.path.join(plot_dir, "06_soc_hist.pdf")
    )
    labels = [">=85%", ">=90%", ">=95%"]
    vals_b = [dwell85_base, dwell90_base, dwell95_base]
    vals_m = [dwell85_mpc, dwell90_mpc, dwell95_mpc]
    save_bar_pdf(
        labels, vals_b, vals_m,
        title="High-SoC Dwell Hours",
        outfile=os.path.join(plot_dir, "07_dwell_hours.pdf"),
        label_a="Baseline",
        label_b="MPC"
    )

    # Residual plot (MPC)
    df_sim["balance_residual_mpc"] = resid_mpc
    save_lineplot_pdf(
        df_sim, xcol="time", ycols=["balance_residual_mpc"],
        title="Energy Balance Residual",
        outfile=os.path.join(plot_dir, "08_balance_residual_mpc.pdf"),
        ylabel="kW"
    )

    # Actuation audit plots (optional for report; useful for appendix)
    df_sim["audit_clamp_delta_charge_kw"] = df_sim["audit_clamp_delta_charge_kw"].fillna(0.0)
    df_sim["audit_projection_delta_throughput_kw"] = df_sim["audit_projection_delta_throughput_kw"].fillna(0.0)

    save_step_series_pdf(
        df_sim, xcol="time", ycol="audit_clamp_delta_charge_kw",
        title="Actuation audit: charge clamp delta (kW) (raw - after measured-surplus cap)",
        outfile=os.path.join(plot_dir, "09_audit_clamp_delta_charge.pdf"),
        ylabel="kW"
    )
    save_step_series_pdf(
        df_sim, xcol="time", ycol="audit_projection_delta_throughput_kw",
        title="Actuation audit: |throughput_after - throughput_raw| (kW)",
        outfile=os.path.join(plot_dir, "10_audit_projection_delta_throughput.pdf"),
        ylabel="kW"
    )

    summary = kpi_summary
    return summary


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs("out", exist_ok=True)

    # >>> Choose presets to run <<<
    presets_to_run = [
        "P2_high_low",
        "P4_high_balanced",
        "P6_target20_balanced",
        "P7_target20_maxlife",
    ]

    df_raw = get_table("out/df.csv", "data/days-range")

    summaries = []
    for preset in presets_to_run:
        print("\n" + "=" * 70)
        print(f"RUNNING PRESET: {preset}")
        print("=" * 70)
        summaries.append(run_one_preset(df_raw, preset))

    # Save comparison table (one row per preset)
    comp = pd.DataFrame(summaries)
    comp.to_csv("out/penalty_compare.csv", index=False)

    # Tradeoff plot: delta energy vs health improvement
    plt.figure(figsize=(3.5, 2.5))
    plt.scatter(comp["delta_energy_eur"], comp["health_improvement_pct"],
                alpha=0.8)

    # Adjust annotation text size for small plots
    for _, r in comp.iterrows():
        plt.text(r["delta_energy_eur"], r["health_improvement_pct"],
                 r["preset_id"], fontsize=7)

    plt.xlabel("$\Delta$ Energy Cost [€]")  # Latex syntax for label
    plt.ylabel("Health Improvement [%]")
    plt.title("Penalty Tradeoff")

    plt.savefig("out/penalty_tradeoff.pdf", format='pdf')
    plt.close()
    print("Saved tradeoff plot: out/penalty_tradeoff.pdf")
    print("\nDone.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
