# pipeline.py
import os
import warnings
import numpy as np
import pandas as pd

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
    MPC_INTERVAL_STEPS = 1

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
        raise KeyError(
            f"Unknown preset_id '{preset_id}'. Available: {list(PENALTY_PRESETS.keys())}")

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

    candidates = ["timestamp", "datetime", "date", "ts", "Time", "Zeit",
                  "Zeitstempel"]
    for c in candidates:
        if c in df.columns:
            t = pd.to_datetime(df[c], utc=True, errors="coerce")
            if t.notna().mean() >= 0.8:
                df["time"] = t.dt.tz_convert("Europe/Berlin").dt.tz_localize(
                    None)
                return df

    start = pd.Timestamp("1970-01-01 00:00:00")
    df["time"] = pd.date_range(start, periods=len(df),
                               freq=pd.Timedelta(minutes=5))
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
        if all(c in df.columns for c in
               ["consumption", "pv_consumption", "charge", "discharge"]):
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
# Simulation Core
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
    clamp_delta_ch = np.full(n, np.nan,
                             dtype=float)  # how much charging got clamped by measured surplus
    proj_delta_thr = np.full(n, np.nan,
                             dtype=float)  # |(ch+dis)_after - (ch+dis)_raw|
    simult_raw_flag = np.full(n, 0, dtype=int)  # raw simult indicator
    clamp_active_flag = np.full(n, 0, dtype=int)  # clamp activation indicator

    soc_mpc[0] = float(soc_base[0])

    mpc = MPCOptimizer(cfg)
    print("MPC Optimizer initialized and compiled.")
    print(
        f"[DEBUG] Imported optimizer from: {getattr(optimizer_mod, '__file__', 'UNKNOWN')}")
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
                pv_forecast, load_forecast, curr_soc,
                n_apply=int(cfg.MPC_INTERVAL_STEPS)
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
                    + (
                                p_ch * cfg.ETA_CH - p_dis / cfg.ETA_DIS) * cfg.DT / cfg.CAPACITY
            )
            soc_mpc[idx + 1] = float(
                np.clip(next_soc, cfg.SOC_MIN_HARD, cfg.SOC_MAX_HARD))

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
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n--- Starting Simulation: {preset_id} ---")
    df = df_raw.copy()
    df, sim_end = run_simulation(df, cfg)
    df_sim = df.iloc[:sim_end].copy()

    # Save Results
    outfile = os.path.join(out_dir, "df_result.csv")
    df_sim.to_csv(outfile, index=False)
    print(f"Saved raw results to: {outfile}")


# ============================================================
# Main Wrapper
# ============================================================
def main():
    # Data Load
    df_raw = get_table("out/df.csv", "data/days-range")

    # Define presets to simulate
    presets_to_run = [
        "P2_high_low",
        "P4_high_balanced",
        "P6_target20_balanced",
        "P7_target20_maxlife",
    ]

    for preset in presets_to_run:
        run_one_preset(df_raw, preset)

    print("\nAll simulations completed. Run 'post-analysis.py' next.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
