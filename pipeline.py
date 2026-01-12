import os.path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from load_forecast_profile import predict_load_horizon
from data_parser import batch_collect
from optimizer import MPCOptimizer


class Config:
    # Time base
    DT = 5.0 / 60.0  # hours per sample (5 min)
    SAMPLES_PER_DAY = 288
    HISTORY_SAMPLES = 60 * 288  # 60 days (used by pipeline for forecast call)
    HORIZON_SAMPLES = 288  # 1 day

    # Run MPC not every step (15 min => every 3 steps)
    MPC_INTERVAL_STEPS = 6  # 3*5min = 15min
    HOLD_CONTROL_BETWEEN_SOLVES = True  # keep last MPC action for intermediate steps

    # Battery model
    CAPACITY = 10.0  # kWh
    P_CH_MAX = 5.0  # kW
    P_DIS_MAX = 5.0  # kW
    ETA_CH = 0.95
    ETA_DIS = 0.95

    SOC_MIN_HARD = 0.05
    SOC_MAX_HARD = 1.00

    # MPC calendar-aging target
    SOC_TARGET = 0.20

    # Cost function weights
    PRICE_BUY = 0.3258
    PRICE_SELL = 0.0794
    COST_WEAR = 0.02

    # Calendar aging proxy weights inside MPC (hinge-based)
    COST_SOC_HOLDING = 0.25  # keep moderate; tweak if MPC too "scared" of SoC
    W_BELOW_TARGET = 0.10
    W_ABOVE_TARGET = 0.15
    W_HIGH_85 = 2.0
    W_HIGH_95 = 8.0

    # Verification / SoH proxy (√t model calibrated at 25°C, 100% SOC)
    SOH_EOL = 0.80
    BATTERY_REPLACEMENT_EUR = 4000.0
    SOH_BUDGET = 1.0 - SOH_EOL  # 0.20
    EUR_PER_SOH_PERCENT = BATTERY_REPLACEMENT_EUR / (SOH_BUDGET * 100.0)  # 200 €/%

    # Calibration point from your plot (25°C @ 100% SOC):
    CAL_DAYS = 160.0
    CAL_SOC = 1.00
    CAL_LOSS_PCT = 9.5  # approx from plot: 100%SOC @25°C ~90.5% after 160 days


def get_table(df_cache, data_folder):
    if os.path.exists(df_cache):
        return pd.read_csv(df_cache)

    df = batch_collect(data_folder)
    df["soc"] = df["soc"] / 100.0

    df["import"] = (
        df["consumption"] -
        df["pv_consumption"] + df["charge"] -
        df["discharge"]
    ).clip(lower=0)

    df["export"] = (
        df["pv"] - df["pv_consumption"]
    ).clip(lower=0)

    df.to_csv(df_cache, index=False)
    return df


def predict_values(df, curr_idx, history_size, horizon_size):
    if curr_idx <= history_size:
        history_size = curr_idx

    return predict_load_horizon(
        df=df,
        curr_idx=int(curr_idx),
        horizon_size=int(horizon_size),
        history_size=int(history_size),
        load_col="consumption",
        pv_col="pv",
        mode="auto",
    )


# -----------------------------
# SoH proxy: √t calendar aging
# -----------------------------
def soc_multiplier_25c(soc: float) -> float:
    """
    Shape proxy: minimum at 20%, late-rising penalty above ~85–95%.
    (Dimensionless multiplier)
    """
    x = np.array([0.00, 0.20, 0.50, 0.70, 0.85, 0.95, 1.00], dtype=float)
    y = np.array([1.10, 1.00, 1.10, 1.25, 1.80, 3.20, 4.00], dtype=float)
    return float(np.interp(float(soc), x, y))


def compute_soh_sqrt_calendar(soc_series: np.ndarray, cfg: Config) -> np.ndarray:
    """
    SOH(t) = 1 - K * sum( mult(soc_k) * (sqrt(t+dt)-sqrt(t)) )
    where t is in days.
    Calibrate K using cfg.CAL_LOSS_PCT at cfg.CAL_DAYS for soc=cfg.CAL_SOC.
    """
    soc_series = np.asarray(soc_series, dtype=float)
    n = len(soc_series)

    dt_days = (cfg.DT / 24.0)  # hours -> days
    # K_frac such that loss_frac = K_frac * mult_cal * sqrt(cal_days)
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
    """
    Scenario estimate: assume the *exposure pattern* continues similarly.
    For √t model, an effective relation is:
      loss_frac(t) = K_frac * m_eff * sqrt(t_days)
    From observed window:
      loss_obs = 1 - soh_end
      t_obs = window_days
      => m_eff = loss_obs / (K_frac*sqrt(t_obs))
    Then solve for t when loss=0.20 (EOL budget).
    """
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

    # Solve loss_target = K_frac*m_eff*sqrt(t)
    loss_target = cfg.SOH_BUDGET  # 0.20
    sqrt_t = loss_target / (K_frac * m_eff)
    t_days = float(sqrt_t ** 2)
    return t_days


# -----------------------------
# MPC step
# -----------------------------
def run_single_mpc_step(df, idx, cfg: Config, mpc: MPCOptimizer, last_action):
    """
    Runs one simulation step.
    If idx is not on MPC interval and HOLD_CONTROL_BETWEEN_SOLVES=True,
    repeats last_action.
    """
    end_idx = idx + cfg.HORIZON_SAMPLES
    if end_idx > len(df):
        return last_action

    current_ts = df.index[idx]
    soc = float(df.at[current_ts, "soc_mpc"])

    # Decide whether to solve MPC this step
    do_solve = (idx % cfg.MPC_INTERVAL_STEPS == 0)

    if do_solve:
        consumption_prediction = predict_values(
            df=df,
            curr_idx=idx,
            history_size=cfg.HISTORY_SAMPLES,
            horizon_size=cfg.HORIZON_SAMPLES,
        )
        # ensure first element = real current consumption
        consumption_prediction[0] = float(df["consumption"].iloc[idx])

        pv_forecast = df["pv"].iloc[idx:end_idx].to_numpy(dtype=float)

        p_charge, p_discharge, p_import, p_export = mpc.solve(
            pv_forecast,
            consumption_prediction,
            soc
        )
        last_action = (p_charge, p_discharge, p_import, p_export)

    else:
        # hold previous action, or do nothing if none
        if cfg.HOLD_CONTROL_BETWEEN_SOLVES and last_action is not None:
            p_charge, p_discharge, p_import, p_export = last_action
        else:
            p_charge, p_discharge, p_import, p_export = (0.0, 0.0, 0.0, 0.0)

    # Write MPC outputs
    df.at[current_ts, "export_mpc"] = p_export
    df.at[current_ts, "import_mpc"] = p_import
    df.at[current_ts, "charge_mpc"] = p_charge
    df.at[current_ts, "discharge_mpc"] = p_discharge

    pv_val = float(df.at[current_ts, "pv"])
    df.at[current_ts, "pv_consumption_mpc"] = max(0.0, pv_val - p_export)

    # Propagate SOC to next step
    if idx + 1 < len(df):
        next_soc = (
            soc +
            (p_charge * cfg.ETA_CH - p_discharge / cfg.ETA_DIS)
            * cfg.DT / cfg.CAPACITY
        )
        next_soc = float(np.clip(next_soc, cfg.SOC_MIN_HARD, cfg.SOC_MAX_HARD))
        next_ts = df.index[idx + 1]
        df.at[next_ts, "soc_mpc"] = next_soc

    return last_action


# -----------------------------
# KPI / reporting
# -----------------------------
def energy_cost_eur(import_series, export_series, cfg: Config) -> float:
    dt = cfg.DT
    imp = np.asarray(import_series, dtype=float)
    exp = np.asarray(export_series, dtype=float)
    return float((cfg.PRICE_BUY * imp - cfg.PRICE_SELL * exp).sum() * dt)


def high_soc_exposure_hours(soc_series, threshold, cfg: Config) -> float:
    soc = np.asarray(soc_series, dtype=float)
    dt_h = cfg.DT
    return float((soc > threshold).sum() * dt_h)


def plot_time_series(df, cols, title):
    fig = px.line(df, x="time", y=cols, title=title)
    fig.show()


def plot_histogram_soc(df, col_a, col_b, title):
    a = df[col_a].to_numpy(dtype=float)
    b = df[col_b].to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=a, name=col_a, opacity=0.55, nbinsx=40))
    fig.add_trace(go.Histogram(x=b, name=col_b, opacity=0.55, nbinsx=40))
    fig.update_layout(barmode="overlay", title=title, xaxis_title="SoC", yaxis_title="count")
    fig.show()


def main():
    df = get_table("out/df.csv", "data/days-range")
    cfg = Config()

    # Ensure time exists (for plotting)
    if "time" not in df.columns:
        raise KeyError("df must contain a 'time' column for plotting")

    # Baseline columns exist in df: soc, pv_consumption, export, import, charge, discharge
    # Create MPC columns (start by copying baseline)
    for column in ["soc", "pv_consumption", "export", "import", "charge", "discharge"]:
        df[f"{column}_mpc"] = df[column].copy()

    # Initialize MPC SOC start (at first sim step)
    # We start MPC simulation at HISTORY_SAMPLES index, but SOC_mpc already copied.
    mpc_optimizer = MPCOptimizer(cfg)
    print("MPC Optimizer initialized and compiled.")

    loop_range = range(1, len(df) - cfg.HORIZON_SAMPLES)
    full = len(loop_range)

    last_action = None

    for i, idx in enumerate(loop_range):
        last_action = run_single_mpc_step(df, idx, cfg, mpc_optimizer, last_action)
        if i % 200 == 0:
            print(f"Iteration {i}/{full}")

    # -----------------------------
    # Verification / KPI computation
    # -----------------------------
    soc_base = df["soc"].to_numpy(dtype=float)
    soc_mpc = df["soc_mpc"].to_numpy(dtype=float)

    soh_base = compute_soh_sqrt_calendar(soc_base, cfg)
    soh_mpc = compute_soh_sqrt_calendar(soc_mpc, cfg)

    df["soh"] = soh_base
    df["soh_mpc"] = soh_mpc

    # Energy costs
    energy_base = energy_cost_eur(df["import"], df["export"], cfg)
    energy_mpc = energy_cost_eur(df["import_mpc"], df["export_mpc"], cfg)

    # SOH drops (simulation window based)
    soh_drop_base_pct = (1.0 - float(soh_base[-1])) * 100.0
    soh_drop_mpc_pct = (1.0 - float(soh_mpc[-1])) * 100.0

    # Wear € proxy (based on 4000€ / 20% budget)
    aging_cost_base = soh_drop_base_pct * cfg.EUR_PER_SOH_PERCENT
    aging_cost_mpc = soh_drop_mpc_pct * cfg.EUR_PER_SOH_PERCENT

    # Exposure KPIs
    hours_gt95_base = high_soc_exposure_hours(soc_base, 0.95, cfg)
    hours_gt95_mpc = high_soc_exposure_hours(soc_mpc, 0.95, cfg)

    hours_gt90_base = high_soc_exposure_hours(soc_base, 0.90, cfg)
    hours_gt90_mpc = high_soc_exposure_hours(soc_mpc, 0.90, cfg)

    # Scenario “days to 80% SOH” using the √t-consistent method
    days_to_80_base = estimate_days_to_eol_from_window(soh_base, cfg)
    days_to_80_mpc = estimate_days_to_eol_from_window(soh_mpc, cfg)

    # Console summary
    print("\n=== GREEDY/BASELINE ===")
    print(f"Energy cost (sim window): {energy_base:,.2f} €")
    print(f"SOH drop    (sim window): {soh_drop_base_pct:,.3f} %")
    print(f"Aging cost  (proxy €):    {aging_cost_base:,.2f} €")
    print(f"Hours SoC>90%:            {hours_gt90_base:,.2f} h")
    print(f"Hours SoC>95%:            {hours_gt95_base:,.2f} h")
    print(f"Scenario days to 80% SOH: {days_to_80_base:,.1f} days")

    print("\n=== MPC ===")
    print(f"Energy cost (sim window): {energy_mpc:,.2f} €")
    print(f"SOH drop    (sim window): {soh_drop_mpc_pct:,.3f} %")
    print(f"Aging cost  (proxy €):    {aging_cost_mpc:,.2f} €")
    print(f"Hours SoC>90%:            {hours_gt90_mpc:,.2f} h")
    print(f"Hours SoC>95%:            {hours_gt95_mpc:,.2f} h")
    print(f"Scenario days to 80% SOH: {days_to_80_mpc:,.1f} days")

    # Save output
    os.makedirs("out", exist_ok=True)
    df.to_csv("out/df_result.csv", index=False)
    print("\nSaved: out/df_result.csv")

    # -----------------------------
    # Plots (many)
    # -----------------------------
    plot_time_series(
        df,
        cols=["pv", "consumption"],
        title="PV and Consumption"
    )

    plot_time_series(
        df,
        cols=["soc", "soc_mpc"],
        title="SoC: Baseline vs MPC"
    )

    plot_time_series(
        df,
        cols=["import", "import_mpc", "export", "export_mpc"],
        title="Grid flows: Import/Export Baseline vs MPC"
    )

    plot_time_series(
        df,
        cols=["charge", "charge_mpc", "discharge", "discharge_mpc"],
        title="Battery power: Charge/Discharge Baseline vs MPC"
    )

    plot_time_series(
        df,
        cols=["soh", "soh_mpc"],
        title="SOH (calendar aging proxy √t): Baseline vs MPC"
    )

    plot_histogram_soc(
        df,
        col_a="soc",
        col_b="soc_mpc",
        title="SoC Distribution (Histogram): Baseline vs MPC"
    )

    # Extra: High-SoC indicator plot (optional)
    df["soc_gt95"] = (df["soc"] > 0.95).astype(int)
    df["soc_mpc_gt95"] = (df["soc_mpc"] > 0.95).astype(int)

    plot_time_series(
        df,
        cols=["soc_gt95", "soc_mpc_gt95"],
        title="High-SoC indicator (>95%): Baseline vs MPC (1=yes)"
    )


if __name__ == '__main__':
    main()
