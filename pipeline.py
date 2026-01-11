# pipeline.py
import os.path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from load_forecast_profile import predict_load_horizon
from data_parser import batch_collect
from optimizer import MPCOptimizer
from soh_pricing import simulate_soh_from_soc, soc_histogram


class Config:
    # --- Time ---
    DT = 5.0 / 60.0  # hours per sample (5 minutes)
    HISTORY_SAMPLES = 60 * 288  # 60 days @ 5min
    HORIZON_SAMPLES = 288  # 1 day horizon @ 5min

    # Run MPC only every N steps (N=3 => every 15 minutes)
    MPC_EVERY_N_STEPS = 3

    # --- Battery ---
    CAPACITY = 10.0  # kWh
    P_CH_MAX = 5.0  # kW
    P_DIS_MAX = 5.0  # kW
    ETA_CH = 0.95
    ETA_DIS = 0.95

    SOC_MIN_HARD = 0.05
    SOC_MAX_HARD = 1.00

    # --- Costs (energy) ---
    PRICE_BUY = 0.30
    PRICE_SELL = 0.08
    COST_WEAR = 0.02  # per kWh throughput proxy
    COST_SOC_HOLDING = 0.0  # not used in KPI mode

    # --- KPI D: replacement-cost proxy ---
    BATTERY_REPLACEMENT_COST_EUR = 4000.0
    SOH_EOL = 0.80  # "battery useless" threshold


def get_table(df_cache: str, data_folder: str) -> pd.DataFrame:
    if os.path.exists(df_cache):
        df = pd.read_csv(df_cache)
    else:
        df = batch_collect(data_folder)
        df["soc"] = df["soc"] / 100.0

        df["import"] = (
            df["consumption"]
            - df["pv_consumption"]
            + df["charge"]
            - df["discharge"]
        ).clip(lower=0)

        df["export"] = (df["pv"] - df["pv_consumption"]).clip(lower=0)

        df.to_csv(df_cache, index=False)

    # Ensure consistent dtype
    for c in ["pv", "consumption", "pv_consumption", "charge", "discharge", "soc", "import", "export"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


def predict_values(df, curr_idx, history_size, horizon_size):
    return predict_load_horizon(
        df=df,
        curr_idx=int(curr_idx),
        horizon_size=int(horizon_size),
        history_size=int(history_size),
        load_col="consumption",
        pv_col="pv",
        mode="auto",
    )


def _compute_import_export(pv, consumption, p_charge, p_discharge):
    """
    Energy balance in pipeline uses:
      import = (cons - pv_consumption + charge - discharge).clip(lower=0)
      export = (pv - pv_consumption).clip(lower=0)
    Here pv_consumption is computed as pv - export (and clipped).
    We'll compute import/export consistently from the model balance:
      grid_import - grid_export = load + p_ch - p_dis - pv
    with nonneg constraints, so:
      import = max(0, load + p_ch - p_dis - pv)
      export = max(0, pv - load - p_ch + p_dis)
    """
    net = consumption + p_charge - p_discharge - pv
    p_import = max(0.0, net)
    p_export = max(0.0, -net)
    pv_consumption = max(0.0, pv - p_export)
    return p_import, p_export, pv_consumption


def run_single_mpc_step(df, idx, cfg, mpc):
    consumption_prediction = predict_values(
        df=df,
        curr_idx=idx,
        history_size=cfg.HISTORY_SAMPLES,
        horizon_size=cfg.HORIZON_SAMPLES,
    )

    # Replace the first element in the prediction with the real data
    consumption_current = float(df["consumption"].iloc[idx])
    consumption_prediction[0] = consumption_current

    end_idx = idx + cfg.HORIZON_SAMPLES
    if end_idx > len(df):
        return None

    pv_forecast = df["pv"].iloc[idx:end_idx].to_numpy()

    current_ts = df.index[idx]
    soc = float(df.at[current_ts, "soc_mpc"])

    p_charge, p_discharge, p_import, p_export = mpc.solve(
        pv_forecast=pv_forecast,
        consumption_forecast=consumption_prediction,
        curr_soc=soc,
    )

    # Write step decisions
    df.at[current_ts, "charge_mpc"] = p_charge
    df.at[current_ts, "discharge_mpc"] = p_discharge
    df.at[current_ts, "import_mpc"] = p_import
    df.at[current_ts, "export_mpc"] = p_export

    pv_val = float(df.at[current_ts, "pv"])
    df.at[current_ts, "pv_consumption_mpc"] = max(0.0, pv_val - p_export)

    # Propagate SOC to next step
    if idx + 1 < len(df):
        next_soc = soc + (p_charge * cfg.ETA_CH - p_discharge / cfg.ETA_DIS) * cfg.DT / cfg.CAPACITY
        next_soc = float(np.clip(next_soc, cfg.SOC_MIN_HARD, cfg.SOC_MAX_HARD))
        next_ts = df.index[idx + 1]
        df.at[next_ts, "soc_mpc"] = next_soc

    return (p_charge, p_discharge)


def run_hold_step(df, idx, cfg, last_u):
    """
    If MPC is not executed this step, we hold last (p_charge, p_discharge),
    recompute import/export from current pv & consumption, and propagate SOC.
    """
    if last_u is None:
        # Fallback: do nothing until first MPC solve
        p_charge, p_discharge = 0.0, 0.0
    else:
        p_charge, p_discharge = float(last_u[0]), float(last_u[1])

    current_ts = df.index[idx]
    pv_val = float(df.at[current_ts, "pv"])
    cons_val = float(df.at[current_ts, "consumption"])
    soc = float(df.at[current_ts, "soc_mpc"])

    p_import, p_export, pv_cons = _compute_import_export(pv_val, cons_val, p_charge, p_discharge)

    df.at[current_ts, "charge_mpc"] = p_charge
    df.at[current_ts, "discharge_mpc"] = p_discharge
    df.at[current_ts, "import_mpc"] = p_import
    df.at[current_ts, "export_mpc"] = p_export
    df.at[current_ts, "pv_consumption_mpc"] = pv_cons

    # Propagate SOC to next step
    if idx + 1 < len(df):
        next_soc = soc + (p_charge * cfg.ETA_CH - p_discharge / cfg.ETA_DIS) * cfg.DT / cfg.CAPACITY
        next_soc = float(np.clip(next_soc, cfg.SOC_MIN_HARD, cfg.SOC_MAX_HARD))
        next_ts = df.index[idx + 1]
        df.at[next_ts, "soc_mpc"] = next_soc


def compute_kpis(df, cfg):
    dt_h = cfg.DT

    # --- KPI A: High-SoC exposure ---
    def hours_above(series, thr):
        return float((series > thr).sum() * dt_h)

    soc_base = df["soc"].astype(float)
    soc_mpc = df["soc_mpc"].astype(float)

    kpiA = {
        "base_hours_soc_gt_80": hours_above(soc_base, 0.80),
        "mpc_hours_soc_gt_80": hours_above(soc_mpc, 0.80),
        "base_hours_soc_gt_95": hours_above(soc_base, 0.95),
        "mpc_hours_soc_gt_95": hours_above(soc_mpc, 0.95),
    }

    # --- KPI C: energy/import/export ---
    # Energy in kWh per step = power(kW)*dt(hours)
    import_kwh_base = float((df["import"] * dt_h).sum())
    export_kwh_base = float((df["export"] * dt_h).sum())
    import_kwh_mpc = float((df["import_mpc"] * dt_h).sum())
    export_kwh_mpc = float((df["export_mpc"] * dt_h).sum())

    energy_cost_base = float(((df["import"] * cfg.PRICE_BUY) - (df["export"] * cfg.PRICE_SELL)).mul(dt_h).sum())
    energy_cost_mpc = float(((df["import_mpc"] * cfg.PRICE_BUY) - (df["export_mpc"] * cfg.PRICE_SELL)).mul(dt_h).sum())

    kpiC = {
        "base_import_kwh": import_kwh_base,
        "base_export_kwh": export_kwh_base,
        "mpc_import_kwh": import_kwh_mpc,
        "mpc_export_kwh": export_kwh_mpc,
        "base_energy_cost_eur": energy_cost_base,
        "mpc_energy_cost_eur": energy_cost_mpc,
    }

    # --- KPI D: SOH proxy + € proxy ---
    soh_base = simulate_soh_from_soc(soc_base.to_numpy(), dt_hours=dt_h)
    soh_mpc_track = simulate_soh_from_soc(soc_mpc.to_numpy(), dt_hours=dt_h)
    df["soh_proxy"] = soh_base
    df["soh_proxy_mpc"] = soh_mpc_track

    soh_drop_base_pct = float((soh_base[0] - soh_base[-1]) * 100.0)
    soh_drop_mpc_pct = float((soh_mpc_track[0] - soh_mpc_track[-1]) * 100.0)

    eur_per_pct = cfg.BATTERY_REPLACEMENT_COST_EUR / ((1.0 - cfg.SOH_EOL) * 100.0)  # = 4000/(20) = 200 €/%
    soh_cost_base = soh_drop_base_pct * eur_per_pct
    soh_cost_mpc = soh_drop_mpc_pct * eur_per_pct
    soh_cost_saved = (soh_drop_base_pct - soh_drop_mpc_pct) * eur_per_pct

    kpiD = {
        "base_soh_drop_pct": soh_drop_base_pct,
        "mpc_soh_drop_pct": soh_drop_mpc_pct,
        "eur_per_soh_pct": float(eur_per_pct),
        "base_soh_cost_proxy_eur": float(soh_cost_base),
        "mpc_soh_cost_proxy_eur": float(soh_cost_mpc),
        "soh_cost_proxy_saved_eur": float(soh_cost_saved),
    }

    # --- KPI B: histogram data (for plotting + small numeric summary) ---
    hist = soc_histogram(
        soc_base=soc_base.to_numpy(),
        soc_mpc=soc_mpc.to_numpy(),
        bin_width=0.05,
    )

    return kpiA, hist, kpiC, kpiD


def plot_histogram(hist):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hist["bin_center"],
            y=hist["count_base"],
            name="Baseline",
            opacity=0.55,
        )
    )
    fig.add_trace(
        go.Bar(
            x=hist["bin_center"],
            y=hist["count_mpc"],
            name="MPC",
            opacity=0.55,
        )
    )
    fig.update_layout(
        barmode="overlay",
        title="KPI B: SOC Histogram (Baseline vs MPC)",
        xaxis_title="SOC",
        yaxis_title="Count",
    )
    fig.show()


def plot_kpiA_bar(kpiA):
    labels = ["Hours SOC>80%", "Hours SOC>95%"]
    base_vals = [kpiA["base_hours_soc_gt_80"], kpiA["base_hours_soc_gt_95"]]
    mpc_vals = [kpiA["mpc_hours_soc_gt_80"], kpiA["mpc_hours_soc_gt_95"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=base_vals, name="Baseline"))
    fig.add_trace(go.Bar(x=labels, y=mpc_vals, name="MPC"))
    fig.update_layout(
        barmode="group",
        title="KPI A: High-SOC Exposure",
        yaxis_title="Hours",
    )
    fig.show()


def main():
    df = get_table("out/df.csv", "data/days-range")
    cfg = Config()

    # Initialize MPC columns (start from baseline values)
    for column in ["soc", "pv_consumption", "export", "import", "charge", "discharge"]:
        df[f"{column}_mpc"] = df[column].copy()

    mpc_optimizer = MPCOptimizer(cfg)
    print("MPC Optimizer initialized and compiled.")

    # Simulation loop
    loop_range = range(cfg.HISTORY_SAMPLES, len(df) - cfg.HORIZON_SAMPLES)
    full = len(loop_range)

    last_u = None  # (p_charge, p_discharge)

    for i, idx in enumerate(loop_range):
        # Execute MPC only every 15 minutes (every 3 steps)
        if (idx % cfg.MPC_EVERY_N_STEPS) == 0:
            last_u = run_single_mpc_step(df, idx, cfg, mpc_optimizer)
        else:
            run_hold_step(df, idx, cfg, last_u)

        if i % 100 == 0:
            print(f"Iteration {i}/{full}")

    # Save results
    df.to_csv("out/df_result.csv", index=False)

    # --- KPIs A..D ---
    kpiA, hist, kpiC, kpiD = compute_kpis(df, cfg)

    print("\n=== KPI A: High-SOC exposure ===")
    print(f"Baseline: Hours SOC > 80%  = {kpiA['base_hours_soc_gt_80']:.2f} h")
    print(f"MPC:      Hours SOC > 80%  = {kpiA['mpc_hours_soc_gt_80']:.2f} h")
    print(f"Baseline: Hours SOC > 95%  = {kpiA['base_hours_soc_gt_95']:.2f} h")
    print(f"MPC:      Hours SOC > 95%  = {kpiA['mpc_hours_soc_gt_95']:.2f} h")

    print("\n=== KPI B: SOC histogram (summary) ===")
    # Print only a compact summary (top 5 bins each)
    top_base = np.argsort(-hist["count_base"])[:5]
    top_mpc = np.argsort(-hist["count_mpc"])[:5]
    print("Top bins Baseline (SOC_center -> count):")
    for j in top_base:
        print(f"  {hist['bin_center'][j]:.2f} -> {int(hist['count_base'][j])}")
    print("Top bins MPC (SOC_center -> count):")
    for j in top_mpc:
        print(f"  {hist['bin_center'][j]:.2f} -> {int(hist['count_mpc'][j])}")

    print("\n=== KPI C: Energy sanity check ===")
    print(f"Energy cost Baseline: {kpiC['base_energy_cost_eur']:.2f} €")
    print(f"Energy cost MPC:      {kpiC['mpc_energy_cost_eur']:.2f} €")
    print(f"Import kWh Baseline:  {kpiC['base_import_kwh']:.1f} kWh")
    print(f"Import kWh MPC:       {kpiC['mpc_import_kwh']:.1f} kWh")
    print(f"Export kWh Baseline:  {kpiC['base_export_kwh']:.1f} kWh")
    print(f"Export kWh MPC:       {kpiC['mpc_export_kwh']:.1f} kWh")

    print("\n=== KPI D: SOH budget + € proxy (calendar-aging proxy) ===")
    print(f"SOH drop Baseline: {kpiD['base_soh_drop_pct']:.3f} %-points")
    print(f"SOH drop MPC:      {kpiD['mpc_soh_drop_pct']:.3f} %-points")
    print(f"€ per 1% SOH:      {kpiD['eur_per_soh_pct']:.1f} €/%-point")
    print(f"SOH cost Baseline: {kpiD['base_soh_cost_proxy_eur']:.2f} € (proxy)")
    print(f"SOH cost MPC:      {kpiD['mpc_soh_cost_proxy_eur']:.2f} € (proxy)")
    print(f"Saved (proxy):     {kpiD['soh_cost_proxy_saved_eur']:.2f} € (proxy)")

    # --- Plots ---
    # KPI A bar + KPI B histogram
    plot_kpiA_bar(kpiA)
    plot_histogram(hist)

    # Existing timeseries plot (optional but handy)
    fig = px.line(
        df,
        x="time",
        y=[
            "pv",
            "consumption",
            "soc",
            "soc_mpc",
            "import",
            "import_mpc",
            "export",
            "export_mpc",
            "soh_proxy",
            "soh_proxy_mpc",
        ],
        title="Baseline vs MPC (SOC, Import/Export, SOH proxy)",
    )
    fig.show()


if __name__ == "__main__":
    main()
