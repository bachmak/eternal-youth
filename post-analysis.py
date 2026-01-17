# post-analysis.py
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipeline import Config, PENALTY_PRESETS
import matplotlib.dates as mdates
import scienceplots

plt.style.use(['science', 'ieee'])


def soc_multiplier_25c(soc: float) -> float:
    x = np.array([0.00, 0.20, 0.50, 0.70, 0.85, 0.95, 1.00], dtype=float)
    y = np.array([1.10, 1.00, 1.10, 1.25, 1.80, 3.20, 4.00], dtype=float)
    return float(np.interp(float(soc), x, y))


def compute_soh_sqrt_calendar(soc_series: np.ndarray,
                              cfg: Config) -> np.ndarray:
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


def estimate_days_to_eol_from_window(soh_series: np.ndarray,
                                     cfg: Config) -> float:
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


def _downsample(x, y, max_points=25000):
    n = len(y)
    if n <= max_points:
        return x, y
    step = max(1, n // max_points)
    return x[::step], y[::step]


import matplotlib.dates as mdates  # Ensure this is imported at the top


def save_lineplot_pdf(df, xcol, y_col_map, title, outfile, ylabel=None):
    x = pd.to_datetime(df[xcol]).dt.to_pydatetime()

    plt.figure(figsize=(7, 2.5))

    for col, label in y_col_map.items():
        y = df[col].to_numpy(dtype=float)
        xs, ys = _downsample(x, y)
        plt.plot(xs, ys, label=label, linewidth=0.8)

    plt.title(title)
    plt.xlabel("Time")
    if ylabel:
        plt.ylabel(ylabel)

    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.minorticks_off()

    plt.legend(loc="best", frameon=True)
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.close()


def save_hist_pdf(df, col_a, label_a, col_b, label_b, title, outfile, bins=60):
    a = df[col_a].to_numpy(dtype=float)
    b = df[col_b].to_numpy(dtype=float)

    plt.figure(figsize=(3.5, 2.5))

    plt.hist(a, bins=bins, alpha=0.6, label=label_a)
    plt.hist(b, bins=bins, alpha=0.6, label=label_b)

    plt.title(title)
    plt.xlabel("SoC")
    plt.ylabel("Count")
    plt.legend(loc="best")

    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.close()


def save_step_series_pdf(df, xcol, ycol, title, outfile, ylabel=None):
    x = pd.to_datetime(df[xcol]).dt.to_pydatetime()
    y = df[ycol].to_numpy(dtype=float)
    xs, ys = _downsample(x, y)

    plt.figure(figsize=(7, 2.0))
    plt.plot(xs, ys, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time")
    if ylabel:
        plt.ylabel(ylabel)

    ax = plt.gca()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.minorticks_off()

    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.close()


def save_bar_pdf(labels, values_a, values_b, title, outfile, label_a, label_b):
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


# ============================================================
# Analysis Logic
# ============================================================
def analyze_one_preset(preset_id: str, cfg):
    infile = os.path.join("out", preset_id, "df_result.csv")
    if not os.path.exists(infile):
        print(f"Skipping {preset_id}: No result file found.")
        return None

    out_dir = os.path.join("out", preset_id)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Analyzing: {preset_id}")
    df = pd.read_csv(infile)

    # SoH proxy
    soc_base = df["soc"].to_numpy(dtype=float)
    soc_mpc = df["soc_mpc"].to_numpy(dtype=float)
    soh_base = compute_soh_sqrt_calendar(soc_base, cfg)
    soh_mpc = compute_soh_sqrt_calendar(soc_mpc, cfg)
    df["soh"] = soh_base
    df["soh_mpc"] = soh_mpc

    # Core KPIs
    energy_base = energy_cost_eur(df["import"], df["export"], cfg)
    energy_mpc = energy_cost_eur(df["import_mpc"], df["export_mpc"],
                                 cfg)

    scr_base = calc_scr(df["pv"].to_numpy(dtype=float),
                        df["pv_consumption"].to_numpy(dtype=float))
    scr_mpc = calc_scr(df["pv"].to_numpy(dtype=float),
                       df["pv_consumption_mpc"].to_numpy(dtype=float))

    ss_base = calc_ss(df["consumption"].to_numpy(dtype=float),
                      df["import"].to_numpy(dtype=float))
    ss_mpc = calc_ss(df["consumption"].to_numpy(dtype=float),
                     df["import_mpc"].to_numpy(dtype=float))

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
    grid_mpc = df["import_mpc"].to_numpy(dtype=float) - df[
        "export_mpc"].to_numpy(dtype=float)
    batt_mpc = df["discharge_mpc"].to_numpy(dtype=float) - df[
        "charge_mpc"].to_numpy(dtype=float)
    resid_mpc = df["consumption"].to_numpy(dtype=float) - df[
        "pv"].to_numpy(dtype=float) - batt_mpc - grid_mpc

    # Baseline residual (dataset consistency)
    grid_base = df["import"].to_numpy(dtype=float) - df[
        "export"].to_numpy(dtype=float)
    batt_base = df["discharge"].to_numpy(dtype=float) - df[
        "charge"].to_numpy(dtype=float)
    resid_base = df["consumption"].to_numpy(dtype=float) - df[
        "pv"].to_numpy(dtype=float) - batt_base - grid_base

    # Audits
    eps = 1e-3
    sim_ch_raw = df["charge_mpc_raw"].to_numpy(dtype=float)
    sim_dis_raw = df["discharge_mpc_raw"].to_numpy(dtype=float)
    simult_steps_raw = int(np.sum((sim_ch_raw > eps) & (sim_dis_raw > eps)))
    simult_rate_raw_pct = 100.0 * simult_steps_raw / max(1, len(df))

    sim_ch = df["charge_mpc"].to_numpy(dtype=float)
    sim_dis = df["discharge_mpc"].to_numpy(dtype=float)
    simult_steps_after = int(np.sum((sim_ch > eps) & (sim_dis > eps)))
    simult_rate_after_pct = 100.0 * simult_steps_after / max(1, len(df))

    # Clamp audit
    clamp_delta_kw = df["audit_clamp_delta_charge_kw"].to_numpy(dtype=float)
    grid_charge_prevented_kwh_equiv = float(np.nansum(clamp_delta_kw) * cfg.DT)
    clamp_active_steps = int(
        np.sum(df["audit_clamp_active_flag"].to_numpy(dtype=int)))
    clamp_active_rate_pct = 100.0 * clamp_active_steps / max(1, len(df))

    # Projection audit (throughput delta)
    proj_delta_thr_kw = df["audit_projection_delta_throughput_kw"].to_numpy(
        dtype=float)
    projection_delta_throughput_kwh_equiv = float(
        np.nansum(proj_delta_thr_kw) * cfg.DT)

    # Cost deltas
    delta_energy_eur = energy_mpc - energy_base
    delta_energy_pct = 100.0 * delta_energy_eur / max(1e-12, energy_base)
    health_improvement_pct = 100.0 * (
            soh_drop_base_pct - soh_drop_mpc_pct) / max(1e-12,
                                                        soh_drop_base_pct)

    # Console summary (human)
    print("\n=== KPI SUMMARY (SIM WINDOW) ===")
    print(f"Preset:               {preset_id}")
    print(f"Energy cost baseline: {energy_base:,.2f} €")
    print(f"Energy cost MPC:      {energy_mpc:,.2f} €")
    print(
        f"Δ Energy cost:        {delta_energy_eur:,.2f} € ({delta_energy_pct:.2f}%)")
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
    print(
        f"Dwell SoC>=0.95 [h]:  base={dwell95_base:,.2f}, mpc={dwell95_mpc:,.2f}")
    print(
        f"Dwell SoC>=0.90 [h]:  base={dwell90_base:,.2f}, mpc={dwell90_mpc:,.2f}")
    print(
        f"Dwell SoC>=0.85 [h]:  base={dwell85_base:,.2f}, mpc={dwell85_mpc:,.2f}")
    print(
        f"QP simult ch/dis:     {simult_steps_raw} steps ({simult_rate_raw_pct:.3f}%)")
    print(
        f"After actuation:      {simult_steps_after} steps ({simult_rate_after_pct:.6f}%)")
    print(
        f"Clamp active:         {clamp_active_steps} steps ({clamp_active_rate_pct:.3f}%)")
    print(
        f"Grid-charge prevented (kWh eq): {grid_charge_prevented_kwh_equiv:.6f}")
    print(
        f"Projection Δthroughput (kWh eq): {projection_delta_throughput_kwh_equiv:.6f}")
    print(f"Residual max|.| BASE: {np.max(np.abs(resid_base)):.6f} kW")
    print(f"Residual RMS  BASE:   {np.sqrt(np.mean(resid_base ** 2)):.6f} kW")
    print(f"Residual max|.| MPC:  {np.max(np.abs(resid_mpc)):.6f} kW")
    print(f"Residual RMS  MPC:    {np.sqrt(np.mean(resid_mpc ** 2)):.6f} kW")

    # Save outputs for this preset
    df.to_csv(os.path.join(out_dir, "df_result.csv"), index=False)
    df.to_csv(os.path.join(out_dir, "df_result_sim.csv"), index=False)

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
        ("mpc_grid_charge_prevented_kwh_equiv", np.nan,
         grid_charge_prevented_kwh_equiv),
        ("mpc_projection_delta_throughput_kwh_equiv", np.nan,
         projection_delta_throughput_kwh_equiv),

        ("baseline_balance_residual_max_abs_kw",
         float(np.max(np.abs(resid_base))), np.nan),
        ("baseline_balance_residual_rms_kw",
         float(np.sqrt(np.mean(resid_base ** 2))), np.nan),
        ("mpc_balance_residual_max_abs_kw", np.nan,
         float(np.max(np.abs(resid_mpc)))),
        ("mpc_balance_residual_rms_kw", np.nan,
         float(np.sqrt(np.mean(resid_mpc ** 2)))),
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
    pd.DataFrame([kpi_summary]).to_csv(os.path.join(out_dir, "kpi_summary.csv"),
                                       index=False)

    # ============================================================
    # Plots
    # ============================================================
    save_lineplot_pdf(
        df, xcol="time",
        y_col_map={"pv": "PV-Generation", "consumption": "Consumption"},
        title="PV-Generation and Consumption",
        outfile=os.path.join(plot_dir, "01_pv_vs_load.pdf"), ylabel="kW"
    )
    save_lineplot_pdf(
        df, xcol="time",
        y_col_map={"soc": "Baseline", "soc_mpc": "MPC"},
        title=f"SoC Comparison",
        outfile=os.path.join(plot_dir, "02_soc_compare.pdf"), ylabel="SoC"
    )
    save_lineplot_pdf(
        df, xcol="time",
        y_col_map={
            "import": "Import Baseline",
            "import_mpc": "Import MPC",
            "export": "Export Baseline",
            "export_mpc": "Export MPC",
        },
        title="Grid flows", outfile=os.path.join(plot_dir, "03_grid_flows.pdf"),
        ylabel="kW"
    )
    save_lineplot_pdf(
        df, xcol="time",
        y_col_map={
            "charge": "Charge Baseline",
            "charge_mpc": "Charge MPC",
            "discharge": "Discharge Baseline",
            "discharge_mpc": "Discharge MPC",
        },
        title="Battery Power",
        outfile=os.path.join(plot_dir, "04_batt_power.pdf"), ylabel="kW"
    )
    save_lineplot_pdf(
        df, xcol="time",
        y_col_map={
            "soh": "SoH Baseline",
            "soh_mpc": "SoH MPC",
        },
        title="SoH Proxy (sqrt calendar)",
        outfile=os.path.join(plot_dir, "05_soh_proxy.pdf"), ylabel="SoH"
    )
    save_hist_pdf(
        df, col_a="soc", col_b="soc_mpc",
        label_a="Baseline",
        label_b="MPC",
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
    df["balance_residual_mpc"] = resid_mpc
    save_lineplot_pdf(
        df, xcol="time",
        y_col_map={"balance_residual_mpc": "MPC",},
        title="Energy Balance Residual",
        outfile=os.path.join(plot_dir, "08_balance_residual_mpc.pdf"),
        ylabel="kW"
    )

    # Actuation audit plots (optional for report; useful for appendix)
    df["audit_clamp_delta_charge_kw"] = df[
        "audit_clamp_delta_charge_kw"].fillna(0.0)
    df["audit_projection_delta_throughput_kw"] = df[
        "audit_projection_delta_throughput_kw"].fillna(0.0)

    save_step_series_pdf(
        df, xcol="time", ycol="audit_clamp_delta_charge_kw",
        title="Actuation audit: charge clamp delta (kW) (raw - after measured-surplus cap)",
        outfile=os.path.join(plot_dir, "09_audit_clamp_delta_charge.pdf"),
        ylabel="kW"
    )
    save_step_series_pdf(
        df, xcol="time", ycol="audit_projection_delta_throughput_kw",
        title="Actuation audit: |throughput_after - throughput_raw| (kW)",
        outfile=os.path.join(plot_dir,
                             "10_audit_projection_delta_throughput.pdf"),
        ylabel="kW"
    )

    summary = kpi_summary
    return summary


def main():
    # Detect presets by looking at output folders
    if not os.path.exists("out"):
        print("No 'out' folder found. Run pipeline.py first.")
        return

    presets = [d for d in os.listdir("out") if
               os.path.isdir(os.path.join("out", d))]
    summaries = []

    cfg = Config()

    for preset in presets:
        res = analyze_one_preset(preset, cfg)
        if res: summaries.append(res)

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