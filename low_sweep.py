# low_sweep_v2.py
# Fine-tuning sweep for LOW penalties with the strengths of sweep_penalties_v2.py
#
# Usage:
#   python low_sweep_v2.py
#
# Outputs:
#   out/low_sweep_v2/low_sweep_v2_all_windows.csv
#   out/low_sweep_v2/low_sweep_v2_agg_by_setting.csv
#   out/low_sweep_v2/plots/*.png
#
# Notes:
# - Robust window slicing includes horizon tail so MPC always has forecast length.
# - KPIs match sweep_penalties_v2 style to be paper-compatible.
# - "grid-charge" audit is measured as steps where (charge_mpc>eps AND import_mpc>eps).
#   (This is a conservative proxy: it flags any charging while importing.)

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pipeline as pl  # uses your existing pipeline.py


# ============================================================
# User knobs (speed vs representativeness)
# ============================================================
OUT_DIR = "out"
RUN_DIR = os.path.join(OUT_DIR, "low_sweep_v2")
PLOT_DIR = os.path.join(RUN_DIR, "plots")

# Windowing (paper-friendly)
WINDOW_DAYS = 7  # 7 is a good "fast but still credible" tuning window; set 14 for more robustness
WINDOW_FRACTIONS = [0.18, 0.55, 0.82]  # early/mid/late across Jul->Jan
WINDOW_NAMES = ["W1_early", "W2_mid", "W3_late"]

# Optional: manual windows by date (must exist in df["time"])
MANUAL_WINDOWS = []  # e.g. [("summer", "2025-08-10"), ("autumn","2025-10-10"), ("winter","2025-12-10")]

# Optional speed knob: only if your pipeline supports it
# We'll try to set a few likely attribute names safely; if pipeline ignores, that's fine.
MPC_INTERVAL_STEPS = 12  # 12=1h at 5-min steps, 24=2h. If unsupported, no effect.

# Filter region for "cost-similar" (your goal ±1–3%)
DELTA_COST_PCT_BAND = 3.0

# Penalty design
SOC_TARGET = 0.20  # keep fixed for low sweep (can add 0.50 later if you want)


# ============================================================
# Helpers (same style as sweep_v2)
# ============================================================
def safe_div(a, b):
    a = float(a)
    b = float(b)
    return a / b if b > 1e-12 else np.nan


def calc_scr(pv, pv_consumption):
    # SCR = PV used locally / PV generated
    return safe_div(np.sum(pv_consumption), np.sum(pv))


def calc_ss(load, imp):
    # SS = 1 - (grid import / total load)
    return 1.0 - safe_div(np.sum(imp), np.sum(load))


def make_cfg(base_cfg, **overrides):
    cfg = copy.copy(base_cfg)
    for k, v in overrides.items():
        setattr(cfg, k, v)

    # Try to apply "replan interval" to speed up sweep if pipeline supports it
    # (multiple likely names, harmless if unused)
    for maybe_name in [
        "MPC_INTERVAL_STEPS",
        "MPC_REPLAN_EVERY",
        "REPLAN_EVERY_STEPS",
        "N_APPLY",  # sometimes used as 'apply n steps per solve'
    ]:
        try:
            setattr(cfg, maybe_name, int(MPC_INTERVAL_STEPS))
        except Exception:
            pass

    return cfg


def slice_window_by_start_idx(df, start_idx, cfg, window_days):
    """
    Slice by index to guarantee enough tail for the MPC horizon.
    window length = window_days * samples_per_day (+ horizon tail)
    """
    base_len = int(window_days * cfg.SAMPLES_PER_DAY)
    need = base_len + int(cfg.HORIZON_SAMPLES) + 10

    start_idx = int(start_idx)
    end_idx = min(len(df), start_idx + need)
    df_win = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    if len(df_win) <= int(cfg.HORIZON_SAMPLES) + 20:
        raise ValueError("Window slice too small for horizon. Choose another start.")

    return df_win, base_len


def build_windows(df, cfg):
    if "time" not in df.columns:
        raise ValueError("df has no 'time' column. Ensure pl.get_table() created it.")

    t = pd.to_datetime(df["time"])
    tmp = df.copy()
    tmp["__date"] = t.dt.normalize()

    unique_dates = np.array(sorted(tmp["__date"].unique()))
    total_days = len(unique_dates)
    if total_days < WINDOW_DAYS + 5:
        raise ValueError("Not enough days in dataset for chosen WINDOW_DAYS.")

    windows = []

    if MANUAL_WINDOWS:
        for name, date_str in MANUAL_WINDOWS:
            d = pd.Timestamp(date_str).normalize()
            possible = tmp.index[tmp["__date"] >= d]
            if len(possible) == 0:
                raise ValueError(f"Manual window date {date_str} not found in dataset range.")
            start_idx = int(possible[0])
            windows.append((name, start_idx))
        return windows

    for name, frac in zip(WINDOW_NAMES, WINDOW_FRACTIONS):
        pos = int(frac * (total_days - 1))
        pos = max(0, min(total_days - 1, pos))
        start_date = unique_dates[pos]
        start_idx = int(tmp.index[tmp["__date"] == start_date][0])
        windows.append((name, start_idx))

    return windows


def compute_kpis(df_sim, cfg):
    # Baseline signals
    pv = df_sim["pv"].to_numpy(dtype=float)
    load = df_sim["consumption"].to_numpy(dtype=float)

    soc_b = df_sim["soc"].to_numpy(dtype=float)
    imp_b = df_sim["import"].to_numpy(dtype=float)
    exp_b = df_sim["export"].to_numpy(dtype=float)
    pvc_b = df_sim["pv_consumption"].to_numpy(dtype=float)

    # MPC signals
    soc_m = df_sim["soc_mpc"].to_numpy(dtype=float)
    imp_m = df_sim["import_mpc"].to_numpy(dtype=float)
    exp_m = df_sim["export_mpc"].to_numpy(dtype=float)
    ch_m  = df_sim["charge_mpc"].to_numpy(dtype=float)
    dis_m = df_sim["discharge_mpc"].to_numpy(dtype=float)
    pvc_m = df_sim["pv_consumption_mpc"].to_numpy(dtype=float)

    # Energy cost
    e_b = pl.energy_cost_eur(imp_b, exp_b, cfg)
    e_m = pl.energy_cost_eur(imp_m, exp_m, cfg)

    # SCR / SS
    scr_b = calc_scr(pv, pvc_b)
    scr_m = calc_scr(pv, pvc_m)

    ss_b = calc_ss(load, imp_b)
    ss_m = calc_ss(load, imp_m)

    # SoH proxy
    soh_b = pl.compute_soh_sqrt_calendar(soc_b, cfg)
    soh_m = pl.compute_soh_sqrt_calendar(soc_m, cfg)
    soh_drop_b = (1.0 - float(soh_b[-1])) * 100.0
    soh_drop_m = (1.0 - float(soh_m[-1])) * 100.0

    aging_eur_b = soh_drop_b * cfg.EUR_PER_SOH_PERCENT
    aging_eur_m = soh_drop_m * cfg.EUR_PER_SOH_PERCENT

    # EoL projection
    days_to_80_b = pl.estimate_days_to_eol_from_window(soh_b, cfg)
    days_to_80_m = pl.estimate_days_to_eol_from_window(soh_m, cfg)

    # Dwell
    d85_b = pl.dwell_hours(soc_b, 0.85, cfg)
    d85_m = pl.dwell_hours(soc_m, 0.85, cfg)
    d90_b = pl.dwell_hours(soc_b, 0.90, cfg)
    d90_m = pl.dwell_hours(soc_m, 0.90, cfg)
    d95_b = pl.dwell_hours(soc_b, 0.95, cfg)
    d95_m = pl.dwell_hours(soc_m, 0.95, cfg)

    # Residual audit (MPC should be ~0)
    grid_m = imp_m - exp_m
    batt_m = dis_m - ch_m
    resid_m = load - pv - batt_m - grid_m
    resid_max = float(np.max(np.abs(resid_m)))
    resid_rms = float(np.sqrt(np.mean(resid_m**2)))

    # Simultaneous charge/discharge & grid-charging audits
    eps = 1e-3
    simult_steps = int(np.sum((ch_m > eps) & (dis_m > eps)))
    simult_rate = 100.0 * simult_steps / max(1, len(df_sim))

    gridchg_steps = int(np.sum((ch_m > eps) & (imp_m > eps)))
    gridchg_rate = 100.0 * gridchg_steps / max(1, len(df_sim))

    # One-number proxy (label as proxy in report!)
    total_proxy_b = e_b + aging_eur_b
    total_proxy_m = e_m + aging_eur_m

    # Health improvement metric (positive is better)
    health_impr_pct = 100.0 * (aging_eur_b - aging_eur_m) / max(1e-9, aging_eur_b)

    # Δcost percent (relative to baseline) for filtering band
    delta_cost_pct = 100.0 * (e_m - e_b) / max(1e-9, abs(e_b))

    return {
        "energy_cost_eur_baseline": e_b,
        "energy_cost_eur_mpc": e_m,
        "delta_energy_eur": e_m - e_b,
        "delta_cost_pct": delta_cost_pct,

        "scr_baseline": scr_b,
        "scr_mpc": scr_m,
        "delta_scr": scr_m - scr_b,

        "ss_baseline": ss_b,
        "ss_mpc": ss_m,
        "delta_ss": ss_m - ss_b,

        "soh_drop_pct_baseline": soh_drop_b,
        "soh_drop_pct_mpc": soh_drop_m,
        "delta_soh_drop_pct": soh_drop_m - soh_drop_b,

        "aging_proxy_eur_baseline": aging_eur_b,
        "aging_proxy_eur_mpc": aging_eur_m,
        "delta_aging_proxy_eur": aging_eur_m - aging_eur_b,
        "health_improvement_pct": health_impr_pct,

        "total_proxy_eur_baseline": total_proxy_b,
        "total_proxy_eur_mpc": total_proxy_m,
        "delta_total_proxy_eur": total_proxy_m - total_proxy_b,

        "days_to_80pct_baseline": days_to_80_b,
        "days_to_80pct_mpc": days_to_80_m,

        "dwell_h_ge_0p85_baseline": d85_b,
        "dwell_h_ge_0p85_mpc": d85_m,
        "dwell_h_ge_0p90_baseline": d90_b,
        "dwell_h_ge_0p90_mpc": d90_m,
        "dwell_h_ge_0p95_baseline": d95_b,
        "dwell_h_ge_0p95_mpc": d95_m,

        "mpc_simult_ch_dis_rate_pct": simult_rate,
        "mpc_grid_charge_rate_pct": gridchg_rate,
        "mpc_balance_residual_max_abs_kw": resid_max,
        "mpc_balance_residual_rms_kw": resid_rms,
    }


def pareto_plot_delta(summary_df, outpath, title, annotate=False):
    """
    Plot: ΔEnergy [€] (x) vs Health Improvement [%] (y).
    """
    x = summary_df["delta_energy_eur"].to_numpy()
    y = summary_df["health_improvement_pct"].to_numpy()

    plt.figure(figsize=(9, 5))
    plt.scatter(x, y)
    plt.axvline(0.0, linewidth=1)
    plt.axhline(0.0, linewidth=1)

    if annotate:
        labels = summary_df["setting_id"].astype(str).to_numpy()
        for xi, yi, lab in zip(x, y, labels):
            plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("Δ Energy cost [€] (MPC - baseline)  (lower is better)")
    plt.ylabel("Health improvement [%] (higher is better)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def band_plot_cost_health(df, outpath, title):
    """
    Highlight the band of ±DELTA_COST_PCT_BAND and show trade-off.
    """
    plt.figure(figsize=(9, 5))
    plt.scatter(df["delta_cost_pct"], df["health_improvement_pct"])
    plt.axvline(+DELTA_COST_PCT_BAND, linewidth=1)
    plt.axvline(-DELTA_COST_PCT_BAND, linewidth=1)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Δ Energy cost [%] vs baseline")
    plt.ylabel("Health improvement [%]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ============================================================
# LOW penalty settings grid (coarse + compact)
# ============================================================
def get_low_settings():
    """
    Compact low-penalty grid:
    - Sweeps COST_SOC_HOLDING (overall strength)
    - Sweeps W_HIGH_85 and W_HIGH_95 (high SoC discouragement)
    - Keeps below/above target terms at 0 for 'low sweep' focus
    """
    c_vals   = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    w85_vals = [0.00, 0.25, 0.50, 0.75, 1.00]
    w95_vals = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

    settings = []
    for c in c_vals:
        for w85 in w85_vals:
            for w95 in w95_vals:
                name = f"LOW_c{c:.2f}_w85{w85:.2f}_w95{w95:.2f}"
                settings.append((
                    name,
                    dict(
                        SOC_TARGET=SOC_TARGET,
                        COST_SOC_HOLDING=float(c),
                        W_BELOW_TARGET=0.0,
                        W_ABOVE_TARGET=0.0,
                        W_HIGH_85=float(w85),
                        W_HIGH_95=float(w95),
                    )
                ))

    # Add a true "calendar off" reference (should be most economic-ish)
    settings.insert(0, (
        "LOW_P0_calendar_off",
        dict(
            SOC_TARGET=SOC_TARGET,
            COST_SOC_HOLDING=0.0,
            W_BELOW_TARGET=0.0,
            W_ABOVE_TARGET=0.0,
            W_HIGH_85=0.0,
            W_HIGH_95=0.0,
        )
    ))

    return settings


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    base_cfg = pl.Config()

    # Load once (same as sweep_v2 usage)
    df = pl.get_table(os.path.join(OUT_DIR, "df.csv"), "data/days-range")

    windows = build_windows(df, base_cfg)
    settings = get_low_settings()

    print(f"Running LOW sweep v2: {len(settings)} settings x {len(windows)} windows")
    print(f"Config: WINDOW_DAYS={WINDOW_DAYS}, MPC_INTERVAL_STEPS={MPC_INTERVAL_STEPS}, SOC_TARGET={SOC_TARGET}")

    all_rows = []

    for win_name, start_idx in windows:
        print("\n==============================")
        print(f"WINDOW {win_name}: start_idx={start_idx}, days={WINDOW_DAYS}")
        print("==============================")

        df_win, base_len = slice_window_by_start_idx(df, start_idx, base_cfg, WINDOW_DAYS)

        win_rows = []
        for i, (setting_id, ov) in enumerate(settings, start=1):
            print(f"[{i}/{len(settings)}] {setting_id}")

            cfg = make_cfg(base_cfg, **ov)

            # Run simulation for this cfg on the window
            try:
                df_out, sim_end = pl.run_simulation(df_win.copy(), cfg)
            except Exception as e:
                # keep sweep running, store failure row
                win_rows.append({
                    "window_id": win_name,
                    "window_days": WINDOW_DAYS,
                    "sim_steps": 0,
                    "start_idx": int(start_idx),
                    "setting_id": setting_id,
                    "failed": True,
                    "fail_reason": str(e)[:200],
                    **ov
                })
                continue

            sim_end = int(min(sim_end, base_len))
            df_sim = df_out.iloc[:sim_end].copy()

            k = compute_kpis(df_sim, cfg)

            row = {
                "window_id": win_name,
                "window_days": WINDOW_DAYS,
                "sim_steps": sim_end,
                "start_idx": int(start_idx),
                "setting_id": setting_id,
                "failed": False,
            }
            row.update(ov)
            row.update(k)

            # Sanity warning (residual should be ~0)
            if k["mpc_balance_residual_max_abs_kw"] > 1e-6:
                print(f"  [WARN] residual max abs = {k['mpc_balance_residual_max_abs_kw']:.3e} kW")

            win_rows.append(row)
            all_rows.append(row)

        win_df = pd.DataFrame(win_rows)

        out_csv = os.path.join(RUN_DIR, f"low_sweep_v2_{win_name}.csv")
        win_df.to_csv(out_csv, index=False)
        print(f"Saved window CSV: {out_csv}")

        # Pareto plot (only successful rows)
        ok = win_df[(win_df["failed"] == False)].copy()
        if len(ok) > 0:
            out_plot = os.path.join(PLOT_DIR, f"pareto_delta_{win_name}.png")
            pareto_plot_delta(ok, out_plot, title=f"LOW Pareto (Δ€ vs Health) - {win_name}", annotate=False)
            print(f"Saved window Pareto plot: {out_plot}")

            out_band = os.path.join(PLOT_DIR, f"band_costpct_vs_health_{win_name}.png")
            band_plot_cost_health(ok, out_band, title=f"LOW sweep band (Δ% vs Health) - {win_name}")
            print(f"Saved window band plot: {out_band}")
        else:
            print("[WARN] No successful rows in this window; plots skipped.")

    # Combined
    all_df = pd.DataFrame(all_rows)
    out_all = os.path.join(RUN_DIR, "low_sweep_v2_all_windows.csv")
    all_df.to_csv(out_all, index=False)
    print(f"\nSaved combined CSV: {out_all}")

    if len(all_df) == 0:
        print("[ERROR] No results at all. Check pipeline.run_simulation and data loading.")
        return

    # Aggregate by setting across windows (mean over windows)
    ok_all = all_df[(all_df["failed"] == False)].copy()
    if len(ok_all) == 0:
        print("[ERROR] All rows failed. Inspect fail_reason in CSV.")
        return

    agg = ok_all.groupby("setting_id", as_index=False).agg({
        "delta_energy_eur": "mean",
        "delta_cost_pct": "mean",
        "health_improvement_pct": "mean",
        "ss_mpc": "mean",
        "scr_mpc": "mean",
        "mpc_simult_ch_dis_rate_pct": "mean",
        "mpc_grid_charge_rate_pct": "mean",
        "mpc_balance_residual_max_abs_kw": "max",
        "total_proxy_eur_mpc": "mean",
        "aging_proxy_eur_mpc": "mean",
        "energy_cost_eur_mpc": "mean",
        "dwell_h_ge_0p95_mpc": "mean",
    })

    out_agg = os.path.join(RUN_DIR, "low_sweep_v2_agg_by_setting.csv")
    agg.to_csv(out_agg, index=False)
    print(f"Saved aggregate CSV: {out_agg}")

    # Plot aggregate Pareto
    out_plot_agg = os.path.join(PLOT_DIR, "pareto_delta_agg.png")
    pareto_plot_delta(agg, out_plot_agg, title="LOW sweep (aggregate over windows): Δ€ vs Health", annotate=False)
    print(f"Saved aggregate Pareto plot: {out_plot_agg}")

    # Focus: cost-similar band ±DELTA_COST_PCT_BAND
    band = agg[(agg["delta_cost_pct"] >= -DELTA_COST_PCT_BAND) & (agg["delta_cost_pct"] <= DELTA_COST_PCT_BAND)].copy()
    if len(band) > 0:
        out_band = os.path.join(PLOT_DIR, "pareto_band_cost_similar.png")
        pareto_plot_delta(band, out_band, title=f"LOW sweep band (±{DELTA_COST_PCT_BAND:.1f}% cost): Δ€ vs Health", annotate=True)
        print(f"Saved band Pareto plot: {out_band}")

        # Top picks within band:
        # 1) best health improvement within band
        pick_health = band.sort_values("health_improvement_pct", ascending=False).head(5)
        # 2) strict operational cleanliness within band (lowest simult+gridchg)
        band["op_badness"] = band["mpc_simult_ch_dis_rate_pct"] + band["mpc_grid_charge_rate_pct"]
        pick_clean = band.sort_values(["op_badness", "delta_cost_pct"], ascending=[True, True]).head(5)

        print("\n==============================")
        print(f"TOP picks within ±{DELTA_COST_PCT_BAND:.1f}% cost band")
        print("==============================")
        cols = ["setting_id", "delta_cost_pct", "delta_energy_eur", "health_improvement_pct",
                "mpc_grid_charge_rate_pct", "mpc_simult_ch_dis_rate_pct", "dwell_h_ge_0p95_mpc"]
        print("\nBest health (band):")
        print(pick_health[cols].to_string(index=False))
        print("\nClean ops (band):")
        print(pick_clean[cols].to_string(index=False))
    else:
        print(f"[WARN] No settings within ±{DELTA_COST_PCT_BAND:.1f}% cost band (aggregate). Consider lowering penalties further.")

    # Also store a compact report-ready table (band only if exists else overall top by health)
    if len(band) > 0:
        report_tbl = band.sort_values(["health_improvement_pct"], ascending=False).head(15)
    else:
        report_tbl = agg.sort_values(["health_improvement_pct"], ascending=False).head(15)

    out_tbl = os.path.join(RUN_DIR, "low_sweep_v2_report_table.csv")
    report_tbl.to_csv(out_tbl, index=False)
    print(f"\nSaved report table: {out_tbl}")


if __name__ == "__main__":
    main()
