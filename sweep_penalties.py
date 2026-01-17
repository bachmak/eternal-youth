# sweep_penalties_v2.py
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pipeline as pl  # uses your existing pipeline.py


# ============================================================
# User knobs
# ============================================================
WINDOW_DAYS = 14                 # 14 is fast + representative (2 weeks)
WINDOW_FRACTIONS = [0.18, 0.55, 0.82]  # roughly summer / autumn / winter across Jul->Jan
WINDOW_NAMES = ["W1_early", "W2_mid", "W3_late"]

# If you want manual windows instead, uncomment and set date strings that exist in your df["time"]:
# MANUAL_WINDOWS = [
#     ("summer_like", "2024-08-10"),
#     ("autumn_like", "2024-10-10"),
#     ("winter_like", "2024-12-10"),
# ]
MANUAL_WINDOWS = []

OUT_DIR = "out"
PLOT_DIR = os.path.join(OUT_DIR, "plots_sweep_v2")


# ============================================================
# Helpers
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
    return cfg


def slice_window_by_start_idx(df, start_idx, cfg, window_days):
    """
    Slice by index to guarantee enough tail for the MPC horizon.
    window length = window_days * 288 steps (+ horizon tail)
    """
    base_len = int(window_days * cfg.SAMPLES_PER_DAY)
    need = base_len + int(cfg.HORIZON_SAMPLES) + 10

    start_idx = int(start_idx)
    end_idx = min(len(df), start_idx + need)
    df_win = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    if len(df_win) <= cfg.HORIZON_SAMPLES + 20:
        raise ValueError("Window slice too small for horizon. Choose another start.")

    # We will evaluate only the simulated part (run_simulation returns sim_end)
    return df_win, base_len


def compute_kpis(df_sim, cfg):
    # Baseline
    pv = df_sim["pv"].to_numpy(dtype=float)
    load = df_sim["consumption"].to_numpy(dtype=float)

    soc_b = df_sim["soc"].to_numpy(dtype=float)
    imp_b = df_sim["import"].to_numpy(dtype=float)
    exp_b = df_sim["export"].to_numpy(dtype=float)
    pvc_b = df_sim["pv_consumption"].to_numpy(dtype=float)

    # MPC
    soc_m = df_sim["soc_mpc"].to_numpy(dtype=float)
    imp_m = df_sim["import_mpc"].to_numpy(dtype=float)
    exp_m = df_sim["export_mpc"].to_numpy(dtype=float)
    ch_m  = df_sim["charge_mpc"].to_numpy(dtype=float)
    dis_m = df_sim["discharge_mpc"].to_numpy(dtype=float)
    pvc_m = df_sim["pv_consumption_mpc"].to_numpy(dtype=float)

    # Energy costs
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

    # Simultaneous charge/discharge audit
    eps = 1e-3
    simult_steps = int(np.sum((ch_m > eps) & (dis_m > eps)))
    simult_rate = 100.0 * simult_steps / max(1, len(df_sim))

    # One-number proxy (clearly label as proxy in report!)
    total_proxy_b = e_b + aging_eur_b
    total_proxy_m = e_m + aging_eur_m

    # Health improvement metric (positive is better)
    health_impr_pct = 100.0 * (aging_eur_b - aging_eur_m) / max(1e-9, aging_eur_b)

    return {
        "energy_cost_eur_baseline": e_b,
        "energy_cost_eur_mpc": e_m,
        "delta_energy_eur": e_m - e_b,

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
        "mpc_balance_residual_max_abs_kw": resid_max,
        "mpc_balance_residual_rms_kw": resid_rms,
    }


def pareto_plot_delta(summary_df, outpath, title):
    """
    Plot: ΔEnergy (x) vs Health Improvement (%) (y).
    Baseline is implicitly at (0,0).
    """
    x = summary_df["delta_energy_eur"].to_numpy()
    y = summary_df["health_improvement_pct"].to_numpy()
    labels = summary_df["setting_id"].astype(str).to_numpy()

    plt.figure(figsize=(9, 5))
    plt.scatter(x, y)
    plt.axvline(0.0, linewidth=1)
    plt.axhline(0.0, linewidth=1)
    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Δ Energy cost [€] (MPC - baseline)  (lower is better)")
    plt.ylabel("Health improvement [%] (higher is better)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def build_windows(df, cfg):
    """
    If MANUAL_WINDOWS is set: pick the first index of that date.
    Else: pick indices based on fractions across unique days.
    Returns list of (window_name, start_idx)
    """
    if "time" not in df.columns:
        raise ValueError("df has no 'time' column. Ensure get_table() created it.")

    t = pd.to_datetime(df["time"])
    df = df.copy()
    df["__date"] = t.dt.normalize()

    unique_dates = np.array(sorted(df["__date"].unique()))
    total_days = len(unique_dates)
    if total_days < WINDOW_DAYS + 5:
        raise ValueError("Not enough days in dataset for chosen WINDOW_DAYS.")

    windows = []

    if MANUAL_WINDOWS:
        for name, date_str in MANUAL_WINDOWS:
            d = pd.Timestamp(date_str).normalize()
            # find first occurrence at/after that date
            possible = df.index[df["__date"] >= d]
            if len(possible) == 0:
                raise ValueError(f"Manual window date {date_str} not found in dataset range.")
            start_idx = int(possible[0])
            windows.append((name, start_idx))
        return windows

    # auto windows based on fractions along the dataset days
    for name, frac in zip(WINDOW_NAMES, WINDOW_FRACTIONS):
        pos = int(frac * (total_days - 1))
        pos = max(0, min(total_days - 1, pos))
        start_date = unique_dates[pos]
        start_idx = int(df.index[df["__date"] == start_date][0])
        windows.append((name, start_idx))

    return windows


# ============================================================
# Settings (wider range to actually create trade-offs)
# ============================================================
def get_settings():
    """
    Curated set: includes 'calendar off' and very mild -> strong.
    Also includes SOC_TARGET=0.5 variants to reduce the "keep it low" bias.
    """
    return [
        # Calendar OFF: pure energy + wear (should be closer to economic behavior)
        ("P0_energy_only", dict(
            COST_SOC_HOLDING=0.0, W_BELOW_TARGET=0.0, W_ABOVE_TARGET=0.0, W_HIGH_85=0.0, W_HIGH_95=0.0,
            SOC_TARGET=0.20
        )),

        # Very mild high-SoC discouragement
        ("P1_high_ultra_low", dict(
            COST_SOC_HOLDING=0.01, W_BELOW_TARGET=0.0, W_ABOVE_TARGET=0.00, W_HIGH_85=0.08, W_HIGH_95=0.30,
            SOC_TARGET=0.20
        )),
        ("P2_high_low", dict(
            COST_SOC_HOLDING=0.03, W_BELOW_TARGET=0.0, W_ABOVE_TARGET=0.00, W_HIGH_85=0.20, W_HIGH_95=0.80,
            SOC_TARGET=0.20
        )),
        ("P3_high_mid", dict(
            COST_SOC_HOLDING=0.06, W_BELOW_TARGET=0.0, W_ABOVE_TARGET=0.00, W_HIGH_85=0.45, W_HIGH_95=1.80,
            SOC_TARGET=0.20
        )),
        ("P4_high_balanced", dict(
            COST_SOC_HOLDING=0.10, W_BELOW_TARGET=0.0, W_ABOVE_TARGET=0.00, W_HIGH_85=0.90, W_HIGH_95=3.20,
            SOC_TARGET=0.20
        )),
        ("P5_high_strong", dict(
            COST_SOC_HOLDING=0.18, W_BELOW_TARGET=0.0, W_ABOVE_TARGET=0.00, W_HIGH_85=1.40, W_HIGH_95=5.00,
            SOC_TARGET=0.20
        )),

        # Balanced with some "above target" shaping
        ("P6_target20_balanced", dict(
            COST_SOC_HOLDING=0.10, W_BELOW_TARGET=0.0, W_ABOVE_TARGET=0.06, W_HIGH_85=0.90, W_HIGH_95=3.20,
            SOC_TARGET=0.20
        )),
        # Your current max-life style
        ("P7_target20_maxlife", dict(
            COST_SOC_HOLDING=0.25, W_BELOW_TARGET=0.10, W_ABOVE_TARGET=0.15, W_HIGH_85=2.0, W_HIGH_95=8.0,
            SOC_TARGET=0.20
        )),

        # SOC_TARGET=0.5 variants (can preserve autonomy better in many cases)
        ("P8_target50_balanced", dict(
            COST_SOC_HOLDING=0.08, W_BELOW_TARGET=0.0, W_ABOVE_TARGET=0.04, W_HIGH_85=0.60, W_HIGH_95=2.00,
            SOC_TARGET=0.50
        )),
        ("P9_target50_strong", dict(
            COST_SOC_HOLDING=0.14, W_BELOW_TARGET=0.0, W_ABOVE_TARGET=0.06, W_HIGH_85=1.20, W_HIGH_95=4.00,
            SOC_TARGET=0.50
        )),
        ("P10_target50_maxlife", dict(
            COST_SOC_HOLDING=0.25, W_BELOW_TARGET=0.05, W_ABOVE_TARGET=0.10, W_HIGH_85=2.00, W_HIGH_95=8.00,
            SOC_TARGET=0.50
        )),
    ]


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    base_cfg = pl.Config()

    # Load once
    df = pl.get_table(os.path.join(OUT_DIR, "df.csv"), "data/days-range")

    windows = build_windows(df, base_cfg)
    settings = get_settings()

    all_rows = []

    for win_name, start_idx in windows:
        print(f"\n==============================")
        print(f"WINDOW {win_name}: start_idx={start_idx}, days={WINDOW_DAYS}")
        print(f"==============================")

        df_win, base_len = slice_window_by_start_idx(df, start_idx, base_cfg, WINDOW_DAYS)

        win_rows = []
        for setting_id, ov in settings:
            print(f"\n--- Running {setting_id} on {win_name} ---")
            cfg = make_cfg(base_cfg, **ov)

            # Run simulation for this cfg on the window
            df_out, sim_end = pl.run_simulation(df_win.copy(), cfg)

            # Limit evaluation to the simulated part; additionally clip to base_len if desired
            sim_end = int(min(sim_end, base_len))
            df_sim = df_out.iloc[:sim_end].copy()

            k = compute_kpis(df_sim, cfg)

            # Audit warning (should be ~0)
            if k["mpc_balance_residual_max_abs_kw"] > 1e-6:
                print(f"[WARN] Residual too large: {k['mpc_balance_residual_max_abs_kw']:.3e} kW")

            row = {
                "window_id": win_name,
                "window_days": WINDOW_DAYS,
                "sim_steps": sim_end,
                "start_idx": int(start_idx),
                "setting_id": setting_id,
            }
            row.update(ov)
            row.update(k)

            win_rows.append(row)
            all_rows.append(row)

        win_df = pd.DataFrame(win_rows)

        # Save per-window CSV
        out_csv = os.path.join(OUT_DIR, f"sweep_v2_{win_name}.csv")
        win_df.to_csv(out_csv, index=False)
        print(f"\nSaved window CSV: {out_csv}")

        # Pareto plot per window
        out_plot = os.path.join(PLOT_DIR, f"pareto_delta_{win_name}.png")
        pareto_plot_delta(
            win_df,
            out_plot,
            title=f"Pareto (ΔEnergy vs Health improvement) - {win_name}"
        )
        print(f"Saved window Pareto plot: {out_plot}")

        # Quick picks for this window
        # 1) Max-life: best health improvement
        maxlife = win_df.sort_values("health_improvement_pct", ascending=False).head(1)
        # 2) Economic: smallest ΔEnergy among those with at least 25% health improvement
        candidates = win_df[win_df["health_improvement_pct"] >= 25.0].copy()
        if len(candidates) > 0:
            economic = candidates.sort_values("delta_energy_eur", ascending=True).head(1)
        else:
            economic = win_df.sort_values("delta_energy_eur", ascending=True).head(1)
        # 3) Balanced: minimal total proxy (energy + aging proxy)
        balanced = win_df.sort_values("total_proxy_eur_mpc", ascending=True).head(1)

        print("\n[Window picks]")
        print("Max-life:", maxlife[["setting_id", "delta_energy_eur", "health_improvement_pct", "ss_mpc", "scr_mpc", "dwell_h_ge_0p85_mpc"]].to_string(index=False))
        print("Economic:", economic[["setting_id", "delta_energy_eur", "health_improvement_pct", "ss_mpc", "scr_mpc", "dwell_h_ge_0p85_mpc"]].to_string(index=False))
        print("Balanced:", balanced[["setting_id", "delta_energy_eur", "health_improvement_pct", "ss_mpc", "scr_mpc", "dwell_h_ge_0p85_mpc"]].to_string(index=False))

    # Combined summary across windows
    all_df = pd.DataFrame(all_rows)
    out_all = os.path.join(OUT_DIR, "sweep_v2_all_windows.csv")
    all_df.to_csv(out_all, index=False)
    print(f"\nSaved combined CSV: {out_all}")

    # Aggregate by setting across windows (mean)
    agg = all_df.groupby("setting_id", as_index=False).agg({
        "delta_energy_eur": "mean",
        "health_improvement_pct": "mean",
        "ss_mpc": "mean",
        "scr_mpc": "mean",
        "mpc_simult_ch_dis_rate_pct": "mean",
        "mpc_balance_residual_max_abs_kw": "max",
        "total_proxy_eur_mpc": "mean",
        "aging_proxy_eur_mpc": "mean",
        "energy_cost_eur_mpc": "mean",
        "dwell_h_ge_0p85_mpc": "mean",
    })
    out_agg = os.path.join(OUT_DIR, "sweep_v2_agg_by_setting.csv")
    agg.to_csv(out_agg, index=False)
    print(f"Saved aggregate CSV: {out_agg}")

    # Global picks across windows
    # Economic across windows: minimal mean ΔEnergy among those with >=25% mean health improvement
    agg_candidates = agg[agg["health_improvement_pct"] >= 25.0].copy()
    if len(agg_candidates) > 0:
        pick_econ = agg_candidates.sort_values("delta_energy_eur", ascending=True).head(1)
    else:
        pick_econ = agg.sort_values("delta_energy_eur", ascending=True).head(1)

    pick_maxlife = agg.sort_values("health_improvement_pct", ascending=False).head(1)
    pick_bal = agg.sort_values("total_proxy_eur_mpc", ascending=True).head(1)

    print("\n==============================")
    print("GLOBAL PICKS (mean over windows)")
    print("==============================")
    print("Economic:", pick_econ[["setting_id", "delta_energy_eur", "health_improvement_pct", "ss_mpc", "scr_mpc"]].to_string(index=False))
    print("Max-life:", pick_maxlife[["setting_id", "delta_energy_eur", "health_improvement_pct", "ss_mpc", "scr_mpc"]].to_string(index=False))
    print("Balanced:", pick_bal[["setting_id", "delta_energy_eur", "health_improvement_pct", "ss_mpc", "scr_mpc"]].to_string(index=False))


if __name__ == "__main__":
    main()
