import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PRESET_ID = "P4_high_balanced"
OUT_DIR = "out"


# -----------------------------
# Plot style: report-ready
# -----------------------------
def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2.0,
    })


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def find_col(df: pd.DataFrame, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def time_axis(df: pd.DataFrame):
    if "time" in df.columns:
        try:
            t = pd.to_datetime(df["time"])
            return t, "Time"
        except Exception:
            return df["time"].values, "Time"
    return np.arange(len(df)), "Time step [-]"


# -----------------------------
# Load KPI row for P4
# -----------------------------
def load_kpis():
    kpi_path = os.path.join(OUT_DIR, "penalty_compare.csv")
    if not os.path.exists(kpi_path):
        raise FileNotFoundError(f"Missing {kpi_path}. Run pipeline first (penalty sweep).")

    kpi = pd.read_csv(kpi_path)

    # Make it tolerant to slightly different column names
    # (If your pipeline changes names, adapt here.)
    if "preset_id" not in kpi.columns:
        raise ValueError(f"{kpi_path} must contain a 'preset_id' column.")

    row = kpi[kpi["preset_id"] == PRESET_ID]
    if row.empty:
        raise ValueError(f"Preset '{PRESET_ID}' not found in {kpi_path}.")
    return row.iloc[0], kpi_path


# -----------------------------
# Find a timeseries CSV inside out/P4_high_balanced/
# -----------------------------
def pick_timeseries_csv(preset_dir: str):
    # Search recursively for CSVs (ignore penalty_compare)
    candidates = [
        p for p in glob.glob(os.path.join(preset_dir, "**", "*.csv"), recursive=True)
        if os.path.basename(p) not in ["penalty_compare.csv"]
    ]
    if not candidates:
        return None, []

    # Score by presence of useful columns
    expected_groups = [
        {"pv", "pv_kw", "P_pv"},
        {"consumption", "load", "P_load"},
        {"soc", "soc_mpc", "soc_base"},
        {"import", "export"},
        {"charge", "discharge"},
    ]

    def score(path):
        try:
            df0 = pd.read_csv(path, nrows=5)
            cols = set(df0.columns)
            s = sum(1 for g in expected_groups if len(cols.intersection(g)) > 0)
            size = os.path.getsize(path)
            return (s, size)
        except Exception:
            return (-1, 0)

    best = max(candidates, key=score)
    best_score = score(best)[0]

    # Require at least PV/load or SoC present
    if best_score < 2:
        return None, candidates
    return best, candidates


def load_timeseries():
    preset_dir = os.path.join(OUT_DIR, PRESET_ID)
    ts_csv, all_csvs = pick_timeseries_csv(preset_dir)

    if ts_csv is None:
        print("\n[ERROR] Could not find a usable time-series CSV for plotting in:")
        print(f"        {preset_dir}")
        print("\n[INFO] CSV files found (but none looked like timeseries with pv/load/soc columns):")
        for p in all_csvs[:30]:
            print("   -", p)
        print("\nFix options:")
        print("  1) Ensure pipeline saves a timeseries CSV per preset (recommended).")
        print("  2) Manually place the simulation dataframe CSV into out/P4_high_balanced/ (e.g., df_sim.csv).")
        raise SystemExit(1)

    print(f"[OK] Using timeseries CSV: {ts_csv}")
    df = pd.read_csv(ts_csv)

    # Coerce numerics where possible
    for c in df.columns:
        if c == "time":
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")

    return df


# -----------------------------
# Plot generators (P4)
# -----------------------------
def plot_pv_vs_load(df, outpath):
    pv = find_col(df, ["pv", "pv_kw", "P_pv", "pv_power"])
    load = find_col(df, ["consumption", "load", "P_load", "load_kw", "house_load"])
    if pv is None or load is None:
        raise ValueError("Need pv + load/consumption columns for PV vs Load plot.")

    x, xlabel = time_axis(df)

    plt.figure(figsize=(11.5, 4.6))
    plt.plot(x, df[pv].values, label="PV [kW]")
    plt.plot(x, df[load].values, label="Load [kW]")
    plt.title("PV generation and household load (P4_high_balanced run)")
    plt.xlabel(xlabel)
    plt.ylabel("Power [kW]")
    plt.legend(loc="best")
    save_fig(outpath)


def plot_soc_compare(df, outpath):
    soc_base = find_col(df, ["soc_base", "soc_dataset", "soc"])
    soc_mpc  = find_col(df, ["soc_mpc", "soc_sim", "soc_pred"])

    if soc_base is None or soc_mpc is None:
        raise ValueError("Need baseline and MPC SoC columns (e.g., soc_base + soc_mpc).")

    x, xlabel = time_axis(df)

    plt.figure(figsize=(11.5, 4.6))
    plt.plot(x, 100.0 * df[soc_base].values, label="Baseline SoC [%]")
    plt.plot(x, 100.0 * df[soc_mpc].values, label="MPC SoC [%]")
    plt.title("SoC comparison: baseline vs MPC (P4_high_balanced)")
    plt.xlabel(xlabel)
    plt.ylabel("SoC [%]")
    plt.ylim(0, 100)
    plt.legend(loc="best")
    save_fig(outpath)


def plot_soc_hist(df, outpath):
    soc_base = find_col(df, ["soc_base", "soc_dataset", "soc"])
    soc_mpc  = find_col(df, ["soc_mpc", "soc_sim", "soc_pred"])
    if soc_base is None or soc_mpc is None:
        raise ValueError("Need baseline and MPC SoC columns for histogram.")

    base = 100.0 * df[soc_base].dropna().values
    mpc  = 100.0 * df[soc_mpc].dropna().values

    bins = np.linspace(0, 100, 41)  # 2.5% bins

    plt.figure(figsize=(9.2, 4.9))
    plt.hist(base, bins=bins, alpha=0.60, label="Baseline")
    plt.hist(mpc,  bins=bins, alpha=0.60, label="MPC (P4)")
    plt.title("SoC distribution (histogram)")
    plt.xlabel("SoC [%]")
    plt.ylabel("Count [-]")
    plt.legend(loc="best")
    save_fig(outpath)


def plot_dwell_hours_from_kpis(kpi_row, outpath):
    # These names must match your penalty_compare.csv columns.
    # If your file uses different headers, rename here.
    required = [
        "dwell_h_ge_0p85_base", "dwell_h_ge_0p85_mpc",
        "dwell_h_ge_0p90_base", "dwell_h_ge_0p90_mpc",
        "dwell_h_ge_0p95_base", "dwell_h_ge_0p95_mpc",
    ]
    missing = [c for c in required if c not in kpi_row.index]
    if missing:
        raise ValueError(
            "penalty_compare.csv is missing required dwell columns:\n"
            + "\n".join(missing)
            + "\nAdapt the column names in plot_dwell_hours_from_kpis()."
        )

    base = [
        float(kpi_row["dwell_h_ge_0p85_base"]),
        float(kpi_row["dwell_h_ge_0p90_base"]),
        float(kpi_row["dwell_h_ge_0p95_base"]),
    ]
    mpc = [
        float(kpi_row["dwell_h_ge_0p85_mpc"]),
        float(kpi_row["dwell_h_ge_0p90_mpc"]),
        float(kpi_row["dwell_h_ge_0p95_mpc"]),
    ]
    labels = ["SoC ≥ 85%", "SoC ≥ 90%", "SoC ≥ 95%"]

    x = np.arange(len(labels))
    w = 0.38

    plt.figure(figsize=(9.2, 4.9))
    plt.bar(x - w/2, base, width=w, label="Baseline")
    plt.bar(x + w/2, mpc,  width=w, label="MPC (P4)")
    plt.xticks(x, labels)
    plt.ylabel("Dwell time [h]")
    plt.title("High-SoC dwell hours (P4_high_balanced)")
    plt.legend(loc="best")
    save_fig(outpath)


def copy_tradeoff_plot(plots_dir):
    src = os.path.join(OUT_DIR, "penalty_tradeoff.png")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(plots_dir, "penalty_tradeoff.png"))
        print("[OK] Copied penalty_tradeoff.png into preset plot folder.")
    else:
        print("[WARN] out/penalty_tradeoff.png not found (skip).")


# -----------------------------
# Main
# -----------------------------
def main():
    set_plot_style()

    preset_dir = os.path.join(OUT_DIR, PRESET_ID)
    plots_dir = os.path.join(preset_dir, "plots")
    safe_mkdir(plots_dir)

    # Load KPI row and validate budget justification
    kpi_row, kpi_path = load_kpis()
    delta_pct = float(kpi_row.get("delta_energy_pct", np.nan))
    health = float(kpi_row.get("health_improvement_pct", np.nan))

    print(f"[INFO] Using preset: {PRESET_ID}")
    print(f"[INFO] KPI from {kpi_path}: ΔEnergy={delta_pct:.2f}% | Health improvement={health:.2f}%")
    if delta_pct > 10.0:
        print("[WARN] This preset violates the 10% budget based on penalty_compare.csv.")

    # Load timeseries for PV/Load + SoC plots
    df = load_timeseries()

    # Generate plots
    plot_pv_vs_load(df, os.path.join(plots_dir, "01_pv_vs_load.png"))
    plot_soc_compare(df, os.path.join(plots_dir, "02_soc_compare.png"))
    plot_soc_hist(df, os.path.join(plots_dir, "06_soc_hist.png"))
    plot_dwell_hours_from_kpis(kpi_row, os.path.join(plots_dir, "07_dwell_hours.png"))
    copy_tradeoff_plot(plots_dir)

    # Save a one-row KPI summary for easy LaTeX table filling
    pd.DataFrame([kpi_row]).to_csv(os.path.join(preset_dir, "kpi_summary_P4.csv"), index=False)

    print(f"[DONE] Plots written to: {plots_dir}")
    print(f"[DONE] KPI summary written to: {os.path.join(preset_dir, 'kpi_summary_P4.csv')}")


if __name__ == "__main__":
    main()
