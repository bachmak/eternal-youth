import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import sys


# ==========================================
# 0. DEBUGGING HELPER
# ==========================================
def log(msg):
    print(f"[INFO] {msg}")


def log_err(msg):
    print(f"!!! [ERROR] {msg}")


# ==========================================
# 1. CONFIGURATION
# ==========================================
# GLOBAL PATH: Adjust this to your local project folder
BASE_PATH_GLOBAL = r"C:\Users\henni\PycharmProjects\eternal-youth\out"


class Config:
    # We define only the scenario we want to analyze
    # Scenario C: Transition (September) - Realistic Load (~7.5 kWh) & Mixed Weather
    SCENARIO_NAME = "Scenario: Transition (September)"

    TARGET_FILE = os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-09-05.csv")

    HISTORY_FILES = [
        os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-08-29.csv"),
        os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-08-30.csv"),
        os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-08-31.csv"),
        os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-09-01.csv"),
        os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-09-02.csv"),
        os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-09-03.csv"),
        os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-09-04.csv")
    ]

    # --- TECHNICAL PARAMETERS ---
    DT = 5.0 / 60.0  # Time step: 5 Minutes
    STEPS_PER_DAY = int(24 / DT)

    # Battery Hardware Limits (Huawei LUNA2000 10kWh)
    CAPACITY = 10.0  # kWh
    P_MAX = 5.0  # kW
    ETA = 0.95  # Roundtrip Efficiency ~90%

    # Hard Constraints (Safety / BMS)
    SOC_MIN_HARD = 0.05
    SOC_MAX_HARD = 1.00
    SOC_START = 0.10  # Initial SoC at 00:00

    # --- MPC STRATEGY PARAMETERS ---
    # Soft Constraints (Health Corridor 20-90%)
    SOC_MIN_SOFT = 0.20
    PENALTY_LOW = 200.0  # High penalty: Only dip below 20% if critical
    SOC_MAX_SOFT = 0.90
    PENALTY_HIGH = 10.0  # Moderate penalty: Avoid >90% unless necessary

    # Cost Function Weights
    PRICE_BUY = 0.30  # Grid Import Price
    PRICE_SELL = 0.08  # Feed-in Tariff
    COST_WEAR = 0.02  # Cyclic Aging Cost (per kWh throughput)
    COST_SOC_HOLDING = 0.01  # Calendar Aging Cost ("Parking Fee")
    VAL_TERMINAL = 0.15  # End-of-Horizon Value


# ==========================================
# 2. HELPER: DATA LOADING
# ==========================================
def load_single_file(filename):
    """Loads a single CSV file and standardizes column names."""
    if not os.path.exists(filename):
        log_err(f"File missing: {filename}")
        return None

    try:
        df = pd.read_csv(filename, sep=None, engine='python')
        df.columns = [c.strip().lower() for c in df.columns]

        # Rename German/inconsistent columns to standard English
        rename_map = {'pv': 'pv_kw', 'generation': 'pv_kw', 'erzeugung': 'pv_kw',
                      'load': 'load_kw', 'consumption': 'load_kw', 'verbrauch': 'load_kw'}
        df = df.rename(columns=rename_map)

        df = df.fillna(0)
        df['pv_kw'] = pd.to_numeric(df['pv_kw'], errors='coerce').fillna(0).abs()
        df['load_kw'] = pd.to_numeric(df['load_kw'], errors='coerce').fillna(0).abs()
        return df
    except Exception as e:
        log_err(f"Error reading {filename}: {e}")
        return None


def learn_load_profile(history_files, cfg):
    """Creates an average load profile from historical data (Training Step)."""
    valid_profiles = []
    for f in history_files:
        df = load_single_file(f)
        if df is not None and len(df) >= cfg.STEPS_PER_DAY:
            # Take only the first 24h
            profile = df['load_kw'].values[:cfg.STEPS_PER_DAY]
            valid_profiles.append(profile)

    if not valid_profiles:
        log_err("No training data found! Using fallback flat profile.")
        return np.full(cfg.STEPS_PER_DAY, 0.5)

        # Calculate mean across all training days
    return np.mean(np.array(valid_profiles), axis=0)


# ==========================================
# 3. SIMULATION LOGIC
# ==========================================
def simulate_greedy(df, cfg):
    """
    Simulates the standard manufacturer logic: 'Fill-First'.
    """
    steps = len(df)
    soc = [cfg.SOC_START * cfg.CAPACITY]
    current_energy = soc[0]
    pv_data = df['pv_kw'].values
    load_data = df['load_kw'].values

    for i in range(steps):
        net = load_data[i] - pv_data[i]
        p = 0.0

        if net < 0:  # Surplus -> Charge immediately
            p = min(-net, cfg.P_MAX)
            max_in = (cfg.CAPACITY - current_energy) / cfg.ETA / cfg.DT
            p = min(p, max_in)
        elif net > 0:  # Deficit -> Discharge immediately
            p = -min(net, cfg.P_MAX)
            avail = current_energy - (cfg.CAPACITY * cfg.SOC_MIN_HARD)
            max_out = max(0, avail) * cfg.ETA / cfg.DT
            p = -min(-p, max_out)

        if p > 0:
            current_energy += p * cfg.ETA * cfg.DT
        else:
            current_energy += p / cfg.ETA * cfg.DT

        # Apply Hard Limits
        current_energy = max(cfg.CAPACITY * cfg.SOC_MIN_HARD, min(cfg.CAPACITY * cfg.SOC_MAX_HARD, current_energy))
        soc.append(current_energy)

    return np.array(soc[:-1])


def simulate_mpc_realistic(df, forecast_load, cfg):
    """
    Simulates the Health-Aware MPC using Convex Optimization.
    """
    T = len(df)
    load_plan = forecast_load
    pv_plan = df['pv_kw'].values

    # CVXPY Variables
    p_c = cp.Variable(T, nonneg=True)  # Charge Power
    p_d = cp.Variable(T, nonneg=True)  # Discharge Power
    e_b = cp.Variable(T + 1)  # Energy State
    p_gr_in = cp.Variable(T, nonneg=True)  # Grid Import
    p_gr_out = cp.Variable(T, nonneg=True)  # Grid Export
    slack_low = cp.Variable(T, nonneg=True)  # Soft Constraint Violation (Low)
    slack_high = cp.Variable(T, nonneg=True)  # Soft Constraint Violation (High)

    constraints = [e_b[0] == cfg.SOC_START * cfg.CAPACITY]
    cost_terms = []

    for k in range(T):
        # 1. Energy Balance
        constraints.append(load_plan[k] + p_c[k] == pv_plan[k] + p_d[k] + p_gr_in[k] - p_gr_out[k])
        # 2. Battery Dynamics
        constraints.append(e_b[k + 1] == e_b[k] + (p_c[k] * cfg.ETA - p_d[k] / cfg.ETA) * cfg.DT)
        # 3. Hardware Limits
        constraints.append(p_c[k] <= cfg.P_MAX)
        constraints.append(p_d[k] <= cfg.P_MAX)
        constraints.append(e_b[k + 1] >= cfg.CAPACITY * cfg.SOC_MIN_HARD)
        constraints.append(e_b[k + 1] <= cfg.CAPACITY * cfg.SOC_MAX_HARD)

        # 4. Soft Constraints (Health Corridor)
        constraints.append(e_b[k + 1] >= (cfg.CAPACITY * cfg.SOC_MIN_SOFT) - slack_low[k])
        constraints.append(e_b[k + 1] <= (cfg.CAPACITY * cfg.SOC_MAX_SOFT) + slack_high[k])

        # 5. Cost Function
        cost_grid = (p_gr_in[k] * cfg.PRICE_BUY - p_gr_out[k] * cfg.PRICE_SELL) * cfg.DT
        cost_wear = cfg.COST_WEAR * (p_c[k] + p_d[k]) * cfg.DT
        cost_hold = cfg.COST_SOC_HOLDING * e_b[k + 1] * cfg.DT
        cost_penalty = cfg.PENALTY_LOW * slack_low[k] + cfg.PENALTY_HIGH * slack_high[k]

        cost_terms.append(cost_grid + cost_wear + cost_hold + cost_penalty)

    cost_terms.append(-cfg.VAL_TERMINAL * e_b[T])

    prob = cp.Problem(cp.Minimize(cp.sum(cost_terms)), constraints)
    try:
        prob.solve(solver=cp.ECOS)
    except:
        prob.solve(solver=cp.SCS)

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        return None
    return e_b.value[:-1]


# ==========================================
# 4. MAIN EXECUTION & PLOTTING (SINGLE FOCUS)
# ==========================================
if __name__ == "__main__":
    log("=== STARTING SINGLE SCENARIO SIMULATION (ENGLISH) ===")
    cfg = Config()

    log(f"Loading data for: {cfg.SCENARIO_NAME}")

    # 1. Learn & Load
    avg_profile = learn_load_profile(cfg.HISTORY_FILES, cfg)
    df_target = load_single_file(cfg.TARGET_FILE)

    if df_target is None:
        log_err("Target file could not be loaded. Exiting.")
        sys.exit(1)

    # Cut to 24h if longer
    if len(df_target) > cfg.STEPS_PER_DAY:
        df_target = df_target.iloc[:cfg.STEPS_PER_DAY]

    # Calculate Totals
    daily_pv_kwh = df_target['pv_kw'].sum() * cfg.DT
    daily_load_kwh = df_target['load_kw'].sum() * cfg.DT

    log(f"Daily Totals -> PV: {daily_pv_kwh:.2f} kWh | Load: {daily_load_kwh:.2f} kWh")

    # 2. Run Simulations
    soc_greedy = simulate_greedy(df_target, cfg)
    soc_mpc = simulate_mpc_realistic(df_target, avg_profile, cfg)

    if soc_mpc is None:
        log_err("MPC Optimization failed.")
        sys.exit(1)

    # 3. Data Smoothing (Moving Average)
    window_size = 6  # 30 Minutes window
    pv_smooth = df_target['pv_kw'].rolling(window=window_size, center=True, min_periods=1).mean()
    load_smooth = df_target['load_kw'].rolling(window=window_size, center=True, min_periods=1).mean()

    # 4. Plotting (Big Single Plot)
    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax2 = ax1.twinx()  # Secondary axis for Power

    time_axis = np.linspace(0, 24, len(soc_greedy))

    # Background: Power (Smoothed)
    ln1 = ax2.fill_between(time_axis, pv_smooth, color='#FDB813', alpha=0.3, label='PV Generation (Smoothed)')
    ln2 = ax2.plot(time_axis, load_smooth, color='#005293', alpha=0.6, linewidth=2, label='Domestic Load (Smoothed)')

    # Foreground: SoC
    soc_g_perc = soc_greedy / cfg.CAPACITY * 100
    soc_m_perc = soc_mpc / cfg.CAPACITY * 100

    ln3 = ax1.plot(time_axis, soc_g_perc, color='#E30613', linestyle='--', alpha=0.9, linewidth=3,
                   label='Standard Logic (Greedy)')
    ln4 = ax1.plot(time_axis, soc_m_perc, color='#009640', linewidth=4, label='Health-Aware MPC')

    # Zones
    ax1.axhspan(0, 5, color='black', alpha=0.2)
    ax1.axhspan(90, 100, color='orange', alpha=0.15)
    ax1.axhspan(5, 20, color='red', alpha=0.08)

    # Labels & Titles
    title_text = f"Performance Analysis: {cfg.SCENARIO_NAME}\n(Total PV: {daily_pv_kwh:.1f} kWh | Total Load: {daily_load_kwh:.1f} kWh)"
    ax1.set_title(title_text, fontsize=16, fontweight='bold', pad=20)

    ax1.set_ylabel('State of Charge (%)', fontsize=14)
    ax1.set_xlabel('Time of Day (h)', fontsize=14)
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(True, alpha=0.3)

    # Scale right axis dynamically (min 6 kW for visibility)
    max_power = max(6.0, pv_smooth.max() * 1.1)
    ax2.set_ylabel('Power (kW)', fontsize=14)
    ax2.set_ylim(0, max_power)
    ax2.tick_params(axis='both', labelsize=12)

    # Unified Legend (bottom center)
    lines = [ln1, ln2[0], ln3[0], ln4[0]]
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02),
               fancybox=True, shadow=True, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Reserve space for legend
    plt.savefig("Analysis_Scenario_Transition_English.png", dpi=300)
    plt.show()
    log("Plot created successfully.")