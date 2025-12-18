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
# 1. KONFIGURATION
# ==========================================
# FIX: BASE_PATH muss ausserhalb der Klasse stehen, damit List Comprehensions funktionieren
BASE_PATH_GLOBAL = r"C:\Users\henni\PycharmProjects\eternal-youth\out"


class Config:
    # Wir definieren die 4 Szenarien mit den ECHTEN Dateien
    SCENARIOS = {
        "Szenario A: Sommer (Saturation)": {
            "target": os.path.join(BASE_PATH_GLOBAL, "sunny", "2025-06-24.csv"),
            "history": [os.path.join(BASE_PATH_GLOBAL, "sunny", f"2025-06-{day}.csv") for day in range(17, 24)]
        },
        "Szenario B: Frühling (Volatile Proxy)": {
            "target": os.path.join(BASE_PATH_GLOBAL, "very-sunny", "2025-04-26.csv"),
            "history": [
                os.path.join(BASE_PATH_GLOBAL, "very-sunny", "2025-04-24.csv"),
                os.path.join(BASE_PATH_GLOBAL, "very-sunny", "2025-04-25.csv")
            ]
        },
        "Szenario C: Übergang (Transition)": {
            "target": os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-09-05.csv"),
            "history": [
                os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-08-29.csv"),
                os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-08-30.csv"),
                os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-08-31.csv"),
                os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-09-01.csv"),
                os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-09-02.csv"),
                os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-09-03.csv"),
                os.path.join(BASE_PATH_GLOBAL, "mixed", "2025-09-04.csv")
            ]
        },
        "Szenario D: Winter (Dark/Dunkelflaute)": {
            "target": os.path.join(BASE_PATH_GLOBAL, "overcast", "2025-11-15.csv"),
            "history": [os.path.join(BASE_PATH_GLOBAL, "overcast", f"2025-11-{day:02d}.csv") for day in range(8, 15)]
        }
    }

    DT = 5.0 / 60.0  # 5 Minuten Schritte
    STEPS_PER_DAY = int(24 / DT)

    # System Limits
    CAPACITY = 10.0  # kWh
    P_MAX = 5.0  # kW
    ETA = 0.95  # Effizienz
    SOC_MIN_HARD = 0.05
    SOC_MAX_HARD = 1.00
    SOC_START = 0.10

    # Strategie
    SOC_MIN_SOFT = 0.20
    PENALTY_LOW = 200.0
    SOC_MAX_SOFT = 0.90
    PENALTY_HIGH = 10.0

    # Kosten
    PRICE_BUY = 0.30
    PRICE_SELL = 0.08
    COST_WEAR = 0.02
    COST_SOC_HOLDING = 0.01
    VAL_TERMINAL = 0.15


# ==========================================
# 2. HELPER: DATEIEN LADEN
# ==========================================
def load_single_file(filename):
    """Lädt eine einzelne CSV Datei und bereinigt sie"""
    if not os.path.exists(filename):
        log_err(f"Datei fehlt: {filename}")
        return None

    try:
        df = pd.read_csv(filename, sep=None, engine='python')
        df.columns = [c.strip().lower() for c in df.columns]
        rename_map = {'pv': 'pv_kw', 'generation': 'pv_kw', 'erzeugung': 'pv_kw',
                      'load': 'load_kw', 'consumption': 'load_kw', 'verbrauch': 'load_kw'}
        df = df.rename(columns=rename_map)
        df = df.fillna(0)
        df['pv_kw'] = pd.to_numeric(df['pv_kw'], errors='coerce').fillna(0).abs()
        df['load_kw'] = pd.to_numeric(df['load_kw'], errors='coerce').fillna(0).abs()
        return df
    except Exception as e:
        log_err(f"Fehler beim Lesen von {filename}: {e}")
        return None


def learn_load_profile(history_files, cfg):
    """
    Liest die History-Dateien ein und erstellt ein Durchschnittsprofil (Training).
    """
    valid_profiles = []

    for f in history_files:
        df = load_single_file(f)
        if df is not None and len(df) >= cfg.STEPS_PER_DAY:
            # Nur den ersten Tag nehmen falls Datei länger ist
            profile = df['load_kw'].values[:cfg.STEPS_PER_DAY]
            valid_profiles.append(profile)

    if not valid_profiles:
        log_err("Keine Trainingsdaten gefunden! Nutze flaches Profil als Fallback.")
        return np.full(cfg.STEPS_PER_DAY, 0.5)

        # Durchschnitt berechnen
    avg_profile = np.mean(np.array(valid_profiles), axis=0)
    log(f"Lastprofil gelernt aus {len(valid_profiles)} Tagen.")
    return avg_profile


# ==========================================
# 3. SIMULATIONEN
# ==========================================
def simulate_greedy(df, cfg):
    """Der dumme Regler"""
    steps = len(df)
    soc = [cfg.SOC_START * cfg.CAPACITY]
    current_energy = soc[0]
    pv_data = df['pv_kw'].values
    load_data = df['load_kw'].values

    for i in range(steps):
        net = load_data[i] - pv_data[i]
        p = 0.0
        if net < 0:  # Laden
            p = min(-net, cfg.P_MAX)
            max_in = (cfg.CAPACITY - current_energy) / cfg.ETA / cfg.DT
            p = min(p, max_in)
        elif net > 0:  # Entladen
            p = -min(net, cfg.P_MAX)
            avail = current_energy - (cfg.CAPACITY * cfg.SOC_MIN_HARD)
            max_out = max(0, avail) * cfg.ETA / cfg.DT
            p = -min(-p, max_out)

        if p > 0:
            current_energy += p * cfg.ETA * cfg.DT
        else:
            current_energy += p / cfg.ETA * cfg.DT

        current_energy = max(cfg.CAPACITY * cfg.SOC_MIN_HARD, min(cfg.CAPACITY * cfg.SOC_MAX_HARD, current_energy))
        soc.append(current_energy)

    return np.array(soc[:-1])


def simulate_mpc_realistic(df, forecast_load, cfg):
    """Der schlaue Regler (Optimiert für LFP-Health)"""
    T = len(df)
    load_plan = forecast_load
    pv_plan = df['pv_kw'].values

    p_c = cp.Variable(T, nonneg=True)
    p_d = cp.Variable(T, nonneg=True)
    e_b = cp.Variable(T + 1)
    p_gr_in = cp.Variable(T, nonneg=True)
    p_gr_out = cp.Variable(T, nonneg=True)
    slack_low = cp.Variable(T, nonneg=True)
    slack_high = cp.Variable(T, nonneg=True)

    constraints = [e_b[0] == cfg.SOC_START * cfg.CAPACITY]
    cost_terms = []

    for k in range(T):
        constraints.append(load_plan[k] + p_c[k] == pv_plan[k] + p_d[k] + p_gr_in[k] - p_gr_out[k])
        constraints.append(e_b[k + 1] == e_b[k] + (p_c[k] * cfg.ETA - p_d[k] / cfg.ETA) * cfg.DT)
        constraints.append(p_c[k] <= cfg.P_MAX)
        constraints.append(p_d[k] <= cfg.P_MAX)
        constraints.append(e_b[k + 1] >= cfg.CAPACITY * cfg.SOC_MIN_HARD)
        constraints.append(e_b[k + 1] <= cfg.CAPACITY * cfg.SOC_MAX_HARD)

        # Soft Constraints
        constraints.append(e_b[k + 1] >= (cfg.CAPACITY * cfg.SOC_MIN_SOFT) - slack_low[k])
        constraints.append(e_b[k + 1] <= (cfg.CAPACITY * cfg.SOC_MAX_SOFT) + slack_high[k])

        # Kostenfunktion
        cost = (p_gr_in[k] * cfg.PRICE_BUY - p_gr_out[k] * cfg.PRICE_SELL) * cfg.DT + \
               cfg.COST_WEAR * (p_c[k] + p_d[k]) * cfg.DT + \
               cfg.COST_SOC_HOLDING * e_b[k + 1] * cfg.DT + \
               cfg.PENALTY_LOW * slack_low[k] + cfg.PENALTY_HIGH * slack_high[k]
        cost_terms.append(cost)

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
# 4. MAIN LOOP ÜBER ALLE 4 SZENARIEN
# ==========================================
if __name__ == "__main__":
    log("=== STARTING 4-SCENARIO VALIDATION ===")
    cfg = Config()

    # 2x2 Matrix Plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    for i, (name, data_paths) in enumerate(cfg.SCENARIOS.items()):
        log(f"Simuliere: {name}")

        # 1. Training (Lastprofil lernen)
        avg_profile = learn_load_profile(data_paths["history"], cfg)

        # 2. Target laden
        df_target = load_single_file(data_paths["target"])
        if df_target is None:
            continue

        # Kürzen auf 24h
        if len(df_target) > cfg.STEPS_PER_DAY:
            df_target = df_target.iloc[:cfg.STEPS_PER_DAY]

        # 3. Simulationen
        soc_greedy = simulate_greedy(df_target, cfg)
        soc_mpc = simulate_mpc_realistic(df_target, avg_profile, cfg)

        if soc_mpc is None:
            log_err(f"MPC fehlgeschlagen bei {name}")
            soc_mpc = np.zeros(len(df_target))

        # 4. Plotten
        ax = axes[i]
        time_axis = np.linspace(0, 24, len(soc_greedy))
        soc_g_perc = soc_greedy / cfg.CAPACITY * 100
        soc_m_perc = soc_mpc / cfg.CAPACITY * 100

        # Zonen Visualisierung
        ax.axhspan(0, 5, color='black', alpha=0.3)  # Hard
        ax.axhspan(90, 100, color='orange', alpha=0.15)  # Stress
        ax.axhspan(5, 20, color='red', alpha=0.1)  # Soft Low

        ax.plot(time_axis, soc_g_perc, color='red', linestyle='--', alpha=0.6, label='Standard (Greedy)')
        ax.plot(time_axis, soc_m_perc, color='green', linewidth=2.5, label='MPC (Healthy)')

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_ylabel('SoC (%)')
        ax.grid(True, alpha=0.3)

        if i == 0: ax.legend(loc='lower center', ncol=2)

    plt.tight_layout()
    plt.savefig("Validierung_4_Szenarien_Final.png", dpi=300)
    plt.show()
    log("Fertig.")