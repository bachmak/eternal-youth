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
class Config:
    # --- DATEILISTE (29. Aug bis 06. Sep) ---
    # Bitte überprüfe, ob diese Pfade auf deinem PC wirklich existieren!
    FILES = [
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\mixed\2025-08-29.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\mixed\2025-08-30.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\mixed\2025-08-31.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\mixed\2025-09-01.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\mixed\2025-09-02.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\mixed\2025-09-03.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\mixed\2025-09-04.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\mixed\2025-09-05.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\mixed\2025-09-06.csv"
    ]

    # --- ZEIT-EINSTELLUNG (WICHTIG!) ---
    # Deine Daten sind im 5-Minuten-Takt.
    # 5 Minuten = 5/60 Stunden = 0.08333...
    DT = 5.0 / 60.0

    # --- SYSTEM LIMITS ---
    CAPACITY = 10.0  # kWh
    P_MAX = 5.0  # kW
    ETA = 0.95  # Effizienz
    SOC_MIN_HARD = 0.05  # 5%
    SOC_MAX_HARD = 1.0  # 100%
    SOC_START = 0.10  # Start mit 10%

    # --- STRATEGIE ---
    SOC_MIN_SOFT = 0.15  # 15% Puffer
    PENALTY_SOFT = 100.0  # Strafe

    # --- PREISE & HEALTH (INNOVATION) ---
    PRICE_BUY = 0.30
    PRICE_SELL = 0.08

    # Faktor 1: Zyklenkosten (Gegen Zittern)
    COST_WEAR = 0.02

    # Faktor 2: SoC-Holding-Cost (Gegen Thermal Stress)
    # Das ist der wichtigste Hebel!
    COST_SOC_HOLDING = 0.005

    # Terminal Value
    VAL_TERMINAL = 0.15


# ==========================================
# 2. DATEN IMPORT (ROBUST)
# ==========================================
def load_and_merge_data(cfg):
    all_days = []
    log(f"Starte Import von {len(cfg.FILES)} Dateien...")

    for i, filename in enumerate(cfg.FILES):
        if not os.path.exists(filename):
            log_err(f"Datei nicht gefunden: {filename}")
            continue

        try:
            # 1. Versuch: Laden (Auto-Separator)
            # Manche CSVs nutzen , manche ;
            df_day = pd.read_csv(filename, sep=None, engine='python')

            # Debug: Was haben wir geladen?
            # log(f"Lade Datei {i+1}: {os.path.basename(filename)} | Shape: {df_day.shape}")

            # 2. Mapping
            # Wir machen alles lowercase, um Tippfehler zu vermeiden
            df_day.columns = [c.strip().lower() for c in df_day.columns]

            # Mapping definieren (alles auf lowercase prüfen)
            rename_map = {
                'pv': 'pv_kw',
                'generation': 'pv_kw',
                'erzeugung': 'pv_kw',
                'load': 'load_kw',
                'consumption': 'load_kw',
                'verbrauch': 'load_kw'
            }

            df_day = df_day.rename(columns=rename_map)

            # 3. Validierung
            if 'pv_kw' not in df_day.columns or 'load_kw' not in df_day.columns:
                log_err(f"Spalten fehlen in {os.path.basename(filename)}!")
                log(f"Gefundene Spalten (lowercase): {df_day.columns.tolist()}")
                log("Überspringe diese Datei...")
                continue

            # 4. Cleanup
            df_day = df_day.fillna(0)
            # Sicherstellen dass es Floats sind
            df_day['pv_kw'] = pd.to_numeric(df_day['pv_kw'], errors='coerce').fillna(0).abs()
            df_day['load_kw'] = pd.to_numeric(df_day['load_kw'], errors='coerce').fillna(0).abs()

            all_days.append(df_day)

        except Exception as e:
            log_err(f"Fehler beim Lesen von {filename}: {e}")

    if not all_days:
        log_err("Keine gültigen Daten geladen! Skript bricht ab.")
        sys.exit(1)

    # Zusammenfügen
    full_df = pd.concat(all_days, ignore_index=True)

    # Info
    total_steps = len(full_df)
    total_hours = total_steps * cfg.DT
    total_days = total_hours / 24.0

    log(f"IMPORT ERFOLGREICH!")
    log(f"Gesamt-Datensatz: {total_steps} Zeitschritte")
    log(f"Entspricht ca. {total_days:.1f} Tagen (bei 5-Min Intervall)")

    return full_df


# ==========================================
# 3. SIMULATIONEN
# ==========================================
def simulate_greedy(df, cfg):
    log("Starte Greedy Simulation...")
    steps = len(df)
    soc = [cfg.SOC_START * cfg.CAPACITY]
    current_energy = soc[0]

    pv_data = df['pv_kw'].values
    load_data = df['load_kw'].values

    for i in range(steps):
        net = load_data[i] - pv_data[i]
        p = 0.0

        if net < 0:  # Überschuss -> Laden
            p = min(-net, cfg.P_MAX)
            # Passt es rein?
            max_in = (cfg.CAPACITY - current_energy) / cfg.ETA / cfg.DT
            p = min(p, max_in)

        elif net > 0:  # Bedarf -> Entladen
            p = -min(net, cfg.P_MAX)
            # Ist was da? (Bis Hard Min)
            avail = current_energy - (cfg.CAPACITY * cfg.SOC_MIN_HARD)
            max_out = max(0, avail) * cfg.ETA / cfg.DT
            p = -min(-p, max_out)

        # Physik
        if p > 0:
            current_energy += p * cfg.ETA * cfg.DT
        else:
            current_energy += p / cfg.ETA * cfg.DT

        soc.append(current_energy)

    return np.array(soc[:-1])


def simulate_mpc(df, cfg):
    T = len(df)
    log(f"Starte MPC Optimierung für T={T} Schritte. Bitte warten...")

    # Vektoren extrahieren (schneller)
    load_vec = df['load_kw'].values
    pv_vec = df['pv_kw'].values

    # --- Variablen ---
    p_c = cp.Variable(T, nonneg=True)
    p_d = cp.Variable(T, nonneg=True)
    e_b = cp.Variable(T + 1)
    slack = cp.Variable(T, nonneg=True)
    p_gr_in = cp.Variable(T, nonneg=True)
    p_gr_out = cp.Variable(T, nonneg=True)

    constraints = []
    cost_terms = []

    # Startbedingung
    constraints.append(e_b[0] == cfg.SOC_START * cfg.CAPACITY)

    # --- LOOP ÜBER ALLE SCHRITTE ---
    # Bei 9 Tagen * 288 Schritten = 2592 Schritte. Das schafft ECOS locker.
    for k in range(T):
        # 1. Power Balance
        constraints.append(load_vec[k] + p_c[k] == pv_vec[k] + p_d[k] + p_gr_in[k] - p_gr_out[k])

        # 2. Battery Dynamics
        constraints.append(e_b[k + 1] == e_b[k] + (p_c[k] * cfg.ETA - p_d[k] / cfg.ETA) * cfg.DT)

        # 3. Limits
        constraints.append(p_c[k] <= cfg.P_MAX)
        constraints.append(p_d[k] <= cfg.P_MAX)
        constraints.append(e_b[k + 1] <= cfg.CAPACITY * cfg.SOC_MAX_HARD)
        constraints.append(e_b[k + 1] >= cfg.CAPACITY * cfg.SOC_MIN_HARD)

        # 4. Soft Constraints
        constraints.append(e_b[k + 1] >= (cfg.CAPACITY * cfg.SOC_MIN_SOFT) - slack[k])

        # 5. Kosten
        # Grid
        cost_grid = (p_gr_in[k] * cfg.PRICE_BUY - p_gr_out[k] * cfg.PRICE_SELL) * cfg.DT
        # Wear (Zyklen)
        cost_wear = cfg.COST_WEAR * (p_c[k] + p_d[k]) * cfg.DT
        # SoC Holding (Health Innovation!)
        cost_holding = cfg.COST_SOC_HOLDING * e_b[k + 1] * cfg.DT
        # Soft Penalty
        cost_soft = cfg.PENALTY_SOFT * slack[k]

        cost_terms.append(cost_grid + cost_wear + cost_holding + cost_soft)

    # Terminal Value
    cost_terms.append(-cfg.VAL_TERMINAL * e_b[T])

    # --- SOLVE ---
    prob = cp.Problem(cp.Minimize(cp.sum(cost_terms)), constraints)

    log("Solver läuft...")
    try:
        prob.solve(solver=cp.ECOS)
    except Exception as e:
        log_err(f"Standard-Solver fehlgeschlagen: {e}. Versuche Fallback...")
        prob.solve()

    log(f"Solver Status: {prob.status}")

    if prob.status != 'optimal' and prob.status != 'optimal_inaccurate':
        log_err("Problem konnte nicht optimal gelöst werden!")
        return None

    return e_b.value[:-1]


# ==========================================
# 4. MAIN & PLOTTING
# ==========================================
if __name__ == "__main__":
    log("=== MPC SIMULATION START ===")

    cfg = Config()

    # 1. Laden
    full_df = load_and_merge_data(cfg)

    # 2. Rechnen
    soc_greedy = simulate_greedy(full_df, cfg)
    soc_mpc = simulate_mpc(full_df, cfg)

    if soc_mpc is None:
        log_err("Abbruch: Keine MPC Ergebnisse.")
        sys.exit(1)

    # 3. Umrechnen in Prozent
    soc_g_perc = soc_greedy / cfg.CAPACITY * 100
    soc_m_perc = soc_mpc / cfg.CAPACITY * 100

    log("Erstelle Plot...")

    # Zeitachse (in Tagen)
    # Ein Tag hat 24h / 0.0833h = 288 Schritte
    steps_per_day = int(24 / cfg.DT)
    days_axis = np.arange(len(full_df)) / steps_per_day

    plt.figure(figsize=(16, 8))

    # Kurven
    plt.plot(days_axis, soc_g_perc, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
             label='Huawei Standard (Greedy)')
    plt.plot(days_axis, soc_m_perc, color='green', linewidth=2.5, label='Health-Aware MPC (Dein Code)')

    # Flächen füllen
    plt.fill_between(days_axis, soc_g_perc, soc_m_perc, where=(soc_g_perc > soc_m_perc),
                     color='red', alpha=0.15, hatch='//', label='Vermeidbarer Stress (Health Gain)')

    # Tages-Markierungen
    start_date = pd.Timestamp("2025-08-29")
    num_days = int(days_axis[-1]) + 1

    for d in range(num_days):
        if d > 0:
            plt.axvline(x=d, color='gray', linestyle=':', alpha=0.5)

        # Datum labeln
        curr_date = start_date + pd.Timedelta(days=d)
        date_label = curr_date.strftime("%d.%m.")
        plt.text(d + 0.5, 102, date_label, ha='center', fontsize=10, fontweight='bold', color='#555')

    # Limits
    plt.axhline(100, color='gray', linewidth=1, alpha=0.5)
    plt.axhline(15, color='orange', linestyle=':', label='Soft Min (15%)')
    plt.axhline(5, color='black', linewidth=1.5, label='Hard Min (5%)')

    plt.ylim(0, 110)
    plt.xlim(0, num_days)
    plt.ylabel('State of Charge [%]', fontsize=12)
    plt.xlabel('Tage im Simulationszeitraum', fontsize=12)
    plt.title('9-Tage Validierung: Standard-Logik vs. Health-Aware MPC', fontsize=16)
    plt.legend(loc='lower center', ncol=3, fontsize=11, frameon=True)
    plt.grid(True, alpha=0.3)

    filename_plot = "mpc_debug_plot.png"
    plt.tight_layout()
    plt.savefig(filename_plot, dpi=300)

    log(f"FERTIG! Plot gespeichert als: {filename_plot}")
    plt.show()