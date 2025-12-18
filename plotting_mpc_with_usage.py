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
    # --- DATEILISTE ---
    FILES = [
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\sunny\2025-06-17.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\sunny\2025-06-18.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\sunny\2025-06-19.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\sunny\2025-06-20.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\sunny\2025-06-21.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\sunny\2025-06-22.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\sunny\2025-06-23.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\sunny\2025-06-24.csv",
        r"C:\Users\henni\PycharmProjects\eternal-youth\out\sunny\2025-06-25.csv"
    ]

    DT = 5.0 / 60.0  # 5 Minuten
    STEPS_PER_DAY = int(24 / DT)

    # --- SYSTEM LIMITS ---
    CAPACITY = 10.0
    P_MAX = 5.0
    ETA = 0.95
    SOC_MIN_HARD = 0.05
    SOC_MAX_HARD = 1.0
    SOC_START = 0.10

    # --- STRATEGIE ---
    SOC_MIN_SOFT = 0.15
    PENALTY_SOFT = 100.0

    # --- PREISE & HEALTH ---
    PRICE_BUY = 0.30
    PRICE_SELL = 0.08
    COST_WEAR = 0.02 #maybe change that later
    COST_SOC_HOLDING = 0.005
    VAL_TERMINAL = 0.15


# ==========================================
# 2. DATEN IMPORT
# ==========================================
def load_and_merge_data(cfg):
    all_days = []
    log(f"Starte Import von {len(cfg.FILES)} Dateien...")

    for i, filename in enumerate(cfg.FILES):
        if not os.path.exists(filename):
            log_err(f"Datei nicht gefunden: {filename}")
            continue

        try:
            df_day = pd.read_csv(filename, sep=None, engine='python')
            df_day.columns = [c.strip().lower() for c in df_day.columns]

            rename_map = {
                'pv': 'pv_kw', 'generation': 'pv_kw', 'erzeugung': 'pv_kw',
                'load': 'load_kw', 'consumption': 'load_kw', 'verbrauch': 'load_kw'
            }
            df_day = df_day.rename(columns=rename_map)

            if 'pv_kw' not in df_day.columns or 'load_kw' not in df_day.columns:
                continue

            df_day = df_day.fillna(0)
            df_day['pv_kw'] = pd.to_numeric(df_day['pv_kw'], errors='coerce').fillna(0).abs()
            df_day['load_kw'] = pd.to_numeric(df_day['load_kw'], errors='coerce').fillna(0).abs()
            all_days.append(df_day)

        except Exception as e:
            log_err(f"Fehler bei {filename}: {e}")

    if not all_days:
        sys.exit(1)

    return pd.concat(all_days, ignore_index=True)


# ==========================================
# 3. PROFIL-GENERATOR
# ==========================================
def calculate_average_profile(df, cfg):
    num_days = len(df) // cfg.STEPS_PER_DAY
    clean_len = num_days * cfg.STEPS_PER_DAY
    load_matrix = df['load_kw'].values[:clean_len].reshape(num_days, cfg.STEPS_PER_DAY)

    avg_day_profile = np.mean(load_matrix, axis=0)
    full_horizon_forecast = np.tile(avg_day_profile, num_days)

    remaining = len(df) - len(full_horizon_forecast)
    if remaining > 0:
        full_horizon_forecast = np.concatenate([full_horizon_forecast, avg_day_profile[:remaining]])

    return full_horizon_forecast


# ==========================================
# 4. SIMULATIONEN
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

        if net < 0:  # Laden
            p = min(-net, cfg.P_MAX)
            max_in = (cfg.CAPACITY - current_energy) / cfg.ETA / cfg.DT
            p = min(p, max_in)
        elif net > 0:  # Entladen
            p = -min(net, cfg.P_MAX)
            avail = current_energy - (cfg.CAPACITY * cfg.SOC_MIN_HARD)
            max_out = max(0, avail) * cfg.ETA / cfg.DT
            p = -min(-p, max_out)

        current_energy += (p * cfg.ETA if p > 0 else p / cfg.ETA) * cfg.DT
        soc.append(current_energy)

    return np.array(soc[:-1])


def simulate_mpc_fast(df, forecast_load, cfg):
    T = len(df)
    log(f"Starte MPC Optimierung (Vektorisiert) für {T} Schritte...")

    load_plan = forecast_load
    pv_plan = df['pv_kw'].values

    p_c = cp.Variable(T, nonneg=True)
    p_d = cp.Variable(T, nonneg=True)
    e_b = cp.Variable(T + 1)
    slack = cp.Variable(T, nonneg=True)
    p_gr_in = cp.Variable(T, nonneg=True)
    p_gr_out = cp.Variable(T, nonneg=True)

    constraints = []
    constraints.append(e_b[0] == cfg.SOC_START * cfg.CAPACITY)
    constraints.append(e_b[1:] == e_b[:-1] + (p_c * cfg.ETA - p_d / cfg.ETA) * cfg.DT)
    constraints.append(load_plan + p_c == pv_plan + p_d + p_gr_in - p_gr_out)
    constraints.append(p_c <= cfg.P_MAX)
    constraints.append(p_d <= cfg.P_MAX)
    constraints.append(e_b[1:] <= cfg.CAPACITY * cfg.SOC_MAX_HARD)
    constraints.append(e_b[1:] >= cfg.CAPACITY * cfg.SOC_MIN_HARD)
    constraints.append(e_b[1:] >= (cfg.CAPACITY * cfg.SOC_MIN_SOFT) - slack)

    cost_term = cp.sum(p_gr_in * cfg.PRICE_BUY - p_gr_out * cfg.PRICE_SELL) * cfg.DT + \
                cp.sum(p_c + p_d) * cfg.COST_WEAR * cfg.DT + \
                cp.sum(e_b[1:]) * cfg.COST_SOC_HOLDING * cfg.DT + \
                cp.sum(slack) * cfg.PENALTY_SOFT - \
                (cfg.VAL_TERMINAL * e_b[T])

    prob = cp.Problem(cp.Minimize(cost_term), constraints)

    log("Solver läuft...")
    try:
        prob.solve(solver=cp.ECOS)
    except:
        prob.solve()

    if prob.status != 'optimal' and prob.status != 'optimal_inaccurate':
        return None, None, None

    # Wir geben jetzt auch die Power-Werte zurück!
    return e_b.value[:-1], p_c.value, p_d.value


# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    log("=== VISUALIZE MPC FLOWS ===")
    cfg = Config()

    # 1. Daten & Profil
    full_df = load_and_merge_data(cfg)
    avg_load_profile = calculate_average_profile(full_df, cfg)

    # 2. Simulationen
    soc_greedy = simulate_greedy(full_df, cfg)
    soc_mpc, p_c_mpc, p_d_mpc = simulate_mpc_fast(full_df, avg_load_profile, cfg)

    if soc_mpc is None:
        sys.exit(1)

    # ----------------------------------------------------
    # BERECHNUNG DER ENERGIEFLÜSSE (Coverage Analysis)
    # ----------------------------------------------------
    # Wir nutzen den MPC-Plan (Batterie) und wenden ihn auf die ECHTE Last an.

    real_load = full_df['load_kw'].values
    real_pv = full_df['pv_kw'].values

    # Grid Bilanz nachberechnen: Load + Charge = PV + Discharge + Grid
    # Grid = (Load + Charge) - (PV + Discharge)
    # Das ist der physikalische Netzbezug, wenn wir den MPC-Plan ausführen
    net_grid_req = (real_load + p_c_mpc) - (real_pv + p_d_mpc)

    grid_import_real = np.maximum(0, net_grid_req)
    grid_export_real = np.maximum(0, -net_grid_req)

    # --- Zerlegung: Wer deckt die Last? ---
    # 1. Batterie deckt Last (soweit entladen wird)
    # Wir nehmen an: Entladung geht primär ins Haus, Überschuss ins Netz.
    covered_by_bat = np.minimum(real_load, p_d_mpc)

    # 2. Grid deckt Last
    # Wenn wir importieren, decken wir damit erst die Last (und dann evtl. Ladung)
    # Import für Last = min(Load - Bat, Import)
    covered_by_grid = np.minimum(np.maximum(0, real_load - covered_by_bat), grid_import_real)

    # 3. PV deckt Last (Rest)
    covered_by_pv = np.maximum(0, real_load - covered_by_bat - covered_by_grid)

    # Plotting Arrays vorbereiten
    days_axis = np.arange(len(full_df)) / cfg.STEPS_PER_DAY

    # ----------------------------------------------------
    # PLOT 1: SoC Validation
    # ----------------------------------------------------
    plt.figure(figsize=(16, 6))
    plt.plot(days_axis, soc_greedy / cfg.CAPACITY * 100, 'r--', alpha=0.6, label='Greedy (Reaktiv)')
    plt.plot(days_axis, soc_mpc / cfg.CAPACITY * 100, 'g-', linewidth=2, label='MPC Plan')
    plt.fill_between(days_axis, soc_greedy / cfg.CAPACITY * 100, soc_mpc / cfg.CAPACITY * 100,
                     where=(soc_greedy > soc_mpc), color='red', alpha=0.15, hatch='//', label='Health Gain')
    plt.axhline(100, color='gray', alpha=0.3);
    plt.ylim(0, 110)
    plt.title('Validation: SoC Behavior', fontsize=14)
    plt.legend(loc='lower center', ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mpc_soc_validation.png", dpi=300)
    log("Plot 1 gespeichert: mpc_soc_validation.png")

    # ----------------------------------------------------
    # PLOT 2: Energy Flow Analysis
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Subplot 1: Last-Vergleich (Was dachte der MPC vs. Realität)
    ax1.plot(days_axis, real_load, color='gray', linewidth=1, alpha=0.5, label='Echte Last (Realität)')
    ax1.plot(days_axis, avg_load_profile, color='blue', linewidth=2, linestyle='-', label='MPC Annahme (Prognose)')
    ax1.set_ylabel('Leistung [kW]')
    ax1.set_title('A. Prognose-Qualität: Durchschnittsprofil vs. Echte Spitzen', loc='left', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Deckungsbeitrag (Stacked Area)
    ax2.stackplot(days_axis,
                  covered_by_pv, covered_by_bat, covered_by_grid,
                  labels=['Direktverbrauch PV (Gelb)', 'Batterie Entladung (Grün)', 'Netzbezug (Rot)'],
                  colors=['#FFD700', '#2E8B57', '#CD5C5C'], alpha=0.7)

    # Echte Lastlinie nochmal drüberlegen zur Orientierung
    ax2.plot(days_axis, real_load, color='black', linewidth=0.5, alpha=0.3)

    ax2.set_ylabel('Verbrauchs-Deckung [kW]')
    ax2.set_xlabel('Tage')
    ax2.set_title('B. Energie-Herkunft: Womit wurde der Hausverbrauch gedeckt?', loc='left', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Tages-Trenner
    for d in range(int(days_axis[-1]) + 1):
        if d > 0:
            ax1.axvline(x=d, color='gray', linestyle=':', alpha=0.5)
            ax2.axvline(x=d, color='gray', linestyle=':', alpha=0.5)
        ax1.text(d + 0.5, ax1.get_ylim()[1] * 0.9, f"Tag {d + 1}", ha='center', color='blue', fontweight='bold')

    plt.tight_layout()
    plt.savefig("mpc_energy_flow.png", dpi=300)
    log("Plot 2 gespeichert: mpc_energy_flow.png")

    plt.show()

    # ----------------------------------------------------
    # ZUSATZ: ZAHLEN & FAKTEN (Statistik-Box)
    # ----------------------------------------------------
    # Wir integrieren die Leistung (kW) über die Zeit (h) -> Energie (kWh)
    total_load_kwh = np.sum(real_load) * cfg.DT
    total_pv_direct_kwh = np.sum(covered_by_pv) * cfg.DT
    total_bat_kwh = np.sum(covered_by_bat) * cfg.DT
    total_grid_kwh = np.sum(covered_by_grid) * cfg.DT

    # Autarkiegrad berechnen (Wie viel % selbst gedeckt?)
    autarkie = (1 - (total_grid_kwh / total_load_kwh)) * 100

    # 1. Print in die Konsole (für dich zum Kopieren)
    print("\n" + "=" * 40)
    print(f"ERGEBNISSE (Hardware Limit: {cfg.P_MAX} kW)")
    print("=" * 40)
    print(f"Gesamtverbrauch (9 Tage): {total_load_kwh:.2f} kWh")
    print(f"----------------------------------------")
    print(
        f"Gedeckt durch PV-Direkt:  {total_pv_direct_kwh:.2f} kWh ({(total_pv_direct_kwh / total_load_kwh) * 100:.1f}%)")
    print(f"Gedeckt durch Batterie:   {total_bat_kwh:.2f} kWh ({(total_bat_kwh / total_load_kwh) * 100:.1f}%)")
    print(f"Gedeckt durch Netzbezug:  {total_grid_kwh:.2f} kWh ({(total_grid_kwh / total_load_kwh) * 100:.1f}%)")
    print(f"----------------------------------------")
    print(f"AUTARKIE-GRAD:            {autarkie:.1f} %")
    print("=" * 40 + "\n")

    # 2. Textbox direkt in den Plot (ax2 ist der Stackplot unten)
    stats_text = (
        f"STATISTIK (Summen):\n"
        f"-------------------\n"
        f"Verbrauch:  {total_load_kwh:.1f} kWh\n\n"
        f"Herkunft:\n"
        f"  - PV:     {total_pv_direct_kwh:.1f} kWh\n"
        f"  - Akku:   {total_bat_kwh:.1f} kWh\n"
        f"  - Netz:   {total_grid_kwh:.1f} kWh\n\n"
        f"Autarkie:   {autarkie:.1f} %"
    )

    # Weiße Box oben links in den unteren Graphen setzen
    ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Speichern wir das Bild nochmal, jetzt mit Zahlen
    plt.savefig(f"mpc_analysis_{cfg.P_MAX}kW.png", dpi=300)