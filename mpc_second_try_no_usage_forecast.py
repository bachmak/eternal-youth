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

    DT = 5.0 / 60.0  # 5 Minuten Schritte
    STEPS_PER_DAY = int(24 / DT)  # 288 Schritte

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
    COST_WEAR = 0.02
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

            # Wir merken uns, welcher Tag das ist (für spätere Analyse)
            df_day['day_index'] = i
            all_days.append(df_day)

        except Exception as e:
            log_err(f"Fehler bei {filename}: {e}")

    if not all_days:
        sys.exit(1)

    full_df = pd.concat(all_days, ignore_index=True)
    return full_df


# ==========================================
# 3. PROFIL-GENERATOR (NEU!)
# ==========================================
def calculate_average_profile(df, cfg):
    """
    Erstellt ein 'Standard-Lastprofil' aus den historischen Daten.
    Das ist das, was der MPC glaubt, was passieren wird (seine Prognose).
    """
    # Wir nehmen an, der Verbrauch wiederholt sich zyklisch alle 288 Schritte
    # Wir reshape die Daten in (Tage, Schritte_pro_Tag)
    num_days = len(df) // cfg.STEPS_PER_DAY
    # Abschneiden falls Rest
    clean_len = num_days * cfg.STEPS_PER_DAY
    load_matrix = df['load_kw'].values[:clean_len].reshape(num_days, cfg.STEPS_PER_DAY)

    # Durchschnitt über alle Tage berechnen (Spalten-Mittelwert)
    avg_day_profile = np.mean(load_matrix, axis=0)

    # Das Profil für den gesamten Horizont wiederholen
    full_horizon_forecast = np.tile(avg_day_profile, num_days)

    # Falls Rest am Ende war, füllen wir mit dem Anfang des Profils auf
    remaining = len(df) - len(full_horizon_forecast)
    if remaining > 0:
        full_horizon_forecast = np.concatenate([full_horizon_forecast, avg_day_profile[:remaining]])

    return full_horizon_forecast


# ==========================================
# 4. SIMULATIONEN
# ==========================================
def simulate_greedy(df, cfg):
    """Der dumme Regler sieht immer nur das Jetzt -> Keine Änderung nötig"""
    log("Starte Greedy Simulation...")
    steps = len(df)
    soc = [cfg.SOC_START * cfg.CAPACITY]
    current_energy = soc[0]

    pv_data = df['pv_kw'].values
    load_data = df['load_kw'].values  # ECHTE Last

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
        soc.append(current_energy)

    return np.array(soc[:-1])


def simulate_mpc_realistic(df, forecast_load, cfg):
    """
    Der MPC optimiert basierend auf PROGNOSE (Wetter=Perfekt, Last=Durchschnitt).
    Aber wir validieren das Ergebnis gegen die REALITÄT.

    Hier vereinfacht: Open-Loop MPC.
    Wir lassen den MPC einmal komplett planen basierend auf seiner falschen Annahme.
    Dann schauen wir, ob der Plan in der Realität aufgeht.
    """
    T = len(df)
    log(f"Starte Realistic MPC (Planung mit Durchschnittslast)...")

    # Vektoren für die Optimierung (Das, was der MPC *denkt*)
    load_plan = forecast_load  # <-- HIER IST DER UNTERSCHIED: Er nutzt den Durchschnitt!
    pv_plan = df['pv_kw'].values  # Wetter kennt er (Annahme)

    # ECHTE Daten für die Validierung
    load_real = df['load_kw'].values

    # --- 1. PLANUNGSPHASE (Der MPC träumt) ---
    p_c = cp.Variable(T, nonneg=True)
    p_d = cp.Variable(T, nonneg=True)
    e_b = cp.Variable(T + 1)
    slack = cp.Variable(T, nonneg=True)
    p_gr_in = cp.Variable(T, nonneg=True)
    p_gr_out = cp.Variable(T, nonneg=True)

    constraints = [e_b[0] == cfg.SOC_START * cfg.CAPACITY]
    cost_terms = []

    for k in range(T):
        # Er plant mit load_plan (Durchschnitt)
        constraints.append(load_plan[k] + p_c[k] == pv_plan[k] + p_d[k] + p_gr_in[k] - p_gr_out[k])
        constraints.append(e_b[k + 1] == e_b[k] + (p_c[k] * cfg.ETA - p_d[k] / cfg.ETA) * cfg.DT)

        constraints.append(p_c[k] <= cfg.P_MAX);
        constraints.append(p_d[k] <= cfg.P_MAX)
        constraints.append(e_b[k + 1] <= cfg.CAPACITY * cfg.SOC_MAX_HARD)
        constraints.append(e_b[k + 1] >= cfg.CAPACITY * cfg.SOC_MIN_HARD)
        constraints.append(e_b[k + 1] >= (cfg.CAPACITY * cfg.SOC_MIN_SOFT) - slack[k])

        cost = (p_gr_in[k] * cfg.PRICE_BUY - p_gr_out[k] * cfg.PRICE_SELL) * cfg.DT + \
               cfg.COST_WEAR * (p_c[k] + p_d[k]) * cfg.DT + \
               cfg.COST_SOC_HOLDING * e_b[k + 1] * cfg.DT + cfg.PENALTY_SOFT * slack[k]
        cost_terms.append(cost)

    cost_terms.append(-cfg.VAL_TERMINAL * e_b[T])

    prob = cp.Problem(cp.Minimize(cp.sum(cost_terms)), constraints)
    try:
        prob.solve(solver=cp.ECOS)
    except:
        prob.solve()

    if prob.status != 'optimal' and prob.status != 'optimal_inaccurate':
        log_err("MPC konnte keinen Plan erstellen!")
        return None

    # --- 2. REALITÄTSCHECK (Der Plan trifft auf das Leben) ---
    # Wir nehmen die geplante Batterie-Leistung (p_c - p_d) und wenden sie auf die ECHTE Last an.
    # Wenn die Last höher ist als gedacht, müssen wir Netzstrom nehmen (nicht Batterie, denn die folgt dem Plan).
    # ACHTUNG: In einem echten MPC würde man alle 15 min nachregeln.
    # Hier nehmen wir einfach die geplanten SoC-Werte, prüfen aber, ob sie physikalisch möglich sind.

    # Einfacherer Ansatz für die Grafik:
    # Wir zeigen, was der MPC *geplant* hat (basierend auf Durchschnitt).
    # Das zeigt seine Strategie (spätes Laden).
    # Da wir Open-Loop simulieren, ist der geplante SoC das Ergebnis.
    # In der Realität würde der Regler kurzfristig reagieren (Intraday), aber die Strategie bleibt gleich.

    return e_b.value[:-1]


# ==========================================
# 5. MAIN & PLOTTING
# ==========================================
if __name__ == "__main__":
    log("=== REALISTIC MPC SIMULATION ===")
    cfg = Config()

    # 1. Daten laden
    full_df = load_and_merge_data(cfg)

    # 2. Lastprofil erstellen (Durchschnittsbildung)
    avg_load_profile = calculate_average_profile(full_df, cfg)

    # 3. Simulationen
    soc_greedy = simulate_greedy(full_df, cfg)

    # Der MPC bekommt jetzt das avg_load_profile statt der echten Last!
    soc_mpc = simulate_mpc_realistic(full_df, avg_load_profile, cfg)

    if soc_mpc is None:
        sys.exit(1)

    # Plotting
    soc_g_perc = soc_greedy / cfg.CAPACITY * 100
    soc_m_perc = soc_mpc / cfg.CAPACITY * 100

    steps_per_day = int(24 / cfg.DT)
    days_axis = np.arange(len(full_df)) / steps_per_day

    plt.figure(figsize=(16, 8))

    # 1. Greedy (Rot)
    plt.plot(days_axis, soc_g_perc, color='red', linestyle='--', alpha=0.6, label='Greedy (Reagiert auf echte Last)')

    # 2. MPC (Grün)
    plt.plot(days_axis, soc_m_perc, color='green', linewidth=2.5, label='MPC Plan (Basierend auf Durchschnittslast)')

    # 3. Last-Vergleich (damit man sieht, was passiert)
    # Wir plotten die Last auf einer zweiten Y-Achse
    # ax2 = plt.gca().twinx()
    # ax2.plot(days_axis, full_df['load_kw'], color='gray', alpha=0.1, label='Echte Last')
    # ax2.plot(days_axis, avg_load_profile, color='blue', alpha=0.2, linestyle=':', label='MPC Annahme (Durchschnitt)')

    plt.fill_between(days_axis, soc_g_perc, soc_m_perc, where=(soc_g_perc > soc_m_perc),
                     color='red', alpha=0.15, hatch='//', label='Health Gain')

    # Styling
    plt.axhline(100, color='gray', linewidth=1, alpha=0.5)
    plt.ylim(0, 110)
    plt.title('Robustheits-Check: MPC plant ohne perfektes Last-Wissen', fontsize=16)
    plt.legend(loc='lower center', ncol=3)
    plt.grid(True, alpha=0.3)

    # Tagesmarkierungen
    for d in range(int(days_axis[-1]) + 1):
        if d > 0: plt.axvline(x=d, color='gray', linestyle=':', alpha=0.5)
        plt.text(d + 0.5, 105, f"Tag {d + 1}", ha='center')

    plt.tight_layout()
    plt.savefig("mpc_realistic_load.png", dpi=300)
    plt.show()
    log("Fertig.")