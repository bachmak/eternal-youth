import os.path
import pandas as pd
import numpy as np
import plotly.express as px
import cvxpy as cp

from load_forecast_profile import predict_load_horizon
from data_parser import batch_collect


class Config:
    DT = 5.0 / 60.0  # 5 minutes per sample
    HISTORY_SAMPLES = 7 * 288  # 7 days
    HORIZON_SAMPLES = 288  # 1 day

    CAPACITY = 10.0  # kWh
    P_CH_MAX = 5.0  # kW
    P_DIS_MAX = 5.0  # kW
    ETA_CH = 0.95  # efficiency
    ETA_DIS = 0.95  # efficiency

    SOC_MIN_HARD = 0.05
    SOC_MAX_HARD = 1.00
    SOC_REF = 0.5

    # Cost Function Weights
    PRICE_BUY = 0.30  # Grid Import Price
    PRICE_SELL = 0.08  # Feed-in Tariff
    COST_WEAR = 0.02  # Cyclic Aging Cost (per kWh throughput)
    COST_SOC_HOLDING = 0.01  # Calendar Aging Cost ("Parking Fee")


def get_table(df_cache, data_folder):
    if os.path.exists(df_cache):
        return pd.read_csv(df_cache)

    df = batch_collect(data_folder)
    df["soc"] = df["soc"] / 100.0

    df["import"] = (
            df["consumption"] -
            df["pv_consumption"] + df["charge"] -
            df["discharge"]
    ).clip(lower=0)

    df["export"] = (
            df["pv"] - df["pv_consumption"]
    ).clip(lower=0)

    df.to_csv(df_cache, index=False)
    return df


def predict_values(
        df,
        curr_idx,
        column,
        history_size,
        horizon_size,
):
    return predict_load_horizon(
        df=df,
        curr_idx=int(curr_idx),
        horizon_size=int(horizon_size),
        history_size=int(history_size),
        load_col="consumption",
        pv_col="pv",
        mode="auto",
    )



def simulate_mpc(
        pv_forecast,
        consumption_forecast,
        curr_soc,
        cfg,
):
    assert len(pv_forecast) == len(consumption_forecast)

    n = len(pv_forecast)

    pv = np.array(pv_forecast)
    load = np.array(consumption_forecast)

    p_ch = cp.Variable(n, nonneg=True)
    p_dis = cp.Variable(n, nonneg=True)
    p_gr_im = cp.Variable(n, nonneg=True)
    p_gr_ex = cp.Variable(n, nonneg=True)

    soc = cp.Variable(n + 1)

    constraints = [
        soc[0] == curr_soc,
        soc >= cfg.SOC_MIN_HARD,
        soc <= cfg.SOC_MAX_HARD,
        p_ch <= cfg.P_CH_MAX,
        p_dis <= cfg.P_DIS_MAX,
    ]

    for k in range(n):
        constraints.append(
            soc[k + 1] == soc[k] +
            (p_ch[k] * cfg.ETA_CH - p_dis[k] / cfg.ETA_DIS) *
            cfg.DT / cfg.CAPACITY
        )

        constraints.append(
            p_gr_im[k] - p_gr_ex[k] ==
            load[k] + p_ch[k] - p_dis[k] - pv[k]
        )

    cost_terms = (
            cfg.PRICE_BUY * cp.sum(p_gr_im)  # import
            - cfg.PRICE_SELL * cp.sum(p_gr_ex)  # export
            + cfg.COST_WEAR * cp.sum(p_ch + p_dis)  # cycles
            + cfg.COST_SOC_HOLDING * cp.sum_squares(soc - cfg.SOC_REF)  # SoC
    )

    problem = cp.Problem(cp.Minimize(cost_terms), constraints)

    problem.solve(
        verbose=True,
        max_iter=20000,
        eps_abs=1e-3,
        eps_rel=1e-3,
    )

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise Exception(f"Problem {problem.status} is not optimal")

    p_charge = float(p_ch.value[0])
    p_discharge = float(p_dis.value[0])
    p_import = float(p_gr_im.value[0])
    p_export = float(p_gr_ex.value[0])

    return p_charge, p_discharge, p_import, p_export


def run_single_mpc_step(df, idx, cfg):
    consumption_prediction = predict_values(
        df,
        idx,
        "consumption",
        cfg.HISTORY_SAMPLES,
        cfg.HORIZON_SAMPLES,
    )

    # Replace the first element in the prediction with the real data
    consumption_current = df["consumption"][idx]
    consumption_prediction[0] = consumption_current

    end_idx = idx + cfg.HORIZON_SAMPLES
    if end_idx > len(df):
        return

    pv_forecast = df["pv"][idx:end_idx].to_numpy()

    current_ts = df.index[idx]
    soc = df.at[current_ts, "soc_mpc"]

    p_charge, p_discharge, p_import, p_export = simulate_mpc(
        pv_forecast,
        consumption_prediction,
        soc,
        cfg
    )

    df.at[current_ts, "export_mpc"] = p_export
    df.at[current_ts, "import_mpc"] = p_import
    df.at[current_ts, "charge_mpc"] = p_charge
    df.at[current_ts, "discharge_mpc"] = p_discharge

    pv_val = df.at[current_ts, "pv"]
    df.at[current_ts, "pv_consumption_mpc"] = max(0, pv_val - p_export)

    if idx + 1 >= len(df):
        return

    next_soc = (
            soc +
            (p_charge * cfg.ETA_CH - p_discharge / cfg.ETA_DIS)
            * cfg.DT / cfg.CAPACITY
    )
    next_ts = df.index[idx + 1]
    df.at[next_ts, "soc_mpc"] = next_soc


def main():
    df = get_table("out/df.csv", "data/days-range")
    cfg = Config()

    for column in [
        "soc",
        "pv_consumption",
        "export",
        "import",
        "charge",
        "discharge",
    ]:
        df[f"{column}_mpc"] = df[column].copy()

    for idx in range(cfg.HISTORY_SAMPLES, len(df) - cfg.HORIZON_SAMPLES):
        run_single_mpc_step(df, idx, cfg)
        full = len(df) - cfg.HISTORY_SAMPLES - cfg.HORIZON_SAMPLES
        print(f"Iteration {idx}/{full}")

    fig = px.line(
        df,
        x="time",
        y=[
            "pv",
            "consumption",

            "soc",
            "pv_consumption",
            "export",
            "import",
            "charge",
            "discharge",

            "soc_mpc",
            "pv_consumption_mpc",
            "export_mpc",
            "import_mpc",
            "charge_mpc",
            "discharge_mpc",
        ],
        title="test visualisation",
    )

    fig.show()


if __name__ == '__main__':
    main()
