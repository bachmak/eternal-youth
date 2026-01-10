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
    P_MAX = 5.0  # kW
    ETA = 0.95  # efficiency

    SOC_MIN_HARD = 0.05
    SOC_MAX_HARD = 1.00

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


def simulate_mpc(pv_forecast, consumption_forecast, curr_soc, cfg):
    assert len(pv_forecast) == len(consumption_forecast)

    horizon = len(pv_forecast)

    # charge power
    p_c = cp.Variable(horizon, nonneg=True)

    # discharge power
    p_d = cp.Variable(horizon, nonneg=True)

    # energy state
    e_b = cp.Variable(horizon + 1)

    # grid import
    p_gr_in = cp.Variable(horizon, nonneg=True)

    # grid export
    p_gr_out = cp.Variable(horizon, nonneg=True)

    # soft constraint violation (low)
    slack_low = cp.Variable(horizon, nonneg=True)

    # soft constraint violation (high)
    slack_high = cp.Variable(horizon, nonneg=True)

    constraints = [e_b[0] == curr_soc * cfg.CAPACITY]
    cost_terms = []

    for k in range(horizon):
        # energy balance
        constraints.append(
            consumption_forecast[k] + p_c[k] ==
            pv_forecast[k] + p_d[k] + p_gr_in[k] - p_gr_out[k]
        )
        # battery dynamics
        constraints.append(
            e_b[k + 1] ==
            e_b[k] + (p_c[k] * cfg.ETA - p_d[k] / cfg.ETA) * cfg.DT
        )
        # hardware limits
        constraints.append(p_c[k] <= cfg.P_MAX)
        constraints.append(p_d[k] <= cfg.P_MAX)
        constraints.append(e_b[k + 1] >= cfg.CAPACITY * cfg.SOC_MIN_HARD)
        constraints.append(e_b[k + 1] <= cfg.CAPACITY * cfg.SOC_MAX_HARD)

        # soft constraints (health corridor)
        constraints.append(
            e_b[k + 1] >= (cfg.CAPACITY * cfg.SOC_MIN_SOFT) - slack_low[k])
        constraints.append(
            e_b[k + 1] <= (cfg.CAPACITY * cfg.SOC_MAX_SOFT) + slack_high[k])

        # cost function
        cost_grid = (p_gr_in[k] * cfg.PRICE_BUY - p_gr_out[k] * cfg.PRICE_SELL) * cfg.DT
        cost_wear = cfg.COST_WEAR * (p_c[k] + p_d[k]) * cfg.DT
        cost_hold = cfg.COST_SOC_HOLDING * e_b[k + 1] * cfg.DT
        cost_penalty = cfg.PENALTY_LOW * slack_low[k] + cfg.PENALTY_HIGH * slack_high[k]

        cost_terms.append(cost_grid + cost_wear + cost_hold + cost_penalty)

    cost_terms.append(-cfg.VAL_TERMINAL * e_b[horizon])

    prob = cp.Problem(cp.Minimize(cp.sum(cost_terms)), constraints)

    # try:
    prob.solve(verbose=True)
    # except:
    #     prob.solve(solver=cp.SCS, verbose=True)

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        return None

    return e_b.value[0]


def run_single_mpc_step(df, idx, cfg):
    consumption_prediction = predict_values(
        df,
        idx,
        "consumption",
        cfg.HISTORY_SAMPLES,
        cfg.HORIZON_SAMPLES,
    )

    pv_forecast = df["pv"][idx:idx + cfg.HORIZON_SAMPLES].to_numpy()
    curr_soc = df["soc_mpc"].iloc[idx]

    soc = simulate_mpc(pv_forecast, consumption_prediction, curr_soc, cfg)
    df.at[idx+1, "soc_mpc"] = soc


def main():
    df = get_table("out/df.csv", "data/days-range")
    cfg = Config()

    df["soc_mpc"] = df["soc"].copy()

    for idx in range(cfg.HISTORY_SAMPLES, len(df)):
        run_single_mpc_step(df, idx, cfg)

    fig = px.line(
        df,
        x="time",
        y=[
            "soc",
            "soc_mpc",
            "pv",
            "consumption",
            "pv_consumption",
            "export",
            "import",
            "charge",
            "discharge",
        ],
        title="test visualisation",
    )

    fig.show()


if __name__ == '__main__':
    main()
