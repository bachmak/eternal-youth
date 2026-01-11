import os.path
import pandas as pd
import numpy as np
import plotly.express as px
import cvxpy as cp

from load_forecast_profile import predict_load_horizon
from data_parser import batch_collect
from optimizer import MPCOptimizer


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

def run_single_mpc_step(df, idx, cfg, mpc):
    consumption_prediction = predict_values(
        df,
        idx,
        "consumption",
        cfg.HISTORY_SAMPLES,
        cfg.HORIZON_SAMPLES,
    )

    # Replace the first element in the prediction with the real data
    consumption_current = df["consumption"].iloc[idx]
    consumption_prediction[0] = consumption_current

    end_idx = idx + cfg.HORIZON_SAMPLES
    if end_idx > len(df):
        return

    pv_forecast = df["pv"].iloc[idx:end_idx].to_numpy()

    current_ts = df.index[idx]
    soc = df.at[current_ts, "soc_mpc"]

    p_charge, p_discharge, p_import, p_export = mpc.solve(
        pv_forecast,
        consumption_prediction,
        soc
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

    mpc_optimizer = MPCOptimizer(cfg)
    print("MPC Optimizer initialized and compiled.")

    loop_range = range(cfg.HISTORY_SAMPLES, len(df) - cfg.HORIZON_SAMPLES)
    full = len(loop_range)

    for i, idx in enumerate(loop_range):
        run_single_mpc_step(df, idx, cfg, mpc_optimizer)
        if i % 100 == 0:
            print(f"Iteration {i}/{full}")

    df.to_csv("out/df_result.csv", index=False)

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
