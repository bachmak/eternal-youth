import os.path
import pandas as pd
import numpy as np
import plotly.express as px

from data_parser import batch_collect

def get_table(df_cache, data_folder):
    if os.path.exists(df_cache):
        return pd.read_csv(df_cache)

    df = batch_collect(data_folder)
    df.to_csv(df_cache, index=False)
    return df


def predict_values(
        df,
        curr_idx,
        column,
        history_size,
        horizon_size,
):
    # TODO: improve prediction quality
    start_idx = max(0, curr_idx - history_size)
    history = df[column].iloc[start_idx:curr_idx]

    if len(history) == 0:
        mean_value = 0.0
    else:
        mean_value = history.mean()

    prediction = np.full(horizon_size, mean_value)
    return prediction


def simulate_mpc(df, idx, predicted_consumption):
    # TODO: implement
    return 42.0


def run_single_mpc_step(df, idx):
    history_samples = 7 * 288
    horizon_samples = 288

    consumption = predict_values(
        df,
        idx,
        "consumption",
        history_samples,
        horizon_samples
    )

    soc = simulate_mpc(df, idx, consumption)
    df.at[idx, "soc_mpc"] = soc


def main():
    df = get_table("out/df.csv", "data/days-range")

    for idx in range(len(df)):
        run_single_mpc_step(df, idx)

    fig = px.line(
        df,
        x="time",
        y=["soc", "soc_mpc", "pv", "consumption"],
        title="test visualisation",
    )

    fig.show()


if __name__ == '__main__':
    main()
