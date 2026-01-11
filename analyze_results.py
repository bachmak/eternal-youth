import pandas as pd
import matplotlib.pyplot as plt

from pipeline import Config


def analyze_results(file_path, cfg):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    start_date = df.index.min()
    end_date = df.index.max()

    duration = end_date - start_date
    days = duration.days
    hours = duration.seconds // 3600

    print(f"Analysed time:")
    print(f"  Start: {start_date.strftime('%d.%m.%Y %H:%M')}")
    print(f"  End:  {end_date.strftime('%d.%m.%Y %H:%M')}")
    print(
        f"  Elapsed time: {days} days and {hours} hours ({len(df)} "
        f"steps)")

    kwh_import_base = df["import"].sum() * cfg.DT
    kwh_export_base = df["export"].sum() * cfg.DT

    cost_import_base = kwh_import_base * cfg.PRICE_BUY
    rev_export_base = kwh_export_base * cfg.PRICE_SELL

    total_cost_base = cost_import_base - rev_export_base

    kwh_import_mpc = df["import_mpc"].sum() * cfg.DT
    kwh_export_mpc = df["export_mpc"].sum() * cfg.DT

    cost_import_mpc = kwh_import_mpc * cfg.PRICE_BUY
    rev_export_mpc = kwh_export_mpc * cfg.PRICE_SELL

    bill_mpc = cost_import_mpc - rev_export_mpc
    total_economic_cost_mpc = bill_mpc

    print("\n" + "=" * 40)
    print("       RESULT COMPARISON")
    print("=" * 40)

    print(
        f"{'Metric':<25} | {'Base':>12} | {'MPC':>12} |"
        f" {'Diff':>12}")
    print("-" * 68)

    print(
        f"{'Import (kWh)':<25} | {kwh_import_base:12.2f} | {kwh_import_mpc:12.2f} | {kwh_import_mpc - kwh_import_base:+12.2f}")
    print(
        f"{'Export (kWh)':<25} | {kwh_export_base:12.2f} | {kwh_export_mpc:12.2f} | {kwh_export_mpc - kwh_export_base:+12.2f}")
    print("-" * 68)
    print(
        f"{'Bill (€)':<25} | {total_cost_base:12.2f} | {bill_mpc:12.2f} |"
        f" {bill_mpc - total_cost_base:+12.2f}")
    print("-" * 68)
    print(
        f"{'Total (€)':<25} | {total_cost_base:12.2f} |"
        f" {total_economic_cost_mpc:12.2f} | {total_economic_cost_mpc - total_cost_base:+12.2f}")
    print("=" * 40)

    savings = total_cost_base - total_economic_cost_mpc
    print(f"\nSummary: You have saved {savings:.2f} € (wear not incl.).")

    step_cost_base = (df["import"] * cfg.PRICE_BUY - df[
        "export"] * cfg.PRICE_SELL) * cfg.DT
    step_cost_mpc = (df["import_mpc"] * cfg.PRICE_BUY - df[
        "export_mpc"] * cfg.PRICE_SELL) * cfg.DT

    cum_base = step_cost_base.cumsum()
    cum_mpc = step_cost_mpc.cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, cum_base, label="Cumulative costs (base)",
             color="red", linestyle="--")
    plt.plot(df.index, cum_mpc, label="Cumulative costs (MPC)",
             color="green")

    plt.title("Costs over the time")
    plt.ylabel("Costs [€]")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_results("out/df_result.csv", Config())
