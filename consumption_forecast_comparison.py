from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_actual_vs_forecast(day_dir: str):
    day_dir = Path(day_dir)

    # Only actual daily files (ignore forecasts)
    actual_files = sorted(
        f for f in day_dir.glob("*.csv")
        if not f.stem.endswith("_forecast")
    )

    if len(actual_files) < 8:
        raise ValueError(
            f"{day_dir}: expected at least 8 actual daily CSV files"
        )

    # Convention:
    # first 7 days = history
    # 8th day = current day
    today_actual = actual_files[7]
    today_forecast = day_dir / f"{today_actual.stem}_forecast.csv"

    if not today_forecast.exists():
        raise FileNotFoundError(
            f"Missing forecast file for current day: {today_forecast.name}"
        )

    actual_df = pd.read_csv(today_actual)
    forecast_df = pd.read_csv(today_forecast)

    if "consumption" not in actual_df.columns:
        raise ValueError("Actual file missing 'consumption' column")

    if "consumption_forecast" not in forecast_df.columns:
        raise ValueError("Forecast file missing 'consumption_forecast' column")

    if len(actual_df) != len(forecast_df):
        raise ValueError("Actual and forecast data length mismatch")

    plt.figure()
    plt.plot(actual_df["consumption"])
    plt.plot(forecast_df["consumption_forecast"])
    plt.xlabel("5-minute timestep")
    plt.ylabel("Consumption")
    plt.title(f"Actual vs Forecast Consumption ({today_actual.stem})")
    plt.legend(["Actual", "Forecast"])
    plt.show()


if __name__ == "__main__":
    plot_actual_vs_forecast("out/mixed")
