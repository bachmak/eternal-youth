from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def daily_energy(series: pd.Series, step_minutes: int) -> float:
    """Integrate power (kW) to energy (kWh)."""
    return series.sum() * (step_minutes / 60.0)


def plot_actual_vs_forecast_with_energy(day_dir: str, timestep_minutes: int = 5):
    day_dir = Path(day_dir)

    # Actual daily files only
    actual_files = sorted(
        f for f in day_dir.glob("*.csv")
        if not f.stem.endswith("_forecast")
    )

    if len(actual_files) < 7:
        raise ValueError(
            f"{day_dir}: expected at least 7 actual CSVs for history"
        )

    # First 7 days = history
    history_files = actual_files[:7]

    # Current day actual (8th day) if it exists
    today_actual = actual_files[7] if len(actual_files) >= 8 else None

    # Forecast file (must exist)
    if today_actual:
        forecast_path = day_dir / f"{today_actual.stem}_forecast.csv"
        day_label = today_actual.stem
    else:
        # fallback: infer forecast from the only forecast file present
        forecast_files = list(day_dir.glob("*_forecast.csv"))
        if not forecast_files:
            raise FileNotFoundError("No forecast file found")
        forecast_path = forecast_files[0]
        day_label = forecast_path.stem.replace("_forecast", "")

    if not forecast_path.exists():
        raise FileNotFoundError(f"Forecast file not found: {forecast_path.name}")

    forecast_df = pd.read_csv(forecast_path)

    if "consumption_forecast" not in forecast_df.columns:
        raise ValueError("Forecast file missing 'consumption_forecast' column")

    forecast_energy = daily_energy(
        forecast_df["consumption_forecast"], timestep_minutes
    )

    plt.figure()

    # Plot actual if available
    if today_actual:
        actual_df = pd.read_csv(today_actual)

        if "consumption" not in actual_df.columns:
            raise ValueError("Actual file missing 'consumption' column")

        if len(actual_df) != len(forecast_df):
            raise ValueError("Actual and forecast length mismatch")

        actual_energy = daily_energy(
            actual_df["consumption"], timestep_minutes
        )

        plt.plot(actual_df["consumption"])
        plt.plot(forecast_df["consumption_forecast"])

        plt.legend(["Actual", "Forecast"])

        title = (
            f"Actual vs Forecast Consumption ({day_label})\n"
            f"Actual: {actual_energy:.2f} kWh | "
            f"Forecast: {forecast_energy:.2f} kWh | "
            f"Error: {forecast_energy - actual_energy:+.2f} kWh"
        )

        print(f"{day_label}")
        print(f"  Actual energy:   {actual_energy:.2f} kWh")
        print(f"  Forecast energy: {forecast_energy:.2f} kWh")
        print(f"  Error: {forecast_energy - actual_energy:+.2f} kWh")

    else:
        # Forecast only
        plt.plot(forecast_df["consumption_forecast"])
        plt.legend(["Forecast"])

        title = (
            f"Forecast Consumption ({day_label})\n"
            f"Forecast energy: {forecast_energy:.2f} kWh"
        )

        print(f"{day_label}")
        print(f"  Forecast energy: {forecast_energy:.2f} kWh")

    plt.xlabel("5-minute timestep")
    plt.ylabel("Consumption (kW)")
    plt.title(title)
    plt.tight_layout()

    # Save plot
    out_path = day_dir / f"{day_label}_actual_vs_forecast.png"
    plt.savefig(out_path)
    plt.close()

    print(f"✔ Plot saved: {out_path.name}")
