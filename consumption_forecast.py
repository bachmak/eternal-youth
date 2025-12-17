from pathlib import Path
import pandas as pd


def load_daily_consumption(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if "consumption" not in df.columns:
        raise ValueError(f"{csv_path.name}: missing 'consumption' column")
    return df["consumption"]


def generate_forecast(day_dir: str):
    day_dir = Path(day_dir)

    if not day_dir.exists():
        raise FileNotFoundError(day_dir)

    # Load and sort daily CSVs
    csv_files = sorted(day_dir.glob("*.csv"))

    if len(csv_files) < 9:
        raise ValueError(
            f"{day_dir}: expected at least 9 CSV files (7 history + today + tomorrow)"
        )

    # First 7 days → history
    history_files = csv_files[:7]

    # Day 8 & 9 filenames
    today_file = csv_files[7]
    tomorrow_file = csv_files[8]

    # Load historical consumption
    history = [load_daily_consumption(f) for f in history_files]

    # Validate equal length
    lengths = {len(s) for s in history}
    if len(lengths) != 1:
        raise ValueError(f"{day_dir}: inconsistent time resolution in history")

    n_steps = lengths.pop()

    # Build intraday profile (mean per timestep)
    history_df = pd.DataFrame(history)
    profile = history_df.mean(axis=0)

    # Create forecasts
    forecast_today = profile.copy()
    forecast_tomorrow = profile.copy()

    # Write forecasts
    def write_forecast(base_file: Path, forecast: pd.Series):
        out_path = base_file.with_name(base_file.stem + "_forecast.csv")
        out_df = pd.DataFrame({
            "consumption_forecast": forecast.values
        })
        out_df.to_csv(out_path, index=False)
        print(f"✔ Forecast written: {out_path.name}")

    write_forecast(today_file, forecast_today)
    write_forecast(tomorrow_file, forecast_tomorrow)


if __name__ == "__main__":
    generate_forecast("out/mixed")
