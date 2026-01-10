import json
import re
from pathlib import Path
import pandas as pd

MAPPING = {
    "xAxis": "time",
    "productPower": "pv",
    "usePower": "consumption",
    "selfUsePower": "pv_consumption",
    "chargePower": "charge",
    "dischargePower": "discharge",
}

DAY_DIR_NAME_PATTERN = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})")


def validate_df(df: pd.DataFrame) -> None:
    assert df["time"].notna().all(), "NaT in 'time'"
    assert (df["pv"] >= 0).all(), "negative PV"
    assert (df["consumption"] >= 0).all(), "negative consumption"
    assert (df["pv_consumption"] >= 0).all(), "negative consumption"
    assert (df["soc"] >= 0).all(), "negative soc"
    assert (df["soc"] <= 100.001).all(), "soc greater than 100"


def extract_json_data(
        day_path: Path,
        mapping: dict[str, str],
) -> pd.DataFrame:
    overview_path = day_path / "overview.json"
    soc_path = day_path / "soc.json"

    with overview_path.open("r", encoding="utf-8") as f:
        overview_json = json.load(f)

    with soc_path.open("r", encoding="utf-8") as f:
        soc_json = json.load(f)

    overview_data = overview_json.get("data", {})

    extracted = {
        col_name: overview_data.get(json_key)
        for json_key, col_name in mapping.items()
    }

    soc_data = (
        soc_json
        .get("data", {})
        .get("30007", {})
        .get("pmDataList", [])
    )

    extracted["soc"] = [item["counterValue"] for item in soc_data]

    df = pd.DataFrame(extracted)

    df["time"] = pd.to_datetime(df["time"].str.replace(" DST", "", regex=False))
    df["time"] = df["time"].dt.tz_localize(
        "Europe/Berlin",
        ambiguous='infer',
        nonexistent='shift_forward',
    )

    float_columns = [
        "pv",
        "consumption",
        "soc",
        "pv_consumption",
        "charge",
        "discharge",
    ]
    for c in float_columns:
        df[c] = df[c].astype(float)

    validate_df(df)

    return df


def batch_collect(
        input_dir: str,
        mapping: dict[str, str] = MAPPING,
        pattern: re.Pattern[str] = DAY_DIR_NAME_PATTERN,
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    days = [
        p for p in input_dir.iterdir()
        if p.is_dir() and pattern.fullmatch(p.name)
    ]

    if not days:
        print(f"No JSON files found in: {input_dir}")
        return None

    dfs = []

    for day in sorted(days):
        df = extract_json_data(day, mapping)
        dfs.append(df)
        print(f"Loaded {day}: {len(df)} samples")

    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values("time").reset_index(drop=True)

    return result


def main():
    data = batch_collect("data/days-range", MAPPING, DAY_DIR_NAME_PATTERN)

    print(data)

    print(f"\nTotal samples: {len(data)}")

if __name__ == "__main__":
    main()
