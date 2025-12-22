import json
import re
from pathlib import Path
import pandas as pd

MAPPING = {
    "xAxis": "time",
    "productPower": "pv",
    "usePower": "consumption",
    "chargePower": "charge_power",
    "dischargePower": "discharge_power",
}

FILENAME_PATTERN = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}).*\.json$")


def validate_df(df: pd.DataFrame) -> None:
    assert df["time"].notna().all(), "NaT in 'time'"
    assert (df["pv"] >= 0).all(), "Negative PV-Werte"
    assert (df["consumption"] >= 0).all(), "Negative consumption-Werte"


def extract_json_data(
        json_path: Path,
        mapping: dict[str, str],
) -> pd.DataFrame:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not payload.get("success"):
        raise ValueError(f"{json_path.name}: JSON indicates success=false")

    data = payload.get("data", {})

    def ensure_list(key, values):
        if not isinstance(values, list):
            raise TypeError(f"{json_path.name}: expected list for '{key}'")
        return values

    extracted = {
        col_name: ensure_list(json_key, data.get(json_key))
        for json_key, col_name in mapping.items()
    }

    df = pd.DataFrame(extracted)

    df["time"] = pd.to_datetime(df["time"].str.replace(" DST", "", regex=False))
    df["time"] = df["time"].dt.tz_localize(
        "Europe/Berlin",
        ambiguous='infer',
        nonexistent='shift_forward',
    )

    float_columns = ["pv", "consumption", "charge_power", "discharge_power"]
    for c in float_columns:
        df[c] = df[c].astype(float)

    validate_df(df)

    return df


def batch_collect(
        input_dir: str,
        mapping: dict[str, str] = MAPPING,
        pattern: re.Pattern[str] = FILENAME_PATTERN,
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    json_files = [
        p for p in input_dir.iterdir()
        if p.is_file() and pattern.fullmatch(p.name)
    ]

    if not json_files:
        print(f"No JSON files found in: {input_dir}")
        return None

    dfs = []

    for json_file in sorted(json_files):
        df = extract_json_data(json_file, mapping)
        dfs.append(df)
        print(f"Loaded {json_file.name}: {len(df)} samples")

    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values("time").reset_index(drop=True)

    return result


if __name__ == "__main__":
    data = batch_collect("data/days-range", MAPPING, FILENAME_PATTERN)

    print(data)

    print(f"\nTotal samples: {len(data)}")
