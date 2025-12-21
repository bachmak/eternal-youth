import json
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Sample:
    time: str
    PV: float
    consumption: float
    charge_power: float
    discharge_power: float


MAPPING = {
    "xAxis": "time",
    "productPower": "PV",
    "usePower": "consumption",
    "chargePower": "charge_power",
    "dischargePower": "discharge_power",
}

FILENAME_PATTERN = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}).*\.json$")


def extract_json_data(
        json_path: Path,
        mapping: dict[str, str],
) -> list[Sample]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not payload.get("success"):
        raise ValueError(f"{json_path.name}: JSON indicates success=false")

    data = payload.get("data", {})

    extracted = {}
    lengths = set()

    for json_key, attr_name in mapping.items():
        values = data.get(json_key)
        if not isinstance(values, list):
            raise TypeError(f"{json_path.name}: expected list for '{json_key}'")

        extracted[attr_name] = values
        lengths.add(len(values))

    if len(lengths) != 1:
        raise ValueError(f"{json_path.name}: array length mismatch")

    n = lengths.pop()

    rows = []
    for i in range(n):
        sample = Sample(
            time=extracted["time"][i],
            PV=extracted["PV"][i],
            consumption=extracted["consumption"][i],
            charge_power=extracted["charge_power"][i],
            discharge_power=extracted["discharge_power"][i],
        )
        rows.append(sample)

    return rows


def batch_collect(
        input_dir: str,
        mapping: dict[str, str] = MAPPING,
        pattern: re.Pattern[str] = FILENAME_PATTERN,
) -> list[Sample]:
    input_dir = Path(input_dir)
    json_files = [
        p for p in input_dir.iterdir()
        if p.is_file() and pattern.fullmatch(p.name)
    ]

    if not json_files:
        print(f"No JSON files found in: {input_dir}")
        return []

    all_rows = []

    for json_file in sorted(json_files):
        rows = extract_json_data(json_file, mapping)
        all_rows.extend(rows)
        print(f"Loaded {json_file.name}: {len(rows)} samples")

    return all_rows


if __name__ == "__main__":
    data = batch_collect("data/days-range", MAPPING, FILENAME_PATTERN)

    for s in data[:5]:
        print(s)

    print(f"\nTotal samples: {len(data)}")
