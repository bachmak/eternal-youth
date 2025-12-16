import json
import csv
import re
from pathlib import Path


# JSON → CSV column mapping
MAPPING = {
    "xAxis": "time",
    "productPower": "PV",
    "usePower": "consumption",
    "chargePower": "charge_power",
    "dischargePower": "discharge_power",
}

# Match filenames like: 28-11-2025.json
FILENAME_PATTERN = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}).*\.json$")


def convert_file(json_path: Path, out_dir: Path):
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not payload.get("success"):
        raise ValueError(f"{json_path.name}: JSON indicates success=false")

    data = payload.get("data", {})

    extracted = {}
    lengths = set()

    for json_key, csv_col in MAPPING.items():
        values = data.get(json_key)

        if not isinstance(values, list):
            raise TypeError(
                f"{json_path.name}: expected list for '{json_key}', got {type(values)}"
            )

        extracted[csv_col] = values
        lengths.add(len(values))

    if len(lengths) != 1:
        detail = {k: len(v) for k, v in extracted.items()}
        raise ValueError(
            f"{json_path.name}: array length mismatch: {detail}"
        )

    row_count = lengths.pop()

    out_path = out_dir / json_path.with_suffix(".csv").name
    fieldnames = list(extracted.keys())

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(row_count):
            writer.writerow({col: extracted[col][i] for col in fieldnames})

    print(f"✔ {json_path.name} → {out_path.name} ({row_count} rows)")


def batch_convert(input_dir: str, output_dir: str = "out"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = [
        p for p in input_dir.iterdir()
        if p.is_file() and FILENAME_PATTERN.fullmatch(p.name)
    ]

    if not json_files:
        print("No matching JSON files found.")
        return

    for json_file in sorted(json_files):
        convert_file(json_file, output_dir)


if __name__ == "__main__":
    batch_convert("data")
