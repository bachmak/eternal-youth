from dataclasses import dataclass
import pandas as pd
import json


@dataclass
class DataSample:
    def __init__(
            self,
            time: str,
            pv: str,
            consumption: str,
            charge_power: str,
            discharge_power: str,
    ):
        self.time = pd.Timestamp(time.replace("DST", ""))
        self.pv = float(pv)
        self.consumption = float(consumption)
        self.charge_power = float(charge_power)
        self.discharge_power = float(discharge_power)

    def __str__(self) -> str:
        return self.to_json()

    def to_dict(self) -> dict:
        return {
            "time": str(self.time),
            "pv": self.pv,
            "consumption": self.consumption,
            "charge_power": self.charge_power,
            "discharge_power": self.discharge_power,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    time: pd.Timestamp
    PV: float
    consumption: float
    charge_power: float
    discharge_power: float
