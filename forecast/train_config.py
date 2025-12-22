from __future__ import annotations

import os
import hashlib
import numpy as np
import json


class TrainConfig:
    def __init__(
            self,
            train_data: np.ndarray,
            seq_len: int,
            epochs: int,
            lr: float,
    ):
        self.train_data = np.array(train_data, dtype=float)
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr

    def __eq__(self, other):
        if not isinstance(other, TrainConfig):
            return False

        return (
                self.train_data.all() == other.train_data.all() and
                self.seq_len == other.seq_len and
                self.epochs == other.epochs and
                self.lr == other.lr
        )

    def __hash__(self):
        data_bytes = self.train_data.tobytes()
        data_hash = int(hashlib.md5(data_bytes).hexdigest(), 16)

        combined = (
            data_hash,
            self.seq_len,
            self.epochs,
            self.lr,
        )
        combined_bytes = str(combined).encode("utf-8")
        combined_hash = int(hashlib.md5(combined_bytes).hexdigest(), 16)

        return combined_hash

    def to_dict(self) -> dict:
        return {
            "train_data": self.train_data.tolist(),
            "seq_len": self.seq_len,
            "epochs": self.epochs,
            "lr": self.lr,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrainConfig:
        return TrainConfig(
            train_data=np.array(data["train_data"], dtype=float),
            seq_len=int(data["seq_len"]),
            epochs=int(data["epochs"]),
            lr=float(data["lr"]),
        )

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, data: dict) -> TrainConfig:
        return TrainConfig.from_dict(json.loads(data))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> TrainConfig:
        with open(path, "r") as f:
            return cls.from_json(f.read())

    def is_cached(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False

        other = TrainConfig.load(file_path)
        return self == other

    def cache(self, file_path: str) -> None:
        folder = os.path.dirname(file_path)
        os.makedirs(folder, exist_ok=True)

        self.save(file_path)
