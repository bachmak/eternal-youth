import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from train_config import TrainConfig
import os


def create_sequences(data, seq_length):
    """Erstellt Input/Output-Sequenzen für LSTM (nur volle 24h Vorhersagen)"""
    xs, ys = [], []
    total_length = seq_length + 288  # Input + 24h Horizon (288 Punkte à 5 min)
    for i in range(len(data) - total_length + 1):
        x = data[i:i + seq_length]
        y = data[i + seq_length:i + total_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 output_size=288):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # letzte Zeitschritt-Ausgabe
        out = self.fc(out)
        return out


def get_model_from_file(file_path):
    model = LSTMForecast()
    model.load_state_dict(torch.load(file_path))
    return model


def train_model(cfg: TrainConfig, model_cache_dir: str):
    cfg_hash = str(abs(hash(cfg)))
    model_dir = os.path.join(model_cache_dir, cfg_hash)
    os.makedirs(model_dir, exist_ok=True)

    cfg_path = os.path.join(model_dir, f"config.json")
    model_path = os.path.join(model_dir, "model.pth")

    if cfg.is_cached(cfg_path):
        print(f"Using cached model from {model_path}")
        model = get_model_from_file(model_path)
        model.eval()
        return model

    xs, ys = create_sequences(cfg.train_data, cfg.seq_len)
    xs = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1)  # (batch, seq, 1)
    ys = torch.tensor(ys, dtype=torch.float32)  # (batch, 288)

    model = LSTMForecast()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(cfg.epochs):
        optimizer.zero_grad()
        output = model(xs)
        loss = loss_fn(output, ys)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{cfg.epochs}, Loss: {loss.item():.6f}")

    cfg.cache(cfg_path)
    torch.save(model.state_dict(), model_path)
    print(f"Model cached to {model_path}")

    return model


# ------------------------
# Rolling Forecast
# ------------------------
def forecast(model, history, realtime_data, seq_length=288):
    model.eval()
    predictions = []
    data = list(history)
    for t in range(len(realtime_data)):
        x = torch.tensor(data[-seq_length:], dtype=torch.float32).unsqueeze(
            0).unsqueeze(-1)
        with torch.no_grad():
            y_pred = model(x).numpy().flatten()
        predictions.append(y_pred)
        # für das nächste Schritt fügen wir echten Wert hinzu
        data.append(realtime_data[t])
    return np.array(predictions)


def visualize(history, realtime, prediction):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history, label="Train Data")
    plt.subplot(1, 2, 2)
    plt.plot( realtime, label="Realtime Data", color="red")
    plt.plot( prediction, label="Prediction", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Data")
    plt.title("Train and Realtime Data")
    plt.legend()
    plt.show()


def func(t):
    freq = 5
    return np.sin(2 * np.pi * freq * t)

def main():
    t = np.linspace(0, 2 * np.pi, 2288)
    train_t = t[0:2000]
    real_t = t[2000:]

    train_data = func(train_t)
    realtime_data = func(real_t)

    train_config = TrainConfig(
        train_data=train_data,
        seq_len=288,
        epochs=10,
        lr=0.01,
    )
    model = train_model(train_config, "out/model-cache")
    predictions = forecast(model, train_data, realtime_data)
    visualize(train_data, realtime_data, predictions[5])


if __name__ == '__main__':
    main()
