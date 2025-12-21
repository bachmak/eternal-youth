import torch
import torch.nn as nn
import numpy as np

# ----------------------------
# 1. LSTM-Modell definieren
# ----------------------------
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=288):
        """
        output_size = 288, weil 24h / 5min = 288 Schritte
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        # nur letzter hidden state für Vorhersage
        out = out[:, -1, :]
        out = self.linear(out)
        return out

# ----------------------------
# 2. Hilfsfunktion für Sliding-Window-Daten
# ----------------------------
def create_sequences(data, seq_length=288):
    """Erstellt Sequenzen für LSTM (nur volle 24h Vorhersagen)"""
    xs, ys = [], []
    total_length = seq_length + 288  # Input + 24h Horizon
    for i in range(len(data) - total_length + 1):
        x = data[i:i+seq_length]
        y = data[i+seq_length:i+total_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# ----------------------------
# 3. Trainingsfunktion
# ----------------------------
def train_model(train_data, seq_length=288, epochs=10, lr=0.001):
    x, y = create_sequences(train_data, seq_length)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # [batch, seq, 1]
    y = torch.tensor(y, dtype=torch.float32)               # [batch, 288]

    model = LSTMForecast()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")
    return model

# ----------------------------
# 4. Rolling Forecast Funktion
# ----------------------------
def rolling_forecast(model, train_data, realtime_data, seq_length=288):
    """
    train_data: historische Daten
    realtime_data: echte Daten, die nach und nach eintreffen
    seq_length: Länge der Input-Sequenz für LSTM
    """
    forecasts = []
    window = list(train_data[-seq_length:])  # Start mit den letzten Trainingswerten

    for t in range(len(realtime_data)):
        # Input vorbereiten
        x_input = torch.tensor(window[-seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        model.eval()
        with torch.no_grad():
            pred = model(x_input).squeeze(0).numpy()  # Vorhersage 24h
        forecasts.append(pred)

        # echtes neues Sample hinzufügen
        window.append(realtime_data[t])

    return np.array(forecasts)

# ----------------------------
# 5. Beispiel-Simulation
# ----------------------------
if __name__ == "__main__":
    # Dummy-Daten erzeugen
    np.random.seed(42)
    train_data = np.sin(np.linspace(0, 50*np.pi, 10000)) + np.random.normal(0, 0.1, 10000)
    realtime_data = np.sin(np.linspace(50*np.pi, 52*np.pi, 288)) + np.random.normal(0, 0.1, 288)

    # Modell trainieren
    model = train_model(train_data, epochs=5)  # kurze Epochs für Demo

    # Rolling Forecast berechnen
    forecasts = rolling_forecast(model, train_data, realtime_data)
    print("Shape forecasts:", forecasts.shape)
    print("Forecast Beispiel (erste 5):\n", forecasts[0][:5])
