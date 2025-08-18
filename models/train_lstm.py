from __future__ import annotations
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from data_utils import load_all_csvs, split_by_tickers, build_sequences, print_metrics


# ---- Device helpers (Apple MPS → CUDA → CPU) ----
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optional: improves matmul stability/speed on CPU/CUDA (harmless on MPS)
torch.set_float32_matmul_precision("high")


class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden_size=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))
    def forward(self, x):
        h, _ = self.lstm(x)
        last = h[:, -1, :]
        return self.head(last).squeeze(-1)


def train_epoch(m, ld, crit, opt, dev):
    m.train(); tot = 0.0
    for xb, yb in ld:
        xb = xb.to(dev)
        yb = yb.to(dev)
        opt.zero_grad()
        logits = m(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(xb)
    return tot / max(1, len(ld.dataset))


@torch.no_grad()
def pred_probs(m, ld, dev):
    m.eval(); outs = []
    for xb, _ in ld:
        xb = xb.to(dev)
        logits = m(xb)
        outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outs)


def main(data_dir, test_tickers=None, target_col="y_5d", window=30, epochs=10, lr=3e-4, bs=256):
    # Data
    full = load_all_csvs(data_dir, target_col=target_col)
    tr, te = split_by_tickers(full, test_tickers=test_tickers)
    Xtr, ytr = build_sequences(tr, window=window, target_col=target_col, extra_drop=("Ticker",))
    Xte, yte = build_sequences(te, window=window, target_col=target_col, extra_drop=("Ticker",))

    # Scale features (fit on train, apply to test)
    B, T, F = Xtr.shape
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr.reshape(-1, F)).reshape(B, T, F).astype("float32")
    Xte = sc.transform(Xte.reshape(-1, F)).reshape(Xte.shape[0], T, F).astype("float32")

    # Tensors (float32 required for BCE on MPS)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32)

    # Device
    dev = get_device()
    print("Using device:", dev)

    # DataLoaders
    tr_ld = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=bs, shuffle=True)
    te_ld = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=bs, shuffle=False)

    # Model / loss / opt
    m = LSTMClassifier(F).to(dev)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)

    # Train
    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(m, tr_ld, crit, opt, dev)
        print(f"Epoch {ep}/{epochs} - train loss {tr_loss:.4f}")

    # Predict & metrics
    y_prob = pred_probs(m, te_ld, dev)
    print("== LSTM (2-layer, MPS-ready) ==")
    print_metrics(yte, y_prob)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--test_tickers", nargs="*", default=None)
    ap.add_argument("--target_col", default="y_5d")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    main(a.data_dir, test_tickers=a.test_tickers, target_col=a.target_col,
         window=a.window, epochs=a.epochs, lr=a.lr, bs=a.bs)
