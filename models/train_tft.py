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

# Optional: harmless on MPS; improves CPU/CUDA matmul stability/speed
torch.set_float32_matmul_precision("high")


class AttentionPooling(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)
    def forward(self, x):                 # x: (B, T, H)
        a = torch.softmax(self.w(x).squeeze(-1), dim=1)  # (B, T)
        return torch.sum(x * a.unsqueeze(-1), dim=1)     # (B, H)


class TFT_Lite(nn.Module):
    """Simplified TFT-style: proj -> LSTM -> attention pooling -> head"""
    def __init__(self, in_dim, hidden=16, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True, dropout=dropout)
        self.attn = AttentionPooling(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x):
        h, _ = self.lstm(self.proj(x))
        return self.head(self.attn(h)).squeeze(-1)


def main(data_dir, test_tickers=None, target_col="y_5d",
         window=30, epochs=10, lr=1e-3, bs=256):

    # --- Load & build sequences ---
    full = load_all_csvs(data_dir, target_col=target_col)
    tr, te = split_by_tickers(full, test_tickers=test_tickers)
    Xtr, ytr = build_sequences(tr, window=window, target_col=target_col, extra_drop=("Ticker",))
    Xte, yte = build_sequences(te, window=window, target_col=target_col, extra_drop=("Ticker",))

    # --- Scale features ---
    B, T, F = Xtr.shape
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr.reshape(-1, F)).reshape(B, T, F).astype("float32")
    Xte = sc.transform(Xte.reshape(-1, F)).reshape(Xte.shape[0], T, F).astype("float32")

    # --- Tensors (force float32 for MPS) ---
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32)

    # --- Device ---
    dev = get_device()
    print("Using device:", dev)

    # --- DataLoaders ---
    tr_ld = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=bs, shuffle=True)
    te_ld = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=bs, shuffle=False)

    # --- Model / loss / opt ---
    m = TFT_Lite(F, hidden=16, dropout=0.1).to(dev)
    crit = nn.BCEWithLogitsLoss()
    opt  = torch.optim.AdamW(m.parameters(), lr=lr)

    # --- Train ---
    for ep in range(1, epochs + 1):
        m.train(); tot = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(m(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        print(f"Epoch {ep}/{epochs} - train loss {tot / max(1, len(tr_ld.dataset)):.4f}")

    # --- Predict ---
    m.eval(); outs = []
    with torch.no_grad():
        for xb, _ in te_ld:
            xb = xb.to(dev)
            outs.append(torch.sigmoid(m(xb)).cpu().numpy())
    y_prob = np.concatenate(outs)

    print("== Temporal Fusion Transformer (lite, MPS-ready) ==")
    print_metrics(yte, y_prob)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--test_tickers", nargs="*", default=None)
    ap.add_argument("--target_col", default="y_5d")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    main(a.data_dir, test_tickers=a.test_tickers, target_col=a.target_col,
         window=a.window, epochs=a.epochs, lr=a.lr, bs=a.bs)
