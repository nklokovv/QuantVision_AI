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

# Optional (harmless on MPS, useful on CPU/CUDA)
torch.set_float32_matmul_precision("high")


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, layers=4, ff=256, dropout=0.2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x):
        B, T, F = x.shape
        cls = self.cls.expand(B, -1, -1)   # (B,1,F)
        h = self.enc(torch.cat([cls, x], dim=1))  # (B,T+1,F)
        return self.head(h[:, 0, :]).squeeze(-1)


class FeatureProjector(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.proj = nn.Identity() if in_dim == d_model else nn.Linear(in_dim, d_model)
    def forward(self, x): return self.proj(x)


def main(data_dir, test_tickers=None, target_col="y_5d", window=30,
         d_model=64, layers=4, heads=4, dropout=0.2,
         epochs=12, lr=6e-4, bs=256):

    # --- Load & sequence ---
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

    # --- Loaders ---
    tr_ld = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=bs, shuffle=True)
    te_ld = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=bs, shuffle=False)

    # --- Model ---
    proj = FeatureProjector(F, d_model).to(dev)
    model = TransformerEncoder(d_model=d_model, nhead=heads, layers=layers, ff=4*d_model, dropout=dropout).to(dev)
    crit = nn.BCEWithLogitsLoss()
    opt  = torch.optim.AdamW(list(proj.parameters()) + list(model.parameters()), lr=lr)

    # --- Train ---
    for ep in range(1, epochs + 1):
        model.train(); proj.train(); tot = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logits = model(proj(xb))
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        print(f"Epoch {ep}/{epochs} - train loss {tot / max(1, len(tr_ld.dataset)):.4f}")

    # --- Predict ---
    model.eval(); proj.eval(); outs = []
    with torch.no_grad():
        for xb, _ in te_ld:
            xb = xb.to(dev)
            logits = model(proj(xb))
            outs.append(torch.sigmoid(logits).cpu().numpy())
    y_prob = np.concatenate(outs)

    print("== TabTransformer (sequence-attention, MPS-ready) ==")
    print_metrics(yte, y_prob)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--test_tickers", nargs="*", default=None)
    ap.add_argument("--target_col", default="y_5d")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    main(a.data_dir, test_tickers=a.test_tickers, target_col=a.target_col,
         window=a.window, d_model=a.d_model, layers=a.layers, heads=a.heads,
         dropout=a.dropout, epochs=a.epochs, lr=a.lr, bs=a.bs)
