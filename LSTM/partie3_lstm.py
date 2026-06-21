"""
=============================================================
PROJET DEEP LEARNING – EMSI 2025-2026
Partie III : RNN / LSTM / GRU / Seq2Seq
Tâche     : Prévision météo pour alerter l'agriculteur
Dataset   : Jena Climate Dataset – 420 000 observations horaires
            (température, humidité, pression, vent…)
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math, warnings
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES MÉTÉO
# ─────────────────────────────────────────────
def load_weather(path="jena_climate_2009_2016.csv", max_rows=50000):
    """
    Charge le Jena Climate Dataset.
    Téléchargement : https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
    Colonnes utilisées : T (°C), rh (%), p (hPa), wv (m/s), rain (mm)
    """
    try:
        df = pd.read_csv(path, nrows=max_rows)
        cols = ['T (degC)', 'rh (%)', 'p (mbar)', 'wv (m/s)', 'rain (mm)']
        cols = [c for c in cols if c in df.columns]
        data = df[cols].values.astype(np.float32)
        print(f"[Data] Jena Climate : {data.shape[0]} obs, {data.shape[1]} features")
    except FileNotFoundError:
        print("[Data] Fichier absent – données synthétiques (demo)")
        n = max_rows
        t = np.linspace(0, 4*np.pi, n)
        data = np.column_stack([
            20 + 10*np.sin(t) + np.random.randn(n)*1.5,    # température
            60 + 20*np.cos(t) + np.random.randn(n)*5,      # humidité
            1013 + 5*np.sin(t*0.3) + np.random.randn(n),   # pression
            3 + 2*np.abs(np.sin(t*2)) + np.random.randn(n)*0.5,  # vent
            np.clip(np.random.exponential(0.5, n), 0, 5)   # pluie
        ]).astype(np.float32)
    return data

def build_sequences(data, seq_len=72, pred_len=24):
    """
    Construit des séquences (X, y) pour la prévision.
    X : fenêtre de seq_len pas → y : pred_len pas suivants (température)
    """
    # Normalisation min-max
    mins  = data.min(0)
    maxs  = data.max(0)
    data_norm = (data - mins) / (maxs - mins + 1e-8)

    Xs, ys = [], []
    for i in range(len(data_norm) - seq_len - pred_len):
        Xs.append(data_norm[i:i+seq_len])
        ys.append(data_norm[i+seq_len:i+seq_len+pred_len, 0])  # température seulement
    return np.array(Xs), np.array(ys), mins, maxs

# ─────────────────────────────────────────────
# 2. PRÉPARATION DES DATASETS
# ─────────────────────────────────────────────
SEQ_LEN  = 72    # 72h passées
PRED_LEN = 24    # prédire 24h
data_raw = load_weather()
X, y, mins, maxs = build_sequences(data_raw, SEQ_LEN, PRED_LEN)

n = len(X)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

X_train, y_train = X[:n_train],          y[:n_train]
X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test,  y_test  = X[n_train+n_val:],    y[n_train+n_val:]

def make_loader(X, y, batch_size=64, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.float32).to(device)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val)
test_loader  = make_loader(X_test,  y_test)

IN_FEATURES = X.shape[2]
print(f"[Data] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. MODÈLES RÉCURRENTS
# ─────────────────────────────────────────────
class WeatherRNN(nn.Module):
    """RNN simple pour prévision météo."""
    def __init__(self, in_features, hidden, pred_len, num_layers=2, dropout=0.2, cell='LSTM'):
        super().__init__()
        self.cell = cell
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[cell]
        self.rnn = rnn_cls(
            in_features, hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc  = nn.Linear(hidden, pred_len)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.rnn(x)          # (B, T, hidden)
        last    = out[:, -1, :]       # dernier état caché
        return self.fc(self.drop(last))

# ─────────────────────────────────────────────
# 4. ARCHITECTURE SEQ2SEQ
# ─────────────────────────────────────────────
class Seq2SeqEncoder(nn.Module):
    def __init__(self, in_features, hidden, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x):
        outputs, hidden = self.gru(x)
        return outputs, hidden

class Seq2SeqDecoder(nn.Module):
    def __init__(self, hidden, pred_len):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)
        self.pred_len = pred_len

    def forward(self, enc_hidden, teacher_forcing=None):
        B = enc_hidden.shape[1]
        dec_input = torch.zeros(B, 1, 1).to(enc_hidden.device)
        hidden    = enc_hidden[-1:, :, :]
        outputs   = []
        for t in range(self.pred_len):
            out, hidden = self.gru(dec_input, hidden)
            pred = self.fc(out)                       # (B, 1, 1)
            outputs.append(pred)
            if teacher_forcing is not None:
                dec_input = teacher_forcing[:, t:t+1, :]
            else:
                dec_input = pred
        return torch.cat(outputs, dim=1).squeeze(-1)  # (B, pred_len)

class Seq2SeqWeather(nn.Module):
    def __init__(self, in_features, hidden, pred_len, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = Seq2SeqEncoder(in_features, hidden, num_layers, dropout)
        self.decoder = Seq2SeqDecoder(hidden, pred_len)

    def forward(self, x, teacher_forcing=None):
        _, enc_hidden = self.encoder(x)
        return self.decoder(enc_hidden, teacher_forcing)

# ─────────────────────────────────────────────
# 5. ENTRAÎNEMENT AVEC GRADIENT CLIPPING
# ─────────────────────────────────────────────
def train_recurrent(model, train_loader, val_loader, epochs=30, lr=1e-3,
                    clip=1.0, model_name="Model"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.MSELoss()
    history   = {'train_loss': [], 'val_loss': [], 'perplexity': []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            # ── Gradient Clipping ──────────────
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            count      += len(yb)
        train_loss = total_loss / count

        model.eval()
        total_loss, count = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred       = model(xb)
                loss       = criterion(pred, yb)
                total_loss += loss.item() * len(yb)
                count      += len(yb)
        val_loss = total_loss / count
        perplexity = math.exp(min(val_loss, 100))

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['perplexity'].append(perplexity)

        if epoch % 5 == 0:
            print(f"[{model_name}] Epoch {epoch:3d} | Train MSE: {train_loss:.5f} "
                  f"| Val MSE: {val_loss:.5f} | Perplexité: {perplexity:.2f}")

    return history

# ─────────────────────────────────────────────
# 6. COMPARAISON RNN / LSTM / GRU
# ─────────────────────────────────────────────
def compare_cells(IN_FEATURES, PRED_LEN, train_loader, val_loader, epochs=20):
    results = {}
    for cell in ['RNN', 'GRU', 'LSTM']:
        print(f"\n{'='*50}\n[Compare] Modèle : {cell}\n{'='*50}")
        model = WeatherRNN(IN_FEATURES, hidden=128, pred_len=PRED_LEN,
                           num_layers=2, dropout=0.2, cell=cell)
        hist = train_recurrent(model, train_loader, val_loader, epochs=epochs, model_name=cell)
        results[cell] = hist

    # Visualisation comparative
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {'RNN': '#E84855', 'LSTM': '#2E86AB', 'GRU': '#F18F01'}
    for cell, hist in results.items():
        axes[0].plot(hist['val_loss'],   label=cell, color=colors[cell], linewidth=2)
        axes[1].plot(hist['perplexity'], label=cell, color=colors[cell], linewidth=2)
    axes[0].set_title("Val MSE Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Perplexité");   axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.suptitle("Comparaison RNN / LSTM / GRU", fontweight='bold')
    plt.tight_layout()
    plt.savefig("rnn_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[Figure] Comparaison sauvegardée : rnn_comparison.png")
    return results

# ─────────────────────────────────────────────
# 7. DÉCODAGE : GLOUTON vs BEAM SEARCH
# ─────────────────────────────────────────────
def greedy_decode(model, x):
    """Décodage glouton : prédiction en un seul passage."""
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()

def beam_search_decode(model, x, beam_width=3, pred_len=24):
    """
    Beam Search pour modèles Seq2Seq.
    Conserve les beam_width meilleures hypothèses à chaque pas.
    """
    model.eval()
    B = x.shape[0]
    with torch.no_grad():
        _, enc_hidden = model.encoder(x)

    beams = [(0.0, torch.zeros(B, 1, 1).to(device), enc_hidden, [])]
    for t in range(pred_len):
        candidates = []
        for score, dec_input, hidden, seq in beams:
            with torch.no_grad():
                out, new_hidden = model.decoder.gru(dec_input, hidden[-1:, :, :])
                pred = model.decoder.fc(out)          # (B, 1, 1)
            new_score = score - pred.mean().item()     # log-prob approx
            candidates.append((new_score, pred, new_hidden, seq + [pred]))
        # Garder les k meilleurs
        candidates.sort(key=lambda c: c[0])
        beams = candidates[:beam_width]

    best_seq = beams[0][3]
    return torch.cat(best_seq, dim=1).squeeze(-1).cpu().numpy()

# ─────────────────────────────────────────────
# 8. PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" PARTIE III – LSTM/Seq2Seq : Prévision météo agricole")
    print("="*60)

    # ── Comparaison RNN / LSTM / GRU ──────────
    print("\n[Phase 1] Comparaison des cellules récurrentes...")
    compare_results = compare_cells(IN_FEATURES, PRED_LEN, train_loader, val_loader, epochs=15)

    # ── Entraînement Seq2Seq ──────────────────
    print("\n[Phase 2] Entraînement du modèle Seq2Seq...")
    seq2seq = Seq2SeqWeather(IN_FEATURES, hidden=128, pred_len=PRED_LEN, num_layers=2)
    hist_s2s = train_recurrent(seq2seq, train_loader, val_loader,
                                epochs=20, model_name="Seq2Seq")

    # ── Décodage Glouton vs Beam Search ───────
    print("\n[Phase 3] Comparaison des stratégies de décodage...")
    xb, yb = next(iter(test_loader))
    xb_sample = xb[:4]

    greedy_preds = greedy_decode(seq2seq, xb_sample)
    beam_preds   = beam_search_decode(seq2seq, xb_sample, beam_width=3, pred_len=PRED_LEN)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(2):
        true  = yb[i].cpu().numpy()
        axes[i].plot(true,          label='Réel',         color='black',   linewidth=2)
        axes[i].plot(greedy_preds[i], label='Glouton',    color='#E84855', linewidth=1.5, linestyle='--')
        axes[i].plot(beam_preds[i],   label='Beam k=3',   color='#2E86AB', linewidth=1.5, linestyle='-.')
        axes[i].set_title(f"Prévision météo – Exemple {i+1}", fontweight='bold')
        axes[i].set_xlabel("Heure")
        axes[i].set_ylabel("Température normalisée")
        axes[i].legend(); axes[i].grid(alpha=0.3)
    plt.suptitle("Glouton vs Beam Search – Prévision 24h", fontweight='bold')
    plt.tight_layout()
    plt.savefig("decoding_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

    # ── Sauvegarde ────────────────────────────
    torch.save(seq2seq.state_dict(), "best_seq2seq.pth")

    print("\n✅ Partie III terminée. Fichiers générés :")
    print("   • best_seq2seq.pth")
    print("   • rnn_comparison.png")
    print("   • decoding_comparison.png")
