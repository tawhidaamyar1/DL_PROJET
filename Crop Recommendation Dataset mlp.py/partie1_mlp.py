"""
=============================================================
PROJET DEEP LEARNING – EMSI 2025-2026
Partie I : MLP sur données tabulaires agricoles
Tâche    : Prédiction de la qualité du sol / recommandation de culture
Dataset  : Crop Recommendation Dataset (Kaggle) – 2 200 exemples
=============================================================
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)

import warnings, os, random
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. REPRODUCTIBILITÉ
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─────────────────────────────────────────────
# 2. DEVICE (CPU ou GPU)
# ─────────────────────────────────────────────
def get_device():
    """Sélectionne automatiquement le meilleur device disponible."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[Device] GPU détecté : {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Utilisation du CPU")
    return device

device = get_device()

# ─────────────────────────────────────────────
# 3. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
def load_data():
    """
    Charge le Crop Recommendation Dataset.
    Colonnes : N, P, K, temperature, humidity, ph, rainfall, label
    2200 exemples, 22 classes de cultures.
    Téléchargement : https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
    """
    try:
        df = pd.read_csv("Crop_recommendation.csv")
        print(f"[Data] Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    except FileNotFoundError:
        # Génération de données synthétiques si le fichier est absent
        print("[Data] Fichier absent – génération de données synthétiques (demo)")
        np.random.seed(SEED)
        n = 2200
        df = pd.DataFrame({
            'N':           np.random.uniform(0, 140, n),
            'P':           np.random.uniform(5, 145, n),
            'K':           np.random.uniform(5, 205, n),
            'temperature': np.random.uniform(8, 44, n),
            'humidity':    np.random.uniform(14, 100, n),
            'ph':          np.random.uniform(3.5, 9.5, n),
            'rainfall':    np.random.uniform(20, 300, n),
            'label':       np.random.choice(
                ['rice','maize','chickpea','kidneybeans','pigeonpeas',
                 'mothbeans','mungbean','blackgram','lentil','pomegranate',
                 'banana','mango','grapes','watermelon','muskmelon',
                 'apple','orange','papaya','coconut','cotton','jute','coffee'],
                n
            )
        })
    return df

df = load_data()
print(df.head())

# ─────────────────────────────────────────────
# 4. PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────
# 4.1 Encodage de la cible
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
num_classes = len(le.classes_)
print(f"[Data] {num_classes} classes : {list(le.classes_)}")

# 4.2 Séparation features / cible
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[FEATURES].values.astype(np.float32)
y = df['label_enc'].values.astype(np.int64)

# 4.3 Division train / val / test (70 / 15 / 15)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
X_val,   X_test, y_val,  y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

# 4.4 Normalisation (fit sur train uniquement)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# 4.5 Conversion en tenseurs
def to_tensor(X, y, device):
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.long).to(device)
    )

train_ds = to_tensor(X_train, y_train, device)
val_ds   = to_tensor(X_val,   y_val,   device)
test_ds  = to_tensor(X_test,  y_test,  device)

# 4.6 DataLoaders
BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ─────────────────────────────────────────────
# 5. MODÈLES MLP
# ─────────────────────────────────────────────

# ── VERSION A : nn.Sequential ────────────────
def build_sequential_mlp(in_features, hidden, num_classes, dropout=0.3):
    """
    MLP construit avec nn.Sequential.
    Architecture : Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2),
        nn.BatchNorm1d(hidden // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden // 2, num_classes)
    )

# ── VERSION B : Classe personnalisée ─────────
class MLP(nn.Module):
    """
    MLP défini comme classe nn.Module.
    Plus flexible : permet branchements, régularisation sur mesure, etc.
    """
    def __init__(self, in_features, hidden, num_classes, dropout=0.3):
        super().__init__()
        self.fc1   = nn.Linear(in_features, hidden)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.fc2   = nn.Linear(hidden, hidden // 2)
        self.bn2   = nn.BatchNorm1d(hidden // 2)
        self.fc3   = nn.Linear(hidden // 2, num_classes)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ─────────────────────────────────────────────
# 6. INITIALISATION DES PARAMÈTRES
# ─────────────────────────────────────────────
def init_gaussian(module):
    """Initialisation gaussienne : poids ~ N(0, 0.01)"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

def init_constant(module):
    """Initialisation constante : tous les poids = 1 (mauvaise pratique, demo)"""
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 1.0)
        nn.init.zeros_(module.bias)

def init_xavier(module):
    """Initialisation Xavier : stabilise la variance dans les couches profondes"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)

# ─────────────────────────────────────────────
# 7. BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=10):
    """
    Entraîne le modèle avec early stopping.
    Retourne l'historique des métriques (loss et accuracy).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        # ── Entraînement ──────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)
        train_loss = total_loss / total
        train_acc  = correct   / total

        # ── Validation ────────────────────────
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits     = model(xb)
                loss       = criterion(logits, yb)
                total_loss += loss.item() * len(yb)
                correct    += (logits.argmax(1) == yb).sum().item()
                total      += len(yb)
        val_loss = total_loss / total
        val_acc  = correct   / total

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} "
                  f"| Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}")

        # ── Early Stopping ────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Early Stopping] Arrêt à l'époque {epoch}")
                break

    model.load_state_dict(best_state)
    return history

# ─────────────────────────────────────────────
# 8. ÉVALUATION
# ─────────────────────────────────────────────
def evaluate(model, loader, class_names):
    """Calcule les métriques : accuracy, precision, recall, F1, matrice de confusion."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            probs  = F.softmax(logits, dim=1)
            preds  = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1   = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"\n{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"{'='*50}\n")

    return all_labels, all_preds, all_probs

# ─────────────────────────────────────────────
# 9. VISUALISATIONS
# ─────────────────────────────────────────────
def plot_history(history, title="Courbes d'apprentissage"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train', color='#2E86AB')
    axes[0].plot(history['val_loss'],   label='Val',   color='#E84855')
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Époque")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train', color='#2E86AB')
    axes[1].plot(history['val_acc'],   label='Val',   color='#E84855')
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Époque")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"learning_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[Figure] Courbes sauvegardées : learning_curves.png")

def plot_confusion(labels, preds, class_names, title="Matrice de confusion"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap='Blues', xticks_rotation=45)
    ax.set_title(title, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[Figure] Matrice sauvegardée : confusion_matrix.png")

# ─────────────────────────────────────────────
# 10. SAUVEGARDE ET RECHARGEMENT
# ─────────────────────────────────────────────
def save_model(model, path="best_mlp.pth"):
    torch.save(model.state_dict(), path)
    print(f"[Save] Modèle sauvegardé : {path}")

def load_model(model_class, path, **kwargs):
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[Load] Modèle rechargé depuis : {path}")
    return model

# ─────────────────────────────────────────────
# 11. COMPARAISON DES INITIALISATIONS
# ─────────────────────────────────────────────
def compare_inits(in_features, num_classes, train_loader, val_loader, epochs=30):
    """Compare 3 stratégies d'initialisation sur les courbes de perte."""
    results = {}
    inits   = {'Xavier': init_xavier, 'Gaussienne': init_gaussian, 'Constante': init_constant}

    for name, init_fn in inits.items():
        print(f"\n[Init] Stratégie : {name}")
        model = MLP(in_features, hidden=256, num_classes=num_classes)
        model.apply(init_fn)
        hist = train_model(model, train_loader, val_loader, epochs=epochs, patience=epochs)
        results[name] = hist['val_loss']

    plt.figure(figsize=(8, 4))
    colors = {'Xavier': '#2E86AB', 'Gaussienne': '#A23B72', 'Constante': '#F18F01'}
    for name, losses in results.items():
        plt.plot(losses, label=name, color=colors[name], linewidth=2)
    plt.title("Comparaison des initialisations – Val Loss", fontweight='bold')
    plt.xlabel("Époque")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("init_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[Figure] Comparaison sauvegardée : init_comparison.png")

# ─────────────────────────────────────────────
# 12. PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────
if __name__ == "__main__":
    IN_FEATURES = len(FEATURES)
    HIDDEN      = 256
    EPOCHS      = 60

    print("\n" + "="*60)
    print(" PARTIE I – MLP : Recommandation de culture agricole")
    print("="*60)

    # ── Inspection des paramètres ─────────────
    model = MLP(IN_FEATURES, HIDDEN, num_classes)
    model.to(device)
    model.apply(init_xavier)
    print(f"\n[Model] Paramètres entraînables : {model.count_parameters():,}")
    print("\n[Model] named_parameters() :")
    for name, param in model.named_parameters():
        print(f"  {name:30s}  shape={str(param.shape):20s}  grad={param.requires_grad}")

    # ── Entraînement ──────────────────────────
    print("\n[Train] Démarrage de l'entraînement...")
    history = train_model(model, train_loader, val_loader, epochs=EPOCHS)

    # ── Évaluation finale ─────────────────────
    print("\n[Eval] Résultats sur le jeu de test :")
    labels, preds, probs = evaluate(model, test_loader, le.classes_)

    # ── Visualisations ────────────────────────
    plot_history(history, title="MLP – Détection de maladies agricoles")
    plot_confusion(labels, preds, le.classes_)

    # ── Sauvegarde du meilleur modèle ─────────
    save_model(model, "best_mlp.pth")

    # ── Comparaison des initialisations ───────
    print("\n[Init] Comparaison des 3 initialisations...")
    compare_inits(IN_FEATURES, num_classes, train_loader, val_loader, epochs=30)

    # ── MLP Sequential (comparaison) ──────────
    print("\n[Seq] Entraînement du MLP Sequential...")
    model_seq = build_sequential_mlp(IN_FEATURES, HIDDEN, num_classes).to(device)
    model_seq.apply(init_xavier)
    hist_seq = train_model(model_seq, train_loader, val_loader, epochs=EPOCHS)
    print("\n[Seq] Évaluation du MLP Sequential :")
    evaluate(model_seq, test_loader, le.classes_)

    print("\n✅ Partie I terminée. Fichiers générés :")
    print("   • best_mlp.pth")
    print("   • learning_curves.png")
    print("   • confusion_matrix.png")
    print("   • init_comparison.png")
