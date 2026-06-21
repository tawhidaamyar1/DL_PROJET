"""
=============================================================
PROJET DEEP LEARNING – EMSI 2025-2026
Partie II : CNN – Classification de chiffres manuscrits
Dataset   : MNIST – 70 000 images 28x28, 10 classes (0-9)
            Téléchargement AUTOMATIQUE via torchvision
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")

# ─────────────────────────────────────────────
# 1. CORRÉLATION CROISÉE 2D MANUELLE
# ─────────────────────────────────────────────
def corr2d(X, K):
    """
    Corrélation croisée 2D implémentée manuellement.
    X : tenseur 2D (H, W)
    K : noyau 2D (kh, kw)
    Taille sortie : (H - kh + 1) x (W - kw + 1)
    """
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

def pool2d(X, pool_size, mode='max'):
    """Max-pooling et Average-pooling manuels."""
    ph, pw = pool_size
    Y = torch.zeros(X.shape[0] - ph + 1, X.shape[1] - pw + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            region = X[i:i+ph, j:j+pw]
            Y[i, j] = region.max() if mode == 'max' else region.mean()
    return Y

def demo_operations():
    """Démonstration numérique : corrélation croisée + pooling."""
    print("\n[Demo] Corrélation croisée 2D manuelle :")
    X = torch.tensor([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]])
    K = torch.tensor([[0.,1.],[2.,3.]])
    Y = corr2d(X, K)
    print(f"  X =\n{X.numpy()}")
    print(f"  K =\n{K.numpy()}")
    print(f"  Y =\n{Y.numpy()}")

    print("\n[Demo] Max-pooling 2x2 :")
    print(f"  Ymax =\n{pool2d(X, (2,2), 'max').numpy()}")
    print(f"  Yavg =\n{pool2d(X, (2,2), 'avg').numpy()}")

    print("\n[Demo] Vérification taille de sortie :")
    print(f"  Entrée 8x8, noyau 3x3, padding=1, stride=1 → {(8-3+2)//1+1}x{(8-3+2)//1+1}")
    print(f"  Entrée 8x8, noyau 3x3, padding=1, stride=2 → {(8-3+2)//2+1}x{(8-3+2)//2+1}")

demo_operations()

# ─────────────────────────────────────────────
# 2. CHARGEMENT MNIST (automatique)
# ─────────────────────────────────────────────
print("\n[Data] Téléchargement automatique de MNIST...")

transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Téléchargement automatique dans ./data/
train_full = datasets.MNIST('./data', train=True,  download=True, transform=transform_train)
test_ds    = datasets.MNIST('./data', train=False, download=True, transform=transform_test)

# Séparation train / val (85% / 15%)
n_val   = int(0.15 * len(train_full))
n_train = len(train_full) - n_val
train_ds, val_ds = random_split(train_full, [n_train, n_val],
                                generator=torch.Generator().manual_seed(SEED))

BATCH_SIZE   = 64
NUM_CLASSES  = 10
CLASS_NAMES  = [str(i) for i in range(10)]

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# Visualisation de quelques images
def show_samples(loader, n=10):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    for i in range(n):
        axes[i].imshow(images[i].squeeze().numpy(), cmap='gray')
        axes[i].set_title(f"Classe : {labels[i].item()}", fontsize=9)
        axes[i].axis('off')
    plt.suptitle("Exemples MNIST", fontweight='bold')
    plt.tight_layout()
    plt.savefig("mnist_samples.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[Figure] Exemples sauvegardés : mnist_samples.png")

show_samples(train_loader)

# ─────────────────────────────────────────────
# 3. ARCHITECTURE CNN (inspirée LeNet-5)
# ─────────────────────────────────────────────
class LeNetMNIST(nn.Module):
    """
    CNN inspiré de LeNet-5 pour MNIST.
    Architecture :
      Conv1(6, 5x5, p=2) → BN → Sigmoid → AvgPool(2x2)
      Conv2(16, 5x5)      → BN → Sigmoid → AvgPool(2x2)
      Conv 1x1 (16→10)    → réduction de canaux
      FC(120) → FC(84) → FC(10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Bloc 1 : Conv → BN → Sigmoid → Pool
        self.conv1 = nn.Conv2d(1, 6,  kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Bloc 2 : Conv → BN → Sigmoid → Pool
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Convolution 1×1 (mélange de canaux)
        self.conv1x1 = nn.Conv2d(16, 16, kernel_size=1)
        # Couches fully-connected
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(16 * 5 * 5, 120)
        self.fc2     = nn.Linear(120, 84)
        self.fc3     = nn.Linear(84, num_classes)
        self.drop    = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool1(torch.sigmoid(self.bn1(self.conv1(x))))  # 28→14
        x = self.pool2(torch.sigmoid(self.bn2(self.conv2(x))))  # 10→5
        x = F.relu(self.conv1x1(x))
        x = self.flatten(x)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)

    def get_feature_maps(self, x):
        """Retourne les feature maps pour visualisation."""
        maps = {}
        x = torch.sigmoid(self.bn1(self.conv1(x))); maps['conv1'] = x.clone()
        x = self.pool1(x)
        x = torch.sigmoid(self.bn2(self.conv2(x))); maps['conv2'] = x.clone()
        return maps

# ─────────────────────────────────────────────
# 4. VISUALISATION DES FEATURE MAPS
# ─────────────────────────────────────────────
def visualize_feature_maps(model, loader, device, n_filters=6):
    """Affiche les feature maps après conv1 et conv2."""
    model.eval()
    images, labels = next(iter(loader))
    img = images[0:1].to(device)

    with torch.no_grad():
        maps = model.get_feature_maps(img)

    for layer_name, fmap in maps.items():
        fmap = fmap[0].cpu()
        n = min(n_filters, fmap.shape[0])
        fig, axes = plt.subplots(1, n+1, figsize=(2*(n+1), 2.5))
        # Image originale
        axes[0].imshow(images[0].squeeze().numpy(), cmap='gray')
        axes[0].set_title("Original", fontsize=9)
        axes[0].axis('off')
        # Feature maps
        for i in range(n):
            axes[i+1].imshow(fmap[i].numpy(), cmap='viridis')
            axes[i+1].set_title(f"Filtre {i+1}", fontsize=9)
            axes[i+1].axis('off')
        fig.suptitle(f"Feature maps – {layer_name}", fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"feature_maps_{layer_name}.png", dpi=150, bbox_inches='tight')
        plt.show()
        print(f"[Figure] Feature maps sauvegardées : feature_maps_{layer_name}.png")

# ─────────────────────────────────────────────
# 5. ENTRAÎNEMENT
# ─────────────────────────────────────────────
def train_cnn(model, train_loader, val_loader, epochs=15, lr=1e-3, name="CNN"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    history   = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += len(yb)
        train_loss = total_loss / total
        train_acc  = correct   / total

        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out        = model(xb)
                loss       = criterion(out, yb)
                total_loss += loss.item() * len(yb)
                correct    += (out.argmax(1) == yb).sum().item()
                total      += len(yb)
        val_loss = total_loss / total
        val_acc  = correct   / total

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"[{name}] Epoch {epoch:2d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

    return history

def plot_history(history, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train', color='#2E86AB', linewidth=2)
    axes[0].plot(history['val_loss'],   label='Val',   color='#E84855', linewidth=2)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['train_acc'], label='Train', color='#2E86AB', linewidth=2)
    axes[1].plot(history['val_acc'],   label='Val',   color='#E84855', linewidth=2)
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig("cnn_learning_curves.png", dpi=150, bbox_inches='tight')
    plt.show()

# ─────────────────────────────────────────────
# 6. ÉTUDE D'ABLATION
# ─────────────────────────────────────────────
def ablation_study(train_loader, val_loader, epochs=8):
    """Compare MLP simple vs CNN sur MNIST."""
    # MLP simple (sans convolution)
    mlp = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128),   nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 10)
    ).to(device)

    # CNN LeNet
    cnn = LeNetMNIST().to(device)

    print("\n[Ablation] Entraînement MLP simple...")
    hist_mlp = train_cnn(mlp, train_loader, val_loader, epochs=epochs, name="MLP")
    print("\n[Ablation] Entraînement CNN LeNet...")
    hist_cnn = train_cnn(cnn, train_loader, val_loader, epochs=epochs, name="CNN")

    # Comparaison visuelle
    plt.figure(figsize=(8, 4))
    plt.plot(hist_mlp['val_acc'], label='MLP simple', color='#E84855', linewidth=2, linestyle='--')
    plt.plot(hist_cnn['val_acc'], label='CNN LeNet',  color='#2E86AB', linewidth=2)
    plt.title("MLP vs CNN – Val Accuracy sur MNIST", fontweight='bold')
    plt.xlabel("Époque"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("mlp_vs_cnn.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[Figure] Comparaison sauvegardée : mlp_vs_cnn.png")

    print(f"\n[Ablation] Résultats finaux :")
    print(f"  MLP simple → Val Acc : {hist_mlp['val_acc'][-1]:.3f}")
    print(f"  CNN LeNet  → Val Acc : {hist_cnn['val_acc'][-1]:.3f}")

# ─────────────────────────────────────────────
# 7. ÉVALUATION FINALE
# ─────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    print("\n[Eval] Rapport de classification :")
    print(classification_report(all_labels, all_preds,
                                 target_names=CLASS_NAMES, digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, colorbar=True, cmap='Blues')
    ax.set_title("Matrice de confusion – MNIST", fontweight='bold')
    plt.tight_layout()
    plt.savefig("confusion_matrix_cnn.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[Figure] Matrice sauvegardée : confusion_matrix_cnn.png")

# ─────────────────────────────────────────────
# 8. PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" PARTIE II – CNN : Classification MNIST (chiffres 0-9)")
    print("="*60)

    model = LeNetMNIST(NUM_CLASSES).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] Paramètres entraînables : {total:,}")

    # Inspection des dimensions couche par couche
    print("\n[Model] Évolution des dimensions :")
    x = torch.randn(1, 1, 28, 28).to(device)
    for name, layer in model.named_children():
        try:
            x = layer(x)
            print(f"  {name:10s} → {tuple(x.shape)}")
        except:
            pass

    # Entraînement
    print("\n[Train] Démarrage entraînement CNN LeNet...")
    history = train_cnn(model, train_loader, val_loader, epochs=15, name="LeNet")
    plot_history(history, "CNN LeNet – MNIST")

    # Feature maps
    print("\n[Viz] Visualisation des feature maps...")
    visualize_feature_maps(model, test_loader, device)

    # Évaluation finale
    print("\n[Eval] Évaluation sur jeu de test...")
    evaluate(model, test_loader)

    # Ablation : MLP vs CNN
    print("\n[Ablation] Comparaison MLP vs CNN...")
    ablation_study(train_loader, val_loader, epochs=8)

    # Sauvegarde
    torch.save(model.state_dict(), "best_cnn.pth")

    print("\n✅ Partie II terminée ! Fichiers générés :")
    print("   • best_cnn.pth")
    print("   • mnist_samples.png")
    print("   • cnn_learning_curves.png")
    print("   • feature_maps_conv1.png / conv2.png")
    print("   • confusion_matrix_cnn.png")
    print("   • mlp_vs_cnn.png")
