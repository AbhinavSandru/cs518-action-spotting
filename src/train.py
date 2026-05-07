import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import WindowDataset
from src.model import ActionSpotter


def compute_class_weights(data_dir, num_classes, device):
    """Compute pos_weight for BCEWithLogitsLoss from actual annotation files.
    pos_weight[c] = num_negative_frames / num_positive_frames for class c.
    """
    import json
    from SoccerNet.Downloader import getListGames
    from src.dataset import LABEL_TO_IDX

    print("Computing class weights from annotations...")
    pos_counts = np.zeros(num_classes, dtype=np.float64)
    total_frames = 0

    game_list = getListGames(split="train")
    for game_path in game_list:
        label_path = os.path.join(data_dir, game_path, "Labels-v2.json")
        feat1_path = os.path.join(data_dir, game_path, "1_ResNET_TF2.npy")
        feat2_path = os.path.join(data_dir, game_path, "2_ResNET_TF2.npy")
        if not all(os.path.exists(p) for p in [label_path, feat1_path, feat2_path]):
            continue
        f1 = np.load(feat1_path, mmap_mode="r")
        f2 = np.load(feat2_path, mmap_mode="r")
        total_frames += f1.shape[0] + f2.shape[0]
        with open(label_path) as f:
            annotation = json.load(f)
        for ann in annotation.get("annotations", []):
            label = ann.get("label", "")
            if label in LABEL_TO_IDX:
                pos_counts[LABEL_TO_IDX[label]] += 1

    # Square-root inverse frequency — moderates extreme imbalance without flattening differences
    neg_counts = total_frames - pos_counts
    raw_weights = neg_counts / (pos_counts + 1e-6)
    weights = np.sqrt(raw_weights)               # sqrt dampens extreme values
    weights = np.clip(weights, 1.0, 50.0)        # soft cap
    print(f"  Pos counts: min={pos_counts.min():.0f}, max={pos_counts.max():.0f}")
    print(f"  Pos weights: min={weights.min():.1f}, max={weights.max():.1f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for features, labels in tqdm(loader, desc="Training", leave=False):
        features = features.to(device)   # (B, T, 512)
        labels = labels.to(device)       # (B, T, 17)

        optimizer.zero_grad()
        logits = model(features)         # (B, T, 17)

        # Flatten time dimension for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1, labels.size(-1))

        loss = criterion(logits_flat, labels_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Evaluating", leave=False):
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1, labels.size(-1))

            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()

    return total_loss / len(loader)


def train(
    data_dir="data/soccernet",
    checkpoint_dir="checkpoints",
    num_epochs=20,
    batch_size=32,
    lr=1e-4,
    window_size=60,
    stride=30,
    d_model=256,
    nhead=4,
    num_layers=4,
    dropout=0.3,
):
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Datasets and loaders
    windows_dir = "/Users/abhinavsandru/CS518_Project"
    print("Loading training data...")
    train_dataset = WindowDataset(windows_dir, split="train")
    print("Loading validation data...")
    val_dataset = WindowDataset(windows_dir, split="valid")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train windows: {len(train_dataset)} | Val windows: {len(val_dataset)}")

    # Model
    model = ActionSpotter(
        input_dim=2048,
        num_classes=17,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Focal loss with class weights — better than BCE for sparse action spotting
    class_weights = compute_class_weights(data_dir=data_dir, num_classes=17, device=device)

    def criterion(logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=class_weights, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** 2   # gamma=2, standard focal loss
        return (focal_weight * bce).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 5

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save per-epoch checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:02d}.pt"))

        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": {"input_dim": 2048, "num_classes": 17, "d_model": d_model, "nhead": nhead, "num_layers": num_layers, "dropout": dropout}
            }, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"  Saved best model (epoch {epoch}).")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stop_patience})")
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Curves")
    plt.savefig("results/loss_curve.png")
    print("\nLoss curve saved to results/loss_curve.png")

    return model


if __name__ == "__main__":
    train()
