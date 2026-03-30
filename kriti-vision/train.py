"""
Training Script — MouseVisualCNN
==================================
Trains the mouse visual threat CNN on simulated Unity data.

Usage:
    python train.py                        # train with defaults
    python train.py --epochs 50 --lr 3e-4  # custom hyperparams
    python train.py --data_dir ./my_data   # custom data folder

Folder structure expected:
    data/
        train/
            frames/        ← .png grayscale frames (128×128)
            labels.csv     ← columns: filename, behavior, looming_rate
        val/
            frames/
            labels.csv

labels.csv example:
    filename,        behavior, looming_rate
    frame_0001.png,  escape,   12.4
    frame_0002.png,  freeze,   3.1
    frame_0003.png,  freeze,   2.8
    ...

behavior must be either "escape" or "freeze".
looming_rate is the pixel/frame expansion rate from Unity (used for aux loss).
"""

import os
import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Import our model (must be in same directory)
from mouse_visual_cnn import MouseVisualCNN, ThreatLoss, LoomingAuxHead


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MouseThreatDataset(Dataset):
    """
    Loads frames + labels from a data split folder.

    Each sample returns:
        frame         (1, 128, 128) float tensor
        behavior_label (2,) float tensor  — [escape, freeze] one-hot
        looming_rate   (1,) float tensor  — expansion rate (pixels/frame)
    """

    def __init__(self, split_dir: str):
        self.frames_dir = os.path.join(split_dir, 'frames')
        self.samples = []

        label_path = os.path.join(split_dir, 'labels.csv')
        with open(label_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                behavior = row['behavior'].strip().lower()
                if behavior == 'escape':
                    label = [1.0, 0.0]
                elif behavior == 'freeze':
                    label = [0.0, 1.0]
                else:
                    raise ValueError(f"Unknown behavior: {behavior}. Must be 'escape' or 'freeze'.")
                self.samples.append({
                    'filename':     row['filename'].strip(),
                    'label':        label,
                    'looming_rate': float(row['looming_rate']),
                })

        self.transform = transforms.Compose([
            transforms.Grayscale(),                         # ensure single channel
            transforms.Resize((128, 128)),                  # standardize size
            transforms.ToTensor(),                          # → [0, 1] float tensor
            transforms.Normalize(mean=[0.5], std=[0.5]),    # → [-1, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = os.path.join(self.frames_dir, s['filename'])
        image = Image.open(img_path).convert('L')           # grayscale PIL image
        frame = self.transform(image)                       # (1, 128, 128)

        behavior_label = torch.tensor(s['label'],        dtype=torch.float32)
        looming_rate   = torch.tensor([s['looming_rate']], dtype=torch.float32)

        return frame, behavior_label, looming_rate


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, aux_head, loader, optimizer, criterion, device):
    model.train()
    aux_head.train()
    total_loss, correct, total = 0.0, 0, 0

    for frames, labels, looming in loader:
        frames  = frames.to(device)
        labels  = labels.to(device)
        looming = looming.to(device)

        optimizer.zero_grad()

        # Forward pass through shared backbone
        threat_pred = model(frames)  # (B, 2)

        # Auxiliary: get SC superficial features for looming regression
        # (we re-run partway through — in a real setup you'd cache this)
        sc_sup_feats = model.sc_superficial(model.retina(frames))
        looming_pred = aux_head(sc_sup_feats)  # (B, 1)

        loss = criterion(threat_pred, labels, looming_pred, looming)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * frames.size(0)

        # Accuracy: predicted behavior = argmax of threat vector
        pred_behavior = threat_pred.argmax(dim=1)   # 0=escape, 1=freeze
        true_behavior = labels.argmax(dim=1)
        correct += (pred_behavior == true_behavior).sum().item()
        total   += frames.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for frames, labels, looming in loader:
        frames  = frames.to(device)
        labels  = labels.to(device)

        threat_pred = model(frames)
        loss = criterion(threat_pred, labels)  # no aux loss at eval time

        total_loss += loss.item() * frames.size(0)
        pred_behavior = threat_pred.argmax(dim=1)
        true_behavior = labels.argmax(dim=1)
        correct += (pred_behavior == true_behavior).sum().item()
        total   += frames.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- Data ----
    train_dataset = MouseThreatDataset(os.path.join(args.data_dir, 'train'))
    val_dataset   = MouseThreatDataset(os.path.join(args.data_dir, 'val'))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ---- Model ----
    model    = MouseVisualCNN(dropout=0.3).to(device)
    aux_head = LoomingAuxHead(in_channels=64).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ---- Loss + Optimizer ----
    criterion = ThreatLoss(aux_weight=0.2)
    optimizer = optim.Adam(
        list(model.parameters()) + list(aux_head.parameters()),
        lr=args.lr,
        weight_decay=1e-4  # L2 regularization
    )

    # Reduce LR by 0.5 if val loss doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ---- Training ----
    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>8}")
    print("-" * 52)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, aux_head, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>10.3f} {val_loss:>10.4f} {val_acc:>8.3f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch':      epoch,
                'model':      model.state_dict(),
                'aux_head':   aux_head.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'val_loss':   val_loss,
                'val_acc':    val_acc,
            }, checkpoint_path)
            print(f"         ✓ Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'best_model.pt')}")


# ---------------------------------------------------------------------------
# Inference helper — use this to get threat vectors for the motor team
# ---------------------------------------------------------------------------

def get_threat_vector(model_path: str, frame_path: str) -> dict:
    """
    Load a saved model and run inference on a single frame.

    Returns a dict with escape_signal and freeze_signal,
    ready to pass to the motor team's dPAG module.

    Example:
        result = get_threat_vector('checkpoints/best_model.pt', 'frame.png')
        print(result)
        # {'escape_signal': 0.82, 'freeze_signal': 0.14, 'predicted_behavior': 'escape'}
    """
    device = torch.device('cpu')

    model = MouseVisualCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    image = Image.open(frame_path).convert('L')
    frame = transform(image).unsqueeze(0)  # (1, 1, 128, 128)

    with torch.no_grad():
        threat = model(frame)  # (1, 2)

    escape_signal = threat[0, 0].item()
    freeze_signal = threat[0, 1].item()
    predicted     = 'escape' if escape_signal > freeze_signal else 'freeze'

    return {
        'escape_signal':      round(escape_signal, 4),
        'freeze_signal':      round(freeze_signal, 4),
        'predicted_behavior': predicted,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MouseVisualCNN')
    parser.add_argument('--data_dir',   type=str,   default='./data',         help='Root data directory')
    parser.add_argument('--save_dir',   type=str,   default='./checkpoints',  help='Where to save checkpoints')
    parser.add_argument('--epochs',     type=int,   default=30,               help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,   default=32,               help='Batch size')
    parser.add_argument('--lr',         type=float, default=1e-3,             help='Learning rate')
    args = parser.parse_args()
    main(args)
