"""
train.py
====================================
Trains MouseVisualMotorModel on T=5 frame trial sequences.

Usage:
    python generate_dummy_data.py   # first time only
    python train.py
    python train.py --epochs 50 --lr 3e-4
"""

import os, csv, argparse
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from mouse_visual_cnn import MouseVisualMotorModel, LoomingAuxHead, FullModelLoss

T = 5


class MouseThreatDataset(Dataset):
    def __init__(self, split_dir):
        self.frames_dir = os.path.join(split_dir, 'frames')
        self.samples = []
        with open(os.path.join(split_dir, 'labels.csv'), newline='') as f:
            for row in csv.DictReader(f):
                b = row['behavior'].strip().lower()
                self.samples.append({
                    'trial_id':     row['trial_id'].strip(),
                    'label':        [1.0, 0.0] if b == 'escape' else [0.0, 1.0],
                    'looming_rate': float(row['looming_rate']),
                })
        self.tf = transforms.Compose([
            transforms.Grayscale(), transforms.Resize((128, 128)),
            transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = torch.cat([
            self.tf(Image.open(os.path.join(self.frames_dir, f"{s['trial_id']}_t{t}.png")).convert('L'))
            for t in range(T)
        ], dim=0)
        return (
            frames,
            torch.tensor(s['label'],          dtype=torch.float32),
            torch.tensor([s['looming_rate']], dtype=torch.float32),
        )


def train_epoch(model, aux_head, loader, opt, crit, device):
    model.train(); aux_head.train()
    total_loss, correct, n = 0., 0, 0
    for frames, labels, looming in loader:
        frames, labels, looming = frames.to(device), labels.to(device), looming.to(device)
        opt.zero_grad()
        out       = model(frames)
        loom_pred = aux_head(out['sc_sup_feats'])
        loss = crit(out, labels, loom_pred, looming)
        loss.backward(); opt.step()
        total_loss += loss.item() * frames.size(0)
        correct    += (out['freeze_escape_comp'].argmax(1) == labels.argmax(1)).sum().item()
        n          += frames.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    total_loss, correct, n = 0., 0, 0
    for frames, labels, _ in loader:
        frames, labels = frames.to(device), labels.to(device)
        out  = model(frames)
        loss = crit(out, labels)
        total_loss += loss.item() * frames.size(0)
        correct    += (out['freeze_escape_comp'].argmax(1) == labels.argmax(1)).sum().item()
        n          += frames.size(0)
    return total_loss / n, correct / n


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = MouseThreatDataset(os.path.join(args.data_dir, 'train'))
    val_ds   = MouseThreatDataset(os.path.join(args.data_dir, 'val'))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    model    = MouseVisualMotorModel(T=T).to(device)
    aux_head = LoomingAuxHead().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    crit = FullModelLoss(aux_weight=0.2, velocity_weight=0.1)
    opt  = optim.Adam(
        list(model.parameters()) + list(aux_head.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    os.makedirs(args.save_dir, exist_ok=True)
    best = float('inf')

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>8}")
    print("-" * 52)

    for epoch in range(1, args.epochs + 1):
        tl, ta = train_epoch(model, aux_head, train_dl, opt, crit, device)
        vl, va = evaluate(model, val_dl, crit, device)
        sched.step(vl)
        print(f"{epoch:>6} {tl:>12.4f} {ta:>10.3f} {vl:>10.4f} {va:>8.3f}")
        if vl < best:
            best = vl
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'aux': aux_head.state_dict(), 'val_loss': vl},
                       os.path.join(args.save_dir, 'best_model.pt'))
            print(f"         Saved (val_loss={vl:.4f})")

    print(f"\nDone. Best val loss: {best:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='./data')
    p.add_argument('--save_dir',   default='./checkpoints')
    p.add_argument('--epochs',     type=int,   default=30)
    p.add_argument('--batch_size', type=int,   default=16)
    p.add_argument('--lr',         type=float, default=1e-3)
    main(p.parse_args())
