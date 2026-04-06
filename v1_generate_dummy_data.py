"""
generate_dummy_data.py
=============================================
Generates T=5 frame trial sequences for the full visual-motor model.

escape: expanding dark disk overhead (high looming_rate)
freeze: small laterally sweeping disk (low looming_rate)
"""

import os, csv, random
from PIL import Image, ImageDraw

random.seed(42)
TRAIN, VAL = 800, 200
SZ, T = 128, 5
DATA_DIR = './data'


def make_trial(behavior):
    frames, bg = [], 128
    if behavior == 'escape':
        cx, cy = random.randint(44, 84), random.randint(44, 84)
        r0, rate = random.randint(8, 18), random.uniform(5.0, 16.0)
        fill = random.randint(20, 70)
        for t in range(T):
            r = int(r0 + rate * t)
            img = Image.new('L', (SZ, SZ), bg)
            ImageDraw.Draw(img).ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill)
            _noise(img); frames.append(img)
    else:
        sx, cy = random.randint(10, 30), random.randint(44, 84)
        r, speed = random.randint(8, 18), random.uniform(4.0, 12.0)
        fill, rate = random.randint(60, 110), random.uniform(0.5, 3.0)
        for t in range(T):
            cx = int(sx + speed * t)
            ri = int(r + rate * t)
            img = Image.new('L', (SZ, SZ), bg)
            ImageDraw.Draw(img).ellipse([cx-ri, cy-ri, cx+ri, cy+ri], fill=fill)
            _noise(img); frames.append(img)
    return frames, rate


def _noise(img, m=8):
    px = img.load()
    for x in range(SZ):
        for y in range(SZ):
            px[x, y] = max(0, min(255, px[x, y] + random.randint(-m, m)))


def generate(split, n):
    d = os.path.join(DATA_DIR, split, 'frames')
    os.makedirs(d, exist_ok=True)
    rows = []
    for i in range(n):
        b = 'escape' if i % 2 == 0 else 'freeze'
        fs, lr = make_trial(b)
        for t, f in enumerate(fs):
            f.save(os.path.join(d, f'trial_{i:05d}_t{t}.png'))
        rows.append({'trial_id': f'trial_{i:05d}', 'behavior': b, 'looming_rate': round(lr, 2)})
    with open(os.path.join(DATA_DIR, split, 'labels.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['trial_id', 'behavior', 'looming_rate'])
        w.writeheader(); w.writerows(rows)
    print(f"  {split}: {n} trials × {T} frames")


if __name__ == '__main__':
    print(f"Generating data (T={T})...")
    generate('train', TRAIN); generate('val', VAL)
    print("Done. Run: python train.py")
