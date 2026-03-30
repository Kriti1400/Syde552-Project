"""
generate_dummy_data.py
======================
Creates fake training data so you can run train.py immediately
without needing real Unity frames.

Generates simple synthetic "looming" images:
  - escape samples: large, high-contrast expanding circle (fast looming)
  - freeze samples: small, low-contrast circle (slow/ambiguous looming)

Run this once before train.py:
    python generate_dummy_data.py

It will create:
    data/
        train/   (800 samples)
        val/     (200 samples)
"""

import os
import csv
import random
import math
from PIL import Image, ImageDraw

random.seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_SAMPLES = 800
VAL_SAMPLES   = 200
IMG_SIZE      = 128
DATA_DIR      = './data'


def make_looming_frame(behavior: str) -> tuple:
    """
    Draws a synthetic looming disk on a gray background.

    escape: large radius (40-60px), high contrast, off-center
    freeze: small radius (10-25px), low contrast, overhead (near center)

    Returns (PIL Image, looming_rate)
    """
    img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=128)
    draw = ImageDraw.Draw(img)

    if behavior == 'escape':
        radius       = random.randint(40, 60)
        fill_color   = random.randint(0, 40)      # dark = high contrast
        looming_rate = random.uniform(8.0, 18.0)  # fast expansion
        cx = random.randint(40, 88)
        cy = random.randint(40, 88)
    else:  # freeze
        radius       = random.randint(10, 25)
        fill_color   = random.randint(80, 120)    # near-gray = low contrast
        looming_rate = random.uniform(1.0, 5.0)   # slow expansion
        cx = random.randint(50, 78)               # more centered = overhead
        cy = random.randint(50, 78)

    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=fill_color
    )

    # Add mild noise
    pixels = img.load()
    for x in range(IMG_SIZE):
        for y in range(IMG_SIZE):
            noise = random.randint(-10, 10)
            pixels[x, y] = max(0, min(255, pixels[x, y] + noise))

    return img, looming_rate


def generate_split(split_name: str, n_samples: int):
    split_dir  = os.path.join(DATA_DIR, split_name)
    frames_dir = os.path.join(split_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    label_path = os.path.join(split_dir, 'labels.csv')
    rows = []

    for i in range(n_samples):
        behavior = 'escape' if i % 2 == 0 else 'freeze'  # balanced split
        img, looming_rate = make_looming_frame(behavior)

        filename = f'frame_{i:05d}.png'
        img.save(os.path.join(frames_dir, filename))
        rows.append({
            'filename':     filename,
            'behavior':     behavior,
            'looming_rate': round(looming_rate, 2),
        })

    with open(label_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'behavior', 'looming_rate'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {split_name}: {n_samples} samples → {split_dir}")


if __name__ == '__main__':
    print("Generating dummy data...")
    generate_split('train', TRAIN_SAMPLES)
    generate_split('val',   VAL_SAMPLES)
    print(f"\nDone. Now run:  python train.py")
