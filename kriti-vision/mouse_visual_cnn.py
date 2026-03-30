"""
Mouse Visual Threat CNN
========================
Models the retina → SC superficial → SC deep (PV+ fork) pathway.

INPUT:  Grayscale frame, shape (B, 1, 128, 128)
        - Simulates wide-field mouse retinal input from Unity camera
        - Single channel (mice are largely dichromatic; luminance dominates threat detection)

OUTPUT: Threat vector, shape (B, 2)
        - output[:, 0] = escape signal  (SC → PBGN stream)
        - output[:, 1] = freeze signal  (SC → LPTN stream)
        - Both are sigmoid-activated scalars in [0, 1]
        - This vector is the handoff to the motor team's dPAG module

Biological mapping:
  RetinalModule       → Retinal ganglion cells (DoG center-surround filters)
  SCSuperficialModule → SC superficial layers (retinotopic spatial processing)
  SCDeepModule        → SC deep PV+ neurons (threat integration)
  PBGNHead            → SC → PBGN projection (escape-relevant stream)
  LPTNHead            → SC → LPTN projection (freeze-relevant stream)

References:
  Shang et al. (2018) Nature Communications — SC PV+ divergent pathways
  Sit & Bhatt (2023) Nature Communications — SC retinotopic tiling
  Martersteck et al. (2017) Cell Reports — RGC type diversity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Utility: Difference-of-Gaussians kernel (biologically motivated retinal filter)
# ---------------------------------------------------------------------------

def dog_kernel(size: int, sigma_center: float, sigma_surround: float) -> torch.Tensor:
    """
    Creates a Difference-of-Gaussians (DoG) convolutional kernel.

    Approximates the center-surround receptive fields of retinal ganglion cells.
    - Center Gaussian = excitatory (On-center) or inhibitory (Off-center) region
    - Surround Gaussian = the opposite polarity annulus

    Args:
        size:            Kernel size (odd integer, e.g. 7)
        sigma_center:    Std of center Gaussian (smaller = tighter center)
        sigma_surround:  Std of surround Gaussian (larger = wider surround)

    Returns:
        Normalized DoG kernel, shape (size, size)
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    r2 = x**2 + y**2

    center   = torch.exp(-r2 / (2 * sigma_center**2))
    surround = torch.exp(-r2 / (2 * sigma_surround**2))

    # On-center DoG: positive center, negative surround
    dog = center / (2 * np.pi * sigma_center**2) - surround / (2 * np.pi * sigma_surround**2)
    dog = dog - dog.mean()  # zero-mean (DC removal, like real RGCs)
    return dog


# ---------------------------------------------------------------------------
# Module 1: Retinal Module
# Biological analogue: Retinal ganglion cells (RGCs)
# ---------------------------------------------------------------------------

class RetinalModule(nn.Module):
    """
    Approximates retinal ganglion cell processing.

    Uses two convolutional layers:
      - Layer 1: DoG-initialized filters for On-center and Off-center channels
                 (mimics ~30 RGC types, here simplified to On/Off)
      - Layer 2: Learned combination of those channels
                 (mimics convergence from RGCs onto SC postsynaptic cells)

    The DoG initialization is biologically motivated — rather than random weights,
    we start from a known retinal filter shape. The network can adapt from there.

    Input:  (B, 1, 128, 128)   — grayscale frame
    Output: (B, 32, 64, 64)    — downsampled feature maps (2× stride)
    """

    def __init__(self, out_channels: int = 32):
        super().__init__()
        kernel_size = 7

        # Layer 1: DoG-initialized conv (On-center + Off-center banks)
        # We use out_channels // 2 On-center and out_channels // 2 Off-center filters
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,           # 128 → 64 spatial downsample
            padding=kernel_size // 2,
            bias=False
        )

        # Initialize with DoG kernels (On-center for first half, Off-center for second half)
        with torch.no_grad():
            dog_on  = dog_kernel(kernel_size, sigma_center=1.0, sigma_surround=2.5)
            dog_off = -dog_on  # Off-center is the polarity-reversed version

            half = out_channels // 2
            for i in range(half):
                # Slight variation in sigma to simulate RGC diversity
                sigma_c = 0.8 + 0.4 * (i / half)
                self.conv1.weight[i, 0] = dog_kernel(kernel_size, sigma_c, sigma_c * 2.5)
            for i in range(half):
                sigma_c = 0.8 + 0.4 * (i / half)
                self.conv1.weight[half + i, 0] = -dog_kernel(kernel_size, sigma_c, sigma_c * 2.5)

        # Layer 2: Learned mixing (like RGC → SC convergence)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 128, 128)
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 32, 64, 64)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 32, 64, 64)
        return x


# ---------------------------------------------------------------------------
# Module 2: SC Superficial Module
# Biological analogue: SC superficial layers (retinotopic map, direction selectivity)
# ---------------------------------------------------------------------------

class SCSuperficialModule(nn.Module):
    """
    Approximates SC superficial layer processing.

    Key biological properties encoded here:
      - Retinotopy: conv layers preserve spatial layout (no global pooling yet)
      - Direction selectivity: 3×3 kernels can learn directional filters
        (looming = expansion from center, which has a characteristic spatial signature)
      - Two conv layers ≈ simple and complex cell hierarchy in SC superficial

    Input:  (B, 32, 64, 64)   — retinal feature maps
    Output: (B, 64, 32, 32)   — further downsampled, richer features
    """

    def __init__(self, in_channels: int = 32, out_channels: int = 64):
        super().__init__()

        # Layer 1: Maintains retinotopy, learns direction-selective features
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        # Layer 2: Integrates over slightly larger spatial extents
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Skip connection (SC receives V1 feedback — residual mimics top-down modulation)
        self.skip  = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 32, 64, 64)
        skip = self.skip(x)                             # (B, 64, 32, 32)
        x    = F.relu(self.bn1(self.conv1(x)))          # (B, 64, 32, 32)
        x    = F.relu(self.bn2(self.conv2(x)) + skip)   # (B, 64, 32, 32) — residual
        return x


# ---------------------------------------------------------------------------
# Module 3: SC Deep / PV+ Module
# Biological analogue: SC deep PV+ neurons (threat salience integration)
# ---------------------------------------------------------------------------

class SCDeepModule(nn.Module):
    """
    Approximates SC deep layer PV+ neuron processing.

    Key differences from superficial module:
      - Integrates across space (global average pooling at end)
        → SC deep layers lose strict retinotopy, integrate threat across the visual field
      - Larger receptive fields (5×5 kernels)
        → SC deep cells have larger RFs than superficial cells
      - Global average pooling → produces a fixed-size threat representation
        regardless of input spatial resolution

    Input:  (B, 64, 32, 32)   — SC superficial features
    Output: (B, 128)           — global threat representation vector
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 128):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Global average pooling: spatial integration across the visual field
        # Biologically: SC deep cells pool over large retinotopic regions
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 128, 16, 16)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 128, 16, 16)
        x = self.gap(x)                        # (B, 128, 1, 1)
        x = x.flatten(1)                       # (B, 128)
        return x


# ---------------------------------------------------------------------------
# Output heads: PBGN and LPTN streams
# These are the two PV+ projection outputs — your handoff to the motor team
# ---------------------------------------------------------------------------

class ThreatHead(nn.Module):
    """
    Single FC output head (used for both PBGN and LPTN streams).

    FC(128 → 64 → 1) with dropout for regularization.
    Sigmoid activation: output is in [0, 1], representing activation level
    of that SC output stream.

    Biologically: each head learns to weight features that are relevant
    to its downstream target — PBGN head responds more to fast/high-contrast
    looming; LPTN head to slower, more ambiguous overhead stimuli.
    """

    def __init__(self, in_features: int = 128, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # (B, 1)


# ---------------------------------------------------------------------------
# Full model: MouseVisualCNN
# ---------------------------------------------------------------------------

class MouseVisualCNN(nn.Module):
    """
    Full mouse visual threat detection CNN.

    Replicates the retina → SC superficial → SC deep → PV+ fork pathway.

    Args:
        dropout: Dropout rate in output heads (default 0.3)

    Input:  (B, 1, 128, 128)  — grayscale frame from Unity simulation
    Output: (B, 2)             — [escape_signal, freeze_signal]
                                  Both in [0, 1], passed to motor team's dPAG module.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.retina        = RetinalModule(out_channels=32)
        self.sc_superficial = SCSuperficialModule(in_channels=32, out_channels=64)
        self.sc_deep        = SCDeepModule(in_channels=64, out_channels=128)
        self.pbgn_head      = ThreatHead(in_features=128, dropout=dropout)  # escape stream
        self.lptn_head      = ThreatHead(in_features=128, dropout=dropout)  # freeze stream

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input frame, shape (B, 1, 128, 128)

        Returns:
            threat_vector: shape (B, 2)
                           [:, 0] = escape signal (PBGN stream)
                           [:, 1] = freeze signal (LPTN stream)
        """
        x = self.retina(x)            # (B, 32, 64, 64)
        x = self.sc_superficial(x)    # (B, 64, 32, 32)
        x = self.sc_deep(x)           # (B, 128)

        escape_signal = self.pbgn_head(x)   # (B, 1)
        freeze_signal = self.lptn_head(x)   # (B, 1)

        return torch.cat([escape_signal, freeze_signal], dim=1)  # (B, 2)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class ThreatLoss(nn.Module):
    """
    Combined training loss.

    Primary loss:
        Binary cross-entropy on [escape_signal, freeze_signal] against
        behavioral labels from the Unity simulation.
        Labels: [1, 0] = escape occurred, [0, 1] = freeze occurred

    Auxiliary loss (optional but recommended):
        MSE on estimated looming expansion rate.
        Forces the SC superficial module to explicitly encode the
        looming magnitude — a key threat parameter — not just the
        final behavioral outcome.
        Requires an additional regression head (see LoomingAuxHead below).

    Args:
        aux_weight: Weight of auxiliary looming loss (default 0.2)
    """

    def __init__(self, aux_weight: float = 0.2):
        super().__init__()
        self.bce        = nn.BCELoss()
        self.mse        = nn.MSELoss()
        self.aux_weight = aux_weight

    def forward(
        self,
        threat_pred:   torch.Tensor,   # (B, 2)  model output
        behavior_label: torch.Tensor,  # (B, 2)  [escape, freeze] one-hot from Unity
        looming_pred:  torch.Tensor = None,  # (B, 1) optional aux output
        looming_label: torch.Tensor = None,  # (B, 1) expansion rate from simulation
    ) -> torch.Tensor:

        primary_loss = self.bce(threat_pred, behavior_label)

        if looming_pred is not None and looming_label is not None:
            aux_loss = self.mse(looming_pred, looming_label)
            return primary_loss + self.aux_weight * aux_loss

        return primary_loss


# ---------------------------------------------------------------------------
# Optional auxiliary head: looming expansion rate regression
# ---------------------------------------------------------------------------

class LoomingAuxHead(nn.Module):
    """
    Auxiliary regression head attached to SC superficial features.

    Predicts the looming expansion rate of the stimulus (pixels/frame).
    This auxiliary objective forces the SC superficial module to explicitly
    encode threat-relevant visual features, not just final behavior.

    Biological motivation: SC superficial cells are known to respond
    to looming stimuli with activity proportional to expansion rate.
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()   # expansion rate is non-negative
        )

    def forward(self, sc_superficial_features: torch.Tensor) -> torch.Tensor:
        # sc_superficial_features: (B, 64, 32, 32)
        x = self.gap(sc_superficial_features).flatten(1)  # (B, 64)
        return self.fc(x)  # (B, 1)


# ---------------------------------------------------------------------------
# Quick test: verify shapes are correct
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    model = MouseVisualCNN(dropout=0.3)
    print(model)
    print()

    B = 4  # batch size
    dummy_input = torch.randn(B, 1, 128, 128)

    # Forward pass
    threat_vector = model(dummy_input)

    print(f"Input shape:         {dummy_input.shape}")
    print(f"Output shape:        {threat_vector.shape}")
    print(f"Escape signals:      {threat_vector[:, 0].detach().numpy().round(3)}")
    print(f"Freeze signals:      {threat_vector[:, 1].detach().numpy().round(3)}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # Test loss
    behavior_label = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])  # escape / freeze alternating
    criterion = ThreatLoss(aux_weight=0.2)
    loss = criterion(threat_vector, behavior_label)
    print(f"Loss (random weights): {loss.item():.4f}")
