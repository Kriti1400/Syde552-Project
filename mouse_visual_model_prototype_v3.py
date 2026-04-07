"""
Mouse Retino-Collicular Visual Threat Detection Model
======================================================
A biologically grounded PyTorch model of the mouse visual system for predator
detection, spanning photoreceptor transduction → retinal interneurons → parallel
RGC output channels → retino-collicular (fast) and retino-geniculate-cortical
(slow) pathways → motor decision (escape / freeze / neutral).

Biological grounding:
  - Naka-Rushton hyperbolic saturation at the photoreceptor layer
  - ON / OFF bipolar cell split via Difference-of-Gaussians (DoG) spatial RFs
  - Starburst Amacrine Cell (SAC) direction-selectivity module
  - VG3 looming-specific amacrine module
  - Parallel RGC channels: W3 alarm, OFF-transient alpha (K-RGC), ON-OFF DSGC,
    sustained alpha
  - Fast retino-collicular pathway  → superior colliculus (SC) LSTM
  - Slow retino-geniculate-cortical pathway → V1 CNN + higher visual areas
  - Decision head: 3-class (escape / freeze / neutral)

Input:  (B, T, 1, H, W)  — grayscale temporal stack of T frames
Output: (B, 3)            — logits over [escape, freeze, neutral]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


# ---------------------------------------------------------------------------
# 1. Photoreceptor Layer  (Naka-Rushton contrast transduction)
# ---------------------------------------------------------------------------

class PhotoreceptorLayer(nn.Module):
    """
    Weber-law contrast sensitivity via the Naka-Rushton hyperbolic saturation:

        R = R_max * I^n / (I^n + I_half^n)

    I_half is a learnable semi-saturation constant that approximates
    light-adaptive gain control. The output represents relative contrast
    (delta-I / I) normalised to [0, 1].

    Rod-dominated mosaic → single achromatic channel (grayscale input expected).
    """

    def __init__(self, r_max: float = 1.0, n: float = 1.0):
        super().__init__()
        # Semi-saturation constant — learnable to allow gain adaptation
        self.log_i_half = nn.Parameter(torch.zeros(1))   # exp gives >0
        self.r_max = r_max
        self.n = n

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, 1, H, W)  pixel intensities in [0, 1]
        Returns:
            r: (B, T, 1, H, W)  normalised photoreceptor responses
        """
        i_half = torch.exp(self.log_i_half)
        x_n = x ** self.n
        r = self.r_max * x_n / (x_n + i_half ** self.n + 1e-6)
        return r


# ---------------------------------------------------------------------------
# 2. ON / OFF Bipolar Cell Layer  (Difference-of-Gaussians spatial filter)
# ---------------------------------------------------------------------------

def _gaussian_kernel(sigma: float, size: int) -> Tensor:
    """Create a 2-D Gaussian kernel of given sigma and odd size."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


class DifferenceOfGaussiansFilter(nn.Module):
    """
    Fixed DoG spatial filter representing bipolar-cell centre-surround RFs:

        D(x,y) = G(sigma_cen) − B * G(sigma_sur)

    Applied as depthwise convolution over the spatial dimensions.
    sign=+1 → ON bipolar (centre excitation, surround inhibition)
    sign=−1 → OFF bipolar (inverted)
    """

    def __init__(self, sigma_cen: float = 1.0, sigma_sur: float = 3.0,
                 b: float = 0.85, kernel_size: int = 9, sign: int = 1):
        super().__init__()
        assert sign in (1, -1)
        self.sign = sign
        k_cen = _gaussian_kernel(sigma_cen, kernel_size)
        k_sur = _gaussian_kernel(sigma_sur, kernel_size)
        dog = sign * (k_cen - b * k_sur)
        # Shape (1, 1, K, K) for depthwise conv over 1-channel input
        self.register_buffer("kernel", dog.unsqueeze(0).unsqueeze(0))
        self.pad = kernel_size // 2

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B*T, 1, H, W)
        Returns:
            out: (B*T, 1, H, W)
        """
        out = F.conv2d(x, self.kernel, padding=self.pad)
        return F.relu(out)   # half-wave rectify (spiking neurons can't go negative)


class BipolarCellLayer(nn.Module):
    """
    Splits photoreceptor output into ON and OFF pathways.
    ON bipolars: mGluR6-mediated sign-inversion → respond to light increments.
    OFF bipolars: AMPA/kainate → respond to light decrements.
    """

    def __init__(self):
        super().__init__()
        self.on_filter  = DifferenceOfGaussiansFilter(sign=+1)
        self.off_filter = DifferenceOfGaussiansFilter(sign=-1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, 1, H, W)
        Returns:
            on:  (B, T, 1, H, W)
            off: (B, T, 1, H, W)
        """
        B, T, C, H, W = x.shape
        flat = x.view(B * T, C, H, W)
        on  = self.on_filter(flat).view(B, T, 1, H, W)
        off = self.off_filter(flat).view(B, T, 1, H, W)
        return on, off


# ---------------------------------------------------------------------------
# 3. Amacrine Cell Modules
# ---------------------------------------------------------------------------

class StarbustAmacrineModule(nn.Module):
    """
    Starburst Amacrine Cells (SACs) — direction-selective via spatiotemporal
    correlation of ON and OFF bipolar inputs.

    Implemented as a lightweight temporal Conv1d over the flattened spatial
    feature map, approximating the asymmetric dendritic delay structure that
    makes SACs direction-tuned.
    """

    def __init__(self, spatial_features: int, hidden: int = 64):
        super().__init__()
        # Project spatial map to compact feature vector
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.flat_dim = hidden * 4 * 4
        # Temporal convolution captures direction via differential timing
        self.temporal_conv = nn.Conv1d(
            in_channels=self.flat_dim,
            out_channels=hidden,
            kernel_size=3,
            padding=1,
        )
        self.out_dim = hidden

    def forward(self, on: Tensor, off: Tensor) -> Tensor:
        """
        Args:
            on, off: (B, T, 1, H, W)
        Returns:
            direction_signal: (B, T, hidden)
        """
        B, T, _, H, W = on.shape
        # Stack ON/OFF as 2-channel input
        x = torch.cat([on, off], dim=2)           # (B, T, 2, H, W)
        x = x.view(B * T, 2, H, W)
        x = self.spatial_proj(x)                   # (B*T, hidden, 4, 4)
        x = x.view(B, T, -1).permute(0, 2, 1)     # (B, flat_dim, T)
        x = F.relu(self.temporal_conv(x))          # (B, hidden, T)
        return x.permute(0, 2, 1)                  # (B, T, hidden)


class VG3LoomingModule(nn.Module):
    """
    VG3 amacrine cells — looming-specific detection via approach-sensitive
    glutamatergic signalling.

    Implemented as a temporal expansion detector: computes the rate of increase
    of the OFF signal energy (dark object getting larger) over successive frames.
    A 1-D causal conv extracts the expansion signature from the temporal profile
    of the spatial OFF energy.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.hidden = hidden
        # Learnable spatial pooling weights
        self.pool_conv = nn.Conv2d(1, hidden, kernel_size=5, padding=2)
        # Causal temporal conv (looming is an expanding signal over time)
        self.temporal = nn.Conv1d(hidden, hidden, kernel_size=5, padding=4)
        self.out_dim = hidden

    def forward(self, off: Tensor) -> Tensor:
        """
        Args:
            off: (B, T, 1, H, W)
        Returns:
            looming_signal: (B, T, hidden)
        """
        B, T, _, H, W = off.shape
        x = off.view(B * T, 1, H, W)
        x = F.relu(self.pool_conv(x))              # (B*T, hidden, H, W)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # (B*T, hidden)
        x = x.view(B, T, self.hidden).permute(0, 2, 1)           # (B, hidden, T)
        # Causal: trim the acausal padding from the right
        x = F.relu(self.temporal(x)[:, :, :T])    # (B, hidden, T)
        return x.permute(0, 2, 1)                  # (B, T, hidden)


# ---------------------------------------------------------------------------
# 4. Parallel RGC Output Channels
# ---------------------------------------------------------------------------

class W3AlarmRGC(nn.Module):
    """
    W3 "alarm" RGCs — nonlinear centre pooling to detect small moving objects
    (predators) against a stationary background.
    Uses a difference between a small-kernel and large-kernel pooling to isolate
    small-figure motion, consistent with nonlinear subunit pooling reported
    for W3 cells (Zhang et al., 2012).
    """

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.local  = nn.Conv2d(1, out_dim, kernel_size=3, padding=1)
        self.global_ = nn.Conv2d(1, out_dim, kernel_size=11, padding=5)
        self.out_dim = out_dim

    def forward(self, off: Tensor) -> Tensor:
        """off: (B*T, 1, H, W) → (B*T, out_dim)"""
        local_r  = F.relu(self.local(off))
        global_r = F.relu(self.global_(off))
        # Small-figure response = local − global (nonlinear subunit suppression)
        alarm = F.relu(local_r - global_r)
        return F.adaptive_avg_pool2d(alarm, 1).flatten(1)


class OffTransientAlphaRGC(nn.Module):
    """
    OFF-transient alpha (K-RGC) — encodes the size of looming threats by
    pooling over a wide spatial extent. Responds transiently to rapid
    decrements (the leading edge of an expanding dark disk).
    Temporal transience is approximated by taking frame-to-frame difference.
    """

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.pool_conv = nn.Conv2d(1, out_dim, kernel_size=7, padding=3)
        self.out_dim = out_dim

    def forward(self, off_t: Tensor, off_prev: Tensor) -> Tensor:
        """
        off_t, off_prev: (B, 1, H, W) — current and previous OFF frames
        Returns: (B, out_dim)
        """
        transient = F.relu(off_t - off_prev)          # temporal differentiation
        r = F.relu(self.pool_conv(transient))
        return F.adaptive_avg_pool2d(r, 1).flatten(1)


class ONOFFDirectionSelectiveRGC(nn.Module):
    """
    ON-OFF DSGCs — encode cardinal motion direction by combining the
    directional signal from the SAC module with a learned spatial projection.
    """

    def __init__(self, sac_dim: int, out_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(sac_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, sac_signal: Tensor) -> Tensor:
        """sac_signal: (B, sac_dim) → (B, out_dim)"""
        return F.relu(self.proj(sac_signal))


class SustainedAlphaRGC(nn.Module):
    """
    Sustained alpha-like RGCs — relay form and contrast details to V1.
    Responds to maintained stimulation; modelled as a standard spatial conv
    on the ON pathway.
    """

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(1, out_dim, kernel_size=5, padding=2)
        self.out_dim = out_dim

    def forward(self, on: Tensor) -> Tensor:
        """on: (B*T, 1, H, W) → (B*T, out_dim)"""
        r = F.relu(self.conv(on))
        return F.adaptive_avg_pool2d(r, 1).flatten(1)


# ---------------------------------------------------------------------------
# 5. Superior Colliculus (Fast Retino-Collicular Pathway)
# ---------------------------------------------------------------------------

class SuperiorColliculusModule(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.out_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, input_dim)  — temporal RGC feature stream
        Returns:
            sc_out: (B, hidden_dim)  — final hidden state
        """
        out, (h_n, _) = self.lstm(x)
        return h_n[-1]   # last layer's final hidden state


# ---------------------------------------------------------------------------
# 6. Primary Visual Cortex / Higher Visual Areas (Slow Retino-Geniculate Path)
# ---------------------------------------------------------------------------

class V1HigherVisualModule(nn.Module):

    def __init__(self, in_channels: int = 1, out_dim: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),          nn.ReLU(),
            nn.Conv2d(32, out_dim, kernel_size=3, padding=1),     nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(out_dim * 4 * 4, out_dim)
        self.out_dim = out_dim

    def forward(self, on: Tensor) -> Tensor:
        """
        Args:
            on: (B, T, 1, H, W)  — ON bipolar output (bright/contrast info)
        Returns:
            ctx: (B, out_dim)  — contextual modulation signal
        """
        B, T, C, H, W = on.shape
        x = on.view(B * T, C, H, W)
        x = self.cnn(x)                            # (B*T, out_dim, 4, 4)
        x = self.fc(x.flatten(1))                  # (B*T, out_dim)
        x = x.view(B, T, -1).mean(dim=1)           # temporal average → (B, out_dim)
        return x


# ---------------------------------------------------------------------------
# 7. Full Mouse Retino-Collicular Visual Threat Model
# ---------------------------------------------------------------------------

class MouseRetinoCollicularModel(nn.Module):
    """
    End-to-end mouse visual threat detection model.

    Architecture (bottom-up):
        Input frames (B, T, 1, H, W)
            │
        PhotoreceptorLayer      [Naka-Rushton contrast gain]
            │
        BipolarCellLayer        [DoG ON/OFF split]
            │
        ┌───┴────────────────────────┐
        │   Amacrine modules         │
        │   ├── StarbustAmacrine     │ direction selectivity
        │   └── VG3Looming          │ looming detection
        │                            │
        │   Parallel RGC channels    │
        │   ├── W3Alarm             │ small moving objects
        │   ├── OffTransientAlpha   │ looming size encoder
        │   ├── ONOFFDirectionSelect│ cardinal motion
        │   └── SustainedAlpha      │ form / contrast
        └───┬────────────────────────┘
            │
        ┌───┴───────────────────────────────┐
        │                                   │
    Fast path: SC LSTM              Slow path: V1+HVA CNN
    (retino-collicular)             (retino-geniculate-cortical)
        │                                   │
        └──────────────┬────────────────────┘
                       │
               Decision head (3-class)
               [escape | freeze | neutral]
    """

    def __init__(
        self,
        frame_h: int = 64,
        frame_w: int = 64,
        sc_hidden: int = 128,
        v1_out_dim: int = 64,
        rgc_out_dim: int = 32,
        num_classes: int = 3,
    ):
        super().__init__()

        # ── Photoreceptor ──────────────────────────────────────────────────
        self.photoreceptor = PhotoreceptorLayer()

        # ── Bipolar ────────────────────────────────────────────────────────
        self.bipolar = BipolarCellLayer()

        # ── Amacrine ───────────────────────────────────────────────────────
        self.sac   = StarbustAmacrineModule(spatial_features=frame_h * frame_w)
        self.vg3   = VG3LoomingModule(hidden=rgc_out_dim)

        # ── RGC channels ──────────────────────────────────────────────────
        self.w3        = W3AlarmRGC(out_dim=rgc_out_dim)
        self.k_rgc     = OffTransientAlphaRGC(out_dim=rgc_out_dim)
        self.dsgc      = ONOFFDirectionSelectiveRGC(
                            sac_dim=self.sac.out_dim, out_dim=rgc_out_dim)
        self.sustained = SustainedAlphaRGC(out_dim=rgc_out_dim)

        # ── SC (fast path) ────────────────────────────────────────────────
        # Input dimension = per-frame features from all RGC channels + amacrine
        sc_input_dim = (
            rgc_out_dim       # W3
            + rgc_out_dim     # K-RGC
            + rgc_out_dim     # DSGC
            + rgc_out_dim     # sustained alpha
            + self.sac.out_dim  # SAC direction
            + self.vg3.out_dim  # VG3 looming
        )
        self.sc = SuperiorColliculusModule(
            input_dim=sc_input_dim, hidden_dim=sc_hidden)

        # ── V1 / HVA (slow path) ─────────────────────────────────────────
        self.v1 = V1HigherVisualModule(in_channels=1, out_dim=v1_out_dim)

        # ── Decision head ─────────────────────────────────────────────────
        self.decision = nn.Sequential(
            nn.Linear(sc_hidden + v1_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        # Class labels for reference
        self.class_names = ["escape", "freeze", "neutral"]

    # -----------------------------------------------------------------------
    def forward(self, frames: Tensor) -> Tensor:
        """
        Args:
            frames: (B, T, 1, H, W)  grayscale, intensities in [0, 1]
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = frames.shape

        # ── 1. Photoreceptor transduction ──────────────────────────────────
        pr = self.photoreceptor(frames)             # (B, T, 1, H, W)

        # ── 2. ON / OFF bipolar split ──────────────────────────────────────
        on, off = self.bipolar(pr)                  # each (B, T, 1, H, W)

        # ── 3. Amacrine modules ────────────────────────────────────────────
        sac_sig = self.sac(on, off)                 # (B, T, sac_dim)
        vg3_sig = self.vg3(off)                     # (B, T, vg3_dim)

        # ── 4. RGC channels — process each time step ───────────────────────
        # Flatten time for spatial convolutions
        on_flat  = on.view(B * T, 1, H, W)
        off_flat = off.view(B * T, 1, H, W)

        # W3: small alarm signal from OFF pathway
        w3_feat = self.w3(off_flat).view(B, T, -1)  # (B, T, rgc_dim)

        # K-RGC: transient looming size from consecutive OFF frames
        # Pad the first frame by repeating it (no previous frame available)
        off_prev = torch.cat([off[:, :1], off[:, :-1]], dim=1)  # (B, T, 1, H, W)
        krgc_feats = []
        for t in range(T):
            krgc_feats.append(
                self.k_rgc(off[:, t], off_prev[:, t])
            )
        krgc_feat = torch.stack(krgc_feats, dim=1)  # (B, T, rgc_dim)

        # DSGC: cardinal motion from SAC
        dsgc_feat = self.dsgc(sac_sig.view(B * T, -1)).view(B, T, -1)

        # Sustained alpha: form/contrast from ON pathway
        sust_feat = self.sustained(on_flat).view(B, T, -1)  # (B, T, rgc_dim)

        # ── 5. Assemble temporal feature stream → SC ──────────────────────
        sc_input = torch.cat(
            [w3_feat, krgc_feat, dsgc_feat, sust_feat, sac_sig, vg3_sig],
            dim=-1,
        )  # (B, T, sc_input_dim)

        sc_out = self.sc(sc_input)                  # (B, sc_hidden)

        # ── 6. V1 slow contextual path ────────────────────────────────────
        v1_out = self.v1(on)                         # (B, v1_out_dim)

        # ── 7. Decision ───────────────────────────────────────────────────
        combined = torch.cat([sc_out, v1_out], dim=-1)
        logits = self.decision(combined)             # (B, 3)
        return logits


# ---------------------------------------------------------------------------
# 8. Loss — biologically motivated class weighting
# ---------------------------------------------------------------------------

def build_loss(device: torch.device) -> nn.CrossEntropyLoss:
    """
    Escape and freeze are rare but critical responses.
    Weight threat classes higher than neutral to penalise missed detections.
    """
    weights = torch.tensor([2.0, 2.0, 1.0], device=device)  # escape, freeze, neutral
    return nn.CrossEntropyLoss(weight=weights)


# ---------------------------------------------------------------------------
# 9. Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torchinfo

    T, H, W = 10, 64, 64
    B = 4

    model = MouseRetinoCollicularModel(frame_h=H, frame_w=W)
    model.eval()

    dummy = torch.rand(B, T, 1, H, W)
    with torch.no_grad():
        logits = model(dummy)

    print("=" * 60)
    print("Mouse Retino-Collicular Visual Threat Detection Model")
    print("=" * 60)
    print(f"Input shape  : {tuple(dummy.shape)}")
    print(f"Output logits: {tuple(logits.shape)}")
    print(f"Classes      : {model.class_names}")
    print(f"Predictions  : {logits.argmax(dim=-1).tolist()}")
    print()

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Module breakdown
    print("\n── Module parameter counts ──")
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:<25s}: {n:>10,}")
