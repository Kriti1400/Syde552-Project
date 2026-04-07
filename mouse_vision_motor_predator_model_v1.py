"""
Mouse Visual-Motor System
===================================================
Implements the complete pipeline described in the report:

  Visual (fast subcortical):
    frames → RetinalModule → SCSuperficialModule → SCDeepModule
           → [escape_signal, freeze_signal, sc_heading]

  Visual (slow cortical):
    frames → V1Module → CORnetMModule (HVAs) → M2Module (GRU)

  Motor (fast pathway):
    escape_signal → sc_to_pgb → pgb_to_bla → bla_to_dpag → dpag_to_cuneiform
    freeze_signal → sc_to_lp  → lp_to_cea  → cea_to_vlpag
    [vlpag, cuneiform] → PAG competition (softmax) → behavior_label
    [cuneiform, sc_heading, m2_heading_bias] → mlr_proj → cpg → velocity[vx,vy]

  Motor (slow pathway):
    M2 hidden state → M1Module (linear readout) → m2_pag_bias, m2_heading_bias
    → modulates PAG thresholds and MLR steering

INPUT:   (B, T, H, W)  — T=5 consecutive greyscale frames, H=W=128
OUTPUTS:
  behavior_label (B,)       — 0=freeze, 1=escape  (argmax of PAG competition)
  velocity       (B, 2)     — [vx, vy] escape velocity (zero when freeze wins)
  threat_vector  (B, 2)     — [escape_signal, freeze_signal] for analysis
  sc_heading     (B, 1)     — threat angular location

Assumptions about Unity data:
  - Frames are greyscale 128×128 (rod-dominated, low-acuity retina)
  - T=5 consecutive frames per trial at ~30fps (~167ms window)
  - Labels come from Unity simulation behavioral outcomes (escape/freeze)
  - Spatial vector (x, y, z, shelter_distance) is available but NOT used
    in this implementation (would plug into M2 as a second GRU input)
  - Looming rate (pixels/frame) is logged by Unity for the auxiliary loss

Key references:
  Shang et al. (2018) Nat Commun   — SC PV+ PBGN/LPTN divergent pathways
  Evans et al. (2018) Nature       — mSC→dPAG threshold, escape decisions
  Zhao et al. (2014) Neuron        — V1 corticotectal gain modulation
  Lee et al. (2020) eLife          — SC superficial→deep sifting
  Yilmaz & Meister (2013) CurrBiol — Looming parameters
  Kubilius et al. (2019)           — CORnet-S (basis for CORnet-M)
  Caggiano et al. (2018) Nature    — MLR locomotor speed/gait
  Tovote et al. (2016) Nature      — vlPAG/dPAG reciprocal inhibition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# UTILITY
# =============================================================================

def dog_kernel(size, sigma_c, sigma_s):
    """Difference-of-Gaussians kernel — approximates RGC center-surround RF."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    r2 = x**2 + y**2
    c = torch.exp(-r2 / (2 * sigma_c**2))
    s = torch.exp(-r2 / (2 * sigma_s**2))
    dog = c / (2 * np.pi * sigma_c**2) - s / (2 * np.pi * sigma_s**2)
    return dog - dog.mean()


# =============================================================================
# VISUAL SYSTEM — FAST SUBCORTICAL STREAM
# =============================================================================

class RetinalModule(nn.Module):
    """
    Retinal ganglion cell processing.
    DoG-initialized On-center / Off-center filters over T input frames.
    Temporal weighting ramp encodes motion onset (transient On-Off RGCs).

    Input:  (B, T, 128, 128)
    Output: (B, 32, 64, 64)
    """
    def __init__(self, T=5, out_ch=32):
        super().__init__()
        ksz = 7
        self.conv1 = nn.Conv2d(T, out_ch, ksz, stride=2, padding=ksz//2, bias=False)
        with torch.no_grad():
            half = out_ch // 2
            for i in range(half):
                sc = 0.8 + 0.4 * (i / half)
                dog = dog_kernel(ksz, sc, sc * 2.5)
                for t in range(T):
                    w = 0.5 + 0.5 * (t / (T - 1))
                    self.conv1.weight[i, t]        =  dog * w
                    self.conv1.weight[half + i, t] = -dog * w
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))


class SCSuperficialModule(nn.Module):
    """
    SC superficial layers — retinotopic, direction-selective.
    Residual connection models V1 corticotectal gain (Zhao et al., 2014).

    Input:  (B, 32, 64, 64)
    Output: (B, 64, 32, 32)
    """
    def __init__(self, in_ch=32, out_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 5, stride=2, padding=2, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False)

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x))


class SCDeepModule(nn.Module):
    """
    SC deep PV+ neurons — global average pooling encodes spatial invariance
    (Lee et al., 2020: deep SC neurons respond across the full upper visual field).

    Input:  (B, 64, 32, 32)
    Output: (B, 128)
    """
    def __init__(self, in_ch=64, out_ch=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 5, stride=2, padding=2, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.gap   = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.gap(x).flatten(1)


class SCHeadingHead(nn.Module):
    """
    SC topographic map → sc_heading[B,1].
    Activation-weighted centroid of SC superficial retinotopic map.
    Feeds into motor mlr_proj for escape direction.

    Input:  (B, 64, 32, 32)
    Output: (B, 1)  in [-1, 1]
    """
    def __init__(self, in_ch=64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 1, 1, bias=False)

    def forward(self, sc_sup):
        attn = torch.sigmoid(self.proj(sc_sup))
        W = sc_sup.shape[-1]
        xc = torch.linspace(-1, 1, W, device=sc_sup.device).view(1, 1, 1, W).expand_as(attn)
        s  = attn.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return ((attn * xc).sum(dim=(2, 3), keepdim=True) / s).view(-1, 1)


# =============================================================================
# VISUAL SYSTEM — SLOW CORTICAL STREAM
# =============================================================================

class V1Module(nn.Module):
    """
    Primary visual cortex (V1/VISp).
    LGN relay → simple cells → complex cells.
    Output is concatenated with SC deep features as gain modulation
    (Zhao et al., 2014: ~50% boost to SC looming responses, retinotopically organized).

    Input:  (B, T, 128, 128)
    Output: (B, 64)
    """
    def __init__(self, T=5):
        super().__init__()
        self.lgn = nn.Sequential(
            nn.Conv2d(T, 16, 7, stride=2, padding=3), nn.BatchNorm2d(16), nn.ReLU()
        )
        self.simple = nn.Sequential(
            nn.Conv2d(16, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.complex = nn.Sequential(
            nn.Conv2d(64, 64, 5, stride=2, padding=2), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.gap(self.complex(self.simple(self.lgn(x)))).flatten(1)


class CORnetMBlock(nn.Module):
    """
    Single CORnet-M recurrent block — mouse-adapted CORnet-S.
    Implements within-area recurrence (2 steps) + skip connection.
    Models one higher visual area (HVA) from Marshel et al. (2011).
    """
    def __init__(self, in_ch, out_ch, stride=1, recurrence=2):
        super().__init__()
        self.recurrence = recurrence
        self.conv_in  = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn_in    = nn.BatchNorm2d(out_ch)
        self.conv_rec = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn_rec   = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        x = F.relu(self.bn_in(self.conv_in(x)))
        for _ in range(self.recurrence):
            x = F.relu(self.bn_rec(self.conv_rec(x)) + x)
        return F.relu(x + skip)


class CORnetMModule(nn.Module):
    """
    CORnet-M: mouse-adapted CORnet-S for higher visual areas (HVAs).
    V1 output fans out to 3 parallel HVA groups (Marshel et al., 2011):
      - motion group (VISl, VISal, VISrl, VISam)
      - pattern group (VISpm)
      - combined convergence at VISpor

    Input:  (B, 64)   — V1 GAP output
    Output: (B, 128)  — HVA representation for M2
    """
    def __init__(self):
        super().__init__()
        # Expand V1 features back to spatial for recurrent processing
        self.expand = nn.Linear(64, 64 * 4 * 4)

        # Parallel HVA branches
        self.motion_branch  = CORnetMBlock(64, 64, stride=1)
        self.pattern_branch = CORnetMBlock(64, 64, stride=1)

        # Convergence at VISpor
        self.vispor = CORnetMBlock(128, 128, stride=1)
        self.gap    = nn.AdaptiveAvgPool2d(1)

    def forward(self, v1):
        x = F.relu(self.expand(v1)).view(-1, 64, 4, 4)
        m = self.motion_branch(x)
        p = self.pattern_branch(x)
        merged = torch.cat([m, p], dim=1)
        return self.gap(self.vispor(merged)).flatten(1)


class M2Module(nn.Module):
    """
    Secondary motor cortex (MOs/M2) — 2-layer GRU.
    Integrates HVA output and fast pathway signals over time,
    maintaining persistent activity for action planning (Michaels et al., 2020).
    Produces:
      m2_pag_bias     (B, 1)  — modulates vlPAG/dPAG threshold
      m2_heading_bias (B, 1)  — modulates MLR escape direction

    Input:  hva_feats (B, 128), fast_feats (B, 128)
    Output: m2_pag_bias (B, 1), m2_heading_bias (B, 1), hidden (B, 128)
    """
    def __init__(self, hva_dim=128, fast_dim=128, hidden=128):
        super().__init__()
        self.gru = nn.GRU(hva_dim + fast_dim, hidden, num_layers=2, batch_first=True)
        self.pag_head     = nn.Linear(hidden, 1)
        self.heading_head = nn.Linear(hidden, 1)

    def forward(self, hva_feats, fast_feats):
        # Treat as single-step sequence (extend to seq for multi-frame recurrence)
        combined = torch.cat([hva_feats, fast_feats], dim=1).unsqueeze(1)
        out, _ = self.gru(combined)
        h = out.squeeze(1)
        return torch.tanh(self.pag_head(h)), torch.tanh(self.heading_head(h))


class M1Module(nn.Module):
    """
    Primary motor cortex (MOp/M1) — linear readout from M2.
    Projects 128-dim M2 state to 15-dim motor manifold, then to velocity.
    (Gallego et al., 2017: motor cortex output lies on ~10-15 dim manifold.)

    Input:  (B, 128)
    Output: (B, 2)  — [vx, vy] velocity, tanh-bounded
    """
    def __init__(self, hidden=128, manifold_dim=15):
        super().__init__()
        self.to_manifold = nn.Linear(hidden, manifold_dim)
        self.to_velocity = nn.Linear(manifold_dim, 2)

    def forward(self, m2_hidden):
        return torch.tanh(self.to_velocity(F.relu(self.to_manifold(m2_hidden))))


# =============================================================================
# MOTOR SYSTEM — FAST PATHWAY
# =============================================================================

def fc_block(in_dim, out_dim):
    """Linear + ReLU + LayerNorm — matches report's fast pathway layer spec."""
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.LayerNorm(out_dim))


class FastMotorPathway(nn.Module):
    """
    Fast subcortical motor pathway.
    SC deep features → divergent PV+ streams → PAG competition → behavior + velocity.

    Freeze branch: sc_deep → LP thalamus → CeA → vlPAG
    Escape branch: sc_deep → PBG → BLA → dPAG → cuneiform → MLR → CPG

    Behavioral selection: softmax([vlpag, cuneiform]) → freeze_escape_competition
    Velocity: mlr_drive element-wise gated by cuneiform (collapses to 0 on freeze)

    Input:  sc_deep (B,128), sc_heading (B,1), m2_pag_bias (B,1), m2_heading_bias (B,1)
    Output: vlpag (B,1), cuneiform (B,1), velocity (B,2)
    """
    def __init__(self, sc_dim=192):  # 128 SC deep + 64 V1 gain = 192
        super().__init__()
        # Freeze branch
        self.sc_to_lp    = fc_block(sc_dim, 64)
        self.lp_to_cea   = fc_block(64, 64)
        self.cea_to_vlpag = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

        # Escape branch
        self.sc_to_pgb         = fc_block(sc_dim, 64)
        self.pgb_to_bla        = fc_block(64, 64)
        self.bla_to_dpag       = fc_block(64, 64)
        self.dpag_to_cuneiform = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

        # MLR + CPG
        self.mlr_proj = nn.Sequential(nn.Linear(3, 16), nn.Tanh())  # [cuneiform, sc_heading, m2_heading_bias]
        self.cpg      = nn.Sequential(nn.Linear(16, 2), nn.Tanh())

    def forward(self, sc_combined, sc_heading, m2_pag_bias, m2_heading_bias):
        # Freeze branch
        lp    = self.sc_to_lp(sc_combined)
        cea   = self.lp_to_cea(lp)
        vlpag = self.cea_to_vlpag(cea + m2_pag_bias)        # M2 bias modulates threshold

        # Escape branch
        pgb        = self.sc_to_pgb(sc_combined)
        bla        = self.pgb_to_bla(pgb)
        dpag       = self.bla_to_dpag(bla)
        cuneiform  = self.dpag_to_cuneiform(dpag + m2_pag_bias)

        # MLR: [cuneiform speed, sc_heading, m2_heading_bias]
        mlr_input = torch.cat([cuneiform, sc_heading, m2_heading_bias], dim=1)
        mlr_drive = self.mlr_proj(mlr_input)

        # CPG: velocity gated by cuneiform (zero velocity when freeze wins)
        velocity = self.cpg(mlr_drive) * cuneiform   # (B, 2)

        return vlpag, cuneiform, velocity


# =============================================================================
# FULL INTEGRATED MODEL
# =============================================================================

class MouseVisualMotorModel(nn.Module):
    """
    Complete mouse visual-motor model.

    Fast subcortical stream:
        frames → Retina → SC superficial → SC deep + V1 gain
               → FastMotorPathway → vlpag, cuneiform, velocity

    Slow cortical stream:
        frames → V1 → CORnet-M HVAs → M2 GRU → M1 readout
               → m2_pag_bias, m2_heading_bias → fed into FastMotorPathway

    Behavioral selection:
        softmax([vlpag, cuneiform]) → freeze_escape_competition
        argmax → behavior_label (0=freeze, 1=escape)
        velocity zeroed when freeze wins (gated by cuneiform)

    Args:
        T: number of input frames (default 5)

    Input:  (B, T, 128, 128)

    Returns dict with keys:
        behavior_label          (B,)    — 0=freeze, 1=escape
        velocity                (B, 2)  — [vx, vy]
        freeze_escape_comp      (B, 2)  — softmax probabilities
        threat_vector           (B, 2)  — [escape_signal, freeze_signal]
        sc_heading              (B, 1)  — threat angular location
        sc_sup_feats            (B, 64, 32, 32)  — for auxiliary loss
        vlpag                   (B, 1)
        cuneiform               (B, 1)
    """

    def __init__(self, T=5):
        super().__init__()
        self.T = T

        # Visual — fast subcortical
        self.retina         = RetinalModule(T=T)
        self.sc_superficial = SCSuperficialModule()
        self.sc_deep        = SCDeepModule()
        self.heading_head   = SCHeadingHead()

        # Visual — slow cortical
        self.v1      = V1Module(T=T)
        self.cornet_m = CORnetMModule()
        self.m2      = M2Module()
        self.m1      = M1Module()

        # Threat output heads (PBGN = escape, LPTN = freeze)
        combined_dim = 192  # 128 SC deep + 64 V1
        self.pbgn_head = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.lptn_head = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        # Motor — fast pathway (takes combined SC+V1 features)
        self.fast_motor = FastMotorPathway(sc_dim=combined_dim)

    def forward(self, x):
        B = x.shape[0]

        # --- Fast subcortical visual stream ---
        rgc_feats     = self.retina(x)                   # (B, 32, 64, 64)
        sc_sup_feats  = self.sc_superficial(rgc_feats)   # (B, 64, 32, 32)
        sc_deep_feats = self.sc_deep(sc_sup_feats)       # (B, 128)
        sc_heading    = self.heading_head(sc_sup_feats)  # (B, 1)

        # --- Slow cortical visual stream ---
        v1_feats  = self.v1(x)                           # (B, 64)
        hva_feats = self.cornet_m(v1_feats)              # (B, 128)

        # --- Combined SC + V1 gain (implements Zhao et al. 2014 gain modulation) ---
        sc_combined = torch.cat([sc_deep_feats, v1_feats], dim=1)  # (B, 192)

        # --- Visual threat output heads ---
        escape_signal = self.pbgn_head(sc_combined)      # (B, 1)
        freeze_signal = self.lptn_head(sc_combined)      # (B, 1)
        threat_vector = torch.cat([escape_signal, freeze_signal], dim=1)  # (B, 2)

        # --- M2: integrates HVA + fast SC features ---
        m2_pag_bias, m2_heading_bias = self.m2(hva_feats, sc_deep_feats)  # (B,1), (B,1)

        # --- Fast motor pathway ---
        vlpag, cuneiform, velocity = self.fast_motor(
            sc_combined, sc_heading, m2_pag_bias, m2_heading_bias
        )

        # --- PAG competition: reciprocal inhibition (Tovote et al., 2016) ---
        pag_stack  = torch.cat([vlpag, cuneiform], dim=1)  # (B, 2)
        freeze_esc = torch.softmax(pag_stack, dim=1)       # (B, 2)
        behavior_label = freeze_esc.argmax(dim=1)          # 0=freeze, 1=escape

        return {
            'behavior_label':     behavior_label,          # (B,)
            'velocity':           velocity,                 # (B, 2)
            'freeze_escape_comp': freeze_esc,               # (B, 2)
            'threat_vector':      threat_vector,            # (B, 2)
            'sc_heading':         sc_heading,               # (B, 1)
            'sc_sup_feats':       sc_sup_feats,             # (B, 64, 32, 32) for aux loss
            'vlpag':              vlpag,                    # (B, 1)
            'cuneiform':          cuneiform,                # (B, 1)
        }


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class FullModelLoss(nn.Module):
    """
    Training loss combining:
      1. Behavioral BCE on freeze_escape_competition vs label (primary)
      2. Auxiliary MSE on looming expansion rate from SC superficial features
         (forces SC to encode threat-relevant temporal features)
      3. Velocity consistency: escape trials should have non-zero velocity

    Args:
        aux_weight:      weight for looming rate regression (default 0.2)
        velocity_weight: weight for velocity consistency loss (default 0.1)
    """
    def __init__(self, aux_weight=0.2, velocity_weight=0.1):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.aux_weight      = aux_weight
        self.velocity_weight = velocity_weight

    def forward(self, outputs, behavior_label, looming_pred=None, looming_label=None):
        # Primary: behavioral output loss
        primary = self.bce(outputs['freeze_escape_comp'], behavior_label)

        loss = primary

        # Auxiliary: looming rate regression on SC superficial
        if looming_pred is not None and looming_label is not None:
            loss = loss + self.aux_weight * self.mse(looming_pred, looming_label)

        # Velocity consistency: escape trials (label[:,1]=1) should have non-zero velocity
        escape_mask = behavior_label[:, 1].unsqueeze(1)   # 1 for escape trials
        vel_mag = outputs['velocity'].norm(dim=1, keepdim=True)
        vel_target = escape_mask                           # target magnitude ~1 on escape
        loss = loss + self.velocity_weight * self.mse(vel_mag * escape_mask, vel_target * escape_mask)

        return loss


class LoomingAuxHead(nn.Module):
    """Predicts looming expansion rate from SC superficial features."""
    def __init__(self, in_ch=64):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(nn.Linear(in_ch, 32), nn.ReLU(), nn.Linear(32, 1), nn.ReLU())

    def forward(self, sc_sup_feats):
        return self.fc(self.gap(sc_sup_feats).flatten(1))


# =============================================================================
# SHAPE TEST
# =============================================================================

if __name__ == '__main__':
    T, B = 5, 4
    model    = MouseVisualMotorModel(T=T)
    aux_head = LoomingAuxHead()
    dummy    = torch.randn(B, T, 128, 128)

    out = model(dummy)
    loom_pred = aux_head(out['sc_sup_feats'])

    print(f"Input:               {dummy.shape}")
    print(f"behavior_label:      {out['behavior_label'].shape}  {out['behavior_label'].tolist()}")
    print(f"velocity:            {out['velocity'].shape}")
    print(f"freeze_escape_comp:  {out['freeze_escape_comp'].shape}")
    print(f"threat_vector:       {out['threat_vector'].shape}")
    print(f"sc_heading:          {out['sc_heading'].shape}")
    print(f"vlpag:               {out['vlpag'].shape}")
    print(f"cuneiform:           {out['cuneiform'].shape}")
    print(f"looming_pred:        {loom_pred.shape}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:    {n:,}")
