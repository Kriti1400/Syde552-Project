"""
Visuomotor Brain Model — SYDE 552 Project
==========================================
Models the mouse visuomotor system with two fully separate pathways that
converge only at the brainstem (PAG / MLR level), not at the SC.

Architecture:
  - FastPathway:        Retina (RGCs) → sSC → dSC
                        Shallow CNN; purely feedforward; no cortical input.
                        The SC processes the retinal signal and routes it
                        directly to brainstem defensive circuits.

  - SlowPathway:        Retina → LGN → V1 → HVAs → PPC → M2 (LSTM)
                        Deep CNN + recurrent; builds deliberate motor plan.
                        M2 output converges with SC output at the brainstem,
                        NOT at the SC.

  - BrainstemPathways:  Two parallel anatomical chains from SC deep layers,
                        with M2 cortical input arriving separately at the
                        MLR (heading refinement) and PAG (urgency modulation).

      Freeze branch:    SC dSC → LP thalamus → CeA → vlPAG
                            → reticulospinal neurons → locomotion arrest
      Escape branch:    SC dSC → PBG → BLA → dPAG → cuneiform → MLR
                            → spinal CPG → [vx, vy]
      M2 → brainstem:   M2 cortical output biases MLR heading (deliberate
                        direction) and modulates PAG urgency threshold —
                        this is the slow pathway's only motor influence.

  - ThalamicFeedback:   SC superficial → LP/Pulvinar → V1 / HVAs
                        Top-down attentional loop; does NOT feed back to SC.

Key design principle:
  The SC is a FAST, REFLEXIVE structure. It does not integrate cortical input
  in this model. Cortical influence on motor output arrives downstream at the
  brainstem, consistent with the known M2 → PAG and M2 → MLR projections in
  mice (Duan et al. 2021; Caggiano et al. 2018).

Input:  Sequence of image frames  [B, T, C, H, W]
            B = batch size, T = timesteps
            C = 2 (mouse dichromatic S/M-cone) or 1 (grayscale)
            H, W = height, width

Outputs (dict) — every tensor maps to a named neural population:

  Fast pathway / SC:
    "sc_superficial"    [B, sc_s_dim]     sSC neurons (retinorecipient)
    "sc_deep"           [B, sc_d_dim]     dSC neurons (premotor output layer)

  Freeze branch (dSC → LP → CeA → vlPAG):
    "lp_thalamus"       [B, 64]           LP nucleus neurons
    "cea_amygdala"      [B, 64]           Central amygdala neurons
    "vlpag"             [B, 1]            vlPAG activation ∈ [0,1]
                                          → reticulospinal tract → locomotion arrest

  Escape branch (dSC → PBG → BLA → dPAG → cuneiform → MLR → CPG):
    "pgb_neurons"       [B, 64]           Parabigeminal nucleus neurons
    "bla_amygdala"      [B, 64]           Basolateral amygdala neurons
    "dpag"              [B, 64]           Dorsal PAG neurons
    "cuneiform"         [B, 1]            Cuneiform nucleus activation ∈ [0,1]
    "mlr_drive"         [B, 2]            MLR output: [speed_signal, heading_signal]
                                          → spinal CPG interneurons
    "velocity"          [B, 2]            Spinal CPG → [vx, vy]
                                          → motor neurons / limb muscles

  Slow pathway cortical contributions (M2 → brainstem):
    "m2_heading_bias"   [B, 1]            M2 → MLR heading correction
    "m2_pag_bias"       [B, 1]            M2 → PAG urgency modulation

  Behavior selection:
    "freeze_escape_competition" [B, 2]    softmax([vlpag, cuneiform])
                                          models vlPAG ↔ dPAG mutual inhibition
    "behavior_label"    [B]               0=freeze, 1=escape

  Thalamic feedback:
    "thalamic_feedback" [B, T, 128]       LP/Pulvinar → visual cortex

Anatomical references:
  Fast SC:  Shang et al. 2015 (Science); Evans et al. 2018 (Neuron)
  Freeze:   Wei et al. 2015 (Neuron); Tovote et al. 2016 (Nature)
  Escape:   Bhatt et al. 2020 (Curr Biol); Duan et al. 2021 (Nat Commun)
  MLR/CPG:  Caggiano et al. 2018 (Nature); Ryczko & Dubuc 2013 (Prog Neurobiol)
  M2→PAG:   Bhatt et al. 2020; Duan et al. 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. FAST PATHWAY — Retina → Superficial SC → Deep SC
# ---------------------------------------------------------------------------
# Biological basis:
#   - Retinal ganglion cells (transient ON-OFF DSGCs) project directly to
#     the superficial SC (sSC), retinotopically organized
#   - sSC sends interlaminar glutamatergic projections to deep SC (dSC)
#   - dSC is the premotor output layer; projects to LP, PBG, brainstem
#   - No cortical input to SC in this model — SC is purely fast and reflexive
#   - Modeled as a shallow 3-layer CNN; shallow = few synapses = fast
# ---------------------------------------------------------------------------

class FastPathway(nn.Module):
    """
    Retina → superficial SC → deep SC.

    sSC layer: shallow CNN mimicking RGC centre-surround → SC feature detection
               (direction selectivity, looming selectivity, motion onset)
    dSC layer: linear projection from sSC; represents the premotor output
               population of deep SC neurons that project to LP and PBG
    No cortical input at any stage.
    """

    def __init__(self, in_channels: int = 1,
                 sc_superficial_dim: int = 128,
                 sc_deep_dim: int = 128):
        super().__init__()

        # ── Retina: RGC centre-surround filtering ────────────────────────────
        # 5×5 kernel ≈ RGC receptive field size at mouse visual acuity
        # MaxPool preserves spatial relationships (retinotopy)
        self.rgc_layer = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # ── Superficial SC (sSC): visual feature selectivity ─────────────────
        # Direction-selective and looming-selective neurons
        # Second conv layer increases selectivity without deep hierarchy
        self.sc_superficial_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),   # preserve spatial structure
        )
        self.sc_superficial_proj = nn.Sequential(
            nn.Linear(64 * 4 * 4, sc_superficial_dim),
            nn.ReLU(),
            nn.LayerNorm(sc_superficial_dim),
        )

        # ── Deep SC (dSC): premotor output layer ──────────────────────────────
        # Interlaminar glutamatergic projection from sSC to dSC
        # dSC neurons are the origin of LP and PBG projections
        self.sc_deep_proj = nn.Sequential(
            nn.Linear(sc_superficial_dim, sc_deep_dim),
            nn.ReLU(),
            nn.LayerNorm(sc_deep_dim),
        )

        # ── SC topographic heading readout ────────────────────────────────────
        # The retinotopic map encodes direction of threat as population location
        # This heading scalar is passed to the MLR for escape direction
        self.sc_heading = nn.Sequential(
            nn.Linear(sc_superficial_dim, 1),
            nn.Tanh(),   # bounded to [-1, 1]; represents angular direction
        )

    def forward(self, frame: torch.Tensor) -> dict:
        """
        Args:
            frame: [B, C, H, W] — single current visual frame
        Returns dict:
            "sc_superficial"  [B, sc_superficial_dim]  sSC neuron activations
            "sc_deep"         [B, sc_deep_dim]          dSC neuron activations
            "sc_heading"      [B, 1]                    topographic direction signal
        """
        # Retina → sSC
        x = self.rgc_layer(frame)
        x = self.sc_superficial_conv(x)
        x = x.view(x.size(0), -1)
        sc_s = self.sc_superficial_proj(x)          # [B, sc_superficial_dim]

        # sSC → dSC (interlaminar projection)
        sc_d = self.sc_deep_proj(sc_s)              # [B, sc_deep_dim]

        # Topographic heading from sSC population
        heading = self.sc_heading(sc_s)             # [B, 1]

        return {
            "sc_superficial": sc_s,
            "sc_deep":        sc_d,
            "sc_heading":     heading,
        }


# ---------------------------------------------------------------------------
# 2. SLOW PATHWAY — Retina → LGN → V1 → HVAs → PPC → M2
# ---------------------------------------------------------------------------
# Biological basis:
#   - The retinogeniculate pathway (LGN) feeds V1 and higher visual areas
#   - V1 → HVAs → PPC builds increasingly abstract spatial representations
#   - M2 (secondary motor cortex) maintains deliberate motor plans via
#     recurrent dynamics (working memory); projects to PAG and MLR
#   - This pathway is SLOW: more synapses, temporal integration via LSTM
#   - M2 output does NOT go to SC; it goes directly to brainstem targets
# ---------------------------------------------------------------------------

class SlowPathway(nn.Module):
    """
    Retina → V1 → HVAs → PPC → M2 LSTM.
    Processes full temporal sequence; outputs M2 activations over time.
    M2 projects to brainstem (PAG urgency modulation, MLR heading bias).
    """

    def __init__(self, in_channels: int = 1, cortical_dim: int = 256,
                 lstm_hidden: int = 256, num_layers: int = 2):
        super().__init__()

        # ── V1: Primary visual cortex ─────────────────────────────────────────
        self.v1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # ── HVAs: Higher visual areas (LM, AL, PM, AM, etc.) ─────────────────
        self.hva = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # ── PPC: Posterior parietal cortex ────────────────────────────────────
        self.ppc = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.vis_proj = nn.Linear(128 * 4 * 4, cortical_dim)

        # ── M2: Secondary motor cortex ────────────────────────────────────────
        # LSTM maintains motor plan across time (working memory)
        # 2 layers mirrors laminar depth of M2 recurrent connections
        self.m2_lstm = nn.LSTM(
            input_size=cortical_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.m2_proj = nn.Linear(lstm_hidden, cortical_dim)

    def forward(self, frames: torch.Tensor):
        """
        Args:
            frames: [B, T, C, H, W]
        Returns:
            m2_seq:    [B, T, cortical_dim]  M2 activations over time
            lstm_state: (h_n, c_n)           final recurrent state
        """
        B, T, C, H, W = frames.shape
        vis_features = []
        for t in range(T):
            x = self.v1(frames[:, t])
            x = self.hva(x)
            x = self.ppc(x)
            x = x.view(B, -1)
            x = F.relu(self.vis_proj(x))
            vis_features.append(x)

        vis_seq = torch.stack(vis_features, dim=1)      # [B, T, cortical_dim]
        lstm_out, lstm_state = self.m2_lstm(vis_seq)
        m2_seq = self.m2_proj(lstm_out)                 # [B, T, cortical_dim]
        return m2_seq, lstm_state


# ---------------------------------------------------------------------------
# 3. BRAINSTEM PATHWAYS — SC deep + M2 cortical → freeze / escape / velocity
# ---------------------------------------------------------------------------
# The two streams converge HERE, at the brainstem level, not at the SC.
#
# SC deep layers (dSC) drive two parallel anatomical chains:
#   Freeze: dSC → LP → CeA → vlPAG → reticulospinal → locomotion arrest
#   Escape: dSC → PBG → BLA → dPAG → cuneiform → MLR → spinal CPG → [vx,vy]
#
# M2 cortical output arrives separately at two brainstem targets:
#   1. MLR heading bias: M2 sends a learned heading correction to the MLR,
#      refining the raw SC topographic direction signal. This is analogous
#      to the known M2 → MLR projection for deliberate locomotion control.
#      Ref: Caggiano et al. 2018 (Nature); Duan et al. 2021 (Nat Commun)
#
#   2. PAG urgency modulation: M2 can modulate PAG activation threshold,
#      raising or lowering the threshold for triggering freeze/escape.
#      This models M2's known top-down suppression of reflexive defense.
#      Ref: Bhatt et al. 2020 (Curr Biol); Duan et al. 2021
#
# vlPAG and dPAG are mutually inhibitory (softmax competition).
# ---------------------------------------------------------------------------

LP_DIM        = 64
CEA_DIM       = 64
PBG_DIM       = 64
BLA_DIM       = 64
DPAG_DIM      = 64
MLR_DIM       = 2
CPG_DIM       = 2

BEHAVIOR_LABELS = {0: "freeze", 1: "escape"}


class BrainstemPathways(nn.Module):
    """
    SC deep layers + M2 → freeze and escape motor programs.

    SC input drives both branches in parallel.
    M2 input arrives separately at MLR (heading) and PAG (urgency).
    The two branches compete via vlPAG ↔ dPAG softmax inhibition.
    """

    def __init__(self, sc_deep_dim: int = 128, cortical_dim: int = 256):
        super().__init__()

        # ── FREEZE BRANCH: dSC → LP → CeA → vlPAG ───────────────────────────

        # dSC → LP thalamus (lateral posterior nucleus)
        # PV+ SC neurons project here; LP relays to amygdala + visual cortex
        self.sc_to_lp = nn.Sequential(
            nn.Linear(sc_deep_dim, LP_DIM),
            nn.ReLU(),
            nn.LayerNorm(LP_DIM),
        )

        # LP → Central Amygdala (CeA)
        # CeA output neurons gate PAG; receive LP input
        self.lp_to_cea = nn.Sequential(
            nn.Linear(LP_DIM, CEA_DIM),
            nn.ReLU(),
            nn.LayerNorm(CEA_DIM),
        )

        # CeA → vlPAG
        # Scalar activation ∈ [0,1]; high = active locomotion arrest motor program
        # (sustained muscle tone, shallow breathing, heart rate drop)
        self.cea_to_vlpag = nn.Sequential(
            nn.Linear(CEA_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # ── ESCAPE BRANCH: dSC → PBG → BLA → dPAG → cuneiform ───────────────

        # dSC → Parabigeminal nucleus (PBG)
        # Cholinergic midbrain nucleus; stimulation evokes escape
        self.sc_to_pgb = nn.Sequential(
            nn.Linear(sc_deep_dim, PBG_DIM),
            nn.ReLU(),
            nn.LayerNorm(PBG_DIM),
        )

        # PBG → Basolateral Amygdala (BLA)
        # BLA drives active flight vs. CeA's passive freeze
        self.pgb_to_bla = nn.Sequential(
            nn.Linear(PBG_DIM, BLA_DIM),
            nn.ReLU(),
            nn.LayerNorm(BLA_DIM),
        )

        # BLA → dorsal PAG (dPAG)
        # dPAG activation → "wild running or backward fleeing" in mice
        self.bla_to_dpag = nn.Sequential(
            nn.Linear(BLA_DIM, DPAG_DIM),
            nn.ReLU(),
            nn.LayerNorm(DPAG_DIM),
        )

        # dPAG → Cuneiform nucleus
        # Scalar ∈ [0,1]; drives MLR speed signal
        self.dpag_to_cuneiform = nn.Sequential(
            nn.Linear(DPAG_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # ── MLR: Mesencephalic Locomotor Region ──────────────────────────────
        # Integrates cuneiform speed + SC topographic heading + M2 bias
        # Output: [speed_signal, heading_signal] → spinal CPG interneurons
        self.mlr_proj = nn.Sequential(
            nn.Linear(3, MLR_DIM),  # [cuneiform, sc_heading, m2_heading_bias]
            nn.Tanh(),
        )

        # ── Spinal CPG ────────────────────────────────────────────────────────
        # MLR drive → [vx, vy]; received by spinal motor neurons
        self.cpg = nn.Sequential(
            nn.Linear(MLR_DIM, 16),
            nn.Tanh(),
            nn.Linear(16, CPG_DIM),
        )

        # ── M2 → brainstem: heading bias (M2 → MLR) ──────────────────────────
        # M2 refines the escape direction; separate from SC reflexive heading
        # Analogous to M2 → MLR projection for deliberate locomotion
        self.m2_heading_bias = nn.Sequential(
            nn.Linear(cortical_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        # ── M2 → brainstem: PAG urgency modulation (M2 → PAG) ────────────────
        # M2 can raise/lower the threshold for vlPAG and cuneiform activation
        # Models cortical suppression of reflexive defense during safe contexts
        self.m2_pag_bias = nn.Sequential(
            nn.Linear(cortical_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # positive = lower threshold (more likely to trigger)
        )                # negative = raise threshold (suppress reflex)

    def forward(self, sc_deep: torch.Tensor,
                sc_heading: torch.Tensor,
                m2_last: torch.Tensor) -> dict:
        """
        Args:
            sc_deep:    [B, sc_deep_dim]   dSC premotor output neurons
            sc_heading: [B, 1]             SC topographic direction signal
            m2_last:    [B, cortical_dim]  M2 output (last timestep)

        Returns dict — one key per named neural population:

          Freeze branch:
            "lp_thalamus"              [B, 64]
            "cea_amygdala"             [B, 64]
            "vlpag"                    [B, 1]   ∈ [0,1]

          Escape branch:
            "pgb_neurons"              [B, 64]
            "bla_amygdala"             [B, 64]
            "dpag"                     [B, 64]
            "cuneiform"                [B, 1]   ∈ [0,1]
            "mlr_drive"                [B, 2]   [speed, heading]
            "velocity"                 [B, 2]   [vx, vy]

          M2 cortical contributions (arrive at brainstem, not at SC):
            "m2_heading_bias"          [B, 1]   M2 → MLR direction refinement
            "m2_pag_bias"              [B, 1]   M2 → PAG urgency modulation

          Behavior:
            "freeze_escape_competition" [B, 2]  softmax([vlpag, cuneiform])
            "behavior_label"            [B]     0=freeze, 1=escape
        """
        # ── M2 cortical contributions (arrive at brainstem) ───────────────────
        m2_head = self.m2_heading_bias(m2_last)   # [B, 1]  M2 → MLR
        m2_pag  = self.m2_pag_bias(m2_last)       # [B, 1]  M2 → PAG threshold

        # ── Freeze branch ─────────────────────────────────────────────────────
        lp     = self.sc_to_lp(sc_deep)
        cea    = self.lp_to_cea(lp)
        # M2 PAG bias modulates vlPAG threshold (additive before sigmoid)
        vlpag_raw = self.cea_to_vlpag[:-1](cea)   # up to but excl. Sigmoid
        vlpag  = torch.sigmoid(vlpag_raw + m2_pag)  # M2 shifts threshold

        # ── Escape branch ─────────────────────────────────────────────────────
        pgb    = self.sc_to_pgb(sc_deep)
        bla    = self.pgb_to_bla(pgb)
        dpag   = self.bla_to_dpag(bla)
        # M2 PAG bias also modulates cuneiform threshold
        cuneiform_raw = self.dpag_to_cuneiform[:-1](dpag)
        cuneiform = torch.sigmoid(cuneiform_raw + m2_pag)

        # ── MLR: SC heading + M2 bias + cuneiform speed ───────────────────────
        # M2 heading bias refines the raw SC topographic direction
        refined_heading = torch.tanh(sc_heading + m2_head)    # [B, 1]
        mlr_input = torch.cat([cuneiform, refined_heading,
                                m2_head], dim=-1)              # [B, 3]
        mlr_drive = self.mlr_proj(mlr_input)                  # [B, 2]

        # ── Spinal CPG → [vx, vy] ─────────────────────────────────────────────
        # Gated by cuneiform: freeze win → cuneiform≈0 → velocity≈0
        velocity_raw = self.cpg(mlr_drive)
        velocity     = cuneiform * velocity_raw                # [B, 2]

        # ── vlPAG ↔ dPAG mutual inhibition ────────────────────────────────────
        competition = F.softmax(
            torch.cat([vlpag, cuneiform], dim=-1), dim=-1)    # [B, 2]
        behavior_label = competition.argmax(dim=-1)            # [B]

        return {
            # Freeze branch
            "lp_thalamus":               lp,
            "cea_amygdala":              cea,
            "vlpag":                     vlpag,
            # Escape branch
            "pgb_neurons":               pgb,
            "bla_amygdala":              bla,
            "dpag":                      dpag,
            "cuneiform":                 cuneiform,
            "mlr_drive":                 mlr_drive,
            "velocity":                  velocity,
            # M2 cortical contributions
            "m2_heading_bias":           m2_head,
            "m2_pag_bias":               m2_pag,
            # Behavior
            "freeze_escape_competition": competition,
            "behavior_label":            behavior_label,
        }


# ---------------------------------------------------------------------------
# 4. THALAMIC FEEDBACK — SC superficial → LP/Pulvinar → Visual Cortex
# ---------------------------------------------------------------------------
# Biological basis:
#   - SC SUPERFICIAL layers (not deep) project to LP thalamus / pulvinar
#   - LP/pulvinar feeds back to V1 and higher visual areas (HVAs)
#   - This creates a top-down attentional loop: SC threat detection modulates
#     ongoing visual processing in cortex — sharpening responses to the
#     threat location and suppressing irrelevant visual input
#   - Does NOT feed back to SC (would create a loop without biological basis)
#   - Modeled as: sSC → LP gain signal → applied to HVA cortical sequence
# ---------------------------------------------------------------------------

class ThalamicFeedback(nn.Module):
    """
    SC superficial → LP/Pulvinar → Visual cortex attention modulation.
    SC superficial activations drive a gain signal through LP thalamus,
    which multiplicatively modulates the slow pathway's visual cortex sequence.
    """

    def __init__(self, sc_superficial_dim: int = 128,
                 cortical_dim: int = 256,
                 vis_feedback_dim: int = 128):
        super().__init__()

        # sSC → LP thalamus projection
        # LP neurons receive sSC input and project to V1 and HVAs
        self.sc_to_lp_feedback = nn.Sequential(
            nn.Linear(sc_superficial_dim, 64),
            nn.ReLU(),
        )

        # LP → Visual cortex gain modulation
        # Sigmoid output: gain factor ∈ [0,1] applied to cortical features
        self.lp_to_vis_gain = nn.Sequential(
            nn.Linear(64, vis_feedback_dim),
            nn.Sigmoid(),
        )

        # Project cortical sequence to feedback dim for gain application
        self.cortical_align = nn.Linear(cortical_dim, vis_feedback_dim)

    def forward(self, sc_superficial: torch.Tensor,
                cortical_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sc_superficial: [B, sc_superficial_dim]   sSC activations
            cortical_seq:   [B, T, cortical_dim]       slow pathway HVA sequence
        Returns:
            feedback: [B, T, vis_feedback_dim]  LP-gated cortical features
        """
        lp_signal = self.sc_to_lp_feedback(sc_superficial)     # [B, 64]
        gain      = self.lp_to_vis_gain(lp_signal)             # [B, vis_feedback_dim]
        cortical_proj = self.cortical_align(cortical_seq)       # [B, T, vis_feedback_dim]
        feedback  = gain.unsqueeze(1) * cortical_proj           # [B, T, vis_feedback_dim]
        return feedback


# ---------------------------------------------------------------------------
# 5. FULL VISUOMOTOR MODEL
# ---------------------------------------------------------------------------

class VisualMotorModel(nn.Module):
    """
    Full visuomotor model. Two fully separate pathways converge at brainstem.

    Forward pass:
      1. Fast pathway   — current frame → RGC → sSC → dSC (no cortical input)
      2. Slow pathway   — frame sequence → V1 → HVAs → PPC → M2 LSTM
      3. Brainstem      — dSC + M2 → freeze branch + escape branch
                          (M2 arrives at PAG/MLR, NOT at SC)
      4. Thalamic loop  — sSC → LP → visual cortex gain modulation
    """

    def __init__(self,
                 in_channels:        int = 1,
                 sc_superficial_dim: int = 128,
                 sc_deep_dim:        int = 128,
                 cortical_dim:       int = 256,
                 lstm_hidden:        int = 256,
                 vis_feedback_dim:   int = 128):
        super().__init__()

        self.fast_pathway       = FastPathway(in_channels,
                                              sc_superficial_dim,
                                              sc_deep_dim)
        self.slow_pathway       = SlowPathway(in_channels,
                                              cortical_dim,
                                              lstm_hidden)
        self.brainstem_pathways = BrainstemPathways(sc_deep_dim, cortical_dim)
        self.thalamic_feedback  = ThalamicFeedback(sc_superficial_dim,
                                                   cortical_dim,
                                                   vis_feedback_dim)

    def forward(self, frames: torch.Tensor) -> dict:
        """
        Args:
            frames: [B, T, C, H, W]

        Returns dict — every key maps to a named neural population:

          Fast pathway / SC (no cortical input):
            "sc_superficial"             [B, sc_superficial_dim]
            "sc_deep"                    [B, sc_deep_dim]

          Freeze branch (dSC → LP → CeA → vlPAG):
            "lp_thalamus"                [B, 64]
            "cea_amygdala"               [B, 64]
            "vlpag"                      [B, 1]   ∈ [0,1]

          Escape branch (dSC → PBG → BLA → dPAG → cuneiform → MLR → CPG):
            "pgb_neurons"                [B, 64]
            "bla_amygdala"               [B, 64]
            "dpag"                       [B, 64]
            "cuneiform"                  [B, 1]   ∈ [0,1]
            "mlr_drive"                  [B, 2]   [speed, heading] → CPG
            "velocity"                   [B, 2]   [vx, vy] → motor neurons

          M2 cortical contributions (arrive at brainstem, NOT at SC):
            "m2_heading_bias"            [B, 1]   M2 → MLR direction refinement
            "m2_pag_bias"                [B, 1]   M2 → PAG urgency modulation

          Behavior:
            "freeze_escape_competition"  [B, 2]   softmax([vlpag, cuneiform])
            "behavior_label"             [B]       0=freeze, 1=escape

          Thalamic loop (sSC → LP → visual cortex):
            "thalamic_feedback"          [B, T, vis_feedback_dim]
        """
        # ── Fast pathway: current frame only, no cortical input ───────────────
        fast_out       = self.fast_pathway(frames[:, -1])
        sc_superficial = fast_out["sc_superficial"]
        sc_deep        = fast_out["sc_deep"]
        sc_heading     = fast_out["sc_heading"]

        # ── Slow pathway: full sequence ───────────────────────────────────────
        m2_seq, _  = self.slow_pathway(frames)       # [B, T, cortical_dim]
        m2_last    = m2_seq[:, -1, :]                # [B, cortical_dim]

        # ── Brainstem: dSC + M2 converge here (not at SC) ────────────────────
        bs_out = self.brainstem_pathways(sc_deep, sc_heading, m2_last)

        # ── Thalamic feedback: sSC → LP → visual cortex ───────────────────────
        feedback = self.thalamic_feedback(sc_superficial, m2_seq)

        return {
            # SC
            "sc_superficial":             sc_superficial,
            "sc_deep":                    sc_deep,
            # Freeze branch
            "lp_thalamus":                bs_out["lp_thalamus"],
            "cea_amygdala":               bs_out["cea_amygdala"],
            "vlpag":                      bs_out["vlpag"],
            # Escape branch
            "pgb_neurons":                bs_out["pgb_neurons"],
            "bla_amygdala":               bs_out["bla_amygdala"],
            "dpag":                       bs_out["dpag"],
            "cuneiform":                  bs_out["cuneiform"],
            "mlr_drive":                  bs_out["mlr_drive"],
            "velocity":                   bs_out["velocity"],
            # M2 brainstem contributions
            "m2_heading_bias":            bs_out["m2_heading_bias"],
            "m2_pag_bias":                bs_out["m2_pag_bias"],
            # Behavior
            "freeze_escape_competition":  bs_out["freeze_escape_competition"],
            "behavior_label":             bs_out["behavior_label"],
            # Thalamic loop
            "thalamic_feedback":          feedback,
        }


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, T, C, H, W = 4, 8, 1, 64, 64

    model = VisualMotorModel()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    dummy = torch.randn(B, T, C, H, W)
    out   = model(dummy)

    groups = {
        "Fast pathway / SC  (no cortical input)": [
            "sc_superficial", "sc_deep"],
        "Freeze branch  dSC → LP → CeA → vlPAG": [
            "lp_thalamus", "cea_amygdala", "vlpag"],
        "Escape branch  dSC → PBG → BLA → dPAG → cuneiform → MLR → CPG": [
            "pgb_neurons", "bla_amygdala", "dpag",
            "cuneiform", "mlr_drive", "velocity"],
        "M2 cortical → brainstem  (NOT via SC)": [
            "m2_heading_bias", "m2_pag_bias"],
        "Behavior selection": [
            "freeze_escape_competition", "behavior_label"],
        "Thalamic feedback  sSC → LP → visual cortex": [
            "thalamic_feedback"],
    }

    for group, keys in groups.items():
        print(f"  {group}")
        for k in keys:
            v = out[k]
            print(f"    {k:<32s}: {tuple(v.shape)}")
        print()

    print("  Per-sample behavior")
    print(f"  {'i':<4} {'label':<8} {'vlPAG':>8} {'cuneiform':>10}"
          f" {'vx':>8} {'vy':>8} {'m2_pag':>8}")
    print("  " + "-" * 60)
    for i in range(B):
        label = BEHAVIOR_LABELS[out["behavior_label"][i].item()]
        print(f"  {i:<4} {label:<8}"
              f" {out['vlpag'][i].item():>8.3f}"
              f" {out['cuneiform'][i].item():>10.3f}"
              f" {out['velocity'][i,0].item():>8.3f}"
              f" {out['velocity'][i,1].item():>8.3f}"
              f" {out['m2_pag_bias'][i].item():>8.3f}")
