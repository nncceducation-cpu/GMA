"""Unsupervised movement representation learning.

THE THESIS
GMA labels cap the supervised model at the GMA's own accuracy. The real endpoint
is CP, and GMA is only a surrogate for it. If CP-predictive structure exists in
movement patterns that the GMA taxonomy never named, it can only be found by
learning representations from ALL the movement — labelled or not — and then
attaching the (scarce, late-arriving) CP outcome to the representation.

TWO APPROACHES, BOTH IMPLEMENTED

1. MOVEMENT MOTIFS (unsupervised, interpretable, works at small n).
   Cut every recording into short segments, cluster them into a vocabulary of
   recurring movement "motifs", then describe each infant as a DISTRIBUTION over
   motifs. This is close to what a GMA expert actually does — they recognise
   recurring patterns — but the vocabulary is discovered, not prescribed.
   An infant is then a histogram, and CP prediction becomes a small-n problem
   over a compact representation.

2. MASKED POSE AUTOENCODER (self-supervised, needs more data).
   Mask spans of joint trajectories and reconstruct them. The encoder learns
   movement dynamics without any label. Fine-tune / linear-probe on the small
   labelled set. Requires torch; see `MaskedPoseAE`.

THE NON-NEGOTIABLE PART
Unsupervised models have nothing forcing them to be clinical. Left alone, they
learn the camera, the site, the blanket and the infant's size. Unlike supervised
leakage there is no label to expose it. So:

  * SSL runs on NORMALISED pose (L2), never raw video — the camera is already
    divided out.
  * Contrastive augmentations are CAMERA transforms (rescale, rotate, resample,
    jitter), so the model is explicitly trained to treat them as noise.
  * Every representation is then run through `probes.py`, which tries to predict
    the site/camera/subject FROM the embedding. If it can, the representation is
    contaminated and must not be used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.normalise import COCO, DISTAL, GMA_JOINTS

logger = logging.getLogger("neogma.represent")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Movement motifs
# ═══════════════════════════════════════════════════════════════════════════

def segment_pose(xy: np.ndarray, fps: float, seg_seconds: float = 1.5,
                 hop_seconds: float = 0.5,
                 joints: Optional[List[str]] = None) -> np.ndarray:
    """Cut normalised pose into short overlapping segments of joint VELOCITY.

    Velocity, not position: position encodes where the infant happens to lie in
    the frame, which is a nuisance. Velocity of a torso-normalised skeleton is
    close to pure movement.
    """
    joints = joints or GMA_JOINTS
    idx = [COCO[j] for j in joints]
    dt = 1.0 / fps
    vel = np.gradient(xy[:, idx, :], dt, axis=0)      # [T, J, 2]

    win = max(4, int(round(seg_seconds * fps)))
    hop = max(1, int(round(hop_seconds * fps)))
    segs = []
    for s in range(0, max(1, len(vel) - win + 1), hop):
        segs.append(vel[s:s + win].reshape(-1))       # flatten [win*J*2]
    if not segs:
        return np.empty((0, win * len(idx) * 2), dtype=np.float32)
    return np.stack(segs).astype(np.float32)


def _resample_segment(seg: np.ndarray, n_joint: int, target_len: int) -> np.ndarray:
    """Make segments comparable when fps differs (they shouldn't, post-normalise,
    but be defensive)."""
    v = seg.reshape(-1, n_joint * 2)
    if len(v) == target_len:
        return v.reshape(-1)
    t_src = np.linspace(0, 1, len(v))
    t_dst = np.linspace(0, 1, target_len)
    out = np.stack([np.interp(t_dst, t_src, v[:, k]) for k in range(v.shape[1])], axis=1)
    return out.reshape(-1)


@dataclass
class MotifVocabulary:
    kmeans: object
    n_motifs: int
    seg_seconds: float
    hop_seconds: float
    joints: List[str]
    seg_len: int


def build_motif_vocabulary(corpus: List[Dict], n_motifs: int = 32,
                           seg_seconds: float = 1.5, hop_seconds: float = 0.5,
                           joints: Optional[List[str]] = None,
                           seed: int = 0, max_segments: int = 200_000
                           ) -> MotifVocabulary:
    """Discover a vocabulary of movement motifs from the WHOLE corpus.

    Labels are never used. Unlabelled recordings are first-class citizens here —
    in a real cohort they will be the large majority.
    """
    from sklearn.cluster import MiniBatchKMeans

    joints = joints or GMA_JOINTS
    fps0 = corpus[0]["fps"]
    seg_len = max(4, int(round(seg_seconds * fps0)))

    pool = []
    for rec in corpus:
        s = segment_pose(rec["xy"], rec["fps"], seg_seconds, hop_seconds, joints)
        if len(s):
            pool.append(s)
    if not pool:
        raise ValueError("no segments — corpus is empty or recordings too short")
    X = np.concatenate(pool)
    if len(X) > max_segments:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), max_segments, replace=False)]

    # scale-normalise each segment so the vocabulary is about SHAPE of movement,
    # not its overall vigour (vigour is captured separately, and is the thing
    # most confounded by "this baby was just more active today").
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / nrm

    km = MiniBatchKMeans(n_clusters=n_motifs, random_state=seed, n_init=10,
                         batch_size=1024).fit(Xn)
    logger.info("motif vocabulary: %d motifs from %d segments", n_motifs, len(Xn))
    return MotifVocabulary(kmeans=km, n_motifs=n_motifs, seg_seconds=seg_seconds,
                           hop_seconds=hop_seconds, joints=list(joints),
                           seg_len=seg_len)


def motif_profile(rec: Dict, vocab: MotifVocabulary) -> np.ndarray:
    """Represent one recording as a distribution over motifs (+ transitions).

    Returns a vector: [motif histogram | motif-transition entropy | vigour stats].
    This is the unsupervised representation of an infant's movement repertoire.
    """
    S = segment_pose(rec["xy"], rec["fps"], vocab.seg_seconds, vocab.hop_seconds,
                     vocab.joints)
    if len(S) == 0:
        return np.full(vocab.n_motifs + 3, np.nan, dtype=np.float32)

    vig = np.linalg.norm(S, axis=1)                       # movement vigour
    Sn = S / (vig[:, None] + 1e-8)
    lab = vocab.kmeans.predict(Sn)

    hist = np.bincount(lab, minlength=vocab.n_motifs).astype(np.float32)
    hist /= hist.sum()

    # repertoire richness: a fidgety infant should visit MANY motifs and switch
    # between them; a non-fidgety infant is monotonous (few motifs, sticky).
    p = hist[hist > 0]
    rep_entropy = float(-(p * np.log2(p)).sum())
    switch_rate = float(np.mean(lab[1:] != lab[:-1])) if len(lab) > 1 else 0.0

    return np.concatenate([hist,
                           [rep_entropy, switch_rate, float(np.median(vig))]
                           ]).astype(np.float32)


def motif_feature_names(vocab: MotifVocabulary) -> List[str]:
    return ([f"motif_{i:02d}" for i in range(vocab.n_motifs)]
            + ["repertoire_entropy", "motif_switch_rate", "movement_vigour"])


# ═══════════════════════════════════════════════════════════════════════════
# 2. Masked pose autoencoder (self-supervised) — needs torch and more data
# ═══════════════════════════════════════════════════════════════════════════

CAMERA_AUGMENTATIONS = """
Contrastive/masked pretraining MUST use camera transforms as the augmentation
family, so the encoder is explicitly taught that they carry no information:
    * random rescale        (camera distance / zoom)
    * random rotation       (phone orientation)
    * random temporal resample (frame rate)
    * random keypoint jitter / dropout (tracker noise)
    * random temporal crop
It must NOT use augmentations that destroy the clinical signal, e.g. large
temporal shuffling (destroys movement dynamics) or left-right flip (destroys
asymmetry, which is itself a CP sign).
"""


def build_masked_pose_ae(n_joints: int = 8, seq_len: int = 45, d_model: int = 128):
    """Masked pose autoencoder. Torch is imported lazily so this module stays
    importable without it.

    Objective: mask contiguous spans of the joint-velocity sequence and
    reconstruct them. The encoder must learn movement dynamics to succeed, with
    no labels at all. Then linear-probe the frozen encoder on the small labelled
    set (subject-level splits, always).
    """
    import torch
    import torch.nn as nn

    class MaskedPoseAE(nn.Module):
        def __init__(self):
            super().__init__()
            din = n_joints * 2
            self.inp = nn.Linear(din, d_model)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
            enc = nn.TransformerEncoderLayer(d_model, nhead=4,
                                             dim_feedforward=4 * d_model,
                                             batch_first=True, dropout=0.1)
            self.encoder = nn.TransformerEncoder(enc, num_layers=4)
            self.head = nn.Linear(d_model, din)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        def forward(self, x, mask=None):
            # x: [B, T, n_joints*2] velocity
            h = self.inp(x) + self.pos[:, : x.shape[1]]
            if mask is not None:
                h = torch.where(mask.unsqueeze(-1), self.mask_token, h)
            z = self.encoder(h)
            return self.head(z), z

        @torch.no_grad()
        def embed(self, x):
            _, z = self.forward(x)
            return z.mean(dim=1)          # recording-level embedding

    return MaskedPoseAE()
