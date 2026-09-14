"""
Shared feature extraction for hand gestures.

Imported by extract_landmarks.py (training) and server.py (inference) so the
two can never drift apart. Previously this logic was duplicated, and any edit
to one without the other would silently produce garbage predictions.

WHAT CHANGED AND WHY
--------------------
The earlier version normalized translation and scale but NOT rotation. A hand
tilted 30 degrees produced a completely different feature vector from the same
gesture held upright, so the model had to memorise every orientation it might
see. That is the main reason training accuracy looked fine while live
performance did not.

Three fixes here, in order of impact:

  1. CANONICAL ROTATION FRAME
     An axis system is built from the palm itself — wrist, index knuckle,
     pinky knuckle — and every landmark is expressed in that frame. Hand
     orientation stops affecting the features at all.

  2. HANDEDNESS COLLAPSE
     Left hands are mirrored onto right. 'two' with the left hand and 'two'
     with the right become the same problem, which effectively doubles the
     data behind every class.

  3. RICHER FEATURES
     Pairwise distances between fingertips and knuckles, plus finger curl
     angles. Distances are rotation-invariant by construction and encode
     precisely what separates these gestures — how far apart the fingertips
     are and how bent each finger is.

OUTPUT: 118 features per hand
    63  canonical xyz for 21 landmarks
    45  pairwise distances among 10 key points
    10  finger curl angles (2 per finger)
"""

import numpy as np
from itertools import combinations

# MediaPipe landmark indices
WRIST = 0
INDEX_MCP, PINKY_MCP, MIDDLE_MCP = 5, 17, 9

# Fingertips and knuckles — the points whose spacing defines a gesture
KEY_POINTS = [4, 8, 12, 16, 20,      # thumb -> pinky tips
              0, 5, 9, 13, 17]        # wrist and knuckles
KEY_PAIRS = list(combinations(range(len(KEY_POINTS)), 2))   # 45 pairs

# (mcp, pip, tip) per finger, for curl angles
FINGERS = [
    (1, 2, 4),      # thumb
    (5, 6, 8),      # index
    (9, 10, 12),    # middle
    (13, 14, 16),   # ring
    (17, 18, 20),   # pinky
]

NUM_FEATURES = 63 + len(KEY_PAIRS) + 2 * len(FINGERS)   # 118


def _angle(a, b, c):
    """Angle at vertex b, in radians. 0 = folded back, pi = straight."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def landmarks_to_array(landmarks):
    """MediaPipe landmark list -> (21, 3) float array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def extract(landmarks, handedness="Right"):
    """21 landmarks (+ which hand) -> 118 orientation-independent features.

    handedness: "Left" or "Right" as reported by MediaPipe. Anything else is
    treated as right — a wrong guess costs accuracy but never crashes.
    """
    pts = landmarks_to_array(landmarks)

    # 1. Collapse handedness. Mirroring x turns a left hand into a right one,
    #    so both hands train the same classes instead of competing for capacity.
    if handedness == "Left":
        pts[:, 0] = -pts[:, 0]

    # 2. Move the wrist to the origin
    pts = pts - pts[WRIST]

    # 3. Build an axis system from the palm and rotate into it.
    #    e1 points along the palm toward the index knuckle.
    #    n  is the palm normal (perpendicular to the palm plane).
    #    e2 completes a right-handed orthonormal frame.
    v1 = pts[INDEX_MCP]
    v2 = pts[PINKY_MCP]

    n1 = np.linalg.norm(v1)
    if n1 < 1e-8:
        e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        e1 = v1 / n1

    normal = np.cross(v1, v2)
    nn = np.linalg.norm(normal)
    if nn < 1e-8:
        # Degenerate palm (all three points collinear) — fall back to a fixed
        # axis rather than dividing by zero
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        normal = normal / nn

    e2 = np.cross(normal, e1)
    basis = np.stack([e1, e2, normal])        # rows are the new axes
    pts = pts @ basis.T                       # express every point in that frame

    # 4. Scale by palm size, so near and far hands look identical.
    #    Wrist-to-middle-knuckle is the most stable reference on the hand —
    #    it barely changes as fingers move, unlike fingertip distances.
    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale > 1e-8:
        pts = pts / scale

    # ---- Feature block 1: canonical coordinates ----
    coords = pts.flatten()

    # ---- Feature block 2: pairwise distances ----
    key = pts[KEY_POINTS]
    dists = np.array(
        [np.linalg.norm(key[i] - key[j]) for i, j in KEY_PAIRS],
        dtype=np.float32,
    )

    # ---- Feature block 3: finger curl angles ----
    angles = []
    for mcp, pip, tip in FINGERS:
        angles.append(_angle(pts[WRIST], pts[mcp], pts[pip]))   # knuckle bend
        angles.append(_angle(pts[mcp], pts[pip], pts[tip]))     # finger bend
    angles = np.array(angles, dtype=np.float32)

    return np.concatenate([coords, dists, angles]).astype(np.float32)


def augment(features, rng):
    """Jitter a feature vector for training-set expansion.

    Only noise and scale are applied. The earlier version rotated the raw
    landmarks, which is now pointless — the canonical frame already removes
    rotation, so rotating the input would produce an identical output.
    """
    out = features.copy()
    out *= rng.uniform(0.95, 1.05)              # overall scale jitter
    out += rng.normal(0, 0.02, out.shape)       # per-feature noise
    return out.astype(np.float32)
