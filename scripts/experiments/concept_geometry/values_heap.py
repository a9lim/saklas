"""Schwartz's 10 basic human values, in their canonical circumplex order.

Schwartz's theory (validated across ~80 cultures) arranges the 10 values in a
RING: adjacent values are motivationally compatible, opposite values conflict.
The two principal axes of the circle are openness-to-change vs conservation and
self-enhancement vs self-transcendence. So if the model encodes the circumplex,
the value centroids should recover a 2-D ring in THIS cyclic order.

LABELS are listed in the canonical circular order, so Schwartz adjacency is just
(i, i+1 mod 10) and the recovery test is "does the activation ring reproduce this
cyclic sequence (up to rotation/reflection)?"
"""

from __future__ import annotations

# canonical circumplex order (Schwartz 1992); the ring closes universalism->self-direction
LABELS = [
    "self-direction",
    "stimulation",
    "hedonism",
    "achievement",
    "power",
    "security",
    "conformity",
    "tradition",
    "benevolence",
    "universalism",
]

# short Schwartz definitions used as the elicitation concept (the {c} slot)
DESCRIPTORS = {
    "self-direction": "independent thought, freedom, and choosing your own path",
    "stimulation":    "excitement, novelty, and seeking out challenges",
    "hedonism":       "pleasure and gratifying the senses",
    "achievement":    "personal success through demonstrating competence",
    "power":          "social status, control, and dominance over people and resources",
    "security":       "safety, stability, and social and personal harmony",
    "conformity":     "restraint, politeness, and obeying rules and expectations",
    "tradition":      "respect for custom, religion, and the ways of the past",
    "benevolence":    "loyalty and devotion to the welfare of those close to you",
    "universalism":   "understanding, tolerance, and protection for all people and nature",
}

# concepts passed to generation, aligned to LABELS
CONCEPTS = [DESCRIPTORS[v] for v in LABELS]

# custom-kind system: {c} is the descriptor. No role swap (pooled standard-assistant).
SYSTEM = (
    "You are a person whose single most important life value is {c}. "
    "It guides every choice you make and how you see the world. "
    "Respond exactly as that person would."
)


def schwartz_adjacency():
    """The 10 canonical adjacent pairs (the ring edges), as a set of frozensets."""
    n = len(LABELS)
    return {frozenset((i, (i + 1) % n)) for i in range(n)}


def ideal_angles():
    """Schwartz ideal ring angles φ_i = 2π i / 10."""
    import numpy as np
    n = len(LABELS)
    return np.array([2 * np.pi * i / n for i in range(n)])
