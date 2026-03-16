"""
Trust matrix helpers for CSQA experiments.
"""

from __future__ import annotations

import os
import random
from typing import Sequence

import numpy as np


def load_trust_matrix(path: str, n: int) -> np.ndarray:
    """Load an NxN trust matrix from CSV."""
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: --trust-matrix file not found: {path}")
    try:
        matrix = np.loadtxt(path, delimiter=",", dtype=float)
    except Exception as exc:
        raise SystemExit(f"ERROR: failed to read --trust-matrix as numeric CSV: {exc}")
    if matrix.ndim != 2 or matrix.shape != (n, n):
        raise SystemExit(f"ERROR: --trust-matrix must be {n}x{n}; got {matrix.shape}")
    return matrix


def build_speaker_trust(n: int, scheme: str, explicit_csv: str, sample_seed: int) -> np.ndarray:
    """Construct a speaker trust vector according to CLI options."""
    if explicit_csv.strip():
        parts = [x.strip() for x in explicit_csv.split(",")]
        if len(parts) != n:
            raise SystemExit(
                f"ERROR: --speaker-trust must have exactly {n} comma-separated weights."
            )
        try:
            weights = np.array([float(x) for x in parts], dtype=float)
        except ValueError:
            raise SystemExit("ERROR: --speaker-trust must be numeric CSV.")
        return weights

    if scheme == "uniform":
        return np.ones(n, dtype=float)
    if scheme == "hub_high":
        weights = np.ones(n, dtype=float)
        if n > 0:
            weights[0] = 2.0
        return weights
    if scheme == "hub_low":
        weights = np.ones(n, dtype=float)
        if n > 0:
            weights[0] = 0.5
        return weights
    if scheme == "random":
        rng = random.Random(sample_seed)
        return np.array([round(rng.uniform(0.5, 1.5), 3) for _ in range(n)], dtype=float)
    return np.ones(n, dtype=float)


def speaker_vector_to_matrix(w: np.ndarray) -> np.ndarray:
    """Tile a speaker trust vector into a matrix (same weights for each listener)."""
    n = w.shape[0]
    return np.tile(w.reshape(1, n), (n, 1))


__all__ = ["load_trust_matrix", "build_speaker_trust", "speaker_vector_to_matrix"]
