"""Experimental design: which calls to make, generated deterministically.

Part B devices: mu(lambda) = (1 - lambda) * base + lambda * nu, where base is a
strictly incentive-compatible CE (public coin over the pure Nash equilibria,
or the welfare CE for Shapley) and nu is a deliberately non-IC device. The
"focal" recommendations are the ones whose deviation gain moves with lambda;
the non-focal recommendations have a constant Delta and are queried at a
single level as anchors.
"""
from __future__ import annotations

import numpy as np

from .games import GAMES
from . import solvers as S

BASE = {
    "chicken": np.array([[0, .5], [.5, 0]]),        # public coin over (X,Y),(Y,X)
    "bos":     np.array([[.5, 0], [0, .5]]),        # public coin over (X,X),(Y,Y)
    "shapley": None,                                # welfare CE (filled lazily)
}
NU = {
    "chicken": np.array([[0, 0], [0, 1.]]),         # always (Y,Y): both tempted to X
    "bos":     np.array([[0, 1.], [0, 0]]),         # always (X,Y): miscoordination
    "shapley": np.eye(3) / 3,                       # diagonal: every rec tempted
}
FOCAL = {   # (player, rec) whose Delta varies with lambda
    "chicken": [(0, 1), (1, 1)],
    "bos":     [(0, 0), (1, 1)],
    "shapley": [(0, 0), (1, 0)],
}
MIN_REC_PROB = 0.05


def base_device(key: str) -> np.ndarray:
    b = BASE[key]
    return S.solve_ce(GAMES[key], "welfare")["mu"] if b is None else b


def lambda_for_delta(key: str, target: float) -> tuple[float, float]:
    """Grid-solve for the lambda whose focal Delta is closest to target."""
    g, b, nu = GAMES[key], base_device(key), NU[key]
    p, a = FOCAL[key][0]
    best = None
    for lam in np.linspace(0, 1, 4001):
        mu = S.mix(b, nu, lam)
        if S.rec_probability(g, mu, p, a) < MIN_REC_PROB:
            continue
        d = S.deviation_gain(g, mu, p, a)["delta"]
        if best is None or abs(d - target) < abs(best[1] - target):
            best = (float(lam), d)
    return best


def part_b_cells(condition: str = "private") -> list[dict]:
    """Cells for one Part B condition ("private" or "public"):
    (game, level, player, rec). Level indices match the pilot's call IDs."""
    from .settings import PART_B as pb
    cells = []
    for key in pb["games"]:
        g = GAMES[key]
        levels = [(t, *lambda_for_delta(key, t)) for t in pb["delta_targets"][key]]
        for li, (target, lam, _) in enumerate(levels):
            mu = S.mix(base_device(key), NU[key], lam)
            for p in (0, 1):
                for a in range(g.shape[p]):
                    if S.rec_probability(g, mu, p, a) < MIN_REC_PROB:
                        continue
                    focal = (p, a) in FOCAL[key]
                    if not focal and li != 0:
                        continue          # constant-Delta anchors are asked once
                    if condition == "public" and not focal:
                        continue          # H3 only needs the tempted recommendations
                    dg = S.deviation_gain(g, mu, p, a)
                    cells.append({"game": key, "condition": condition, "level": li,
                                  "target": target, "lam": lam, "player": p,
                                  "rec": a, "focal": focal,
                                  "delta": dg["delta"], "delta_norm": dg["delta_norm"],
                                  "mu": mu.round(6).tolist()})
    return cells


def part_a_cells() -> list[dict]:
    from .settings import PART_A as pa
    return [{"game": g, "objective": o} for g in pa["games"] for o in pa["objectives"]]