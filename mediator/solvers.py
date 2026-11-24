"""Exact game-theoretic references and the automated obedience checker.

Everything an LLM produces is scored by functions in this module; nothing is
checked by hand.

Notation: a device mu is an (m, n) array of probabilities over joint profiles.
Obedience (incentive-compatibility) constraint for player i, recommendation a_i,
deviation a_i':
    sum_{a_-i} mu(a_i, a_-i) [u_i(a_i, a_-i) - u_i(a_i', a_-i)] >= 0
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .games import Game

OBJECTIVES = ("welfare", "egalitarian", "p1")
TOL = 1e-7


# ---------------------------------------------------------------------------
# Obedience constraints
# ---------------------------------------------------------------------------
def obedience_rows(game: Game):
    """Yield (player, rec, dev, coeff) with coeff an (m, n) array such that
    the constraint is  sum(coeff * mu) >= 0."""
    m, n = game.shape
    A, B = game.A, game.B
    for i in range(m):
        for k in range(m):
            if k == i:
                continue
            c = np.zeros((m, n))
            c[i, :] = A[i, :] - A[k, :]
            yield 0, i, k, c
    for j in range(n):
        for l in range(n):
            if l == j:
                continue
            c = np.zeros((m, n))
            c[:, j] = B[:, j] - B[:, l]
            yield 1, j, l, c


@dataclass
class ObedienceReport:
    n_constraints: int
    n_violated: int
    max_violation: float            # raw, in payoff units (>= 0)
    max_violation_norm: float       # divided by game payoff range
    violated: list                  # [(player, rec, dev, slack)]

    @property
    def is_ce(self) -> bool:
        return self.n_violated == 0


def check_obedience(game: Game, mu: np.ndarray, tol: float = 1e-6) -> ObedienceReport:
    mu = np.asarray(mu, dtype=float)
    viol = []
    worst = 0.0
    rows = list(obedience_rows(game))
    for p, a, d, c in rows:
        slack = float((c * mu).sum())
        if slack < -tol:
            viol.append((p, a, d, slack))
            worst = max(worst, -slack)
    return ObedienceReport(
        n_constraints=len(rows),
        n_violated=len(viol),
        max_violation=worst,
        max_violation_norm=worst / game.payoff_range,
        violated=viol,
    )


# ---------------------------------------------------------------------------
# CE linear programs
# ---------------------------------------------------------------------------
def _ce_constraint_matrix(game: Game):
    m, n = game.shape
    G = [-c.flatten() for _, _, _, c in obedience_rows(game)]
    return np.array(G), np.zeros(len(G))


def solve_ce(game: Game, objective: str) -> dict:
    """Optimal correlated equilibrium for 'welfare', 'egalitarian'
    (max-min expected payoff) or 'p1' (player 1's payoff)."""
    m, n = game.shape
    N = m * n
    G, h = _ce_constraint_matrix(game)
    A, B = game.A.flatten(), game.B.flatten()

    if objective in ("welfare", "p1"):
        w = A + B if objective == "welfare" else A
        res = linprog(-w, A_ub=G, b_ub=h, A_eq=[np.ones(N)], b_eq=[1],
                      bounds=[(0, 1)] * N, method="highs")
        x = res.x
    elif objective == "egalitarian":
        # variables [mu (N), t]; maximise t s.t. t <= E[u_i]
        c = np.zeros(N + 1); c[-1] = -1
        G2 = np.hstack([G, np.zeros((G.shape[0], 1))])
        rows = [np.append(-A, 1), np.append(-B, 1)]
        G2 = np.vstack([G2, rows]); h2 = np.append(h, [0, 0])
        Aeq = [np.append(np.ones(N), 0)]
        res = linprog(c, A_ub=G2, b_ub=h2, A_eq=Aeq, b_eq=[1],
                      bounds=[(0, 1)] * N + [(None, None)], method="highs")
        x = res.x[:N]
    else:
        raise ValueError(objective)
    if not res.success:
        raise RuntimeError(f"CE LP failed: {res.message}")
    mu = np.clip(x, 0, None).reshape(m, n)
    mu /= mu.sum()
    return {"mu": mu, "value": objective_value(game, mu, objective),
            "payoffs": expected_payoffs(game, mu)}


def expected_payoffs(game: Game, mu: np.ndarray) -> tuple:
    return float((mu * game.A).sum()), float((mu * game.B).sum())


def objective_value(game: Game, mu: np.ndarray, objective: str) -> float:
    u1, u2 = expected_payoffs(game, mu)
    return {"welfare": u1 + u2, "egalitarian": min(u1, u2), "p1": u1}[objective]


def distance_to_ce(game: Game, mu: np.ndarray) -> float:
    """L1 distance from mu to the nearest correlated equilibrium (an LP)."""
    m, n = game.shape
    N = m * n
    G, h = _ce_constraint_matrix(game)
    target = np.asarray(mu, float).flatten()
    # variables [x (N), s (N)]  min sum s,  s >= |x - target|
    c = np.concatenate([np.zeros(N), np.ones(N)])
    I = np.eye(N)
    A_ub = np.vstack([
        np.hstack([G, np.zeros_like(G)]),
        np.hstack([I, -I]),
        np.hstack([-I, -I]),
    ])
    b_ub = np.concatenate([h, target, -target])
    A_eq = [np.concatenate([np.ones(N), np.zeros(N)])]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1],
                  bounds=[(0, 1)] * N + [(0, None)] * N, method="highs")
    return float(res.fun)


# ---------------------------------------------------------------------------
# Nash equilibria (no-mediator benchmark)
# ---------------------------------------------------------------------------
def nash_equilibria(game: Game) -> list[dict]:
    import nashpy as nash
    g = nash.Game(game.A, game.B)
    out = []
    for s1, s2 in g.support_enumeration():
        mu = np.outer(s1, s2)
        out.append({"s1": s1, "s2": s2, "mu": mu,
                    "payoffs": expected_payoffs(game, mu)})
    return out


def best_nash(game: Game, objective: str) -> dict:
    return max(nash_equilibria(game),
               key=lambda e: objective_value(game, e["mu"], objective))


def public_nash_mix(game: Game, objective: str) -> dict:
    """Best public randomisation over Nash equilibria: the convex hull of NE
    payoffs. For a linear objective (welfare, p1) it is the best single NE;
    for egalitarian it can mix two NE."""
    eqs = nash_equilibria(game)
    best = best_nash(game, objective)
    if objective != "egalitarian":
        return best
    best_val = objective_value(game, best["mu"], objective)
    for e1, e2 in itertools.combinations(eqs, 2):
        for w in np.linspace(0, 1, 201):
            mu = w * e1["mu"] + (1 - w) * e2["mu"]
            v = objective_value(game, mu, objective)
            if v > best_val + 1e-9:
                best, best_val = {"mu": mu, "payoffs": expected_payoffs(game, mu),
                                  "mix": (w, 1 - w)}, v
    return best


# ---------------------------------------------------------------------------
# Player-side temptation
# ---------------------------------------------------------------------------
def rec_probability(game: Game, mu: np.ndarray, player: int, rec: int) -> float:
    return float(mu[rec, :].sum() if player == 0 else mu[:, rec].sum())


def conditional_opponent(mu: np.ndarray, player: int, rec: int) -> np.ndarray:
    """Posterior over the opponent's recommended action given own rec."""
    row = mu[rec, :] if player == 0 else mu[:, rec]
    return row / row.sum()


def deviation_gain(game: Game, mu: np.ndarray, player: int, rec: int) -> dict:
    """Delta_i(a_i) = max_{a'} E[u_i(a', a_-i) - u_i(a_i, a_-i) | a_i].

    Delta > 0 means a rational expected-utility maximiser should deviate.
    """
    post = conditional_opponent(mu, player, rec)
    U = game.payoffs(player)
    # expected payoff of each own action given the posterior
    ev = U @ post if player == 0 else U.T @ post
    alts = [a for a in range(len(ev)) if a != rec]
    best_alt = max(alts, key=lambda a: ev[a])
    return {"delta": float(ev[best_alt] - ev[rec]),
            "delta_norm": float((ev[best_alt] - ev[rec]) / game.payoff_range),
            "best_response": int(np.argmax(ev)),
            "ev": ev.tolist(), "posterior": post.tolist()}


def mix(mu_a: np.ndarray, mu_b: np.ndarray, lam: float) -> np.ndarray:
    return (1 - lam) * mu_a + lam * mu_b


def random_devices(game: Game, k: int, rng: np.random.Generator) -> np.ndarray:
    m, n = game.shape
    return rng.dirichlet(np.ones(m * n), size=k).reshape(k, m, n)
