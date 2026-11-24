"""Normal-form games used in the study.

Each game is a pair of payoff matrices (A for the row player / player 1,
B for the column player / player 2) plus action labels. Labels are neutral
letters in the prompts so that models cannot pattern-match on game names.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Game:
    key: str
    name: str
    A: np.ndarray            # player 1 payoffs, shape (m, n)
    B: np.ndarray            # player 2 payoffs, shape (m, n)
    row_labels: tuple
    col_labels: tuple
    notes: str = ""

    @property
    def shape(self):
        return self.A.shape

    @property
    def payoff_range(self) -> float:
        lo = min(self.A.min(), self.B.min())
        hi = max(self.A.max(), self.B.max())
        return float(hi - lo)

    def payoffs(self, player: int) -> np.ndarray:
        return self.A if player == 0 else self.B

    def labels(self, player: int) -> tuple:
        return self.row_labels if player == 0 else self.col_labels


CHICKEN = Game(
    key="chicken",
    name="Chicken",
    # actions: 0 = Dare, 1 = Swerve
    A=np.array([[0, 7], [2, 6]], dtype=float),
    B=np.array([[0, 2], [7, 6]], dtype=float),
    row_labels=("X", "Y"),
    col_labels=("X", "Y"),
    notes="X = Dare, Y = Swerve. Best CE beats every Nash equilibrium.",
)

BOS = Game(
    key="bos",
    name="Battle of the Sexes",
    A=np.array([[3, 0], [0, 2]], dtype=float),
    B=np.array([[2, 0], [0, 3]], dtype=float),
    row_labels=("X", "Y"),
    col_labels=("X", "Y"),
    notes="Separates welfare from fairness objectives.",
)

SHAPLEY = Game(
    key="shapley",
    name="Shapley's game",
    A=np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),
    B=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float),
    row_labels=("X", "Y", "Z"),
    col_labels=("X", "Y", "Z"),
    notes="Optimal CE lies outside the convex hull of Nash equilibria.",
)

GAMES = {g.key: g for g in (CHICKEN, BOS, SHAPLEY)}
