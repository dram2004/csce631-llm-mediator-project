"""Prompt builders. Games use neutral action letters and no game names so that
models must reason from payoffs rather than recall a textbook answer."""
from __future__ import annotations

import numpy as np

from .games import Game

SYSTEM = ("You are a careful decision-maker in a game-theory experiment. "
          "Follow the output format exactly.")

OBJECTIVE_TEXT = {
    "welfare": "Maximise the SUM of the two players' expected payoffs.",
    "egalitarian": "Maximise the expected payoff of whichever player is worse off "
                   "(make the smaller of the two expected payoffs as large as possible).",
    "p1": "Maximise Player 1's expected payoff.",
}


def nonce_line(nonce: str) -> str:
    """The gateway returns cached replies to byte-identical requests, so every
    call carries a unique request ID that does not affect the task."""
    return f"Request ID: {nonce} (a unique label for this request; it has no bearing on the task)."


def matrix_text(game: Game) -> str:
    m, n = game.shape
    head = "        " + "  ".join(f"P2={c:<9}" for c in game.col_labels)
    rows = [head]
    for i in range(m):
        cells = "  ".join(f"({game.A[i, j]:g}, {game.B[i, j]:g})".ljust(12) for j in range(n))
        rows.append(f"P1={game.row_labels[i]:<4}" + cells)
    return ("Payoff matrix. Player 1 (P1) chooses a row, Player 2 (P2) chooses a column. "
            "Each cell is (P1 payoff, P2 payoff).\n" + "\n".join(rows))


def profile_keys(game: Game) -> list[str]:
    return [r + c for r in game.row_labels for c in game.col_labels]


def device_text(game: Game, mu: np.ndarray) -> str:
    lines = []
    for i, r in enumerate(game.row_labels):
        for j, c in enumerate(game.col_labels):
            lines.append(f"  P1={r}, P2={c}: {mu[i, j]:.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------- Part A
def mediator_messages(game: Game, objective: str, nonce: str) -> list[dict]:
    keys = profile_keys(game)
    example = ", ".join(f'"{k}": <prob>' for k in keys)
    user = f"""{nonce_line(nonce)}

You are a MEDIATOR for a two-player game.

{matrix_text(game)}

How mediation works: you choose a probability distribution over joint action
profiles (a "device"). Before play, one profile is drawn from your device, and
each player is told PRIVATELY only their own recommended action. Both players
know the device you chose. Each player is free to ignore the recommendation.
A device only works if, for every recommendation a player can receive,
following it gives that player at least as high an expected payoff as any
other action, given what the recommendation reveals about the other player's
recommendation.

Your goal: {OBJECTIVE_TEXT[objective]}
Subject to that, the device must be one that both players would always want to follow.

Reply with a single JSON object and nothing else:
{{"device": {{{example}}}, "reason": "<at most 40 words>"}}
Keys "{keys[0]}" etc. mean P1's action followed by P2's action. Probabilities
must be non-negative and sum to 1."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_device(game: Game):
    keys = profile_keys(game)
    m, n = game.shape

    def parse(text: str) -> dict:
        from .llm import extract_json  # noqa: E402
        js = extract_json(text)
        d = js["device"]
        vals = np.array([float(d[k]) for k in keys])
        if np.any(vals < -1e-9):
            raise ValueError("negative probability")
        s = vals.sum()
        return {"device_raw": vals.tolist(), "sum": float(s),
                "sums_to_one": bool(abs(s - 1) <= 0.01),
                "reason": str(js.get("reason", ""))[:400]}
    return parse


# ---------------------------------------------------------------- Part B
def player_messages(game: Game, mu: np.ndarray | None, player: int, rec: int | None,
                    condition: str, opp_rec: int | None, nonce: str) -> list[dict]:
    me, other = ("Player 1 (P1)", "Player 2 (P2)") if player == 0 else ("Player 2 (P2)", "Player 1 (P1)")
    labels = game.labels(player)
    opts = " or ".join(f'"{a}"' for a in labels)
    if condition == "none":
        body = (f"You are {me}. You and {other} choose actions simultaneously and "
                f"without communicating. Choose the action that is best for you.")
    else:
        rec_txt = labels[rec]
        body = (f"You are {me}. A mediator uses the device below, which both players know. "
                f"One joint profile has been drawn from it.\n\nDevice (probability of each profile):\n"
                f"{device_text(game, mu)}\n\n")
        if condition == "private":
            body += (f"The mediator has told you, privately, to play {rec_txt}. "
                     f"{other.split(' (')[0]} has received their own private recommendation, "
                     f"which you do not see.")
        else:
            olab = game.labels(1 - player)[opp_rec]
            body += (f"The mediator has announced publicly: {me.split(' (')[0]} should play {rec_txt} "
                     f"and {other.split(' (')[0]} should play {olab}. Both players saw this announcement.")
        body += (" You are free to follow or ignore the recommendation. "
                 "Choose the action that maximises your own expected payoff.")
    user = f"""{nonce_line(nonce)}

{matrix_text(game)}

{body}

Reply with a single JSON object and nothing else:
{{"action": {opts}, "reason": "<at most 30 words>"}}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_action(game: Game, player: int):
    labels = game.labels(player)

    def parse(text: str) -> dict:
        from .llm import extract_json  # noqa: E402
        js = extract_json(text)
        a = str(js["action"]).strip().strip('"').upper()
        if a not in labels:
            raise ValueError(f"invalid action {a!r}")
        return {"action": labels.index(a), "reason": str(js.get("reason", ""))[:300]}
    return parse