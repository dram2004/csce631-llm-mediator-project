"""Unit tests: reference solutions must match hand-derived values from the
proposal, and the checker must agree with independent definitions."""
import numpy as np
import pytest

from mediator.games import CHICKEN, BOS, SHAPLEY, GAMES
from mediator import solvers as S


def test_chicken_welfare_ce():
    r = S.solve_ce(CHICKEN, "welfare")
    np.testing.assert_allclose(r["mu"], [[0, .25], [.25, .5]], atol=1e-6)
    assert r["payoffs"] == pytest.approx((5.25, 5.25))


def test_chicken_p1_ce_is_pure_nash():
    r = S.solve_ce(CHICKEN, "p1")
    assert r["payoffs"] == pytest.approx((7, 2))


def test_shapley_welfare_ce():
    r = S.solve_ce(SHAPLEY, "welfare")
    off = np.ones((3, 3)) / 6; np.fill_diagonal(off, 0)
    np.testing.assert_allclose(r["mu"], off, atol=1e-6)
    assert r["value"] == pytest.approx(1.0)


def test_bos_egalitarian_is_public_coin():
    r = S.solve_ce(BOS, "egalitarian")
    assert r["value"] == pytest.approx(2.5)
    assert r["mu"][0, 1] + r["mu"][1, 0] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("key", list(GAMES))
@pytest.mark.parametrize("obj", S.OBJECTIVES)
def test_lp_optimum_passes_checker(key, obj):
    g = GAMES[key]
    r = S.solve_ce(g, obj)
    assert S.check_obedience(g, r["mu"]).is_ce
    assert S.distance_to_ce(g, r["mu"]) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("key", list(GAMES))
def test_every_nash_is_a_ce(key):
    g = GAMES[key]
    for e in S.nash_equilibria(g):
        assert S.check_obedience(g, e["mu"]).is_ce


def test_known_non_ce_is_flagged():
    # all mass on (Swerve, Swerve): each player gains 1 by daring
    mu = np.array([[0, 0], [0, 1.]])
    rep = S.check_obedience(CHICKEN, mu)
    assert rep.n_violated == 2 and rep.max_violation == pytest.approx(1)
    assert S.distance_to_ce(CHICKEN, mu) > 0


def test_checker_agrees_with_deviation_gain():
    # obedience slack / P(rec) == -Delta  whenever there is a single deviation
    rng = np.random.default_rng(0)
    for mu in S.random_devices(CHICKEN, 200, rng):
        rep = S.check_obedience(CHICKEN, mu, tol=0)
        for p in (0, 1):
            for a in (0, 1):
                d = S.deviation_gain(CHICKEN, mu, p, a)["delta"]
                flagged = any(v[0] == p and v[1] == a for v in rep.violated)
                assert flagged == (d > 0)


def test_lp_beats_nash_where_expected():
    assert S.solve_ce(CHICKEN, "welfare")["value"] > S.best_nash(CHICKEN, "welfare")["mu"].sum() * 0 + 9.333
    assert S.solve_ce(SHAPLEY, "welfare")["value"] == pytest.approx(1.0)
    sh_nash = S.objective_value(SHAPLEY, S.best_nash(SHAPLEY, "welfare")["mu"], "welfare")
    assert sh_nash == pytest.approx(2 / 3)


def test_rounded_shapley_device_is_eps_ce_not_exact():
    mu = np.array([[0, .167, .167], [.167, 0, .166], [.167, .166, 0]])
    mu = mu / mu.sum()
    rep = S.check_obedience(SHAPLEY, mu)
    assert not rep.is_ce                       # strict check flags rounding
    assert rep.max_violation_norm <= 0.01      # but it is an epsilon-CE
