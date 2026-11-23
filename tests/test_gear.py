"""Analytic gear selection: exactness, singular covariances, no enumeration."""
import math
import random
import time
from fractions import Fraction

import pytest

from app.gear import (MAHALANOBIS_LIMIT, OperatingPoint, review_gear,
                      select_gear)
from app.model import RidgeModel, train
from app.stats import Stats


def stats_from_rows(rows):
    n = len(rows)
    sums = [sum(r[i] for r in rows) for i in range(3)]
    outer = [[sum(r[i] * r[j] for r in rows) for j in range(3)]
             for i in range(3)]
    return Stats.build(n, sums, outer)


def make_rows(n, speed_mu, press_mu, coef, intercept, noise, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = int(round(rng.gauss(speed_mu, 25)))
        p = int(round(rng.gauss(press_mu, 12)))
        y = coef[0] * s + coef[1] * p + intercept + rng.gauss(0, noise)
        rows.append((s, p, int(round(y))))
    return rows


@pytest.fixture(scope="module")
def trained():
    factories = [
        make_rows(300, 1450 + 5 * i, 120, (1.8, 0.6), 40, 5, seed=10 + i)
        for i in range(4)
    ]
    stats = [stats_from_rows(r) for r in factories]
    result = train(stats)
    return OperatingPoint.build(stats[0], 0), result.model


def feasible(model, op, speed, p_lo, p_hi, power_limit):
    for p in (p_lo, p_hi):
        if model.predict(speed, p) > power_limit + 1e-9:
            return False
        d = op.distance_sq(speed, p)
        if d is None or d > MAHALANOBIS_LIMIT ** 2:
            return False
    return True


def brute_best(model, op, p_lo, p_hi, speed_min, speed_max, power_limit):
    best = None
    for speed in range(speed_min, speed_max + 1):
        if feasible(model, op, speed, p_lo, p_hi, power_limit):
            best = speed
    return best


def test_selection_matches_bruteforce(trained):
    op, model = trained
    rng = random.Random(1)
    for _ in range(400):
        p_lo = rng.randint(90, 150)
        p_hi = p_lo + rng.randint(0, 60)
        speed_min = rng.randint(1300, 1500)
        speed_max = speed_min + rng.randint(0, 300)
        limit = rng.uniform(3200, 4200)
        got = select_gear(op, model, p_lo, p_hi, speed_min, speed_max, limit)
        want = brute_best(model, op, p_lo, p_hi, speed_min, speed_max, limit)
        assert (got.gear if got.available else None) == want
        if got.available:
            assert feasible(model, op, got.gear, p_lo, p_hi, limit)
            if got.gear < speed_max:
                assert not feasible(model, op, got.gear + 1,
                                    p_lo, p_hi, limit)


def test_huge_speed_range_is_solved_analytically(trained):
    op, model = trained
    started = time.monotonic()
    got = select_gear(op, model, 100, 140, 0, 10**15, 1e9)
    elapsed = time.monotonic() - started
    assert got.available
    assert 0 <= got.gear < 10**15
    assert elapsed < 1.0  # a per-gear loop could never finish this range


def test_no_usable_gear(trained):
    op, model = trained
    assert not select_gear(op, model, 100, 140, 1300, 1600, 1).available
    # pressure far outside the operating range: no speed can fix the distance
    assert not select_gear(op, model, 10**6, 10**6, 0, 10**6, 1e12).available
    # speed range entirely to the left of the feasible interval
    assert not select_gear(op, model, 100, 140, 0, 1000, 1e12).available


def test_speed_range_is_respected(trained):
    op, model = trained
    got = select_gear(op, model, 100, 140, 1300, 1508, 1e9)
    assert got.available and got.gear == 1508


def test_interval_endpoints_are_the_worst_case(trained):
    op, model = trained
    got = select_gear(op, model, 100, 140, 1300, 1600, 4000)
    assert got.available
    assert got.max_power == pytest.approx(
        max(model.predict(got.gear, 100), model.predict(got.gear, 140))
    )
    assert got.max_distance == pytest.approx(
        max(op.distance(got.gear, 100), op.distance(got.gear, 140))
    )


def test_zero_pressure_variance_must_match_mean():
    # pressure constant -> the pressure direction has zero variance
    rows = [(10 + i, 7, 3) for i in range(20)]
    op = OperatingPoint.build(stats_from_rows(rows), 0)
    model = RidgeModel(mean=[19.5, 7.0, 3.0], std={}, weights=[],
                       intercept=3.0, kept=[])
    assert op.det == 0
    assert op.distance_sq(20, 7) is not None
    assert op.distance_sq(20, 8) is None
    ok = select_gear(op, model, 7, 7, 0, 100, 1e9)
    assert ok.available and ok.gear == 36  # mean 19.5 + 3 * sqrt(30)
    assert not select_gear(op, model, 6, 8, 0, 100, 1e9).available


def test_zero_speed_variance_must_match_mean():
    # speed constant -> only the mean speed may be used
    rows = [(5, 3 + i, 2) for i in range(20)]
    op = OperatingPoint.build(stats_from_rows(rows), 0)
    model = RidgeModel(mean=[5.0, 12.5, 2.0], std={}, weights=[],
                       intercept=2.0, kept=[])
    assert op.det == 0
    assert op.distance_sq(5, 12) is not None
    assert op.distance_sq(6, 12) is None
    got = select_gear(op, model, 5, 15, 0, 100, 1e9)
    assert got.available and got.gear == 5


def test_zero_variance_in_both_directions_requires_the_mean():
    rows = [(5, 7, 3)] * 20
    op = OperatingPoint.build(stats_from_rows(rows), 0)
    model = RidgeModel(mean=[5.0, 7.0, 3.0], std={}, weights=[],
                       intercept=3.0, kept=[])
    assert (op.a, op.b, op.c, op.det) == (0, 0, 0, 0)
    assert op.distance_sq(5, 7) == 0
    assert op.distance_sq(6, 7) is None
    got = select_gear(op, model, 7, 7, 0, 100, 1e9)
    assert got.available and got.gear == 5
    assert not select_gear(op, model, 7, 8, 0, 100, 1e9).available


def test_rank_one_covariance_uses_pseudoinverse():
    # all points lie on p = 2s -> singular covariance, zero-variance
    # direction (2, -1) must be matched exactly.
    rows = [(s, 2 * s, 5) for s in range(10, 20)]
    op = OperatingPoint.build(stats_from_rows(rows), 0)
    model = RidgeModel(mean=[14.5, 29.0, 5.0], std={}, weights=[],
                       intercept=5.0, kept=[])
    assert op.det == 0 and op.b != 0
    assert op.distance_sq(15, 30) == Fraction(1, 33)
    assert op.distance_sq(15, 31) is None  # off the line
    assert op.distance_sq(100, 200) is not None
    # pressure interval 28..30 includes off-line points -> no gear at all
    assert not select_gear(op, model, 28, 30, 0, 100, 1e9).available
    # a single on-line pressure admits gears
    assert select_gear(op, model, 30, 30, 10, 20, 1e9).available


def test_huge_base_values_exact():
    # speed around 1e12 with a spread of 1: exact integer distance survives
    rows = [(10**12 + i, 5 * 10**11 + 2 * i, 7) for i in range(100)]
    op = OperatingPoint.build(stats_from_rows(rows), 0)
    # rank 1: every reading lies on p = 5e11 + 2*(s - 1e12)
    p_of = lambda s: 5 * 10**11 + 2 * (s - 10**12)
    mean_s = 10**12 + 49.5
    sd = math.sqrt(float(Fraction(op.a, op.n ** 2)))
    far = int(round(mean_s + 4 * sd))
    d2 = op.distance_sq(far, p_of(far))
    assert d2 is not None and d2 > 9
    assert op.distance_sq(far, p_of(far) + 1) is None  # off the line
    assert op.distance_sq(10**12 + 49, p_of(10**12 + 49)) is not None


def test_exact_inclusive_distance_boundary():
    # speed 97..103 -> mean 100, population sd 2; pressure constant 50.
    # The distance limit (3) is reached exactly at speed 100 + 3*2 = 106,
    # which must be accepted; 107 must not.
    rows = [(s, 50, 10) for s in range(97, 104)]
    op = OperatingPoint.build(stats_from_rows(rows), 0)
    model = RidgeModel(mean=[100.0, 50.0, 10.0], std={}, weights=[],
                       intercept=10.0, kept=[])
    assert op.distance_sq(106, 50) == 9
    assert op.distance_sq(107, 50) is not None
    got = select_gear(op, model, 50, 50, 90, 200, 1e9)
    assert got.available and got.gear == 106
    assert got.max_distance == pytest.approx(3.0)


def test_conditions_hold_at_interior_pressures(trained):
    # The solver only inspects the endpoints.  Power is linear and the
    # squared distance is convex in pressure (exact Fraction arithmetic
    # below), so both must be bounded by their endpoint values inside.
    op, model = trained
    rng = random.Random(7)
    checked = 0
    for _ in range(20):
        p_lo = rng.randint(90, 150)
        p_hi = p_lo + rng.randint(0, 60)
        limit = rng.uniform(3500, 4200)
        got = select_gear(op, model, p_lo, p_hi, 1200, 1700, limit)
        if not got.available:
            continue
        checked += 1
        for k in range(201):
            p = p_lo + Fraction(p_hi - p_lo) * k / 200
            assert model.predict(got.gear, float(p)) <= limit + 1e-9
            d2 = _distance_sq_fraction(op, got.gear, p)
            assert d2 is None or d2 <= 9
    assert checked > 0


def _distance_sq_fraction(op, speed, pressure):
    """Distance at a rational (possibly non-integer) pressure."""
    mean_s = Fraction(op.sx, op.n)
    mean_p = Fraction(op.sy, op.n)
    ds = Fraction(speed) - mean_s
    dp = pressure - mean_p
    cov = [[Fraction(op.a, op.n ** 2), Fraction(op.b, op.n ** 2)],
           [Fraction(op.b, op.n ** 2), Fraction(op.c, op.n ** 2)]]
    det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0]
    if det == 0:
        return None  # not exercised here (test data is non-singular)
    inv = [[cov[1][1] / det, -cov[0][1] / det],
           [-cov[1][0] / det, cov[0][0] / det]]
    return (ds * (inv[0][0] * ds + inv[0][1] * dp)
            + dp * (inv[1][0] * ds + inv[1][1] * dp))


def test_review_accepts_usable_and_reports_reasons(trained):
    op, model = trained
    got = select_gear(op, model, 100, 140, 1300, 1600, 1e9)
    review = review_gear(op, model, got.gear, 100, 140, 1300, 1600, 1e9)
    assert review.accepted and review.reasons == ()

    low = review_gear(op, model, got.gear, 100, 140, 1300, 1600, 1)
    assert not low.accepted and "power_limit" in low.reasons

    outside = review_gear(op, model, 1000, 100, 140, 1300, 1600, 1e9)
    assert not outside.accepted and "speed_range" in outside.reasons

    # far-away pressure: even the mean gear fails the distance test
    far = review_gear(op, model, 1450, 100000, 100010, 1300, 1600, 1e12)
    assert not far.accepted and "mahalanobis" in far.reasons
