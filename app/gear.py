"""Analytic selection and review of the highest usable integer gear (speed).

For a factory, a corrected pressure interval ``[p_lo, p_hi]``, an integer
speed range and a per-minute power limit, a gear ``s`` is *usable* when

1. ``model.predict(s, p) <= power_limit`` for every ``p`` in the interval;
2. the squared Mahalanobis distance from ``(s, p)`` to the factory's own
   operating mean ``(mean_speed, mean_pressure)`` is at most
   ``MAHALANOBIS_LIMIT ** 2`` for every ``p`` in the interval.

The mean and the *population* covariance are built from the factory's
speed and corrected-pressure sufficient statistics.  Zero-variance
directions follow the pseudo-inverse limit: a component in such a
direction must equal the mean exactly, otherwise the distance is
infinite.

Both constraints are well behaved in ``p``: predicted power is linear and
the squared Mahalanobis distance is convex (pseudo-inverse included), so
only the two interval endpoints matter.  At a fixed endpoint the usable
integer speeds form one interval: the distance bound is solved as an
exact integer quadratic inequality and the power bound as an exact
rational linear inequality.  Intersecting the two endpoint intervals and
the requested speed range leaves one interval; its right end is the
highest usable gear.  No gear is ever tried one by one.

All distance arithmetic is done with Python integers (the covariance
numerator is exact), so huge reading base values and small variances are
handled without cancellation.  Power is treated as an exact rational
linear function of the stored model parameters.  Decisions therefore use
exact arithmetic; the reported ``max_power`` / ``max_distance`` are the
same quantities rounded to ``float`` for JSON.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from .model import RidgeModel
from .stats import Stats, apply_correction, cov_numerator

#: Inclusive limit on the Mahalanobis distance.
MAHALANOBIS_LIMIT = 3
#: Its square; the integer tests compare against this.
MAHALANOBIS_LIMIT_SQ = MAHALANOBIS_LIMIT * MAHALANOBIS_LIMIT

#: Largest finite float; reported values are clamped here so JSON stays valid.
_FLOAT_MAX = 1.7976931348623157e308


def _finite_float(value: Fraction) -> float:
    """float(value) clamped to the finite float range.

    Decisions never use this -- they compare exact Fractions -- it only
    keeps the reported ``max_power`` / ``max_distance`` JSON-encodable for
    pathological inputs (e.g. extreme summary statistics).
    """
    try:
        return float(value)
    except OverflowError:
        return _FLOAT_MAX if value > 0 else -_FLOAT_MAX


def _distance_float(distance_sq: Fraction) -> float:
    """Reported distance = sqrt of an exact squared distance, no overflow."""
    try:
        return math.sqrt(float(distance_sq))
    except OverflowError:
        return _FLOAT_MAX


# --------------------------------------------------------------- utilities
def _ceil_div(a: int, b: int) -> int:
    """ceil(a / b) for integers and b > 0."""
    return -((-a) // b)


def _floor_div(a: int, b: int) -> int:
    """floor(a / b) for integers and b > 0."""
    return a // b


def _ceil_frac(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _floor_frac(value: Fraction) -> int:
    return value.numerator // value.denominator


# ------------------------------------------------------- factory operating
@dataclass(frozen=True)
class OperatingPoint:
    """Exact 2-D operating statistics of one factory after correction.

    ``a, b, c`` are the entries of the *covariance numerator* C:
        C = [[a, b], [b, c]] = n * sum(x x^T) - sum(x) sum(x)^T
    for x = (speed, corrected pressure).  The population covariance is
    C / n^2.  ``sx``/``sy`` are the corrected integer sums; means are
    sx / n and sy / n.
    """

    n: int
    sx: int
    sy: int
    a: int
    b: int
    c: int
    det: int

    @staticmethod
    def build(stats: Stats, correction: int) -> "OperatingPoint":
        corrected = apply_correction(stats, correction)
        cm = cov_numerator(corrected)
        a, b, c = cm[0][0], cm[0][1], cm[1][1]
        return OperatingPoint(
            n=corrected.n,
            sx=corrected.sums[0],
            sy=corrected.sums[1],
            a=a,
            b=b,
            c=c,
            det=a * c - b * b,
        )

    @property
    def trace(self) -> int:
        return self.a + self.c

    def deviations(self, speed: int, pressure: int) -> tuple[int, int]:
        """Integer-scaled deviations (A, B) = (n*(s-mean_s), n*(p-mean_p))."""
        return self.n * speed - self.sx, self.n * pressure - self.sy

    # -------------------------------------------------- exact distance
    def distance_sq(self, speed: int, pressure: int) -> Optional[Fraction]:
        """Squared Mahalanobis distance, or None when it is infinite.

        For a positive-definite covariance this is delta' Sigma^-1 delta;
        for a singular covariance the pseudo-inverse is used, so a point
        with any component in a zero-variance direction has an infinite
        distance.
        """
        A, B = self.deviations(speed, pressure)
        if self.det > 0:
            # delta' C^-1 delta with delta scaled by n.
            numer = self.c * A * A - 2 * self.b * A * B + self.a * B * B
            return Fraction(numer, self.det)

        # Singular covariance: rank 0 or rank 1.
        if self.a == 0 and self.b == 0 and self.c == 0:
            # Both directions have zero variance: the point must be the mean.
            return Fraction(0) if (A == 0 and B == 0) else None
        if self.b == 0:
            if self.a > 0:
                # Pressure is the zero-variance direction: p must equal mean.
                if B != 0:
                    return None
                return Fraction(A * A, self.a)
            # Speed is the zero-variance direction: s must equal mean.
            if A != 0:
                return None
            return Fraction(B * B, self.c)
        # Rank 1 with b != 0: the zero-variance direction is (b, -a);
        # the point must lie in the range direction (a, b).
        if self.b * A - self.a * B != 0:
            return None
        trace = self.trace
        numer = self.a * A * A + 2 * self.b * A * B + self.c * B * B
        return Fraction(numer, trace * trace)

    def distance(self, speed: int, pressure: int) -> Optional[float]:
        d2 = self.distance_sq(speed, pressure)
        return None if d2 is None else math.sqrt(float(d2))


# --------------------------------------------------------- speed intervals
def _speed_interval_from_A(op: OperatingPoint, lo_a: int, hi_a: int):
    """Convert an integer interval of A = n*s - sx into an interval of s."""
    lo = _ceil_div(lo_a + op.sx, op.n)
    hi = _floor_div(hi_a + op.sx, op.n)
    if lo > hi:
        return None
    return lo, hi


def _floor_sqrt_ratio(x: int, k: int, c: int) -> int:
    """floor((x + sqrt(k)) / c) for integers x, k >= 0 and c > 0."""
    t = math.isqrt(k)
    q = _floor_div(x + t, c)
    # sqrt(k) - t < 1, so q is exact or one below the true floor.
    v = c * (q + 1) - x
    if v <= 0 or v * v <= k:
        return q + 1
    return q


def _mahalanobis_speed_interval(op: OperatingPoint, pressure: int):
    """Integer speed interval meeting the distance limit at one pressure.

    Returns ``(lo, hi)`` or ``None`` when no integer speed qualifies.
    The returned interval is not clamped to the caller's speed range.
    """
    B = op.n * pressure - op.sy

    if op.det > 0:
        # c*A^2 - 2*b*B*A + (a*B^2 - 9*det) <= 0, with roots
        # (b*B +- sqrt(det*(9*c - B^2))) / c.  det > 0 implies c > 0.
        k = op.det * (MAHALANOBIS_LIMIT_SQ * op.c - B * B)
        if k < 0:
            return None
        x = op.b * B
        hi_a = _floor_sqrt_ratio(x, k, op.c)
        lo_a = -_floor_sqrt_ratio(-x, k, op.c)
        return _speed_interval_from_A(op, lo_a, hi_a)

    # Singular covariance: at most one speed (or none) can qualify.
    if op.a == 0 and op.b == 0 and op.c == 0:
        if B != 0 or op.sx % op.n != 0:
            return None
        s = op.sx // op.n
        return s, s

    if op.b == 0:
        if op.a > 0:
            # Zero pressure variance: p must equal the mean, speed within 3 sd.
            if B != 0:
                return None
            limit_a = math.isqrt(MAHALANOBIS_LIMIT_SQ * op.a)
            return _speed_interval_from_A(op, -limit_a, limit_a)
        # Zero speed variance: s must equal the mean, p within 3 sd.
        if B * B > MAHALANOBIS_LIMIT_SQ * op.c or op.sx % op.n != 0:
            return None
        s = op.sx // op.n
        return s, s

    # Rank 1, b != 0: the point must lie on the range line b*A = a*B.
    numerator = op.a * B
    if numerator % op.b != 0:
        return None
    A = numerator // op.b
    numer = op.a * A * A + 2 * op.b * A * B + op.c * B * B
    trace = op.trace
    if numer > MAHALANOBIS_LIMIT_SQ * trace * trace:
        return None
    if (A + op.sx) % op.n != 0:
        return None
    s = (A + op.sx) // op.n
    return s, s


@dataclass(frozen=True)
class _PowerTerms:
    """Exact rational form  P(s, p) = base + alpha_s*s + alpha_p*p."""

    alpha_s: Fraction
    alpha_p: Fraction
    base: Fraction


def _power_terms(model: RidgeModel) -> _PowerTerms:
    alpha = {0: Fraction(0), 1: Fraction(0)}
    for weight, index in zip(model.weights, model.kept):
        alpha[index] = Fraction(weight) / Fraction(model.std[index])
    base = Fraction(model.intercept) - sum(
        (alpha[i] * Fraction(model.mean[i]) for i in (0, 1)), Fraction(0)
    )
    return _PowerTerms(alpha_s=alpha[0], alpha_p=alpha[1], base=base)


def _power_at(terms: _PowerTerms, speed: int, pressure: int) -> Fraction:
    return terms.base + terms.alpha_s * speed + terms.alpha_p * pressure


def _power_speed_interval(terms: _PowerTerms, pressure: int,
                          power_limit: Fraction):
    """Speed interval whose predicted power at ``pressure`` is within limit."""
    constant = terms.base + terms.alpha_p * pressure
    if terms.alpha_s == 0:
        return None if constant > power_limit else (-math.inf, math.inf)
    bound = (power_limit - constant) / terms.alpha_s
    if terms.alpha_s > 0:
        return (-math.inf, _floor_frac(bound))
    return (_ceil_frac(bound), math.inf)


def _intersect(interval, other):
    """Intersect ``interval`` with ``other``; ``None`` when either is empty."""
    if interval is None or other is None:
        return None
    lo = max(interval[0], other[0])
    hi = min(interval[1], other[1])
    return None if lo > hi else (lo, hi)


# --------------------------------------------------------------- results
@dataclass(frozen=True)
class GearSelection:
    available: bool
    gear: Optional[int]
    max_power: Optional[float]
    max_distance: Optional[float]


@dataclass(frozen=True)
class GearReview:
    accepted: bool
    reasons: tuple
    max_power: Optional[float]
    max_distance: Optional[float]


# ---------------------------------------------------------------- solver
def _max_over_endpoints(op: OperatingPoint, terms: _PowerTerms, gear: int,
                        p_lo: int, p_hi: int):
    """Max predicted power and max distance over the pressure interval."""
    powers = [_power_at(terms, gear, p) for p in (p_lo, p_hi)]
    distances = [op.distance_sq(gear, p) for p in (p_lo, p_hi)]
    max_power = max(powers)
    if any(d is None for d in distances):
        max_distance = None
    else:
        max_distance = _distance_float(max(distances))
    return max_power, max_distance


def select_gear(op: OperatingPoint, model: RidgeModel, p_lo: int, p_hi: int,
                speed_min: int, speed_max: int,
                power_limit: float) -> GearSelection:
    """Return the highest usable integer gear, analytically."""
    terms = _power_terms(model)
    limit = Fraction(power_limit)
    interval = (speed_min, speed_max)
    for pressure in (p_lo, p_hi):
        interval = _intersect(
            interval, _mahalanobis_speed_interval(op, pressure)
        )
        if interval is None:
            break
        interval = _intersect(
            interval, _power_speed_interval(terms, pressure, limit)
        )
        if interval is None:
            break
    if interval is None:
        return GearSelection(False, None, None, None)
    gear = interval[1]
    max_power, max_distance = _max_over_endpoints(
        op, terms, gear, p_lo, p_hi
    )
    return GearSelection(True, gear, _finite_float(max_power), max_distance)


def review_gear(op: OperatingPoint, model: RidgeModel, gear: int,
                p_lo: int, p_hi: int, speed_min: int, speed_max: int,
                power_limit: float) -> GearReview:
    """Check a chosen gear against every condition; list the failures."""
    terms = _power_terms(model)
    limit = Fraction(power_limit)
    reasons = []
    if gear < speed_min or gear > speed_max:
        reasons.append("speed_range")

    powers = []
    distances = []
    for pressure in (p_lo, p_hi):
        powers.append(_power_at(terms, gear, pressure))
        distances.append(op.distance_sq(gear, pressure))
    if any(p > limit for p in powers):
        reasons.append("power_limit")
    if any(d is None or d > MAHALANOBIS_LIMIT_SQ for d in distances):
        reasons.append("mahalanobis")

    max_power = _finite_float(max(powers))
    if any(d is None for d in distances):
        max_distance = None
    else:
        max_distance = _distance_float(max(distances))
    return GearReview(not reasons, tuple(reasons), max_power, max_distance)
