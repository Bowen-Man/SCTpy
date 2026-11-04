"""Independent integer-count references for the existing fixed-mu objective."""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from sctpy._fit import (
    _cox_reid_score_batch,
    fit_cox_reid_offset,
    fit_cox_reid_offset_batch,
)


def integer_score(log_phi, counts, means):
    """80-digit finite sums, without SciPy special functions or production helpers."""
    with localcontext() as context:
        context.prec = 80
        phi = Decimal.from_float(float(np.exp(log_phi)))
        score = Decimal(0)
        weights = []
        for count, mean in zip(counts, means, strict=True):
            y = int(count)
            mu = Decimal.from_float(float(mean))
            x = phi * mu
            score += sum((phi * k / (1 + phi * k) for k in range(y)), Decimal(0))
            score += ((1 + x).ln() - x / (1 + x)) / phi - y * x / (1 + x)
            weights.append(mu / (1 + x))
        score += (
            Decimal("0.5")
            * Decimal.from_float(0.99)
            * phi
            * sum(w * w for w in weights)
            / (sum(weights) + Decimal.from_float(1e-6))
        )
        return float(score)


def test_integer_laurent_remainder_is_nonnegative_exactly():
    for count, phi in ((2, Fraction(1, 10)), (20, Fraction(1, 1000))):
        value = sum((1 / (1 + phi * k) for k in range(count)), Fraction(0))
        lower = count - phi * count * (count - 1) / 2
        remainder = sum(
            ((phi * k) ** 2 / (1 + phi * k) for k in range(count)), Fraction(0)
        )
        assert value - lower == remainder
        assert 0 <= remainder <= phi**2 * count * (count - 1) * (2 * count - 1) / 6
        assert lower < value <= count


@pytest.mark.parametrize("phi", [1e-12, 1e-8, 9e-6, 1.1e-5, 1e-3, 0.1, 1.0])
@pytest.mark.parametrize(
    "counts,means",
    [
        ([0, 1, 2, 7], [0.1, 0.8, 2.0, 6.0]),
        ([0, 200, 1000, 2000], [0.01, 210.0, 1000.0, 1990.0]),
    ],
)
def test_fixed_mean_score_matches_integer_reference(phi, counts, means):
    counts = np.asarray(counts, dtype=float)
    means = np.asarray(means, dtype=float)
    log_phi = float(np.log(phi))
    expected = integer_score(log_phi, counts, means)
    actual = _cox_reid_score_batch(
        np.array([log_phi]), counts[None, :], means[None, :]
    )[0]
    tolerance = 1e-10 + 1e-12 * counts.sum() + 1e-10 * abs(expected)
    assert abs(actual - expected) <= tolerance


@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
def test_score_at_former_cubic_transition(factor):
    counts = np.array([0.0, 20.0, 1000.0])
    means = np.array([0.01, 19.0, 1001.0])
    fourth_sum = sum(sum(k**4 for k in range(int(y))) for y in counts)
    phi = factor * (np.finfo(float).eps * counts.sum() / fourth_sum) ** 0.25
    log_phi = float(np.log(phi))
    expected = integer_score(log_phi, counts, means)
    actual = _cox_reid_score_batch(np.array([log_phi]), counts[None], means[None])[0]
    assert abs(actual - expected) <= 1e-10 + 1e-12 * counts.sum() + 1e-10 * abs(
        expected
    )


@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
def test_score_across_digamma_remainder_bound_switch(factor):
    counts = np.array([0.0, 20.0, 1000.0])
    means = np.array([0.01, 19.0, 1001.0])
    phi = factor * (6.0 * np.finfo(float).eps) ** (1.0 / 13.0)
    t = float(np.log(phi))
    expected = integer_score(t, counts, means)
    actual = _cox_reid_score_batch(np.array([t]), counts[None], means[None])[0]
    assert abs(actual - expected) <= 1e-10 + 1e-12 * counts.sum() + 1e-10 * abs(
        expected
    )


def test_depth_explained_high_counts_reach_poisson_boundary():
    counts = np.array([80000.0, 100000.0, 120000.0])
    # A finite-sum score is negative at the lower dispersion boundary. The
    # former invalid upper clipping instead produced a spurious theta=1e5.
    assert integer_score(np.log(1e-8), counts, counts) < 0
    assert np.isinf(fit_cox_reid_offset(counts, counts).theta)
    fitted = fit_cox_reid_offset_batch(counts[None], counts)[0]
    assert np.isinf(fitted.theta)


def test_cupy_score_uses_the_same_proven_bound(cupy_device):
    from cupyx.scipy.special import digamma

    from sctpy._gpu import _cox_reid_score_cupy

    cp = cupy_device
    counts = np.array([[0.0, 200.0, 1000.0, 2000.0]])
    means = np.array([[0.01, 210.0, 1000.0, 1990.0]])
    t = np.array([np.log(9e-6)])
    actual = cp.asnumpy(
        _cox_reid_score_cupy(
            cp, digamma, cp.asarray(t), cp.asarray(counts), cp.asarray(means)
        )
    )[0]
    expected = integer_score(t[0], counts[0], means[0])
    assert abs(actual - expected) <= 1e-10 + 1e-12 * counts.sum() + 1e-10 * abs(
        expected
    )
