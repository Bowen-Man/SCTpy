"""Explicit exposure and residual-floor universes, with filtered count rows."""

import numpy as np
import pytest
from scipy import sparse

from sctpy import SCTConfig, fit_transform


def floor_fixture():
    high = np.full((2, 6), 20)
    low = np.array([[1, 1, 1, 0, 0, 0]])
    excluded = np.zeros((20, 6), dtype=int)
    excluded[np.arange(20), np.arange(20) % 6] = 1
    return np.vstack((high, low, excluded, np.zeros((1, 6), dtype=int)))


def transform(counts, config):
    return fit_transform(
        counts,
        gene_names=tuple(f"g{i}" for i in range(counts.shape[0])),
        cell_names=tuple(f"c{i}" for i in range(counts.shape[1])),
        config=config,
    )


@pytest.mark.parametrize(
    "constructor", [np.asarray, sparse.csr_matrix, sparse.csc_matrix]
)
def test_compatibility_floor_uses_only_retained_gene_counts(constructor):
    counts = floor_fixture()
    result = transform(
        constructor(counts), SCTConfig.r_v2(min_cells=3, n_genes=None, n_cells=None)
    )
    assert result.nonzero_median == 20.0
    np.testing.assert_array_equal(result.library_size, counts.sum(axis=0))
    eligible = result.eligible_gene
    mu = np.exp(result.regularized_intercept[eligible, None]) * result.library_size
    sd = np.sqrt(mu + mu**2 / result.regularized_theta[eligible, None])
    expected = np.clip(
        (counts[eligible] - mu) / np.maximum(sd, 4.0), *result.clip_range
    )
    np.testing.assert_allclose(result.residuals[eligible], expected, rtol=0, atol=1e-13)


def test_native_keeps_documented_full_input_floor():
    result = transform(
        floor_fixture(), SCTConfig(min_cells=3, n_genes=None, n_cells=None)
    )
    assert result.nonzero_median == 1.0


@pytest.mark.parametrize("burst", [0, 100, 10000])
def test_native_poisson_intercept_uses_prediction_exposure(burst):
    counts = np.array([[1, 1, 1, 1], [burst, 0, 0, 0], [0, 0, 0, 0]])
    result = transform(counts, SCTConfig(min_cells=2, n_genes=None, n_cells=None))
    mu = np.exp(result.raw_intercept[0]) * result.library_size
    assert abs(np.sum(counts[0] - mu)) < 1e-12


def test_compatibility_ratio_is_labeled_as_a_reference_convention():
    counts = np.array([[1, 1, 1, 1], [100, 0, 0, 0]])
    result = transform(counts, SCTConfig.r_v2(min_cells=2, n_genes=None, n_cells=None))
    assert result.raw_intercept[0] == pytest.approx(0.0)
    # Preserve the documented R compatibility behavior, but do not claim it
    # solves the full-library Poisson score (which equals -100 in this case).
    score = np.sum(counts[0] - np.exp(result.raw_intercept[0]) * result.library_size)
    assert score == pytest.approx(-100.0)
    assert "compatibility" in result.fit_message[0]


def test_zero_gene_does_not_change_exposure_or_floor():
    counts = floor_fixture()
    config = SCTConfig.r_v2(min_cells=3, n_genes=None, n_cells=None)
    original = transform(counts[:-1], config)
    padded = transform(counts, config)
    np.testing.assert_array_equal(original.library_size, padded.library_size)
    assert original.nonzero_median == padded.nonzero_median
    np.testing.assert_allclose(
        original.residuals, padded.residuals[:-1], equal_nan=True
    )
