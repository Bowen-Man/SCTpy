import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from sctransform_native import SCTConfig, fit_transform


def _overdispersed_fixture() -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    rng = np.random.default_rng(7103)
    cells = 96
    exposure = rng.lognormal(mean=0.0, sigma=0.55, size=cells)
    rows: list[np.ndarray] = []
    for mean, theta in zip(
        (1.5, 2.2, 3.0, 4.5, 6.0, 8.0, 10.0, 12.0),
        (0.7, 1.0, 1.3, 1.7, 2.0, 2.5, 3.0, 4.0),
        strict=True,
    ):
        mu = mean * exposure
        probability = theta / (theta + mu)
        rows.append(rng.negative_binomial(theta, probability))
    rows.append(np.ones(cells, dtype=np.int64))
    counts = np.asarray(rows, dtype=np.int64)
    genes = tuple(f"g{index}" for index in range(counts.shape[0]))
    cell_names = tuple(f"c{index}" for index in range(cells))
    return counts, genes, cell_names


def test_all_poisson_path_matches_closed_form_residuals() -> None:
    counts = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
    result = fit_transform(
        counts,
        gene_names=("a", "b"),
        cell_names=tuple(f"c{index}" for index in range(6)),
        config=SCTConfig(min_cells=3, n_genes=None, n_cells=None),
    )

    assert_array_equal(result.poisson_gene, np.array([True, True]))
    assert_allclose(result.raw_intercept, np.log(0.5))
    assert_array_equal(result.raw_theta, np.array([np.inf, np.inf]))
    expected = (counts - 0.5) / np.sqrt(0.5)
    assert_allclose(result.residuals, expected)
    assert result.nonzero_median == 1.0
    assert result.regularization_bandwidth is None
    assert_array_equal(result.hvg_order, np.array([0, 1]))


def test_cpu_core_produces_complete_float64_result() -> None:
    counts, genes, cells = _overdispersed_fixture()
    result = fit_transform(
        counts,
        gene_names=genes,
        cell_names=cells,
        config=SCTConfig(min_cells=5, n_genes=None, n_cells=None, seed=9),
    )

    assert result.residuals.shape == counts.shape
    assert result.residuals.dtype == np.float64
    assert np.count_nonzero(result.converged) >= 4
    assert np.all(result.slope[result.eligible_gene] == np.log(10.0))
    assert np.all(np.isfinite(result.regularized_intercept[result.eligible_gene]))
    assert np.all(result.regularized_theta[result.eligible_gene] > 0)
    assert np.all(np.isfinite(result.residual_variance[result.eligible_gene]))
    assert_array_equal(np.sort(result.hvg_order), np.flatnonzero(result.eligible_gene))
    assert result.highly_variable_genes(3) == tuple(
        genes[index] for index in result.hvg_order[:3]
    )
    assert set(result.timing) == {
        "input_validation",
        "input_statistics",
        "selection",
        "fit",
        "regularization",
        "residuals_and_hvg",
        "total",
    }
    assert result.timing["total"] >= sum(
        value for key, value in result.timing.items() if key != "total"
    )


def test_sparse_dense_and_explicit_orientation_are_equivalent() -> None:
    counts, genes, cells = _overdispersed_fixture()
    config = SCTConfig(min_cells=5, n_genes=None, n_cells=None, seed=13)
    dense = fit_transform(
        counts, gene_names=genes, cell_names=cells, config=config
    )
    sparse_result = fit_transform(
        sparse.csc_matrix(counts),
        gene_names=genes,
        cell_names=cells,
        config=config,
        selection=dense.selection,
    )
    cells_by_genes = fit_transform(
        counts.T,
        gene_names=genes,
        cell_names=cells,
        matrix_orientation="cells_by_genes",
        config=config,
        selection=dense.selection,
    )

    for other in (sparse_result, cells_by_genes):
        assert_allclose(other.raw_intercept, dense.raw_intercept, equal_nan=True)
        assert_allclose(other.raw_theta, dense.raw_theta, equal_nan=True)
        assert_allclose(other.residuals, dense.residuals, equal_nan=True)
        assert_array_equal(other.hvg_order, dense.hvg_order)


def test_genes_below_min_cells_are_visible_and_excluded() -> None:
    counts, genes, cells = _overdispersed_fixture()
    counts = np.vstack([counts, np.r_[1, np.zeros(counts.shape[1] - 1, dtype=int)]])
    genes = (*genes, "rare")
    result = fit_transform(
        counts,
        gene_names=genes,
        cell_names=cells,
        config=SCTConfig(min_cells=5, n_genes=None, n_cells=None),
    )
    assert not result.eligible_gene[-1]
    assert result.fit_status[-1] == "insufficient_cells"
    assert np.all(np.isnan(result.residuals[-1]))
    assert result.hvg_rank[-1] == -1
