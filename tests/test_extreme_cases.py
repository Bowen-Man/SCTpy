import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from sctpy import SCTConfig, fit_transform
from sctpy._matrix import matrix_statistics, prepare_counts


def test_all_zero_gene_remains_explicitly_excluded() -> None:
    counts = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 4, 0, 2, 0, 6],
        ],
        dtype=np.int64,
    )
    result = fit_transform(
        sparse.csr_matrix(counts),
        gene_names=("zero", "constant", "burst"),
        cell_names=tuple(f"cell_{index}" for index in range(counts.shape[1])),
        config=SCTConfig(min_cells=1, n_genes=None, n_cells=None),
    )

    assert not result.eligible_gene[0]
    assert result.fit_status[0] == "insufficient_cells"
    assert np.all(np.isnan(result.residuals[0]))
    assert result.hvg_rank[0] == -1


def test_large_integer_counts_have_finite_dense_and_sparse_statistics() -> None:
    counts = np.array(
        [
            [2**40, 2**40 + 1, 2**40 + 2, 2**40 + 3],
            [1, 2**32, 2**36, 2**38],
        ],
        dtype=np.int64,
    )
    names = ("large_constant", "large_range")
    cells = ("c0", "c1", "c2", "c3")
    dense, _, _ = prepare_counts(
        counts,
        gene_names=names,
        cell_names=cells,
        orientation="genes_by_cells",
    )
    sparse_counts, _, _ = prepare_counts(
        sparse.csr_matrix(counts),
        gene_names=names,
        cell_names=cells,
        orientation="genes_by_cells",
    )

    dense_statistics = matrix_statistics(dense)
    sparse_statistics = matrix_statistics(sparse_counts)
    for dense_value, sparse_value in zip(
        dense_statistics, sparse_statistics, strict=True
    ):
        assert np.all(np.isfinite(dense_value))
        assert_allclose(sparse_value, dense_value, rtol=1e-15)


def test_r_v2_handles_extreme_library_sizes_and_sparse_dense_parity() -> None:
    rng = np.random.default_rng(20261103)
    exposure = np.geomspace(1.0, 1.0e5, 64)
    rows = []
    for rate, theta in zip(
        (0.015, 0.025, 0.04, 0.07, 0.11, 0.18),
        (0.5, 0.8, 1.2, 2.0, 3.5, 6.0),
        strict=True,
    ):
        mean = rate * exposure
        rows.append(rng.negative_binomial(theta, theta / (theta + mean)))
    rows.append(np.ones(exposure.size, dtype=np.int64))
    counts = np.asarray(rows, dtype=np.int64)
    genes = tuple(f"gene_{index}" for index in range(counts.shape[0]))
    cells = tuple(f"cell_{index}" for index in range(counts.shape[1]))
    config = SCTConfig.r_v2(min_cells=5, n_genes=None, n_cells=None)

    dense = fit_transform(counts, gene_names=genes, cell_names=cells, config=config)
    sparse_result = fit_transform(
        sparse.csr_matrix(counts),
        gene_names=genes,
        cell_names=cells,
        config=config,
        selection=dense.selection,
    )

    modeled = dense.eligible_gene
    assert dense.library_size.max() / dense.library_size.min() > 1.0e4
    assert np.all(np.isfinite(dense.regularized_intercept[modeled]))
    assert np.all(dense.regularized_theta[modeled] > 0.0)
    assert np.all(np.isfinite(dense.residuals[modeled]))
    assert_allclose(sparse_result.raw_intercept, dense.raw_intercept, equal_nan=True)
    assert_allclose(sparse_result.raw_theta, dense.raw_theta, equal_nan=True)
    assert_allclose(sparse_result.residuals, dense.residuals, equal_nan=True)
    assert_array_equal(sparse_result.hvg_order, dense.hvg_order)
