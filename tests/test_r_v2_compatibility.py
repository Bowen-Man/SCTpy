import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from benchmarks._common import count_sha256
from benchmarks.correctness import (
    numeric_mask_metrics,
    numeric_metrics,
    sparse_numeric_metrics,
)
from benchmarks.generate_fixture import generate_counts
from sctpy import SCTConfig, SelectionManifest, fit_transform

GOLDEN_PATH = Path(__file__).parent / "data" / "r_v2_small_golden.npz"
PROVENANCE_PATH = GOLDEN_PATH.with_suffix(".provenance.json")
COUNTS_SHA256 = "daa4251f89ad449d0508ebef01bca762ea3960830b0160623d809db1131c2337"


def _names(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _assert_error_bounds(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rmse: float,
    maximum: float,
) -> None:
    metrics = numeric_metrics(actual, expected)
    assert float(metrics["rmse"]) <= rmse
    assert float(metrics["max_absolute_error"]) <= maximum


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_r_v2_small_frozen_correctness_gate(backend, request) -> None:
    if backend == "cupy":
        request.getfixturevalue("cupy_device")
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert (
        hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest()
        == provenance["artifact"]["sha256"]
    )
    counts = generate_counts(genes=800, cells=240, seed=20261102)
    assert count_sha256(counts) == COUNTS_SHA256
    gene_names = tuple(f"gene_{index:05d}" for index in range(counts.shape[0]))
    cell_names = tuple(f"cell_{index:05d}" for index in range(counts.shape[1]))

    with np.load(GOLDEN_PATH, allow_pickle=False) as golden:
        selection = SelectionManifest(
            schema_version=1,
            seed=20261102,
            algorithm="uniform_without_replacement_pcg64_v1",
            source_shape=counts.shape,
            selected_gene_indices=tuple(
                int(value) for value in golden["selected_gene_indices"]
            ),
            selected_gene_names=_names(golden["selected_gene_names"]),
            selected_cell_indices=tuple(
                int(value) for value in golden["selected_cell_indices"]
            ),
            selected_cell_names=_names(golden["selected_cell_names"]),
            min_cells=5,
            requested_genes=None,
            requested_cells=None,
        )
        result = fit_transform(
            sparse.csr_matrix(counts),
            gene_names=gene_names,
            cell_names=cell_names,
            config=SCTConfig.r_v2(
                n_genes=None, n_cells=None, seed=20261102, cox_reid_backend=backend
            ),
            selection=selection,
            return_corrected_umi=True,
        )

        eligible_indices = np.flatnonzero(result.eligible_gene)
        fit_indices = np.asarray(
            [gene_names.index(name) for name in _names(golden["fit_genes"])],
            dtype=np.int64,
        )
        assert tuple(gene_names[index] for index in eligible_indices) == _names(
            golden["genes"]
        )

        _assert_error_bounds(
            result.raw_intercept[fit_indices],
            golden["raw_intercept"],
            rmse=1e-7,
            maximum=1e-6,
        )
        assert numeric_mask_metrics(result.raw_theta[fit_indices], golden["raw_theta"])[
            "exact"
        ]
        finite_theta = np.isfinite(golden["raw_theta"])
        _assert_error_bounds(
            np.log10(result.raw_theta[fit_indices][finite_theta]),
            np.log10(golden["raw_theta"][finite_theta]),
            rmse=1e-5,
            maximum=1e-3,
        )
        _assert_error_bounds(
            result.regularized_intercept[eligible_indices],
            golden["regularized_intercept"],
            rmse=1e-4,
            maximum=1e-3,
        )
        _assert_error_bounds(
            np.log10(result.regularized_theta[eligible_indices]),
            np.log10(golden["regularized_theta"]),
            rmse=1e-6,
            maximum=1e-5,
        )
        _assert_error_bounds(
            result.residual_mean[eligible_indices],
            golden["residual_mean"],
            rmse=1e-4,
            maximum=1e-3,
        )
        _assert_error_bounds(
            result.residual_variance[eligible_indices],
            golden["residual_variance"],
            rmse=1e-4,
            maximum=1e-3,
        )
        expected_corrected = sparse.csr_matrix(
            (
                golden["corrected_umi_data"],
                golden["corrected_umi_indices"],
                golden["corrected_umi_indptr"],
            ),
            shape=tuple(golden["corrected_umi_shape"]),
        )
        assert result.corrected_umi is not None
        corrected_metrics = sparse_numeric_metrics(
            result.corrected_umi[eligible_indices], expected_corrected
        )
        assert float(corrected_metrics["mae"]) <= 0.01
        assert float(corrected_metrics["max_absolute_error"]) <= 2.0
        assert float(corrected_metrics["exact_fraction"]) >= 0.99

        assert result.highly_variable_genes(len(eligible_indices)) == _names(
            golden["hvg_genes"]
        )
        assert tuple(
            gene_names[index] for index in np.flatnonzero(result.poisson_gene)
        ) == _names(golden["poisson_genes"])
        assert tuple(
            gene_names[index] for index in np.flatnonzero(result.parameter_outlier)
        ) == _names(golden["outlier_genes"])
