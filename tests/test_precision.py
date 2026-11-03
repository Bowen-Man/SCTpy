import numpy as np
from scipy import sparse

from benchmarks.correctness import (
    MIXED_PRECISION_THRESHOLDS,
    gate_metrics,
    jaccard,
    numeric_metrics,
)
from benchmarks.generate_fixture import generate_counts
from sctransform_native import SCTConfig, fit_transform


def test_float32_residual_path_passes_preregistered_gate() -> None:
    counts = generate_counts(genes=800, cells=240, seed=20261102)
    genes = tuple(f"gene_{index:05d}" for index in range(counts.shape[0]))
    cells = tuple(f"cell_{index:05d}" for index in range(counts.shape[1]))
    float64 = fit_transform(
        sparse.csr_matrix(counts),
        gene_names=genes,
        cell_names=cells,
        config=SCTConfig.r_v2(n_genes=None, n_cells=None),
        return_corrected_umi=True,
    )
    float32 = fit_transform(
        sparse.csr_matrix(counts),
        gene_names=genes,
        cell_names=cells,
        config=SCTConfig.r_v2(
            n_genes=None,
            n_cells=None,
            residual_dtype="float32",
        ),
        selection=float64.selection,
        return_corrected_umi=True,
    )

    assert float64.residuals is not None
    assert float32.residuals is not None
    assert float32.residuals.dtype == np.float32
    np.testing.assert_array_equal(float32.raw_intercept, float64.raw_intercept)
    np.testing.assert_array_equal(float32.raw_theta, float64.raw_theta)
    np.testing.assert_array_equal(
        float32.regularized_intercept, float64.regularized_intercept
    )
    np.testing.assert_array_equal(float32.regularized_theta, float64.regularized_theta)
    for field in ("residuals", "residual_mean", "residual_variance"):
        metrics = numeric_metrics(getattr(float32, field), getattr(float64, field))
        threshold = MIXED_PRECISION_THRESHOLDS[
            "residual" if field == "residuals" else field
        ]
        assert gate_metrics(metrics, threshold), (field, metrics)

    for top in (100, 500):
        hvg_jaccard = jaccard(float32.hvg_order[:top], float64.hvg_order[:top])
        assert hvg_jaccard >= MIXED_PRECISION_THRESHOLDS["hvg"]["jaccard"]
    assert float32.corrected_umi is not None
    assert float64.corrected_umi is not None
    assert (float32.corrected_umi != float64.corrected_umi).nnz == 0
