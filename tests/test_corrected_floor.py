import numpy as np
import pytest

from sctpy import SCTConfig, fit_transform
from sctpy.core import _residuals_and_summary


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_reconstruction_uses_unclipped_floored_residual(dtype):
    # Original mean 0.0002/0.02; target mean 0.02, target SD sqrt(0.0204).
    result = _residuals_and_summary(
        np.array([[36.0, 0.0]]),
        library_size=np.array([1.0, 100.0]),
        eligible=np.array([True]),
        intercept=np.log(np.array([0.0002])),
        theta=np.array([1.0]),
        clip_range=(-1.0, 1.0),
        standard_deviation_floor=0.2,
        block_size=1,
        store_residuals=True,
        residual_dtype=dtype,
        corrected_umi_depth=100.0,
    )
    residuals, corrected = result[:2]
    assert residuals[0, 0] == 1.0
    # round(0.02 + (36 - 0.0002) / 0.2 * sqrt(0.0204)) = 26.
    np.testing.assert_array_equal(corrected.toarray(), [[26, 0]])


def test_v2_poisson_intercept_uses_eligible_gene_total():
    counts = np.array([[1, 1, 1, 1], [0, 2, 0, 8], [100, 0, 0, 0]])
    result = fit_transform(
        counts,
        gene_names=("constant", "burst", "excluded"),
        cell_names=("a", "b", "c", "d"),
        config=SCTConfig.r_v2(min_cells=2, n_genes=None, n_cells=None),
    )
    assert result.poisson_gene[0] and not result.eligible_gene[2]
    assert result.regularized_intercept[0] == pytest.approx(np.log(4 / 14))
    np.testing.assert_array_equal(result.library_size, [101, 3, 1, 9])
