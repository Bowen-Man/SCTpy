from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from sctransform_native import (
    ImplicitDeviceTransferError,
    SCTConfig,
    fit_transform,
    fit_transform_anndata,
)
from sctransform_native.backend import infer_device


def _adata(counts: object) -> SimpleNamespace:
    return SimpleNamespace(
        X=counts,
        obs_names=("c0", "c1", "c2", "c3"),
        var_names=("g0", "g1", "g2"),
        layers={},
    )


@pytest.mark.parametrize(
    "container",
    [np.asarray, sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array],
)
def test_anndata_adapter_performs_one_explicit_orientation_conversion(
    container: Callable[[np.ndarray], object],
) -> None:
    counts_cxg = np.array([[2, 0, 1], [4, 1, 0], [3, 1, 2], [5, 2, 1]])
    config = SCTConfig(min_cells=1, n_genes=None, n_cells=None)
    adapted = fit_transform_anndata(_adata(container(counts_cxg)), config=config)
    direct = fit_transform(
        counts_cxg.T,
        gene_names=("g0", "g1", "g2"),
        cell_names=("c0", "c1", "c2", "c3"),
        config=config,
    )
    np.testing.assert_allclose(adapted.residuals, direct.residuals)


def test_anndata_adapter_rejects_hidden_cuda_to_cpu_transfer() -> None:
    class FakeCudaArray:
        __cuda_array_interface__ = {
            "shape": (4, 3),
            "typestr": "<f8",
            "data": (0, False),
            "version": 3,
        }

    counts = FakeCudaArray()
    assert infer_device(counts).kind == "cuda"
    with pytest.raises(ImplicitDeviceTransferError, match="explicitly transfer"):
        fit_transform_anndata(_adata(counts))


def test_missing_anndata_layer_is_explicit() -> None:
    with pytest.raises(KeyError, match="does not exist"):
        fit_transform_anndata(_adata(np.ones((4, 3), dtype=int)), layer="counts")
