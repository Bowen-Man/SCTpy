import importlib
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from benchmarks.generate_fixture import generate_counts
from sctransform_native import _gpu, fit_transform
from sctransform_native._fit import fit_cox_reid_offset_batch
from sctransform_native._gpu import (
    _import_cupy,
    fit_cox_reid_offset_batch_cupy,
)
from sctransform_native.core import SCTConfig


def test_cupy_backend_configuration_is_explicit() -> None:
    config = SCTConfig.r_v2(cox_reid_backend="cupy")
    assert config.cox_reid_backend == "cupy"
    assert config.statistical_mode == "r_v2"
    with pytest.raises(ValueError, match="batched Cox-Reid"):
        SCTConfig(fit_method="nb_mle", cox_reid_backend="cupy")
    with pytest.raises(ValueError, match="batched Cox-Reid"):
        SCTConfig.r_v2(cox_reid_backend="cupy", cox_reid_batch_size=None)


def test_missing_cupy_dependency_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_import(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing_import)
    with pytest.raises(ImportError, match="optional CUDA extra"):
        _import_cupy()


def test_cupy_roots_match_numpy(cupy_device) -> None:
    exposure = np.array(
        [80, 95, 110, 130, 90, 150, 105, 125, 100, 140, 115, 85],
        dtype=np.float64,
    )
    counts = np.array(
        [
            [0, 1, 0, 2, 1, 3, 0, 1, 4, 2, 1, 0],
            [0, 0, 0, 0, 1, 0, 10, 0, 0, 20, 0, 0],
            [5, 6, 4, 5, 7, 6, 5, 8, 7, 6, 5, 4],
            [1, 4, 2, 8, 1, 12, 2, 7, 3, 11, 2, 1],
        ],
        dtype=np.float64,
    )
    expected = fit_cox_reid_offset_batch(counts, exposure)
    actual = fit_cox_reid_offset_batch_cupy(counts, exposure)

    expected_theta = np.asarray([fit.theta for fit in expected])
    actual_theta = np.asarray([fit.theta for fit in actual])
    np.testing.assert_array_equal(np.isinf(actual_theta), np.isinf(expected_theta))
    finite = np.isfinite(expected_theta)
    np.testing.assert_allclose(
        np.log10(actual_theta[finite]),
        np.log10(expected_theta[finite]),
        rtol=0.0,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        [fit.intercept for fit in actual],
        [fit.intercept for fit in expected],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        [fit.objective for fit in actual],
        [fit.objective for fit in expected],
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_array_equal(
        [fit.iterations for fit in actual],
        [fit.iterations for fit in expected],
    )


@pytest.mark.parametrize(
    "container", [np.asarray, sparse.csr_matrix, sparse.csc_matrix]
)
@pytest.mark.parametrize(
    "output,dtype", [("full", "float64"), ("full", "float32"), ("summary", "float64")]
)
def test_full_pipeline_parity(cupy_device, container, output, dtype):
    counts = container(generate_counts(genes=200, cells=96, seed=20261103))
    names = dict(
        gene_names=tuple(f"g{i}" for i in range(200)),
        cell_names=tuple(f"c{i}" for i in range(96)),
    )
    config = SCTConfig.r_v2(
        n_genes=None, n_cells=None, residual_output=output, residual_dtype=dtype
    )
    expected = fit_transform(counts, **names, config=config, return_corrected_umi=True)
    actual = fit_transform(
        counts,
        **names,
        config=replace(config, cox_reid_backend="cupy"),
        selection=expected.selection,
        return_corrected_umi=True,
    )
    for field in (
        "poisson_gene",
        "theta_postprocessed",
        "parameter_outlier",
        "hvg_order",
        "converged",
    ):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    for field in ("raw_theta", "regularized_theta"):
        np.testing.assert_allclose(
            np.log10(getattr(actual, field)),
            np.log10(getattr(expected, field)),
            atol=1e-6,
            rtol=0,
        )
    for field in (
        "raw_intercept",
        "regularized_intercept",
        "residual_mean",
        "residual_variance",
    ):
        np.testing.assert_allclose(
            getattr(actual, field), getattr(expected, field), atol=1e-6, rtol=0
        )
    if output == "full":
        np.testing.assert_allclose(
            actual.residuals, expected.residuals, atol=1e-5, rtol=0
        )
    else:
        assert actual.residuals is None
    assert (actual.corrected_umi != expected.corrected_umi).nnz == 0


@pytest.mark.parametrize("allocator_name", ["cupy", "rmm"])
def test_respects_stream_and_allocator(cupy_device, allocator_name):
    cp = cupy_device
    if allocator_name == "rmm":
        allocator = pytest.importorskip("rmm.allocators.cupy").rmm_cupy_allocator
    else:
        allocator = cp.cuda.MemoryPool().malloc
    previous_allocator = cp.cuda.get_allocator()
    previous_stream = cp.cuda.get_current_stream()
    counts = np.array([[0, 2, 0, 5, 1, 9], [1, 2, 3, 4, 5, 6]], dtype=float)
    exposure = np.full(6, 100.0)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream, cp.cuda.using_allocator(allocator):
        actual = fit_cox_reid_offset_batch_cupy(counts, exposure)
        assert cp.cuda.get_current_stream() == stream
        assert cp.cuda.get_allocator() is allocator
    assert cp.cuda.get_current_stream() == previous_stream
    assert cp.cuda.get_allocator() is previous_allocator
    expected = fit_cox_reid_offset_batch(counts, exposure)
    np.testing.assert_allclose(
        [fit.theta for fit in actual], [fit.theta for fit in expected], rtol=1e-8
    )


@pytest.mark.parametrize("mask", ["poisson", "ceiling"])
def test_cpu_branch_guard_repairs_device_disagreement(cupy_device, monkeypatch, mask):
    roots = _gpu._dispersion_roots_cupy

    def changed_branch(*args):
        result = roots(*args)
        flags = getattr(result, mask).copy()
        flags[0] = ~flags[0]
        theta = result.theta.copy()
        theta[0] = 123.0
        return result._replace(theta=theta, **{mask: flags})

    monkeypatch.setattr(_gpu, "_dispersion_roots_cupy", changed_branch)
    counts = np.array([[0, 2, 0, 5, 1, 9], [1, 2, 3, 4, 5, 6]], dtype=float)
    exposure = np.full(6, 100.0)
    expected = fit_cox_reid_offset_batch(counts, exposure)
    actual = fit_cox_reid_offset_batch_cupy(counts, exposure)
    assert actual[0] == expected[0]
    np.testing.assert_allclose(actual[1].theta, expected[1].theta, rtol=1e-8)


def test_anndata_outputs_feed_scanpy_and_rapids_pca(cupy_device):
    ad = pytest.importorskip("anndata")
    sc = pytest.importorskip("scanpy")
    rsc = pytest.importorskip("rapids_singlecell")
    from sctransform_native import fit_transform_anndata, write_anndata

    counts = sparse.csr_matrix(generate_counts(genes=800, cells=240, seed=20261102).T)
    adata = ad.AnnData(counts)
    adata.layers["counts"] = counts.copy()
    result = fit_transform_anndata(
        adata,
        layer="counts",
        config=SCTConfig.r_v2(residual_dtype="float32", cox_reid_backend="cupy"),
    )
    write_anndata(adata, result, n_top_genes=200)
    cpu = adata.copy()
    sc.pp.pca(
        cpu,
        layer="sct_residuals",
        mask_var="highly_variable",
        n_comps=20,
        svd_solver="arpack",
    )
    rsc.get.anndata_to_GPU(adata, layer="sct_residuals")
    rsc.pp.pca(
        adata,
        layer="sct_residuals",
        mask_var="highly_variable",
        n_comps=20,
        svd_solver="jacobi",
    )
    assert adata.obsm["X_pca"].shape == (240, 20)
    np.testing.assert_allclose(
        adata.uns["pca"]["variance_ratio"],
        cpu.uns["pca"]["variance_ratio"],
        rtol=1e-4,
        atol=1e-6,
    )
    assert (adata.layers["counts"] != counts).nnz == 0
