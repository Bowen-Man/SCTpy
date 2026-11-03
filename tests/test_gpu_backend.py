import importlib

import numpy as np
import pytest

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


def test_cupy_roots_match_numpy() -> None:
    cp = pytest.importorskip("cupy", reason="CuPy is an optional GPU dependency")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime is unavailable")

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
