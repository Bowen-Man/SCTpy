import numpy as np
import pytest


@pytest.mark.gpu
def test_cupy_microbenchmark_primitive_matches_cpu() -> None:
    cp = pytest.importorskip("cupy", reason="optional CuPy dependency is not installed")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device is available")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime is unavailable")

    from benchmarks.gpu_sj_functional import _cupy_functional
    from sctransform_native._regularize import _normal_derivative_functional

    x = np.linspace(-1.0, 1.0, 32, dtype=np.float64)
    bandwidth = 0.4
    expected = _normal_derivative_functional(x, bandwidth=bandwidth, order=4)
    observed = float(
        cp.asnumpy(
            _cupy_functional(cp, cp.asarray(x), bandwidth=bandwidth, block_size=16)
        )
    )
    assert observed == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)
