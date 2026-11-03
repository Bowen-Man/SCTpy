import pytest


@pytest.fixture
def cupy_device():
    cp = pytest.importorskip("cupy", reason="CuPy is an optional GPU dependency")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime is unavailable")
    return cp
