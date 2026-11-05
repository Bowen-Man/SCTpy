from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from scipy.stats import nbinom, poisson

from benchmarks.generate_fixture import generate_counts
from sctpy import SCTConfig, _fit, _gpu, core, fit_transform


@pytest.mark.parametrize("container", [np.asarray, sparse.csr_matrix, sparse.csc_array])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("output", ["full", "summary"])
def test_corrected_counts_int64_boundary(container, dtype, output):
    counts = container(np.ones((1, 6), dtype=np.int64))
    kwargs = dict(
        gene_names=("g",),
        cell_names=tuple(f"c{i}" for i in range(6)),
        config=SCTConfig.r_v2(
            min_cells=1,
            n_cells=None,
            n_genes=None,
            residual_dtype=dtype,
            residual_output=output,
        ),
        return_corrected_umi=True,
    )
    safe = np.nextafter(float(2**63), 0.0)
    result = fit_transform(counts, **kwargs, corrected_umi_depth=safe)
    np.testing.assert_array_equal(result.corrected_umi.toarray(), int(safe))
    for depth in (float(2**63), np.nextafter(float(2**63), np.inf)):
        with np.errstate(invalid="raise"), pytest.raises(OverflowError, match="int64"):
            fit_transform(counts, **kwargs, corrected_umi_depth=depth)


@pytest.mark.parametrize("backend", ["scalar", "numpy", "cupy"])
@pytest.mark.parametrize(
    "case",
    [
        "root",
        "poisson",
        "ceiling",
        "nan",
        "inf",
        "midpoint_nan",
        "objective_nan",
        "exhausted",
    ],
)
def test_cox_reid_termination_status(request, monkeypatch, backend, case):
    counts = np.array([[0, 2, 0, 5, 1, 9]], dtype=float)
    exposure = np.full(6, 100.0)
    calls = 0
    if case == "objective_nan":
        monkeypatch.setattr(
            _fit,
            "_negative_log_likelihood_and_gradient",
            lambda *args: (np.nan, np.full(2, np.nan)),
        )

    def score(x, y, mu):
        nonlocal calls
        calls += 1
        if case == "inf":
            return np.full(x.shape, np.inf)
        if case in ("nan", "midpoint_nan") and (case == "nan" or calls == 3):
            return np.full(x.shape, np.nan)
        if case in ("poisson", "ceiling"):
            return np.full(x.shape, -1.0 if case == "poisson" else 1.0)
        return np.log(0.1) - x

    monkeypatch.setattr(_fit, "_cox_reid_score_batch", score)
    if backend == "scalar":
        if case == "exhausted":
            original = _fit.brentq

            def limited(function, *args, **kwargs):
                if function is _fit._cox_reid_score:
                    kwargs["maxiter"] = 0
                return original(function, *args, **kwargs)

            monkeypatch.setattr(_fit, "brentq", limited)
        result = _fit.fit_cox_reid_offset(counts[0], exposure)
    elif backend == "numpy":
        if case == "exhausted":
            monkeypatch.setattr(_fit, "_DISPERSION_ROOT_STEPS", 0)
        result = _fit.fit_cox_reid_offset_batch(counts, exposure)[0]
    else:
        request.getfixturevalue("cupy_device")

        def gpu_score(cp, digamma, x, y, mu):
            return cp.asarray(score(cp.asnumpy(x), None, None))

        monkeypatch.setattr(_gpu, "_cox_reid_score_cupy", gpu_score)
        # Align the existing host branch guard; it must not mask a device failure.
        monkeypatch.setattr(_gpu, "_cox_reid_score_batch", score)
        if case == "exhausted":
            monkeypatch.setattr(_gpu, "_DISPERSION_ROOT_STEPS", 0)
        result = _gpu.fit_cox_reid_offset_batch_cupy(counts, exposure)[0]

    assert result.converged == (case in ("root", "poisson"))
    if case == "root":
        assert result.theta == pytest.approx(10.0)
        assert "root" in result.message
    elif case == "poisson":
        assert result.theta == np.inf
        assert "Poisson boundary" in result.message
    elif case == "ceiling":
        assert "ceiling" in result.message
    elif case in ("nan", "inf", "midpoint_nan", "objective_nan"):
        assert "non-finite" in result.message
    else:
        assert "did not converge" in result.message


@pytest.mark.parametrize("batch_size", [None, 8])
@pytest.mark.parametrize("postprocess", [False, True])
def test_final_raw_objective_matches_independent_distribution(batch_size, postprocess):
    counts = generate_counts(genes=80, cells=96, seed=42)
    config = SCTConfig.r_v2(n_genes=None, n_cells=64, cox_reid_batch_size=batch_size)
    if postprocess:
        config = replace(config, theta_moment_ratio_threshold=1.0)
    result = fit_transform(
        counts,
        gene_names=tuple(f"g{i}" for i in range(80)),
        cell_names=tuple(f"c{i}" for i in range(96)),
        config=config,
    )
    if postprocess:
        assert result.theta_postprocessed.any()
    else:
        assert np.isfinite(result.raw_theta[result.selected_gene]).any()
    for index in np.flatnonzero(result.selected_gene | result.poisson_gene):
        cells = (
            np.arange(96)
            if result.poisson_gene[index]
            else np.asarray(result.selection.selected_cell_indices)
        )
        y = counts[index, cells]
        mu = np.exp(result.raw_intercept[index]) * result.library_size[cells]
        theta = result.raw_theta[index]
        logpmf = (
            poisson.logpmf(y, mu)
            if np.isinf(theta)
            else nbinom.logpmf(y, theta, theta / (theta + mu))
        )
        np.testing.assert_allclose(
            result.fit_objective[index], -logpmf.sum(), rtol=1e-9, atol=1e-7
        )


def test_failed_refit_is_excluded_from_regularization(monkeypatch):
    counts = generate_counts(genes=80, cells=96, seed=42)
    kwargs = dict(
        gene_names=tuple(f"g{i}" for i in range(80)),
        cell_names=tuple(f"c{i}" for i in range(96)),
        config=SCTConfig.r_v2(n_genes=None, n_cells=None),
    )
    baseline = fit_transform(counts, **kwargs)
    failed = np.flatnonzero(baseline.selected_gene & baseline.converged)[0]
    original = core.offset_intercept

    def fail_one(y, *args):
        if np.array_equal(y, counts[failed]):
            raise RuntimeError("simulated solver failure")
        return original(y, *args)

    monkeypatch.setattr(core, "offset_intercept", fail_one)
    result = fit_transform(counts, **kwargs, selection=baseline.selection)
    assert not result.converged[failed]
    assert result.fit_status[failed] == "regularized_after_fit_failure"
    assert "refit failed" in result.fit_message[failed]
    assert np.isfinite(result.regularized_intercept[failed])


def test_analytic_poisson_and_unfitted_states():
    counts = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    result = fit_transform(
        counts,
        gene_names=("constant", "excluded"),
        cell_names=("a", "b", "c", "d"),
        config=SCTConfig.r_v2(min_cells=1, n_cells=None, n_genes=None),
    )
    np.testing.assert_array_equal(result.converged, [True, False])
    assert result.fit_status == ("poisson", "insufficient_cells")
    assert result.fit_objective[0] == pytest.approx(4.0)
    assert np.isnan(result.fit_objective[1])
