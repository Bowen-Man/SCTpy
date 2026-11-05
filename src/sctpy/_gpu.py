from __future__ import annotations

import importlib
from typing import Any, NamedTuple

import numpy as np

from ._fit import (
    _COX_REID_CORRECTION,
    _DISPERSION_CEILING,
    _DISPERSION_FLOOR,
    _DISPERSION_ROOT_STEPS,
    _LOG_DISPERSION_TOLERANCE,
    GeneFit,
    _cox_reid_fit_result,
    _cox_reid_score_batch,
    _scaled_digamma_difference,
    fit_cox_reid_offset_batch,
    offset_intercept,
)


def _import_cupy() -> tuple[Any, Any]:
    try:
        cp = importlib.import_module("cupy")
        special = importlib.import_module("cupyx.scipy.special")
    except ImportError as error:
        raise ImportError(
            "the CuPy Cox-Reid backend requires an optional CUDA extra"
        ) from error
    return cp, special.digamma


def _cox_reid_score_cupy(
    cp: Any, digamma: Any, log_dispersion: Any, counts: Any, mu: Any
) -> Any:
    dispersion = cp.exp(log_dispersion)
    inverse_dispersion = 1.0 / dispersion
    scaled_mu = mu * dispersion[:, None]
    suggestion = cp.log1p(scaled_mu) - mu / (mu + inverse_dispersion[:, None])
    upper = scaled_mu**2 / (1.0 + scaled_mu)
    lower = scaled_mu**2 * (1.0 / (1.0 + scaled_mu) - 0.5)
    likelihood_term = cp.where(
        scaled_mu < 1.0e-10,
        lower,
        cp.where(
            scaled_mu < 1.0e-4,
            cp.clip(suggestion, lower, upper),
            suggestion,
        ),
    )
    likelihood_score = inverse_dispersion * cp.sum(
        likelihood_term + counts / (mu + inverse_dispersion[:, None]),
        axis=1,
        dtype=cp.float64,
    )

    digamma_score = cp.sum(
        _scaled_digamma_difference(dispersion, counts, xp=cp, psi=digamma), axis=1
    )

    weights = mu / (1.0 + dispersion[:, None] * mu)
    adjustment = (
        0.5
        * dispersion
        * _COX_REID_CORRECTION
        * cp.sum(weights * weights, axis=1, dtype=cp.float64)
        / (cp.sum(weights, axis=1, dtype=cp.float64) + 1.0e-6)
    )
    return likelihood_score - digamma_score + adjustment


class _CupyRootResult(NamedTuple):
    theta: Any
    poisson: Any
    ceiling: Any
    converged: Any
    expansion_iterations: Any
    root_iterations: Any
    expansion_rounds: int
    host_synchronizations: int


def _dispersion_roots_cupy(
    cp: Any, digamma: Any, counts: Any, mu: Any, initial_dispersion: Any
) -> _CupyRootResult:
    lower = cp.full(counts.shape[0], np.log(_DISPERSION_FLOOR), dtype=cp.float64)
    lower_score = _cox_reid_score_cupy(cp, digamma, lower, counts, mu)
    valid = cp.isfinite(lower_score)
    poisson = lower_score < 0.0
    upper_dispersion = cp.minimum(
        cp.maximum(initial_dispersion, _DISPERSION_FLOOR * 4.0),
        _DISPERSION_CEILING,
    )
    upper_score = _cox_reid_score_cupy(
        cp, digamma, cp.log(upper_dispersion), counts, mu
    )
    valid &= poisson | cp.isfinite(upper_score)
    expansion_iterations = cp.zeros(counts.shape[0], dtype=cp.int64)
    expansion_rounds = 0
    host_synchronizations = 0
    while True:
        expanding = (
            valid
            & (~poisson)
            & (upper_score > 0.0)
            & (upper_dispersion < _DISPERSION_CEILING)
        )
        host_synchronizations += 1
        if not bool(cp.any(expanding).get()):
            break
        expansion_rounds += 1
        upper_dispersion = cp.where(
            expanding,
            cp.minimum(upper_dispersion * 4.0, _DISPERSION_CEILING),
            upper_dispersion,
        )
        candidate_score = _cox_reid_score_cupy(
            cp, digamma, cp.log(upper_dispersion), counts, mu
        )
        upper_score = cp.where(expanding, candidate_score, upper_score)
        valid &= poisson | cp.isfinite(upper_score)
        expansion_iterations += expanding

    ceiling = (~poisson) & (upper_score > 0.0)
    bracketed = valid & ~(poisson | ceiling)
    upper = cp.log(upper_dispersion)
    root_iterations = cp.zeros(counts.shape[0], dtype=cp.int64)
    for _ in range(_DISPERSION_ROOT_STEPS):
        midpoint = 0.5 * (lower + upper)
        score = _cox_reid_score_cupy(cp, digamma, midpoint, counts, mu)
        root_iterations += bracketed
        valid &= ~bracketed | cp.isfinite(score)
        bracketed &= valid
        positive = (score > 0.0) & bracketed
        lower = cp.where(positive, midpoint, lower)
        upper = cp.where((~positive) & bracketed, midpoint, upper)

    dispersion = cp.exp(0.5 * (lower + upper))
    dispersion = cp.where(poisson, 0.0, dispersion)
    dispersion = cp.where(ceiling, _DISPERSION_CEILING, dispersion)
    dispersion = cp.where(valid, dispersion, cp.nan)
    theta = cp.where(dispersion == 0.0, cp.inf, 1.0 / dispersion)
    converged = valid & (
        poisson | (bracketed & ((upper - lower) <= _LOG_DISPERSION_TOLERANCE))
    )
    return _CupyRootResult(
        theta,
        poisson,
        ceiling,
        converged,
        expansion_iterations,
        root_iterations,
        expansion_rounds,
        host_synchronizations,
    )


def fit_cox_reid_offset_batch_cupy(
    counts: np.ndarray, exposure: np.ndarray
) -> list[GeneFit]:
    """Fit one CPU-resident gene block with CuPy dispersion roots.

    Preliminary intercepts, objectives, inputs, and outputs remain on CPU. The current
    CuPy device, stream, and allocator are respected.
    """
    counts = np.asarray(counts, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != exposure.size:
        raise ValueError("batched counts must be genes by exposure-sized cells")
    means = np.mean(counts, axis=1, dtype=np.float64)
    if np.any(means == 0.0):
        raise ValueError("batched Cox-Reid rows must contain at least one count")

    variances = np.var(counts, axis=1, ddof=1, dtype=np.float64)
    initial_dispersion = np.maximum(
        (variances - means / float(np.mean(exposure))) / means**2,
        0.0,
    )
    preliminary = [
        offset_intercept(row, exposure, float(dispersion))
        for row, dispersion in zip(counts, initial_dispersion, strict=True)
    ]
    intercept = np.asarray([value[0] for value in preliminary])
    intercept_iterations = np.asarray([value[1] for value in preliminary])
    mu = np.exp(intercept[:, None]) * exposure[None, :]

    cp, digamma = _import_cupy()
    device_result = _dispersion_roots_cupy(
        cp,
        digamma,
        cp.asarray(counts),
        cp.asarray(mu),
        cp.asarray(initial_dispersion),
    )
    theta = cp.asnumpy(device_result.theta)
    expansion_iterations = cp.asnumpy(device_result.expansion_iterations)
    root_iterations = cp.asnumpy(device_result.root_iterations)
    gpu_poisson = cp.asnumpy(device_result.poisson)
    gpu_ceiling = cp.asnumpy(device_result.ceiling)
    root_converged = cp.asnumpy(device_result.converged)

    lower_score = _cox_reid_score_batch(
        np.full(counts.shape[0], np.log(_DISPERSION_FLOOR)), counts, mu
    )
    cpu_poisson = lower_score < 0.0
    ceiling_score = _cox_reid_score_batch(
        np.full(counts.shape[0], np.log(_DISPERSION_CEILING)), counts, mu
    )
    cpu_ceiling = (~cpu_poisson) & (ceiling_score > 0.0)
    fallback_indices = np.flatnonzero(
        (gpu_poisson != cpu_poisson) | (gpu_ceiling != cpu_ceiling)
    )
    fallback = (
        dict(
            zip(
                (int(index) for index in fallback_indices),
                fit_cox_reid_offset_batch(counts[fallback_indices], exposure),
                strict=True,
            )
        )
        if fallback_indices.size
        else {}
    )

    output: list[GeneFit] = []
    for index in range(counts.shape[0]):
        if index in fallback:
            output.append(fallback[index])
            continue
        if gpu_ceiling[index]:
            message = "Cox-Reid dispersion ceiling reached without a root"
        elif gpu_poisson[index]:
            message = "Cox-Reid Poisson boundary"
        else:
            message = (
                "Cox-Reid dispersion root"
                if root_converged[index]
                else "Cox-Reid root did not converge"
            )
        output.append(
            _cox_reid_fit_result(
                counts[index],
                exposure,
                float(intercept[index]),
                float(theta[index]),
                converged=bool(root_converged[index]),
                iterations=int(
                    intercept_iterations[index]
                    + expansion_iterations[index]
                    + root_iterations[index]
                ),
                message=message,
            )
        )
    return output
