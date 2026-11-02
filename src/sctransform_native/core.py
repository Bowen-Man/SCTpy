from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ._fit import fit_nb_offset, poisson_offset_intercept
from ._matrix import (
    CountMatrix,
    MatrixOrientation,
    dense_gene_block,
    dense_gene_row,
    global_nonzero_median,
    matrix_statistics,
    prepare_counts,
)
from ._regularize import DispersionRegularization, regularize_parameters
from .selection import (
    SelectionManifest,
    create_selection_manifest,
    validate_selection_manifest,
)


@dataclass(frozen=True)
class SCTConfig:
    min_cells: int = 5
    n_genes: int | None = 2000
    n_cells: int | None = 2000
    seed: int = 1448145
    clip_range: tuple[float, float] | None = None
    bandwidth_adjust: float = 3.0
    outlier_mad_threshold: float = 10.0
    dispersion_regularization: DispersionRegularization = "native"
    theta_min: float = 1e-4
    theta_max: float = 1e8
    optimizer_maxiter: int = 200
    optimizer_ftol: float = 1e-10
    optimizer_gtol: float = 1e-6
    residual_block_size: int = 256

    def __post_init__(self) -> None:
        if self.min_cells < 1:
            raise ValueError("min_cells must be at least 1")
        if self.n_genes is not None and self.n_genes < 1:
            raise ValueError("n_genes must be positive or None")
        if self.n_cells is not None and self.n_cells < 2:
            raise ValueError("n_cells must be at least 2 or None")
        if self.bandwidth_adjust <= 0:
            raise ValueError("bandwidth_adjust must be positive")
        if self.outlier_mad_threshold <= 0:
            raise ValueError("outlier_mad_threshold must be positive")
        if self.dispersion_regularization not in {"native", "sct_v2"}:
            raise ValueError("dispersion_regularization must be 'native' or 'sct_v2'")
        if not 0 < self.theta_min < self.theta_max:
            raise ValueError("theta bounds must satisfy 0 < theta_min < theta_max")
        if self.optimizer_maxiter < 1 or self.residual_block_size < 1:
            raise ValueError("iteration and block sizes must be positive")
        if self.clip_range is not None:
            low, high = self.clip_range
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError("clip_range must contain two finite increasing values")


@dataclass(frozen=True)
class SCTResult:
    gene_names: tuple[str, ...]
    cell_names: tuple[str, ...]
    selection: SelectionManifest
    config: SCTConfig
    library_size: np.ndarray
    gene_mean: np.ndarray
    gene_variance: np.ndarray
    detection_count: np.ndarray
    geometric_mean: np.ndarray
    raw_intercept: np.ndarray
    raw_theta: np.ndarray
    regularized_intercept: np.ndarray
    regularized_theta: np.ndarray
    slope: np.ndarray
    poisson_gene: np.ndarray
    eligible_gene: np.ndarray
    selected_gene: np.ndarray
    converged: np.ndarray
    parameter_outlier: np.ndarray
    fit_objective: np.ndarray
    fit_iterations: np.ndarray
    fit_message: tuple[str, ...]
    fit_status: tuple[str, ...]
    residuals: np.ndarray
    residual_mean: np.ndarray
    residual_variance: np.ndarray
    hvg_order: np.ndarray
    hvg_rank: np.ndarray
    clip_range: tuple[float, float]
    nonzero_median: float
    regularization_bandwidth: float | None
    timing: dict[str, float]

    def highly_variable_genes(self, n_top_genes: int) -> tuple[str, ...]:
        if n_top_genes < 1:
            raise ValueError("n_top_genes must be positive")
        indices = self.hvg_order[:n_top_genes]
        return tuple(self.gene_names[index] for index in indices)

    def parameter_dict(self) -> dict[str, np.ndarray]:
        return {
            "intercept": self.raw_intercept.copy(),
            "slope": self.slope.copy(),
            "theta": self.raw_theta.copy(),
            "regularized_intercept": self.regularized_intercept.copy(),
            "regularized_theta": self.regularized_theta.copy(),
        }


def _residuals_and_summary(
    counts_gxc: CountMatrix,
    *,
    library_size: np.ndarray,
    eligible: np.ndarray,
    intercept: np.ndarray,
    theta: np.ndarray,
    clip_range: tuple[float, float],
    standard_deviation_floor: float,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    genes, cells = counts_gxc.shape
    residuals = np.full((genes, cells), np.nan, dtype=np.float64)
    modeled_indices = np.flatnonzero(
        eligible & np.isfinite(intercept) & ~np.isnan(theta) & (theta > 0)
    )
    for start in range(0, modeled_indices.size, block_size):
        indices = modeled_indices[start : start + block_size]
        observed = dense_gene_block(counts_gxc, indices)
        mu = np.exp(intercept[indices, None]) * library_size[None, :]
        inverse_theta = np.zeros(indices.size, dtype=np.float64)
        finite_theta = np.isfinite(theta[indices])
        inverse_theta[finite_theta] = 1.0 / theta[indices[finite_theta]]
        variance = mu + mu**2 * inverse_theta[:, None]
        denominator = np.maximum(np.sqrt(variance), standard_deviation_floor)
        block = (observed - mu) / denominator
        residuals[indices] = np.clip(block, clip_range[0], clip_range[1])

    residual_mean = np.full(genes, np.nan, dtype=np.float64)
    residual_variance = np.full(genes, np.nan, dtype=np.float64)
    if modeled_indices.size:
        residual_mean[modeled_indices] = np.mean(residuals[modeled_indices], axis=1)
        if cells > 1:
            residual_variance[modeled_indices] = np.var(
                residuals[modeled_indices], axis=1, ddof=1
            )

    valid_hvg = modeled_indices[np.isfinite(residual_variance[modeled_indices])]
    order_within = np.lexsort((valid_hvg, -residual_variance[valid_hvg]))
    hvg_order = valid_hvg[order_within]
    hvg_rank = np.full(genes, -1, dtype=np.int64)
    hvg_rank[hvg_order] = np.arange(hvg_order.size, dtype=np.int64)
    return residuals, residual_mean, residual_variance, hvg_order, hvg_rank


def fit_transform(
    counts: object,
    *,
    gene_names: Sequence[str],
    cell_names: Sequence[str],
    matrix_orientation: MatrixOrientation = "genes_by_cells",
    config: SCTConfig | None = None,
    selection: SelectionManifest | None = None,
    return_corrected_umi: bool = False,
    batch: object | None = None,
    covariates: object | None = None,
) -> SCTResult:
    """Fit the documented SCT v2 core and return float64 Pearson residuals."""
    total_start = perf_counter()
    timing: dict[str, float] = {}
    if return_corrected_umi:
        raise NotImplementedError("corrected UMI counts are not implemented in 0.1.0a1")
    if batch is not None:
        raise NotImplementedError("batch models are not implemented in 0.1.0a1")
    if covariates is not None:
        raise NotImplementedError(
            "additional covariates are not implemented in 0.1.0a1"
        )
    resolved_config = config if config is not None else SCTConfig()
    stage_start = perf_counter()
    counts_gxc, checked_genes, checked_cells = prepare_counts(
        counts,
        gene_names=gene_names,
        cell_names=cell_names,
        orientation=matrix_orientation,
    )
    timing["input_validation"] = perf_counter() - stage_start
    genes, cells = counts_gxc.shape
    stage_start = perf_counter()
    library_size = np.asarray(counts_gxc.sum(axis=0), dtype=np.float64).ravel()
    gene_mean, gene_variance, detection_count, geometric_mean = matrix_statistics(
        counts_gxc
    )
    eligible = detection_count >= resolved_config.min_cells
    if not np.any(eligible):
        raise ValueError("no genes satisfy min_cells")
    timing["input_statistics"] = perf_counter() - stage_start

    stage_start = perf_counter()
    if selection is None:
        selection = create_selection_manifest(
            gene_names=checked_genes,
            cell_names=checked_cells,
            eligible_genes=eligible,
            min_cells=resolved_config.min_cells,
            requested_genes=resolved_config.n_genes,
            requested_cells=resolved_config.n_cells,
            seed=resolved_config.seed,
        )
    validate_selection_manifest(
        selection,
        gene_names=checked_genes,
        cell_names=checked_cells,
        eligible_genes=eligible,
        min_cells=resolved_config.min_cells,
    )
    timing["selection"] = perf_counter() - stage_start

    selected_genes = np.asarray(selection.selected_gene_indices, dtype=np.int64)
    selected_cells = np.asarray(selection.selected_cell_indices, dtype=np.int64)
    selected_gene_mask = np.zeros(genes, dtype=bool)
    selected_gene_mask[selected_genes] = True
    fit_counts = counts_gxc[:, selected_cells]
    fit_library_size = library_size[selected_cells]
    fit_mean, fit_variance, _, _ = matrix_statistics(fit_counts)
    poisson_gene = eligible & ((fit_mean < 0.001) | (fit_variance <= fit_mean))

    raw_intercept = np.full(genes, np.nan, dtype=np.float64)
    raw_theta = np.full(genes, np.nan, dtype=np.float64)
    converged = np.zeros(genes, dtype=bool)
    fit_objective = np.full(genes, np.nan, dtype=np.float64)
    fit_iterations = np.zeros(genes, dtype=np.int64)
    fit_message = np.full(genes, "not fit", dtype=object)
    status = np.full(genes, "insufficient_cells", dtype=object)

    stage_start = perf_counter()
    for gene_index in np.flatnonzero(eligible & poisson_gene):
        y = dense_gene_row(fit_counts, int(gene_index))
        raw_intercept[gene_index] = poisson_offset_intercept(y, fit_library_size)
        raw_theta[gene_index] = np.inf
        fit_message[gene_index] = "analytic Poisson offset"
        status[gene_index] = "poisson"

    for gene_index in selected_genes:
        if poisson_gene[gene_index]:
            continue
        y = dense_gene_row(fit_counts, int(gene_index))
        gene_fit = fit_nb_offset(
            y,
            fit_library_size,
            theta_min=resolved_config.theta_min,
            theta_max=resolved_config.theta_max,
            maxiter=resolved_config.optimizer_maxiter,
            ftol=resolved_config.optimizer_ftol,
            gtol=resolved_config.optimizer_gtol,
        )
        raw_intercept[gene_index] = gene_fit.intercept
        raw_theta[gene_index] = gene_fit.theta
        converged[gene_index] = gene_fit.converged
        fit_objective[gene_index] = gene_fit.objective
        fit_iterations[gene_index] = gene_fit.iterations
        fit_message[gene_index] = gene_fit.message
        status[gene_index] = "fit" if gene_fit.converged else "fit_failed"
    timing["fit"] = perf_counter() - stage_start

    non_poisson = eligible & ~poisson_gene
    training_mask = non_poisson & selected_gene_mask & converged
    regularized_intercept = np.full(genes, np.nan, dtype=np.float64)
    regularized_theta = np.full(genes, np.nan, dtype=np.float64)
    parameter_outlier = np.zeros(genes, dtype=bool)
    bandwidth: float | None = None
    stage_start = perf_counter()
    if np.any(non_poisson):
        (
            predicted_intercept,
            predicted_theta,
            parameter_outlier,
            bandwidth,
        ) = regularize_parameters(
            gene_mean=fit_mean,
            geometric_mean=geometric_mean,
            raw_intercept=raw_intercept,
            raw_theta=raw_theta,
            training_mask=training_mask,
            prediction_mask=non_poisson,
            bandwidth_adjust=resolved_config.bandwidth_adjust,
            outlier_mad_threshold=resolved_config.outlier_mad_threshold,
            theta_min=resolved_config.theta_min,
            theta_max=resolved_config.theta_max,
            dispersion_regularization=resolved_config.dispersion_regularization,
        )
        regularized_intercept[non_poisson] = predicted_intercept[non_poisson]
        regularized_theta[non_poisson] = predicted_theta[non_poisson]
        for gene_index in np.flatnonzero(non_poisson & ~selected_gene_mask):
            status[gene_index] = "regularized_only"
        for gene_index in np.flatnonzero(non_poisson & selected_gene_mask & ~converged):
            status[gene_index] = "regularized_after_fit_failure"

    for gene_index in np.flatnonzero(eligible & poisson_gene):
        regularized_intercept[gene_index] = raw_intercept[gene_index]
        regularized_theta[gene_index] = np.inf
    timing["regularization"] = perf_counter() - stage_start

    slope = np.full(genes, np.nan, dtype=np.float64)
    slope[eligible] = np.log(10.0)
    clip_range = (
        resolved_config.clip_range
        if resolved_config.clip_range is not None
        else (-float(np.sqrt(cells)), float(np.sqrt(cells)))
    )
    nonzero_median = global_nonzero_median(counts_gxc)
    stage_start = perf_counter()
    (
        residuals,
        residual_mean,
        residual_variance,
        hvg_order,
        hvg_rank,
    ) = _residuals_and_summary(
        counts_gxc,
        library_size=library_size,
        eligible=eligible,
        intercept=regularized_intercept,
        theta=regularized_theta,
        clip_range=clip_range,
        standard_deviation_floor=nonzero_median / 5.0,
        block_size=resolved_config.residual_block_size,
    )
    timing["residuals_and_hvg"] = perf_counter() - stage_start
    timing["total"] = perf_counter() - total_start

    return SCTResult(
        gene_names=checked_genes,
        cell_names=checked_cells,
        selection=selection,
        config=resolved_config,
        library_size=library_size,
        gene_mean=gene_mean,
        gene_variance=gene_variance,
        detection_count=detection_count,
        geometric_mean=geometric_mean,
        raw_intercept=raw_intercept,
        raw_theta=raw_theta,
        regularized_intercept=regularized_intercept,
        regularized_theta=regularized_theta,
        slope=slope,
        poisson_gene=poisson_gene,
        eligible_gene=eligible,
        selected_gene=selected_gene_mask,
        converged=converged,
        parameter_outlier=parameter_outlier,
        fit_objective=fit_objective,
        fit_iterations=fit_iterations,
        fit_message=tuple(str(value) for value in fit_message),
        fit_status=tuple(str(value) for value in status),
        residuals=residuals,
        residual_mean=residual_mean,
        residual_variance=residual_variance,
        hvg_order=hvg_order,
        hvg_rank=hvg_rank,
        clip_range=clip_range,
        nonzero_median=nonzero_median,
        regularization_bandwidth=bandwidth,
        timing=timing,
    )
