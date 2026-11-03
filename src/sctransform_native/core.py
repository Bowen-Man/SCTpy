from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy import sparse

from ._fit import (
    FitMethod,
    GeneFit,
    ThetaPostprocess,
    fit_cox_reid_offset,
    fit_cox_reid_offset_batch,
    fit_nb_offset,
    local_median_dispersion_trend,
    offset_intercept,
    poisson_offset_intercept,
    theta_moment_ratio_mask,
)
from ._matrix import (
    CountMatrix,
    MatrixOrientation,
    dense_gene_block,
    dense_gene_row,
    global_nonzero_median,
    matrix_statistics,
    prepare_counts,
)
from ._regularize import (
    DispersionRegularization,
    RegularizationMethod,
    RegularizationSupport,
    regularize_parameters,
)
from .selection import (
    SelectionManifest,
    create_selection_manifest,
    validate_selection_manifest,
)

StatisticalMode = Literal["native", "r_v2", "experimental"]
ResidualOutput = Literal["full", "summary"]
ResidualDType = Literal["float64", "float32"]


@dataclass(frozen=True)
class SCTConfig:
    min_cells: int = 5
    n_genes: int | None = 2000
    n_cells: int | None = 2000
    seed: int = 1448145
    clip_range: tuple[float, float] | None = None
    fit_method: FitMethod = "nb_mle"
    bandwidth_adjust: float = 3.0
    outlier_mad_threshold: float = 10.0
    dispersion_regularization: DispersionRegularization = "native"
    regularization_method: RegularizationMethod = "native"
    regularization_support: RegularizationSupport = "extrapolate"
    theta_postprocess: ThetaPostprocess = "none"
    theta_moment_ratio_threshold: float = 1.0e-3
    theta_min: float = 1e-4
    theta_max: float = 1e8
    optimizer_maxiter: int = 200
    optimizer_ftol: float = 1e-10
    optimizer_gtol: float = 1e-6
    residual_block_size: int = 256
    residual_output: ResidualOutput = "full"
    residual_dtype: ResidualDType = "float64"
    cox_reid_gene_bin_size: int | None = None
    cox_reid_batch_size: int | None = None

    @classmethod
    def r_v2(
        cls,
        *,
        min_cells: int = 5,
        n_genes: int | None = 2000,
        n_cells: int | None = 2000,
        seed: int = 1448145,
        residual_block_size: int = 256,
        residual_output: ResidualOutput = "full",
        residual_dtype: ResidualDType = "float64",
        cox_reid_batch_size: int | None = 64,
    ) -> SCTConfig:
        """Build the validated R-v2 compatibility configuration."""
        return cls(
            min_cells=min_cells,
            n_genes=n_genes,
            n_cells=n_cells,
            seed=seed,
            fit_method="cox_reid",
            cox_reid_gene_bin_size=500,
            cox_reid_batch_size=cox_reid_batch_size,
            dispersion_regularization="sct_v2",
            regularization_method="r_stats",
            regularization_support="clamp",
            theta_postprocess="moment_ratio",
            residual_block_size=residual_block_size,
            residual_output=residual_output,
            residual_dtype=residual_dtype,
        )

    @property
    def statistical_mode(self) -> StatisticalMode:
        """Classify the selected statistical path without changing its behavior."""
        signature = (
            self.fit_method,
            self.cox_reid_gene_bin_size,
            self.dispersion_regularization,
            self.regularization_method,
            self.regularization_support,
            self.theta_postprocess,
        )
        if signature == (
            "nb_mle",
            None,
            "native",
            "native",
            "extrapolate",
            "none",
        ):
            return "native"
        if (
            signature == ("cox_reid", 500, "sct_v2", "r_stats", "clamp", "moment_ratio")
            and self.clip_range is None
            and self.bandwidth_adjust == 3.0
            and self.outlier_mad_threshold == 10.0
            and self.theta_moment_ratio_threshold == 1.0e-3
            and self.theta_min == 1.0e-4
            and self.theta_max == 1.0e8
        ):
            return "r_v2"
        return "experimental"

    def __post_init__(self) -> None:
        if self.min_cells < 1:
            raise ValueError("min_cells must be at least 1")
        if self.n_genes is not None and self.n_genes < 1:
            raise ValueError("n_genes must be positive or None")
        if self.n_cells is not None and self.n_cells < 2:
            raise ValueError("n_cells must be at least 2 or None")
        if self.fit_method not in {"nb_mle", "cox_reid"}:
            raise ValueError("fit_method must be 'nb_mle' or 'cox_reid'")
        if self.cox_reid_gene_bin_size is not None and self.cox_reid_gene_bin_size < 1:
            raise ValueError("cox_reid_gene_bin_size must be positive or None")
        if self.cox_reid_batch_size is not None and self.cox_reid_batch_size < 1:
            raise ValueError("cox_reid_batch_size must be positive or None")
        if self.bandwidth_adjust <= 0:
            raise ValueError("bandwidth_adjust must be positive")
        if self.outlier_mad_threshold <= 0:
            raise ValueError("outlier_mad_threshold must be positive")
        if self.dispersion_regularization not in {"native", "sct_v2"}:
            raise ValueError("dispersion_regularization must be 'native' or 'sct_v2'")
        if self.regularization_method not in {"native", "r_stats"}:
            raise ValueError("regularization_method must be 'native' or 'r_stats'")
        if self.regularization_support not in {"extrapolate", "clamp"}:
            raise ValueError("regularization_support must be 'extrapolate' or 'clamp'")
        if self.theta_postprocess not in {"none", "moment_ratio"}:
            raise ValueError("theta_postprocess must be 'none' or 'moment_ratio'")
        if (
            not np.isfinite(self.theta_moment_ratio_threshold)
            or self.theta_moment_ratio_threshold <= 0.0
        ):
            raise ValueError("theta_moment_ratio_threshold must be finite and positive")
        if not 0 < self.theta_min < self.theta_max:
            raise ValueError("theta bounds must satisfy 0 < theta_min < theta_max")
        if self.optimizer_maxiter < 1 or self.residual_block_size < 1:
            raise ValueError("iteration and block sizes must be positive")
        if self.residual_output not in {"full", "summary"}:
            raise ValueError("residual_output must be 'full' or 'summary'")
        if self.residual_dtype not in {"float64", "float32"}:
            raise ValueError("residual_dtype must be 'float64' or 'float32'")
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
    theta_postprocessed: np.ndarray
    parameter_outlier: np.ndarray
    fit_objective: np.ndarray
    fit_iterations: np.ndarray
    fit_message: tuple[str, ...]
    fit_status: tuple[str, ...]
    residuals: np.ndarray | None
    corrected_umi: sparse.csr_matrix | None
    corrected_umi_depth: float | None
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
    store_residuals: bool,
    residual_dtype: ResidualDType,
    corrected_umi_depth: float | None,
) -> tuple[
    np.ndarray | None,
    sparse.csr_matrix | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    genes, cells = counts_gxc.shape
    working_dtype = np.dtype(residual_dtype)
    working_library_size = library_size.astype(working_dtype, copy=False)
    residuals = (
        np.full((genes, cells), np.nan, dtype=working_dtype)
        if store_residuals
        else None
    )
    residual_mean = np.full(genes, np.nan, dtype=np.float64)
    residual_variance = np.full(genes, np.nan, dtype=np.float64)
    corrected_columns: list[np.ndarray] = []
    corrected_values: list[np.ndarray] = []
    corrected_row_nnz = (
        np.zeros(genes, dtype=np.int64) if corrected_umi_depth is not None else None
    )
    modeled_indices = np.flatnonzero(
        eligible & np.isfinite(intercept) & ~np.isnan(theta) & (theta > 0)
    )
    for start in range(0, modeled_indices.size, block_size):
        indices = modeled_indices[start : start + block_size]
        observed_float64 = dense_gene_block(counts_gxc, indices)
        observed = observed_float64.astype(working_dtype, copy=False)
        block_intercept = intercept[indices].astype(working_dtype, copy=False)
        mu = np.exp(block_intercept[:, None]) * working_library_size[None, :]
        inverse_theta = np.zeros(indices.size, dtype=working_dtype)
        finite_theta = np.isfinite(theta[indices])
        inverse_theta[finite_theta] = 1.0 / theta[indices[finite_theta]].astype(
            working_dtype, copy=False
        )
        variance = mu + mu**2 * inverse_theta[:, None]
        model_standard_deviation = np.sqrt(variance)
        denominator = np.maximum(model_standard_deviation, standard_deviation_floor)
        block = (observed - mu) / denominator
        block = np.clip(block, clip_range[0], clip_range[1])
        residual_mean[indices] = np.mean(block, axis=1, dtype=np.float64)
        if cells > 1:
            residual_variance[indices] = np.var(block, axis=1, ddof=1, dtype=np.float64)
        if residuals is not None:
            residuals[indices] = block

        if corrected_umi_depth is not None:
            if working_dtype == np.dtype(np.float64):
                corrected_mu = mu
                corrected_standard_deviation = model_standard_deviation
                corrected_inverse_theta = inverse_theta
            else:
                corrected_mu = np.exp(intercept[indices, None]) * library_size[None, :]
                corrected_inverse_theta = np.zeros(indices.size, dtype=np.float64)
                corrected_inverse_theta[finite_theta] = (
                    1.0 / theta[indices[finite_theta]]
                )
                corrected_standard_deviation = np.sqrt(
                    corrected_mu + corrected_mu**2 * corrected_inverse_theta[:, None]
                )
            target_mu = np.exp(intercept[indices, None]) * corrected_umi_depth
            target_variance = (
                target_mu + target_mu**2 * corrected_inverse_theta[:, None]
            )
            pearson = (observed_float64 - corrected_mu) / corrected_standard_deviation
            corrected = np.maximum(
                np.rint(target_mu + pearson * np.sqrt(target_variance)), 0.0
            )
            if not np.all(np.isfinite(corrected)):
                raise OverflowError(
                    "corrected UMI calculation produced non-finite values"
                )
            if np.any(corrected > np.iinfo(np.int64).max):
                raise OverflowError("corrected UMI values exceed int64 range")
            corrected_block = sparse.csr_matrix(corrected.astype(np.int64))
            assert corrected_row_nnz is not None
            corrected_row_nnz[indices] = np.diff(corrected_block.indptr)
            corrected_columns.append(corrected_block.indices)
            corrected_values.append(corrected_block.data)

    valid_hvg = modeled_indices[np.isfinite(residual_variance[modeled_indices])]
    order_within = np.lexsort((valid_hvg, -residual_variance[valid_hvg]))
    hvg_order = valid_hvg[order_within]
    hvg_rank = np.full(genes, -1, dtype=np.int64)
    hvg_rank[hvg_order] = np.arange(hvg_order.size, dtype=np.int64)
    corrected_umi: sparse.csr_matrix | None = None
    if corrected_umi_depth is not None:
        assert corrected_row_nnz is not None
        if corrected_values:
            indptr = np.empty(genes + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(corrected_row_nnz, out=indptr[1:])
            corrected_umi = sparse.csr_matrix(
                (
                    np.concatenate(corrected_values),
                    np.concatenate(corrected_columns),
                    indptr,
                ),
                shape=(genes, cells),
                dtype=np.int64,
            )
        else:
            corrected_umi = sparse.csr_matrix((genes, cells), dtype=np.int64)
    return (
        residuals,
        corrected_umi,
        residual_mean,
        residual_variance,
        hvg_order,
        hvg_rank,
    )


def fit_transform(
    counts: object,
    *,
    gene_names: Sequence[str],
    cell_names: Sequence[str],
    matrix_orientation: MatrixOrientation = "genes_by_cells",
    config: SCTConfig | None = None,
    selection: SelectionManifest | None = None,
    return_corrected_umi: bool = False,
    corrected_umi_depth: float | None = None,
    batch: object | None = None,
    covariates: object | None = None,
) -> SCTResult:
    """Fit the documented SCT v2 core and return configured Pearson residuals."""
    total_start = perf_counter()
    timing: dict[str, float] = {}
    if corrected_umi_depth is not None and not return_corrected_umi:
        raise ValueError("corrected_umi_depth requires return_corrected_umi=True")
    if corrected_umi_depth is not None and (
        not np.isfinite(corrected_umi_depth) or corrected_umi_depth <= 0.0
    ):
        raise ValueError("corrected_umi_depth must be finite and positive")
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
    step1_candidate = eligible & (gene_variance > gene_mean)
    poisson_gene = eligible & ((gene_mean < 0.001) | (gene_variance <= gene_mean))
    timing["input_statistics"] = perf_counter() - stage_start

    stage_start = perf_counter()
    if selection is None:
        selection = create_selection_manifest(
            counts_gxc=counts_gxc,
            gene_names=checked_genes,
            cell_names=checked_cells,
            eligible_genes=step1_candidate,
            geometric_mean=geometric_mean,
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
    if np.any(eligible & ~poisson_gene) and not np.any(
        selected_gene_mask & ~poisson_gene
    ):
        raise ValueError("selection must include at least one non-Poisson gene")
    fit_counts = counts_gxc[:, selected_cells]
    fit_library_size = library_size[selected_cells]

    raw_intercept = np.full(genes, np.nan, dtype=np.float64)
    raw_theta = np.full(genes, np.nan, dtype=np.float64)
    converged = np.zeros(genes, dtype=bool)
    fit_objective = np.full(genes, np.nan, dtype=np.float64)
    fit_iterations = np.zeros(genes, dtype=np.int64)
    fit_message = np.full(genes, "not fit", dtype=object)
    status = np.full(genes, "insufficient_cells", dtype=object)
    poisson_intercept = np.full(genes, np.nan, dtype=np.float64)

    stage_start = perf_counter()
    for gene_index in np.flatnonzero(eligible & poisson_gene):
        y = dense_gene_row(counts_gxc, int(gene_index))
        poisson_intercept[gene_index] = poisson_offset_intercept(y, library_size)
        raw_intercept[gene_index] = poisson_intercept[gene_index]
        raw_theta[gene_index] = np.inf
        fit_message[gene_index] = "analytic Poisson offset"
        status[gene_index] = "poisson"

    fit_candidates = selected_genes[step1_candidate[selected_genes]]
    gene_fits: list[tuple[int, GeneFit]] = []
    if (
        resolved_config.fit_method == "cox_reid"
        and resolved_config.cox_reid_batch_size is not None
    ):
        batch_size = resolved_config.cox_reid_batch_size
        for start in range(0, fit_candidates.size, batch_size):
            batch_indices = fit_candidates[start : start + batch_size]
            batch_counts = dense_gene_block(fit_counts, batch_indices)
            gene_fits.extend(
                zip(
                    (int(index) for index in batch_indices),
                    fit_cox_reid_offset_batch(batch_counts, fit_library_size),
                    strict=True,
                )
            )
    else:
        for gene_index in fit_candidates:
            y = dense_gene_row(fit_counts, int(gene_index))
            if resolved_config.fit_method == "cox_reid":
                gene_fit = fit_cox_reid_offset(y, fit_library_size)
            else:
                gene_fit = fit_nb_offset(
                    y,
                    fit_library_size,
                    theta_min=resolved_config.theta_min,
                    theta_max=resolved_config.theta_max,
                    maxiter=resolved_config.optimizer_maxiter,
                    ftol=resolved_config.optimizer_ftol,
                    gtol=resolved_config.optimizer_gtol,
                )
            gene_fits.append((int(gene_index), gene_fit))

    for gene_index, gene_fit in gene_fits:
        raw_intercept[gene_index] = gene_fit.intercept
        raw_theta[gene_index] = gene_fit.theta
        converged[gene_index] = gene_fit.converged
        fit_objective[gene_index] = gene_fit.objective
        fit_iterations[gene_index] = gene_fit.iterations
        fit_message[gene_index] = gene_fit.message
        status[gene_index] = "fit" if gene_fit.converged else "fit_failed"

    if resolved_config.fit_method == "cox_reid":
        fitted = selected_genes[
            step1_candidate[selected_genes] & converged[selected_genes]
        ]
        raw_dispersion = np.zeros(fitted.size, dtype=np.float64)
        finite_theta = np.isfinite(raw_theta[fitted])
        raw_dispersion[finite_theta] = 1.0 / raw_theta[fitted[finite_theta]]
        preliminary_abundance = np.exp(raw_intercept[fitted]) * float(
            np.mean(fit_library_size)
        )
        dispersion_trend = local_median_dispersion_trend(
            preliminary_abundance,
            raw_dispersion,
            gene_bin_size=resolved_config.cox_reid_gene_bin_size,
        )
        for gene_index, dispersion in zip(fitted, dispersion_trend, strict=True):
            y = dense_gene_row(fit_counts, int(gene_index))
            raw_intercept[gene_index], iterations = offset_intercept(
                y, fit_library_size, float(dispersion)
            )
            fit_iterations[gene_index] += iterations
            fit_message[gene_index] = "Cox-Reid dispersion with local-median refit"
    timing["fit"] = perf_counter() - stage_start

    stage_start = perf_counter()
    theta_postprocessed = np.zeros(genes, dtype=bool)
    if resolved_config.theta_postprocess == "moment_ratio":
        candidates = np.flatnonzero(
            step1_candidate & selected_gene_mask & ~np.isnan(raw_theta)
        )
        theta_postprocessed[candidates] = theta_moment_ratio_mask(
            gene_mean[candidates],
            gene_variance[candidates],
            raw_theta[candidates],
            threshold=resolved_config.theta_moment_ratio_threshold,
        )
        raw_theta[theta_postprocessed] = np.inf
        for gene_index in np.flatnonzero(theta_postprocessed):
            status[gene_index] = "theta_postprocessed"
    timing["theta_postprocess"] = perf_counter() - stage_start

    non_poisson = eligible & ~poisson_gene
    nonfinite_theta = non_poisson & selected_gene_mask & ~np.isfinite(raw_theta)
    training_mask = (
        non_poisson
        & selected_gene_mask
        & converged
        & np.isfinite(raw_theta)
        & ~theta_postprocessed
    )
    regularized_intercept = np.full(genes, np.nan, dtype=np.float64)
    regularized_theta = np.full(genes, np.nan, dtype=np.float64)
    parameter_outlier = np.zeros(genes, dtype=bool)
    bandwidth: float | None = None
    stage_start = perf_counter()
    if np.any(non_poisson):
        (
            predicted_intercept,
            predicted_theta,
            smoother_outlier,
            bandwidth,
        ) = regularize_parameters(
            gene_mean=gene_mean,
            geometric_mean=geometric_mean,
            raw_intercept=raw_intercept,
            raw_theta=raw_theta,
            training_mask=training_mask,
            outlier_evaluation_mask=(non_poisson & selected_gene_mask & converged),
            prediction_mask=non_poisson,
            bandwidth_adjust=resolved_config.bandwidth_adjust,
            outlier_mad_threshold=resolved_config.outlier_mad_threshold,
            theta_min=resolved_config.theta_min,
            theta_max=resolved_config.theta_max,
            dispersion_regularization=resolved_config.dispersion_regularization,
            regularization_method=resolved_config.regularization_method,
            regularization_support=resolved_config.regularization_support,
        )
        parameter_outlier |= smoother_outlier | theta_postprocessed | nonfinite_theta
        regularized_intercept[non_poisson] = predicted_intercept[non_poisson]
        regularized_theta[non_poisson] = predicted_theta[non_poisson]
        for gene_index in np.flatnonzero(non_poisson & ~selected_gene_mask):
            status[gene_index] = "regularized_only"
        for gene_index in np.flatnonzero(non_poisson & selected_gene_mask & ~converged):
            status[gene_index] = "regularized_after_fit_failure"

    for gene_index in np.flatnonzero(eligible & poisson_gene):
        regularized_intercept[gene_index] = poisson_intercept[gene_index]
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
    resolved_corrected_umi_depth = (
        float(np.median(library_size))
        if return_corrected_umi and corrected_umi_depth is None
        else corrected_umi_depth
    )
    stage_start = perf_counter()
    (
        residuals,
        corrected_umi,
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
        store_residuals=resolved_config.residual_output == "full",
        residual_dtype=resolved_config.residual_dtype,
        corrected_umi_depth=resolved_corrected_umi_depth,
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
        theta_postprocessed=theta_postprocessed,
        parameter_outlier=parameter_outlier,
        fit_objective=fit_objective,
        fit_iterations=fit_iterations,
        fit_message=tuple(str(value) for value in fit_message),
        fit_status=tuple(str(value) for value in status),
        residuals=residuals,
        corrected_umi=corrected_umi,
        corrected_umi_depth=resolved_corrected_umi_depth,
        residual_mean=residual_mean,
        residual_variance=residual_variance,
        hvg_order=hvg_order,
        hvg_rank=hvg_rank,
        clip_range=clip_range,
        nonzero_median=nonzero_median,
        regularization_bandwidth=bandwidth,
        timing=timing,
    )
