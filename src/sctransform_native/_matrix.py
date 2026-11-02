from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy import sparse

MatrixOrientation = Literal["genes_by_cells", "cells_by_genes"]
CountMatrix = np.ndarray | sparse.csr_matrix


def _validate_names(
    names: Sequence[str], *, expected: int, kind: str
) -> tuple[str, ...]:
    values = tuple(names)
    if len(values) != expected:
        raise ValueError(f"{kind}_names has length {len(values)}; expected {expected}")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{kind}_names must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{kind}_names must be unique")
    return values


def prepare_counts(
    counts: object,
    *,
    gene_names: Sequence[str],
    cell_names: Sequence[str],
    orientation: MatrixOrientation,
) -> tuple[CountMatrix, tuple[str, ...], tuple[str, ...]]:
    """Validate counts and return a float64 genes-by-cells matrix."""
    if orientation not in ("genes_by_cells", "cells_by_genes"):
        raise ValueError(
            "orientation must be 'genes_by_cells' or 'cells_by_genes'"
        )

    if sparse.issparse(counts):
        matrix = counts.T if orientation == "cells_by_genes" else counts
        if matrix.ndim != 2:
            raise ValueError("counts must be a two-dimensional matrix")
        matrix = matrix.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        values = matrix.data
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise TypeError("counts must contain real numeric values")
        if not np.all(np.isfinite(values)):
            raise ValueError("counts must be finite")
        if np.any(values < 0):
            raise ValueError("counts must be nonnegative")
        if not np.all(values == np.floor(values)):
            raise ValueError("counts must be integer-valued raw UMI counts")
        prepared: CountMatrix = matrix.astype(np.float64, copy=False)
    else:
        array = np.asarray(counts)
        if array.ndim != 2:
            raise ValueError("counts must be a two-dimensional matrix")
        if orientation == "cells_by_genes":
            array = array.T
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise TypeError("counts must contain real numeric values")
        if not np.all(np.isfinite(array)):
            raise ValueError("counts must be finite")
        if np.any(array < 0):
            raise ValueError("counts must be nonnegative")
        if not np.all(array == np.floor(array)):
            raise ValueError("counts must be integer-valued raw UMI counts")
        prepared = np.asarray(array, dtype=np.float64, order="C")

    genes, cells = prepared.shape
    checked_genes = _validate_names(gene_names, expected=genes, kind="gene")
    checked_cells = _validate_names(cell_names, expected=cells, kind="cell")
    if genes == 0 or cells == 0:
        raise ValueError("counts must contain at least one gene and one cell")

    library_size = np.asarray(prepared.sum(axis=0), dtype=np.float64).ravel()
    if np.any(library_size <= 0):
        bad = np.flatnonzero(library_size <= 0)
        raise ValueError(
            "every cell must have positive library size; zero-count cell indices: "
            f"{bad.tolist()}"
        )
    return prepared, checked_genes, checked_cells


def matrix_statistics(
    counts_gxc: CountMatrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, sample variance, detection count, and pseudocount geometric mean."""
    n_cells = counts_gxc.shape[1]
    sums = np.asarray(counts_gxc.sum(axis=1), dtype=np.float64).ravel()
    means = sums / n_cells
    if sparse.issparse(counts_gxc):
        squared = counts_gxc.copy()
        squared.data **= 2
        sum_squares = np.asarray(squared.sum(axis=1), dtype=np.float64).ravel()
        detected = np.asarray(counts_gxc.getnnz(axis=1), dtype=np.int64)
        log_matrix = counts_gxc.copy().astype(np.float64)
        log_matrix.data = np.log1p(log_matrix.data)
        log_sums = np.asarray(log_matrix.sum(axis=1), dtype=np.float64).ravel()
    else:
        sum_squares = np.einsum("gc,gc->g", counts_gxc, counts_gxc)
        detected = np.count_nonzero(counts_gxc, axis=1).astype(np.int64)
        log_sums = np.log1p(counts_gxc).sum(axis=1)

    if n_cells > 1:
        variances = (sum_squares - n_cells * means**2) / (n_cells - 1)
        variances = np.maximum(variances, 0.0)
    else:
        variances = np.zeros_like(means)
    geometric_means = np.expm1(log_sums / n_cells)
    return means, variances, detected, geometric_means


def global_nonzero_median(counts_gxc: CountMatrix) -> float:
    if sparse.issparse(counts_gxc):
        values = counts_gxc.data
    else:
        values = counts_gxc[counts_gxc > 0]
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def dense_gene_row(counts_gxc: CountMatrix, gene_index: int) -> np.ndarray:
    if sparse.issparse(counts_gxc):
        return counts_gxc.getrow(gene_index).toarray().ravel()
    return np.asarray(counts_gxc[gene_index], dtype=np.float64)


def dense_gene_block(counts_gxc: CountMatrix, indices: np.ndarray) -> np.ndarray:
    block = counts_gxc[indices]
    if sparse.issparse(block):
        return block.toarray()
    return np.asarray(block, dtype=np.float64)
