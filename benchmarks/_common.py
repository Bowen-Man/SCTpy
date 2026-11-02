from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from sctransform_native import SelectionManifest


def load_fixture(
    path: str | Path,
) -> tuple[np.ndarray | sparse.csr_matrix, tuple[str, ...], tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as fixture:
        if "counts" in fixture.files:
            counts: np.ndarray | sparse.csr_matrix = np.asarray(
                fixture["counts"], dtype=np.int64
            )
        elif str(fixture["format"]) == "csr":
            shape = tuple(int(value) for value in fixture["counts_shape"])
            counts = sparse.csr_matrix(
                (
                    np.asarray(fixture["counts_data"], dtype=np.int64),
                    np.asarray(fixture["counts_indices"]),
                    np.asarray(fixture["counts_indptr"]),
                ),
                shape=shape,
            )
            counts.sum_duplicates()
            counts.eliminate_zeros()
            counts.sort_indices()
        else:
            raise ValueError("fixture must contain dense counts or a CSR matrix")
        genes = tuple(str(value) for value in fixture["gene_names"])
        cells = tuple(str(value) for value in fixture["cell_names"])
    return counts, genes, cells


def load_manifest(path: str | Path) -> SelectionManifest:
    return SelectionManifest.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def count_sha256(counts: np.ndarray | sparse.spmatrix) -> str:
    if not sparse.issparse(counts):
        return hashlib.sha256(np.ascontiguousarray(counts).view(np.uint8)).hexdigest()

    matrix = sparse.csr_matrix(counts, dtype=np.int64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    digest = hashlib.sha256(b"canonical-csr-int64-v1\0")
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<i8").tobytes())
    return digest.hexdigest()


def save_sparse_fixture(
    path: str | Path,
    counts: sparse.spmatrix,
    gene_names: tuple[str, ...],
    cell_names: tuple[str, ...],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = sparse.csr_matrix(counts, dtype=np.int64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    np.savez_compressed(
        destination,
        format=np.asarray("csr"),
        counts_data=matrix.data,
        counts_indices=matrix.indices,
        counts_indptr=matrix.indptr,
        counts_shape=np.asarray(matrix.shape, dtype=np.int64),
        gene_names=np.asarray(gene_names),
        cell_names=np.asarray(cell_names),
    )


def matrix_summary(matrix: np.ndarray | sparse.spmatrix) -> dict[str, Any]:
    if sparse.issparse(matrix):
        return {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "format": matrix.format,
            "nnz": int(matrix.nnz),
            "nbytes": int(
                matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
            ),
        }
    array = np.asarray(matrix)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "format": "dense",
        "nnz": int(np.count_nonzero(array)),
        "nbytes": int(array.nbytes),
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
