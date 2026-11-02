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
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as fixture:
        counts = np.asarray(fixture["counts"], dtype=np.int64)
        genes = tuple(str(value) for value in fixture["gene_names"])
        cells = tuple(str(value) for value in fixture["cell_names"])
    return counts, genes, cells


def load_manifest(path: str | Path) -> SelectionManifest:
    return SelectionManifest.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def count_sha256(counts: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(counts).view(np.uint8)).hexdigest()


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
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
