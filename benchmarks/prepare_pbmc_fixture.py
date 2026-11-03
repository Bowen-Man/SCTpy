from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy import io as scipy_io
from scipy import sparse

from sctpy._matrix import matrix_statistics

from ._common import count_sha256, save_sparse_fixture, write_json

SOURCE_BYTES = 39_503_734
SOURCE_SHA256 = "42a789ff3f92f0e117d1ff4ffcb2fa8edee87dd2a03d83a81502709344a38bf0"
SOURCE_URL = (
    "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
    "10k_PBMC_3p_nextgem_Chromium_X/"
    "10k_PBMC_3p_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def _ordered_name_sha256(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def prepare_fixture(source: Path, output: Path, *, cells: int, seed: int) -> None:
    if source.stat().st_size != SOURCE_BYTES or _sha256(source) != SOURCE_SHA256:
        raise ValueError("source HDF5 size or SHA-256 does not match fixed provenance")

    rng = np.random.Generator(np.random.PCG64(seed))
    with h5py.File(source, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(value) for value in group["shape"][:])
        barcodes = _decode(group["barcodes"][:])
        feature_ids = _decode(group["features/id"][:])
        feature_types = _decode(group["features/feature_type"][:])
        if shape != (len(feature_ids), len(barcodes)):
            raise ValueError(
                "10x matrix shape does not match feature and barcode arrays"
            )
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("10x feature IDs must be unique")
        if len(set(barcodes)) != len(barcodes):
            raise ValueError("10x barcodes must be unique")

        gene_indices = np.flatnonzero(np.asarray(feature_types) == "Gene Expression")
        if cells < 2 or cells > shape[1]:
            raise ValueError(
                "requested cells must be between 2 and the source cell count"
            )
        cell_indices = np.sort(rng.choice(shape[1], size=cells, replace=False))
        matrix = sparse.csc_matrix(
            (
                np.asarray(group["data"][:], dtype=np.int64),
                np.asarray(group["indices"][:]),
                np.asarray(group["indptr"][:]),
            ),
            shape=shape,
        )[gene_indices][:, cell_indices]

    counts = matrix.tocsr()
    genes = tuple(feature_ids[index] for index in gene_indices)
    selected_cells = tuple(barcodes[index] for index in cell_indices)
    save_sparse_fixture(output, counts, genes, selected_cells)
    scipy_io.mmwrite(output.with_suffix(".mtx"), counts, symmetry="general")
    output.with_suffix(".genes.txt").write_text(
        "\n".join(genes) + "\n", encoding="utf-8"
    )
    output.with_suffix(".cells.txt").write_text(
        "\n".join(selected_cells) + "\n", encoding="utf-8"
    )

    means, variances, detection, _ = matrix_statistics(counts)
    eligible = detection >= 5
    candidates = eligible & (variances > means)
    write_json(
        output.with_suffix(".provenance.json"),
        {
            "schema_version": 1,
            "generated_on": datetime.now(timezone.utc).date().isoformat(),
            "source": {
                "url": SOURCE_URL,
                "bytes": SOURCE_BYTES,
                "sha256": SOURCE_SHA256,
                "license": "CC BY 4.0",
            },
            "derivation": {
                "algorithm": "pcg64_cell_subset_source_order_v1",
                "seed": seed,
                "source_shape_genes_by_cells": list(shape),
                "selected_source_cell_count": cells,
                "feature_filter": "feature_type == Gene Expression",
                "biological_qc": "none",
            },
            "fixture": {
                "format": "csr",
                "shape_genes_by_cells": list(counts.shape),
                "nnz": int(counts.nnz),
                "counts_sha256": count_sha256(counts),
                "gene_names_sha256": _ordered_name_sha256(genes),
                "cell_names_sha256": _ordered_name_sha256(selected_cells),
                "eligible_gene_count_min_cells_5": int(np.count_nonzero(eligible)),
                "step1_candidate_count": int(np.count_nonzero(candidates)),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    prepare_fixture(args.input, args.output, cells=args.cells, seed=args.seed)


if __name__ == "__main__":
    main()
