from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import io as scipy_io
from scipy import sparse

from sctransform_native import create_selection_manifest
from sctransform_native._matrix import matrix_statistics

from ._common import count_sha256, write_json


def generate_counts(*, genes: int, cells: int, seed: int) -> np.ndarray:
    """Generate a deterministic synthetic UMI matrix from a Gamma-Poisson mixture."""
    if genes < 2 or cells < 2:
        raise ValueError("fixture requires at least two genes and two cells")
    rng = np.random.Generator(np.random.PCG64(seed))
    cell_factor = rng.lognormal(mean=0.0, sigma=0.45, size=cells)
    cell_factor /= np.mean(cell_factor)
    gene_mean = rng.lognormal(mean=np.log(2.5), sigma=0.9, size=genes)
    theta = np.clip(0.8 + 2.5 * np.sqrt(gene_mean), 0.5, 50.0)

    rare_count = max(1, int(round(genes * 0.03)))
    rare_indices = rng.choice(genes, size=rare_count, replace=False)
    gene_mean[rare_indices] = rng.uniform(0.001, 0.02, size=rare_count)
    poisson_mask = np.zeros(genes, dtype=bool)
    poisson_mask[rng.choice(genes, size=max(1, genes // 10), replace=False)] = True

    counts = np.empty((genes, cells), dtype=np.int64)
    block_size = 256
    for start in range(0, genes, block_size):
        stop = min(start + block_size, genes)
        mu = gene_mean[start:stop, None] * cell_factor[None, :]
        local_poisson = poisson_mask[start:stop]
        block = np.empty((stop - start, cells), dtype=np.int64)
        if np.any(local_poisson):
            block[local_poisson] = rng.poisson(mu[local_poisson])
        if np.any(~local_poisson):
            local_theta = theta[start:stop][~local_poisson, None]
            probability = local_theta / (local_theta + mu[~local_poisson])
            block[~local_poisson] = rng.negative_binomial(local_theta, probability)
        counts[start:stop] = block

    zero_library = np.flatnonzero(counts.sum(axis=0) == 0)
    counts[0, zero_library] = 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--genes", type=int, default=800)
    parser.add_argument("--cells", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--min-cells", type=int, default=5)
    args = parser.parse_args()

    counts = generate_counts(genes=args.genes, cells=args.cells, seed=args.seed)
    gene_names = tuple(f"gene_{index:05d}" for index in range(args.genes))
    cell_names = tuple(f"cell_{index:05d}" for index in range(args.cells))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        counts=counts,
        gene_names=np.asarray(gene_names),
        cell_names=np.asarray(cell_names),
    )

    gene_mean, gene_variance, detection, geometric_mean = matrix_statistics(counts)
    eligible = detection >= args.min_cells
    step1_candidate = eligible & (gene_variance > gene_mean)
    manifest = create_selection_manifest(
        counts_gxc=counts,
        gene_names=gene_names,
        cell_names=cell_names,
        eligible_genes=step1_candidate,
        geometric_mean=geometric_mean,
        min_cells=args.min_cells,
        requested_genes=None,
        requested_cells=None,
        seed=args.seed,
    )
    manifest_path = args.output.with_suffix(".selection.json")
    write_json(manifest_path, manifest.to_dict())

    matrix_market_path = args.output.with_suffix(".mtx")
    scipy_io.mmwrite(matrix_market_path, sparse.coo_matrix(counts), symmetry="general")
    args.output.with_suffix(".genes.txt").write_text(
        "\n".join(gene_names) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".cells.txt").write_text(
        "\n".join(cell_names) + "\n", encoding="utf-8"
    )
    write_json(
        args.output.with_suffix(".provenance.json"),
        {
            "schema_version": 1,
            "generator": "benchmarks.generate_fixture.generate_counts/v1",
            "generated_on": datetime.now(timezone.utc).date().isoformat(),
            "seed": args.seed,
            "shape_genes_by_cells": [args.genes, args.cells],
            "dtype": str(counts.dtype),
            "counts_sha256": count_sha256(counts),
            "eligible_gene_count": int(np.count_nonzero(eligible)),
            "selected_gene_count": len(manifest.selected_gene_indices),
            "license": "BSD-3-Clause (synthetic project-generated fixture)",
            "selection_manifest": manifest_path.name,
        },
    )


if __name__ == "__main__":
    main()
