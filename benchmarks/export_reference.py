from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from sctransform_native._matrix import matrix_statistics

from ._common import count_sha256, load_fixture, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the GPL reference as an external black-box oracle."
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    version = importlib.metadata.version("sctransform-cpu")
    if version != "0.4.0":
        raise RuntimeError(f"expected sctransform-cpu 0.4.0, found {version}")
    from sctransform_cpu import (
        SCTConfig,
        sctransform,
    )
    from sctransform_cpu import (
        SelectionManifest as ReferenceSelectionManifest,
    )

    counts, genes, cells = load_fixture(args.fixture)
    manifest = load_manifest(args.manifest)
    gene_lookup = {name: index for index, name in enumerate(genes)}
    selected_gene_indices = np.asarray(
        [gene_lookup[name] for name in manifest.selected_gene_names], dtype=np.int64
    )
    selected_cell_indices = np.asarray(manifest.selected_cell_indices, dtype=np.int64)
    selected_counts = counts[selected_gene_indices][:, selected_cell_indices]
    selected_mean, selected_variance, _, _ = matrix_statistics(selected_counts)
    poisson = (selected_mean < 0.001) | (selected_variance <= selected_mean)
    reference_fit_genes = tuple(
        name
        for name, is_poisson in zip(manifest.selected_gene_names, poisson, strict=True)
        if not is_poisson
    )
    reference_manifest = ReferenceSelectionManifest(
        seed=manifest.seed,
        cells=manifest.selected_cell_names,
        genes=reference_fit_genes,
    )
    result = sctransform(
        sparse.csr_matrix(counts),
        gene_names=genes,
        cell_names=cells,
        config=SCTConfig(
            min_cells=5,
            fit_cells=None,
            fit_genes=None,
            features=len(genes),
            output="full",
        ),
        selection_manifest=reference_manifest,
    )
    residual_genes = result.genes[result.residual_gene_indices]
    fit_genes = result.genes[result.fit_gene_indices]
    hvg_genes = result.genes[result.hvg_indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        genes=result.genes,
        cells=result.cells,
        regularized_intercept=result.intercept,
        regularized_slope=result.slope,
        regularized_theta=result.theta,
        fit_genes=fit_genes,
        raw_intercept=result.fit_intercept,
        raw_theta=result.fit_theta,
        residual_genes=residual_genes,
        residuals=result.residuals,
        residual_mean=result.residual_mean,
        residual_variance=result.residual_variance,
        hvg_genes=hvg_genes,
        fit_gene_indices=result.fit_gene_indices,
        fit_cell_indices=result.fit_cell_indices,
        poisson_genes=result.genes[np.isinf(result.theta)],
    )
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "oracle": "Bowen-Man/sctransform-cpu",
                "release": "v0.4.0",
                "commit": "b79041796e3f7ca3b33029adde5d7132eedd2430",
                "execution": "external black-box public API",
                "fixture_counts_sha256": count_sha256(counts),
                "selection_manifest_sha256": hashlib.sha256(
                    args.manifest.read_bytes()
                ).hexdigest(),
                "selection_adapter": (
                    "same persisted cells and candidate genes; remove genes selected by "
                    "the documented v2 Poisson rule because the Level-B public manifest "
                    "accepts only NB fit genes"
                ),
                "timing_seconds": result.timing,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
