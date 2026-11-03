from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy import sparse

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
    reference_manifest = ReferenceSelectionManifest(
        seed=manifest.seed,
        cells=manifest.selected_cell_names,
        genes=manifest.selected_gene_names,
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
            return_corrected_umi=True,
        ),
        selection_manifest=reference_manifest,
    )
    if result.corrected_umi is None:
        raise RuntimeError("reference did not return corrected UMI counts")
    residual_genes = result.genes[result.residual_gene_indices]
    fit_genes = result.genes[result.fit_gene_indices]
    fit_cells = result.cells[result.fit_cell_indices]
    if (
        tuple(fit_genes) != manifest.selected_gene_names
        or tuple(fit_cells) != manifest.selected_cell_names
    ):
        raise RuntimeError("reference did not replay the ordered selection manifest")
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
        corrected_umi_data=result.corrected_umi.data,
        corrected_umi_indices=result.corrected_umi.indices,
        corrected_umi_indptr=result.corrected_umi.indptr,
        corrected_umi_shape=result.corrected_umi.shape,
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
                    "replay persisted gene/cell names in their original order; "
                    "no additional gene filtering"
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
