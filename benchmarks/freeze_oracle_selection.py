from __future__ import annotations

import argparse
from pathlib import Path

from sctransform_native import SelectionManifest
from sctransform_native._matrix import matrix_detection_count

from ._common import load_fixture, write_json


def _read_names(path: Path) -> tuple[str, ...]:
    return tuple(
        value for value in path.read_text(encoding="utf-8").splitlines() if value
    )


def freeze_oracle_selection(
    fixture: Path,
    r_oracle: Path,
    output: Path,
    *,
    seed: int,
    min_cells: int,
) -> SelectionManifest:
    counts, genes, cells = load_fixture(fixture)
    fit_genes = _read_names(r_oracle / "fit_genes.txt")
    fit_cells = _read_names(r_oracle / "fit_cells.txt")
    gene_lookup = {name: index for index, name in enumerate(genes)}
    cell_lookup = {name: index for index, name in enumerate(cells)}
    if len(set(fit_genes)) != len(fit_genes) or len(set(fit_cells)) != len(fit_cells):
        raise ValueError("oracle selection names must be unique")
    if (
        not set(fit_genes) <= gene_lookup.keys()
        or not set(fit_cells) <= cell_lookup.keys()
    ):
        raise ValueError("oracle selection contains names absent from the fixture")
    gene_indices = tuple(sorted(gene_lookup[name] for name in fit_genes))
    cell_indices = tuple(sorted(cell_lookup[name] for name in fit_cells))
    if not gene_indices or len(cell_indices) < 2:
        raise ValueError("oracle selection must contain genes and at least two cells")
    fit_genes = tuple(genes[index] for index in gene_indices)
    fit_cells = tuple(cells[index] for index in cell_indices)
    detected = matrix_detection_count(counts[:, cell_indices])
    if any(detected[index] < min_cells for index in gene_indices):
        raise ValueError("oracle selection contains a gene below min_cells")

    manifest = SelectionManifest(
        schema_version=1,
        seed=seed,
        algorithm="r_oracle_named_selection_v1",
        source_shape=counts.shape,
        selected_gene_indices=gene_indices,
        selected_gene_names=fit_genes,
        selected_cell_indices=cell_indices,
        selected_cell_names=fit_cells,
        min_cells=min_cells,
        requested_genes=len(fit_genes),
        requested_cells=len(fit_cells),
    )
    write_json(output, manifest.to_dict())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--r-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--min-cells", type=int, default=5)
    args = parser.parse_args()
    freeze_oracle_selection(
        args.fixture,
        args.r_oracle,
        args.output,
        seed=args.seed,
        min_cells=args.min_cells,
    )


if __name__ == "__main__":
    main()
