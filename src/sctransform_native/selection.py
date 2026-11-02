from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SelectionManifest:
    """Complete, serializable record of cells and genes used for parameter fitting."""

    schema_version: int
    seed: int
    algorithm: str
    source_shape: tuple[int, int]
    selected_gene_indices: tuple[int, ...]
    selected_gene_names: tuple[str, ...]
    selected_cell_indices: tuple[int, ...]
    selected_cell_names: tuple[str, ...]
    min_cells: int
    requested_genes: int | None
    requested_cells: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "algorithm": self.algorithm,
            "source_shape": list(self.source_shape),
            "selected_gene_indices": list(self.selected_gene_indices),
            "selected_gene_names": list(self.selected_gene_names),
            "selected_cell_indices": list(self.selected_cell_indices),
            "selected_cell_names": list(self.selected_cell_names),
            "min_cells": self.min_cells,
            "requested_genes": self.requested_genes,
            "requested_cells": self.requested_cells,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectionManifest":
        return cls(
            schema_version=int(value["schema_version"]),
            seed=int(value["seed"]),
            algorithm=str(value["algorithm"]),
            source_shape=tuple(int(item) for item in value["source_shape"]),  # type: ignore[arg-type]
            selected_gene_indices=tuple(
                int(item) for item in value["selected_gene_indices"]
            ),
            selected_gene_names=tuple(str(item) for item in value["selected_gene_names"]),
            selected_cell_indices=tuple(
                int(item) for item in value["selected_cell_indices"]
            ),
            selected_cell_names=tuple(str(item) for item in value["selected_cell_names"]),
            min_cells=int(value["min_cells"]),
            requested_genes=(
                None
                if value["requested_genes"] is None
                else int(value["requested_genes"])
            ),
            requested_cells=(
                None
                if value["requested_cells"] is None
                else int(value["requested_cells"])
            ),
        )


def create_selection_manifest(
    *,
    gene_names: Sequence[str],
    cell_names: Sequence[str],
    eligible_genes: np.ndarray,
    min_cells: int,
    requested_genes: int | None,
    requested_cells: int | None,
    seed: int,
) -> SelectionManifest:
    """Select without replacement using a named, reproducible PCG64 stream."""
    rng = np.random.Generator(np.random.PCG64(seed))
    eligible_indices = np.flatnonzero(eligible_genes)
    gene_count = (
        eligible_indices.size
        if requested_genes is None
        else min(requested_genes, eligible_indices.size)
    )
    cell_count = (
        len(cell_names)
        if requested_cells is None
        else min(requested_cells, len(cell_names))
    )
    selected_genes = np.sort(
        rng.choice(eligible_indices, size=gene_count, replace=False)
    )
    selected_cells = np.sort(
        rng.choice(len(cell_names), size=cell_count, replace=False)
    )
    return SelectionManifest(
        schema_version=1,
        seed=seed,
        algorithm="uniform_without_replacement_pcg64_v1",
        source_shape=(len(gene_names), len(cell_names)),
        selected_gene_indices=tuple(int(item) for item in selected_genes),
        selected_gene_names=tuple(gene_names[item] for item in selected_genes),
        selected_cell_indices=tuple(int(item) for item in selected_cells),
        selected_cell_names=tuple(cell_names[item] for item in selected_cells),
        min_cells=min_cells,
        requested_genes=requested_genes,
        requested_cells=requested_cells,
    )


def validate_selection_manifest(
    manifest: SelectionManifest,
    *,
    gene_names: Sequence[str],
    cell_names: Sequence[str],
    eligible_genes: np.ndarray,
    min_cells: int,
) -> None:
    if manifest.schema_version != 1:
        raise ValueError(f"unsupported selection schema {manifest.schema_version}")
    expected_shape = (len(gene_names), len(cell_names))
    if manifest.source_shape != expected_shape:
        raise ValueError(
            f"selection source_shape {manifest.source_shape} does not match {expected_shape}"
        )
    if manifest.min_cells != min_cells:
        raise ValueError(
            f"selection min_cells {manifest.min_cells} does not match config {min_cells}"
        )
    gene_indices = np.asarray(manifest.selected_gene_indices, dtype=np.int64)
    cell_indices = np.asarray(manifest.selected_cell_indices, dtype=np.int64)
    if gene_indices.size == 0:
        raise ValueError("selection must include at least one gene")
    if cell_indices.size < 2:
        raise ValueError("selection must include at least two cells")
    if np.any(gene_indices < 0) or np.any(gene_indices >= len(gene_names)):
        raise ValueError("selection contains an out-of-range gene index")
    if np.any(cell_indices < 0) or np.any(cell_indices >= len(cell_names)):
        raise ValueError("selection contains an out-of-range cell index")
    if not np.array_equal(gene_indices, np.unique(gene_indices)):
        raise ValueError("selected gene indices must be unique and sorted")
    if not np.array_equal(cell_indices, np.unique(cell_indices)):
        raise ValueError("selected cell indices must be unique and sorted")
    if not np.all(eligible_genes[gene_indices]):
        raise ValueError("selection includes genes below min_cells")
    expected_gene_names = tuple(gene_names[item] for item in gene_indices)
    expected_cell_names = tuple(cell_names[item] for item in cell_indices)
    if manifest.selected_gene_names != expected_gene_names:
        raise ValueError("selection gene names do not match indices")
    if manifest.selected_cell_names != expected_cell_names:
        raise ValueError("selection cell names do not match indices")
