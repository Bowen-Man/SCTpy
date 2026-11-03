from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ._matrix import CountMatrix, matrix_detection_count


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
    def from_dict(cls, value: Mapping[str, Any]) -> SelectionManifest:
        return cls(
            schema_version=int(value["schema_version"]),
            seed=int(value["seed"]),
            algorithm=str(value["algorithm"]),
            source_shape=tuple(int(item) for item in value["source_shape"]),  # type: ignore[arg-type]
            selected_gene_indices=tuple(
                int(item) for item in value["selected_gene_indices"]
            ),
            selected_gene_names=tuple(
                str(item) for item in value["selected_gene_names"]
            ),
            selected_cell_indices=tuple(
                int(item) for item in value["selected_cell_indices"]
            ),
            selected_cell_names=tuple(
                str(item) for item in value["selected_cell_names"]
            ),
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
    counts_gxc: CountMatrix,
    gene_names: Sequence[str],
    cell_names: Sequence[str],
    eligible_genes: np.ndarray,
    geometric_mean: np.ndarray,
    min_cells: int,
    requested_genes: int | None,
    requested_cells: int | None,
    seed: int,
) -> SelectionManifest:
    """Create the independently specified v2 cell/gene selection manifest."""
    genes, cells = counts_gxc.shape
    if genes != len(gene_names) or cells != len(cell_names):
        raise ValueError("count matrix shape does not match gene and cell names")
    if eligible_genes.shape != (genes,) or geometric_mean.shape != (genes,):
        raise ValueError("gene selection arrays must match the number of genes")

    rng = np.random.Generator(np.random.PCG64(seed))
    cell_count = cells if requested_cells is None else min(requested_cells, cells)
    selected_cells = (
        np.arange(cells, dtype=np.int64)
        if cell_count == cells
        else np.sort(rng.choice(cells, size=cell_count, replace=False))
    )

    detected = matrix_detection_count(counts_gxc[:, selected_cells])
    candidates = np.flatnonzero(eligible_genes & (detected >= min_cells))
    gene_count = (
        candidates.size
        if requested_genes is None
        else min(requested_genes, candidates.size)
    )
    if gene_count == candidates.size:
        selected_genes = candidates
    else:
        abundance = np.log10(
            np.maximum(geometric_mean[candidates], np.finfo(np.float64).tiny)
        )
        selected_genes = np.sort(
            rng.choice(
                candidates,
                size=gene_count,
                replace=False,
                p=_inverse_density_weights(abundance),
            )
        )
    return SelectionManifest(
        schema_version=1,
        seed=seed,
        algorithm="sct_v2_inverse_density_pcg64_v1",
        source_shape=(genes, cells),
        selected_gene_indices=tuple(int(item) for item in selected_genes),
        selected_gene_names=tuple(gene_names[int(item)] for item in selected_genes),
        selected_cell_indices=tuple(int(item) for item in selected_cells),
        selected_cell_names=tuple(cell_names[int(item)] for item in selected_cells),
        min_cells=min_cells,
        requested_genes=requested_genes,
        requested_cells=requested_cells,
    )


def _inverse_density_weights(values: np.ndarray) -> np.ndarray:
    """Return stable inverse Gaussian-density weights on a 512-point grid."""
    size = values.size
    if size < 2 or np.all(values == values[0]):
        return np.full(size, 1.0 / size, dtype=np.float64)

    quartiles = np.quantile(values, (0.25, 0.75))
    scale = min(float(np.std(values, ddof=1)), float(np.diff(quartiles)[0] / 1.34))
    bandwidth = 1.06 * scale * size ** (-0.2)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        return np.full(size, 1.0 / size, dtype=np.float64)

    lower = float(np.min(values) - 3.0 * bandwidth)
    upper = float(np.max(values) + 3.0 * bandwidth)
    grid = np.linspace(lower, upper, 512)
    spacing = grid[1] - grid[0]
    position = (values - lower) / spacing
    left = np.minimum(position.astype(np.int64), grid.size - 2)
    fraction = position - left
    binned = np.zeros(grid.size, dtype=np.float64)
    np.add.at(binned, left, 1.0 - fraction)
    np.add.at(binned, left + 1, fraction)
    density = gaussian_filter1d(
        binned / (size * spacing),
        bandwidth / spacing,
        mode="constant",
        truncate=4.0,
    )
    sampled_density = np.interp(values, grid, density)
    weights = 1.0 / (sampled_density + np.finfo(np.float64).eps)
    return weights / np.sum(weights)


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
    if cell_indices.size < 2:
        raise ValueError("selection must include at least two cells")
    if np.any(gene_indices < 0) or np.any(gene_indices >= len(gene_names)):
        raise ValueError("selection contains an out-of-range gene index")
    if np.any(cell_indices < 0) or np.any(cell_indices >= len(cell_names)):
        raise ValueError("selection contains an out-of-range cell index")
    if np.unique(gene_indices).size != gene_indices.size:
        raise ValueError("selected gene indices must be unique")
    if np.unique(cell_indices).size != cell_indices.size:
        raise ValueError("selected cell indices must be unique")
    if not np.all(eligible_genes[gene_indices]):
        raise ValueError("selection includes genes below min_cells")
    expected_gene_names = tuple(gene_names[int(item)] for item in gene_indices)
    expected_cell_names = tuple(cell_names[int(item)] for item in cell_indices)
    if manifest.selected_gene_names != expected_gene_names:
        raise ValueError("selection gene names do not match indices")
    if manifest.selected_cell_names != expected_cell_names:
        raise ValueError("selection cell names do not match indices")
