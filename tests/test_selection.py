import json

import numpy as np
import pytest

from sctransform_native import (
    SCTConfig,
    SelectionManifest,
    create_selection_manifest,
    fit_transform,
)


def test_manifest_is_reproducible_and_json_round_trips() -> None:
    genes = tuple(f"g{index}" for index in range(10))
    cells = tuple(f"c{index}" for index in range(20))
    eligible = np.ones(10, dtype=bool)
    first = create_selection_manifest(
        gene_names=genes,
        cell_names=cells,
        eligible_genes=eligible,
        min_cells=2,
        requested_genes=4,
        requested_cells=8,
        seed=123,
    )
    second = create_selection_manifest(
        gene_names=genes,
        cell_names=cells,
        eligible_genes=eligible,
        min_cells=2,
        requested_genes=4,
        requested_cells=8,
        seed=123,
    )
    assert first == second
    assert first.algorithm == "uniform_without_replacement_pcg64_v1"
    restored = SelectionManifest.from_dict(json.loads(json.dumps(first.to_dict())))
    assert restored == first


def test_manifest_name_mismatch_is_rejected() -> None:
    counts = np.ones((3, 5), dtype=int)
    config = SCTConfig(min_cells=1, n_genes=None, n_cells=None)
    result = fit_transform(
        counts,
        gene_names=("g0", "g1", "g2"),
        cell_names=("c0", "c1", "c2", "c3", "c4"),
        config=config,
    )
    payload = result.selection.to_dict()
    payload["selected_gene_names"][0] = "wrong"
    invalid = SelectionManifest.from_dict(payload)
    with pytest.raises(ValueError, match="gene names do not match"):
        fit_transform(
            counts,
            gene_names=result.gene_names,
            cell_names=result.cell_names,
            config=config,
            selection=invalid,
        )
