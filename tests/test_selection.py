import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from sctpy import (
    SCTConfig,
    SelectionManifest,
    create_selection_manifest,
    fit_transform,
)
from sctpy.selection import _inverse_density_weights


def test_manifest_is_reproducible_and_json_round_trips() -> None:
    genes = tuple(f"g{index}" for index in range(10))
    cells = tuple(f"c{index}" for index in range(20))
    eligible = np.ones(10, dtype=bool)
    counts = np.arange(1, 201, dtype=np.int64).reshape(10, 20)
    geometric_mean = np.expm1(np.mean(np.log1p(counts), axis=1))
    first = create_selection_manifest(
        counts_gxc=counts,
        gene_names=genes,
        cell_names=cells,
        eligible_genes=eligible,
        geometric_mean=geometric_mean,
        min_cells=2,
        requested_genes=4,
        requested_cells=8,
        seed=123,
    )
    second = create_selection_manifest(
        counts_gxc=counts,
        gene_names=genes,
        cell_names=cells,
        eligible_genes=eligible,
        geometric_mean=geometric_mean,
        min_cells=2,
        requested_genes=4,
        requested_cells=8,
        seed=123,
    )
    assert first == second
    assert first.algorithm == "sct_v2_inverse_density_pcg64_v1"
    restored = SelectionManifest.from_dict(json.loads(json.dumps(first.to_dict())))
    assert restored == first


def test_selection_rechecks_detection_after_cell_sampling_and_stays_sparse() -> None:
    counts = np.array(
        [
            [2, 2, 2, 2, 2, 2],
            [4, 4, 4, 0, 0, 0],
            [0, 0, 0, 5, 5, 5],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.int64,
    )
    genes = ("g0", "g1", "g2", "poisson")
    cells = tuple(f"c{index}" for index in range(6))
    geometric_mean = np.expm1(np.mean(np.log1p(counts), axis=1))
    eligible = np.array([True, True, True, False])
    dense = create_selection_manifest(
        counts_gxc=counts,
        gene_names=genes,
        cell_names=cells,
        eligible_genes=eligible,
        geometric_mean=geometric_mean,
        min_cells=2,
        requested_genes=None,
        requested_cells=3,
        seed=17,
    )
    sparse_result = create_selection_manifest(
        counts_gxc=sparse.csr_matrix(counts),
        gene_names=genes,
        cell_names=cells,
        eligible_genes=eligible,
        geometric_mean=geometric_mean,
        min_cells=2,
        requested_genes=None,
        requested_cells=3,
        seed=17,
    )

    assert dense == sparse_result
    selected_cells = np.asarray(dense.selected_cell_indices)
    selected_genes = np.asarray(dense.selected_gene_indices)
    assert_array_equal(
        selected_genes,
        np.flatnonzero(
            eligible & (np.count_nonzero(counts[:, selected_cells], axis=1) >= 2)
        ),
    )


def test_inverse_density_weights_match_fixed_r_golden() -> None:
    abundance = np.array([-2.0, -1.0, -0.9, -0.8, -0.7, 0.0, 0.05, 0.1, 1.5])
    expected = np.array(
        [
            0.227975288657343,
            0.0700111680638377,
            0.0658154745791617,
            0.0632827404049104,
            0.0620799459702391,
            0.0744238598248327,
            0.0769569044989045,
            0.0799612996897074,
            0.279493318311063,
        ]
    )
    weights = _inverse_density_weights(abundance)

    assert_allclose(np.sum(weights), 1.0)
    assert np.all(np.isfinite(weights))
    assert np.corrcoef(weights, expected)[0, 1] > 0.999999
    assert np.max(np.abs(weights - expected)) < 1.0e-4


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
    payload["selected_cell_names"][0] = "wrong"
    invalid = SelectionManifest.from_dict(payload)
    with pytest.raises(ValueError, match="cell names do not match"):
        fit_transform(
            counts,
            gene_names=result.gene_names,
            cell_names=result.cell_names,
            config=config,
            selection=invalid,
        )
