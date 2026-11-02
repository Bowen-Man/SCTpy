import json
import subprocess
import sys

import numpy as np
from scipy import sparse

from benchmarks._common import count_sha256, load_fixture, save_sparse_fixture
from benchmarks.freeze_oracle_selection import freeze_oracle_selection
from benchmarks.generate_fixture import generate_counts
from sctransform_native import SelectionManifest


def test_fixture_generator_is_deterministic_and_positive_by_cell() -> None:
    first = generate_counts(genes=32, cells=24, seed=7)
    second = generate_counts(genes=32, cells=24, seed=7)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.int64
    assert np.all(first >= 0)
    assert np.all(first.sum(axis=0) > 0)


def test_fixture_cli_writes_replayable_artifacts(tmp_path) -> None:
    output = tmp_path / "fixture.npz"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.generate_fixture",
            "--output",
            str(output),
            "--genes",
            "32",
            "--cells",
            "24",
            "--seed",
            "7",
        ],
        check=True,
    )

    with np.load(output, allow_pickle=False) as fixture:
        assert fixture["counts"].shape == (32, 24)
        assert fixture["gene_names"].shape == (32,)
        assert fixture["cell_names"].shape == (24,)

    selection = SelectionManifest.from_dict(
        json.loads(output.with_suffix(".selection.json").read_text(encoding="utf-8"))
    )
    provenance = json.loads(
        output.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    assert selection.source_shape == (32, 24)
    assert provenance["selected_gene_count"] == len(selection.selected_gene_indices)
    assert output.with_suffix(".mtx").is_file()
    assert output.with_suffix(".genes.txt").is_file()
    assert output.with_suffix(".cells.txt").is_file()


def test_sparse_fixture_round_trip_and_hash_are_canonical(tmp_path) -> None:
    counts = sparse.coo_matrix(
        ([1, 2, 3, 4], ([0, 0, 1, 2], [0, 0, 2, 1])), shape=(3, 4)
    )
    expected = sparse.csr_matrix(
        np.array([[3, 0, 0, 0], [0, 0, 3, 0], [0, 4, 0, 0]], dtype=np.int64)
    )
    path = tmp_path / "sparse.npz"
    save_sparse_fixture(path, counts, ("g0", "g1", "g2"), ("c0", "c1", "c2", "c3"))

    actual, genes, cells = load_fixture(path)

    assert sparse.isspmatrix_csr(actual)
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    assert genes == ("g0", "g1", "g2")
    assert cells == ("c0", "c1", "c2", "c3")
    assert count_sha256(actual) == count_sha256(expected)


def test_oracle_selection_is_frozen_by_name_in_source_order(tmp_path) -> None:
    fixture = tmp_path / "fixture.npz"
    counts = sparse.csr_matrix(np.ones((4, 5), dtype=np.int64))
    genes = ("g0", "g1", "g2", "g3")
    cells = ("c0", "c1", "c2", "c3", "c4")
    save_sparse_fixture(fixture, counts, genes, cells)
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "fit_genes.txt").write_text("g2\ng0\n", encoding="utf-8")
    (oracle / "fit_cells.txt").write_text("c4\nc1\nc3\n", encoding="utf-8")

    manifest = freeze_oracle_selection(
        fixture,
        oracle,
        tmp_path / "selection.json",
        seed=7,
        min_cells=1,
    )

    assert manifest.algorithm == "r_oracle_named_selection_v1"
    assert manifest.selected_gene_indices == (0, 2)
    assert manifest.selected_cell_indices == (1, 3, 4)
    assert manifest.selected_gene_names == ("g0", "g2")
    assert manifest.selected_cell_names == ("c1", "c3", "c4")
