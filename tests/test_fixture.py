import json
import subprocess
import sys

import numpy as np

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
