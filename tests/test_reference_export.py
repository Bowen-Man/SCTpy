import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import export_reference
from sctpy import SelectionManifest


def test_reference_receives_complete_ordered_manifest(monkeypatch, tmp_path):
    # The sampled cells alone make both rows non-overdispersed. The adapter must
    # still replay the oracle's selection, including its original gene order.
    counts = np.array([[1, 1, 1, 1, 1, 1, 30, 0], [2, 2, 2, 2, 2, 2, 0, 40]])
    genes, cells = ("a", "b"), tuple(f"c{i}" for i in range(8))
    manifest = SelectionManifest(
        schema_version=1,
        seed=42,
        algorithm="r_oracle_named_selection_v1",
        source_shape=counts.shape,
        selected_gene_indices=(1, 0),
        selected_gene_names=("b", "a"),
        selected_cell_indices=(5, 4, 3, 2, 1, 0),
        selected_cell_names=tuple(f"c{i}" for i in range(5, -1, -1)),
        min_cells=5,
        requested_genes=2,
        requested_cells=6,
    )
    received = {}

    def oracle(matrix, **kwargs):
        np.testing.assert_array_equal(matrix.toarray(), counts)
        received.update(kwargs)
        raise RuntimeError("oracle boundary reached")

    monkeypatch.setitem(
        sys.modules,
        "sctransform_cpu",
        SimpleNamespace(
            SCTConfig=SimpleNamespace,
            SelectionManifest=SimpleNamespace,
            sctransform=oracle,
        ),
    )
    monkeypatch.setattr(
        export_reference.importlib.metadata, "version", lambda _: "0.4.0"
    )
    monkeypatch.setattr(
        export_reference, "load_fixture", lambda _: (counts, genes, cells)
    )
    monkeypatch.setattr(export_reference, "load_manifest", lambda _: manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_reference",
            "--fixture",
            "unused.npz",
            "--manifest",
            "unused.json",
            "--output",
            str(tmp_path / "reference.npz"),
        ],
    )
    with pytest.raises(RuntimeError, match="oracle boundary reached"):
        export_reference.main()
    actual = received["selection_manifest"]
    assert actual.genes == manifest.selected_gene_names
    assert actual.cells == manifest.selected_cell_names
    assert actual.seed == manifest.seed
    assert received["gene_names"] == genes and received["cell_names"] == cells

    monkeypatch.setattr(
        sys.modules["sctransform_cpu"],
        "sctransform",
        lambda *args, **kwargs: SimpleNamespace(
            corrected_umi=object(),
            genes=np.asarray(genes),
            cells=np.asarray(cells),
            residual_gene_indices=np.array([0, 1]),
            fit_gene_indices=np.array([0, 1]),
            fit_cell_indices=np.asarray(manifest.selected_cell_indices),
        ),
    )
    with pytest.raises(RuntimeError, match="ordered selection manifest"):
        export_reference.main()
