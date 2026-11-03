import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from scipy import sparse

from sctpy import SCTConfig, fit_transform_anndata, write_anndata

ad = pytest.importorskip("anndata")


@pytest.fixture
def fitted():
    counts = sparse.csr_matrix([[2, 0, 0], [4, 1, 0], [3, 1, 0], [5, 2, 0]])
    adata = ad.AnnData(counts.copy())
    adata.layers["counts"] = counts.copy()
    result = fit_transform_anndata(
        adata,
        layer="counts",
        config=SCTConfig.r_v2(min_cells=1, n_genes=None, n_cells=None),
        return_corrected_umi=True,
    )
    return adata, result


def test_write_and_h5ad_roundtrip(fitted, tmp_path):
    adata, result = fitted
    assert write_anndata(adata, result, n_top_genes=1) is None
    path = tmp_path / "sct.h5ad"
    adata.write_h5ad(path)
    saved = ad.read_h5ad(path)
    np.testing.assert_array_equal(saved.X.toarray(), adata.layers["counts"].toarray())
    np.testing.assert_allclose(saved.layers["sct_residuals"], result.residuals.T)
    assert sparse.isspmatrix_csr(saved.layers["sct_corrected_umi"])
    np.testing.assert_array_equal(
        saved.layers["sct_corrected_umi"].toarray(), result.corrected_umi.T.toarray()
    )
    for name, values in result.parameter_dict().items():
        np.testing.assert_allclose(saved.var[f"sct_{name}"], values)
    np.testing.assert_array_equal(
        saved.var_names[saved.var.highly_variable], result.highly_variable_genes(1)
    )
    assert np.isnan(saved.var.highly_variable_rank.iloc[-1])
    assert saved.uns["sct"]["corrected_umi_depth"] == result.corrected_umi_depth
    assert json.loads(saved.uns["sct"]["selection_json"]) == result.selection.to_dict()
    assert json.loads(saved.uns["sct"]["config_json"]) == asdict(result.config)
    saved.layers["sct_residuals"][0, 0] = 123
    assert result.residuals[0, 0] != 123


def test_copy_view_and_summary_replace_stale_results(fitted):
    adata, result = fitted
    view = adata[:, :]
    with pytest.raises(ValueError, match="view"):
        write_anndata(view, result)
    copied = write_anndata(view, result, copy=True)
    assert copied is not None and not copied.is_view
    assert "sct" not in adata.uns
    summary = fit_transform_anndata(
        adata, config=replace(result.config, residual_output="summary")
    )
    write_anndata(copied, summary)
    assert set(copied.layers) == set(adata.layers)
    assert "corrected_umi_depth" not in copied.uns["sct"]
    np.testing.assert_allclose(copied.var.residual_variances, result.residual_variance)


def test_reject_backed_and_reordered_names_before_mutation(fitted, tmp_path):
    adata, result = fitted
    path = tmp_path / "backed.h5ad"
    adata.write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")
    try:
        with pytest.raises(ValueError, match="to_memory"):
            write_anndata(backed, result, copy=True)
    finally:
        backed.file.close()
    reordered = adata[:, ::-1].copy()
    with pytest.raises(ValueError, match="names and order"):
        write_anndata(reordered, result)
    assert not reordered.uns and reordered.var.empty


@pytest.mark.parametrize("n_top", [0, -1, 1.5, True])
def test_invalid_hvg_count_does_not_mutate(fitted, n_top):
    adata, result = fitted
    with pytest.raises(ValueError, match="positive integer"):
        write_anndata(adata, result, n_top_genes=n_top)
    assert not adata.uns and adata.var.empty
