import numpy as np
import pytest
from scipy import sparse

from sctransform_native import SCTConfig, fit_transform


def _fit(counts: object, **kwargs: object) -> object:
    return fit_transform(
        counts,
        gene_names=("g0", "g1"),
        cell_names=("c0", "c1"),
        config=SCTConfig(min_cells=1, n_genes=None, n_cells=None),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (np.array([[1.0, -1.0], [0.0, 2.0]]), "nonnegative"),
        (np.array([[1.5, 0.0], [0.0, 2.0]]), "integer-valued"),
        (np.array([[np.nan, 1.0], [1.0, 1.0]]), "finite"),
    ],
)
def test_invalid_count_values_are_rejected(counts: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _fit(counts)


def test_sparse_input_is_validated() -> None:
    counts = sparse.csr_matrix(np.array([[1.0, 0.25], [0.0, 2.0]]))
    with pytest.raises(ValueError, match="integer-valued"):
        _fit(counts)


def test_sparse_duplicate_entries_cannot_hide_negative_counts() -> None:
    counts = sparse.coo_matrix(([-1.0, 2.0, 1.0], ([0, 0, 1], [0, 0, 1])), shape=(2, 2))
    with pytest.raises(ValueError, match="nonnegative"):
        _fit(counts)


def test_zero_library_cell_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive library size"):
        _fit(np.array([[1, 0], [2, 0]]))


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="gene_names must be unique"):
        fit_transform(
            np.ones((2, 2), dtype=int),
            gene_names=("g", "g"),
            cell_names=("c0", "c1"),
            config=SCTConfig(min_cells=1, n_genes=None, n_cells=None),
        )


def test_unknown_orientation_is_rejected() -> None:
    with pytest.raises(ValueError, match="orientation"):
        _fit(np.ones((2, 2), dtype=int), matrix_orientation="unknown")  # type: ignore[arg-type]


def test_unknown_dispersion_regularization_is_rejected() -> None:
    with pytest.raises(ValueError, match="dispersion_regularization"):
        SCTConfig(dispersion_regularization="unknown")  # type: ignore[arg-type]


def test_unknown_fit_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="fit_method"):
        SCTConfig(fit_method="unknown")  # type: ignore[arg-type]


def test_unknown_regularization_support_is_rejected() -> None:
    with pytest.raises(ValueError, match="regularization_support"):
        SCTConfig(regularization_support="unknown")  # type: ignore[arg-type]


def test_unknown_regularization_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="regularization_method"):
        SCTConfig(regularization_method="unknown")  # type: ignore[arg-type]


def test_unknown_theta_postprocess_is_rejected() -> None:
    with pytest.raises(ValueError, match="theta_postprocess"):
        SCTConfig(theta_postprocess="unknown")  # type: ignore[arg-type]


def test_r_v2_preset_is_fixed_and_sampling_is_configurable() -> None:
    config = SCTConfig.r_v2(
        min_cells=7,
        n_genes=None,
        n_cells=500,
        seed=42,
        residual_block_size=64,
    )

    assert config.statistical_mode == "r_v2"
    assert config.min_cells == 7
    assert config.n_genes is None
    assert config.n_cells == 500
    assert config.seed == 42
    assert config.residual_block_size == 64
    assert config.fit_method == "cox_reid"
    assert config.cox_reid_gene_bin_size == 500
    assert config.dispersion_regularization == "sct_v2"
    assert config.regularization_method == "r_stats"
    assert config.regularization_support == "clamp"
    assert config.theta_postprocess == "moment_ratio"
    assert config.theta_moment_ratio_threshold == 1.0e-3


def test_statistical_modes_do_not_change_the_default_path() -> None:
    assert SCTConfig().statistical_mode == "native"
    assert SCTConfig(fit_method="cox_reid").statistical_mode == "experimental"
    assert SCTConfig.r_v2().statistical_mode == "r_v2"
    assert SCTConfig.r_v2().clip_range is None
    assert SCTConfig.r_v2() == SCTConfig(
        fit_method="cox_reid",
        cox_reid_gene_bin_size=500,
        dispersion_regularization="sct_v2",
        regularization_method="r_stats",
        regularization_support="clamp",
        theta_postprocess="moment_ratio",
    )


def test_cox_reid_gene_bin_size_is_explicit_and_validated() -> None:
    assert SCTConfig(fit_method="cox_reid").cox_reid_gene_bin_size is None
    assert (
        SCTConfig(fit_method="cox_reid", cox_reid_gene_bin_size=500).statistical_mode
        == "experimental"
    )
    with pytest.raises(ValueError, match="cox_reid_gene_bin_size"):
        SCTConfig(cox_reid_gene_bin_size=0)


@pytest.mark.parametrize(
    ("keyword", "message"),
    [
        ({"return_corrected_umi": True}, "corrected UMI"),
        ({"batch": np.array([0, 1])}, "batch models"),
        ({"covariates": np.ones((2, 1))}, "additional covariates"),
    ],
)
def test_unsupported_features_fail_explicitly(
    keyword: dict[str, object], message: str
) -> None:
    with pytest.raises(NotImplementedError, match=message):
        _fit(np.ones((2, 2), dtype=int), **keyword)
