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


def test_unknown_regularization_support_is_rejected() -> None:
    with pytest.raises(ValueError, match="regularization_support"):
        SCTConfig(regularization_support="unknown")  # type: ignore[arg-type]


def test_unknown_regularization_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="regularization_method"):
        SCTConfig(regularization_method="unknown")  # type: ignore[arg-type]


def test_unknown_theta_postprocess_is_rejected() -> None:
    with pytest.raises(ValueError, match="theta_postprocess"):
        SCTConfig(theta_postprocess="unknown")  # type: ignore[arg-type]


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
