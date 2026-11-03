from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .core import SCTConfig, SCTResult, fit_transform
from .selection import SelectionManifest

if TYPE_CHECKING:
    from anndata import AnnData


class ImplicitDeviceTransferError(TypeError):
    """Raised when the adapter would otherwise hide a device transfer."""


def _is_cuda_array(array: object) -> bool:
    module = type(array).__module__.split(".", maxsplit=1)[0]
    return module in {"cupy", "cupyx"} or hasattr(array, "__cuda_array_interface__")


class AnnDataLike(Protocol):
    X: object
    obs_names: Sequence[object]
    var_names: Sequence[object]
    layers: Mapping[str, object]


def fit_transform_anndata(
    adata: AnnDataLike,
    *,
    layer: str | None = None,
    config: SCTConfig | None = None,
    selection: SelectionManifest | None = None,
    return_corrected_umi: bool = False,
    corrected_umi_depth: float | None = None,
) -> SCTResult:
    """Read cells-by-genes counts from AnnData without mutating the object.

    Inputs and outputs are CPU-resident. CUDA-backed AnnData counts are rejected instead
    of being copied to host as a hidden side effect; an explicitly selected fitting
    backend may still move bounded internal blocks to a device.
    """
    if layer is None:
        counts = adata.X
    else:
        try:
            counts = adata.layers[layer]
        except KeyError as error:
            raise KeyError(f"AnnData layer {layer!r} does not exist") from error

    if _is_cuda_array(counts):
        raise ImplicitDeviceTransferError(
            "the AnnData adapter received CUDA-backed counts; explicitly transfer the "
            "selected matrix to CPU before calling the CPU backend"
        )
    return fit_transform(
        counts,
        gene_names=tuple(str(value) for value in adata.var_names),
        cell_names=tuple(str(value) for value in adata.obs_names),
        matrix_orientation="cells_by_genes",
        config=config,
        selection=selection,
        return_corrected_umi=return_corrected_umi,
        corrected_umi_depth=corrected_umi_depth,
    )


def write_anndata(
    adata: AnnData,
    result: SCTResult,
    *,
    n_top_genes: int = 3000,
    copy: bool = False,
) -> AnnData | None:
    """Store an SCT result in AnnData, returning a copy only when requested.

    Names and order must match the fitted data. Replaces standard HVG annotations,
    ``var['sct_*']`` parameters, ``uns['sct']``, and the ``sct_residuals`` and
    ``sct_corrected_umi`` layers. X, raw, and other layers are preserved. Summary
    results remove any previous residual layer. Excluded genes have NaN residuals
    and HVG ranks; subset to eligible genes before downstream PCA.

    Results are copied to CPU cells-by-genes layers. In-place writes to views or
    backed objects are rejected; use copy=True for views or to_memory() for backed
    objects. Configuration and selection are stored as JSON for H5AD portability.
    """
    from . import __version__

    if adata.isbacked:
        raise ValueError("write_anndata requires an in-memory object; use to_memory()")
    if adata.is_view and not copy:
        raise ValueError("cannot write to an AnnData view; use copy=True")
    if (
        tuple(adata.var_names) != result.gene_names
        or tuple(adata.obs_names) != result.cell_names
    ):
        raise ValueError("AnnData gene/cell names and order must match the SCT result")
    if (
        isinstance(n_top_genes, bool)
        or not isinstance(n_top_genes, (int, np.integer))
        or n_top_genes < 1
    ):
        raise ValueError("n_top_genes must be a positive integer")

    target = adata.copy() if copy else adata
    rank = result.hvg_rank.astype(np.float64)
    rank[rank < 0] = np.nan
    target.var["highly_variable"] = rank < n_top_genes
    target.var["highly_variable_rank"] = rank
    target.var["residual_variances"] = result.residual_variance.copy()
    for name, values in result.parameter_dict().items():
        target.var[f"sct_{name}"] = values
    for name in ("eligible_gene", "poisson_gene", "parameter_outlier", "residual_mean"):
        target.var[f"sct_{name}"] = getattr(result, name).copy()
    for key in ("sct_residuals", "sct_corrected_umi"):
        target.layers.pop(key, None)
    if result.residuals is not None:
        target.layers["sct_residuals"] = result.residuals.T.copy()
    if result.corrected_umi is not None:
        target.layers["sct_corrected_umi"] = result.corrected_umi.T.tocsr()
    target.uns["sct"] = {
        "version": __version__,
        "config_json": json.dumps(asdict(result.config), sort_keys=True),
        "selection_json": json.dumps(result.selection.to_dict(), sort_keys=True),
        "clip_range": np.asarray(result.clip_range),
        "n_top_genes": int(n_top_genes),
    }
    if result.corrected_umi_depth is not None:
        target.uns["sct"]["corrected_umi_depth"] = result.corrected_umi_depth
    return target if copy else None
