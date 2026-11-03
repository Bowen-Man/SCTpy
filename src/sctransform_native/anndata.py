from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .core import SCTConfig, SCTResult, fit_transform
from .selection import SelectionManifest


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

    This alpha adapter is CPU-only. CUDA-backed AnnData inputs are rejected instead of
    being copied to host as a hidden side effect.
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
