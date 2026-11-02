"""Native Python components for the documented SCTransform v2 model."""

from importlib.metadata import PackageNotFoundError, version

from .anndata import fit_transform_anndata
from .backend import (
    ArrayBackend,
    Device,
    ImplicitDeviceTransferError,
    TransferRecord,
    TransferResult,
    infer_device,
)
from .core import SCTConfig, SCTResult, fit_transform
from .selection import SelectionManifest, create_selection_manifest

try:
    __version__ = version("sctransform-native")
except PackageNotFoundError:  # pragma: no cover - only an unpackaged source tree
    __version__ = "0+unknown"

__all__ = [
    "SCTConfig",
    "SCTResult",
    "SelectionManifest",
    "ArrayBackend",
    "Device",
    "ImplicitDeviceTransferError",
    "TransferRecord",
    "TransferResult",
    "__version__",
    "create_selection_manifest",
    "fit_transform",
    "fit_transform_anndata",
    "infer_device",
]
