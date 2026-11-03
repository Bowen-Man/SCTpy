"""Native Python components for the documented SCTransform v2 model."""

from importlib.metadata import PackageNotFoundError, version

from .anndata import ImplicitDeviceTransferError, fit_transform_anndata, write_anndata
from .core import SCTConfig, SCTResult, fit_transform
from .selection import SelectionManifest, create_selection_manifest

try:
    __version__ = version("sctpy")
except PackageNotFoundError:  # pragma: no cover - only an unpackaged source tree
    __version__ = "0+unknown"

__all__ = [
    "SCTConfig",
    "SCTResult",
    "SelectionManifest",
    "ImplicitDeviceTransferError",
    "__version__",
    "create_selection_manifest",
    "fit_transform",
    "fit_transform_anndata",
    "write_anndata",
]
