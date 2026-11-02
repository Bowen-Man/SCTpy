"""Native Python components for the documented SCTransform v2 model."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sctransform-native")
except PackageNotFoundError:  # pragma: no cover - only an unpackaged source tree
    __version__ = "0+unknown"

__all__ = ["__version__"]
