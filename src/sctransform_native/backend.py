from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

DeviceKind = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class Device:
    kind: DeviceKind
    index: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "cpu" and self.index is not None:
            raise ValueError("a CPU device does not have a CUDA index")
        if self.kind == "cuda" and (self.index is None or self.index < 0):
            raise ValueError("a CUDA device requires a nonnegative index")


@dataclass(frozen=True)
class TransferRecord:
    source: Device
    destination: Device
    nbytes: int
    elapsed_seconds: float


class ImplicitDeviceTransferError(TypeError):
    """Raised when an adapter would otherwise hide a device transfer."""


def infer_device(array: object) -> Device:
    """Infer CPU versus CUDA without importing an optional GPU package."""
    module = type(array).__module__.split(".", maxsplit=1)[0]
    if module in {"cupy", "cupyx"} or hasattr(array, "__cuda_array_interface__"):
        device = getattr(array, "device", None)
        index = getattr(device, "id", 0)
        return Device("cuda", int(index))
    return Device("cpu")


class ArrayBackend(Protocol):
    """Minimal boundary for a future same-device numerical implementation."""

    name: str

    def device(self, array: object) -> Device: ...

    def asarray(
        self, array: object, *, dtype: object | None = None, copy: bool = False
    ) -> object: ...

    def to_device(
        self, array: object, destination: Device
    ) -> tuple[object, TransferRecord]: ...

    def synchronize(self) -> None: ...
