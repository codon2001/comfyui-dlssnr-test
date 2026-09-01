from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BRIDGE_DLL = ROOT / "native" / "dlssnr_bridge.dll"
_LOCK = threading.Lock()
_CACHE: dict[int, "GpuMapping"] = {}
_LIBRARY = None


@dataclass(frozen=True)
class GpuMapping:
    cuda_index: int
    dxgi_adapter_index: int
    dxgi_nvidia_index: int
    adapter_name: str

    @property
    def label_suffix(self) -> str:
        return (f"DXGI {self.dxgi_adapter_index} / NVIDIA #{self.dxgi_nvidia_index}")


def _library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    bridge_path = BRIDGE_DLL
    # Reuse the package's verified .bin recovery path when imported normally.
    # The fallback keeps this helper usable in isolated diagnostics.
    try:
        from .nodes import _bridge_path
        bridge_path = _bridge_path()
    except (ImportError, AttributeError):
        pass
    library = ctypes.CDLL(str(bridge_path))
    function = library.dlssnr_map_cuda_device
    function.argtypes = (
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_wchar_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int,
    )
    function.restype = ctypes.c_int
    _LIBRARY = library
    return library


def resolve_cuda_gpu(cuda_index: int) -> GpuMapping:
    index = max(0, int(cuda_index))
    with _LOCK:
        cached = _CACHE.get(index)
        if cached is not None:
            return cached
        dxgi = ctypes.c_int(-1)
        nvidia = ctypes.c_int(-1)
        name = ctypes.create_unicode_buffer(256)
        error = ctypes.create_string_buffer(1024)
        ok = _library().dlssnr_map_cuda_device(
            index, ctypes.byref(dxgi), ctypes.byref(nvidia),
            name, len(name), error, len(error),
        )
        if not ok:
            detail = error.value.decode("utf-8", "replace")
            raise RuntimeError(detail or f"无法映射 CUDA GPU {index} 到 DXGI")
        result = GpuMapping(index, dxgi.value, nvidia.value, name.value)
        _CACHE[index] = result
        return result


def directml_device_index(cuda_index: int) -> int:
    """Return the global DXGI index required by DirectML."""
    return resolve_cuda_gpu(cuda_index).dxgi_adapter_index
