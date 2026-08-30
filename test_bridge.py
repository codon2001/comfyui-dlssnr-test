import ctypes
import hashlib
import sys
from pathlib import Path

import numpy as np


class Settings(ctypes.Structure):
    _fields_ = [
        ("style", ctypes.c_int),
        ("intensity", ctypes.c_float),
        ("localTone", ctypes.c_float),
        ("localStructure", ctypes.c_float),
        ("autoMask", ctypes.c_int),
        ("reset", ctypes.c_int),
        ("skinStructure", ctypes.c_float),
        ("uiCorrection", ctypes.c_int),
        ("depthInverted", ctypes.c_int),
        ("motionScaleX", ctypes.c_float),
        ("motionScaleY", ctypes.c_float),
        ("paperWhiteScale", ctypes.c_float),
    ]


root = Path(__file__).resolve().parent
bridge = ctypes.CDLL(str(root / "native" / "dlssnr_bridge.dll"))
bridge.dlssnr_create.restype = ctypes.c_void_p
bridge.dlssnr_create.argtypes = [
    ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_int,
    ctypes.c_char_p, ctypes.c_int,
]
bridge.dlssnr_process.restype = ctypes.c_int
bridge.dlssnr_process.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.POINTER(Settings), ctypes.c_void_p, ctypes.c_uint32,
    ctypes.c_char_p, ctypes.c_int,
]
bridge.dlssnr_destroy.argtypes = [ctypes.c_void_p]

if len(sys.argv) < 2:
    raise SystemExit("用法：python test_bridge.py <nvngx_dlssnr.dll 的绝对路径>")
runtime = sys.argv[1]
core = next(Path(r"C:\Windows\System32\DriverStore\FileRepository").rglob("nvngx.dll"))
w = h = 512
y, x = np.mgrid[0:h, 0:w]
rgb = np.stack([
    x * 255 // (w - 1), y * 255 // (h - 1), ((x // 8 + y // 8) % 2) * 255,
], axis=-1).astype(np.uint8)
alpha = np.full((h, w, 1), 255, dtype=np.uint8)
image = np.ascontiguousarray(np.concatenate([rgb, alpha], axis=-1))
output = np.empty_like(image)
error = ctypes.create_string_buffer(2048)
handle = bridge.dlssnr_create(runtime, str(core), w, h, 0, error, len(error))
if not handle:
    raise RuntimeError(error.value.decode("utf-8", "replace"))

settings = Settings(0, 1.0, 1.0, 1.0, 0, 0, 1.0, 0, 1, 1.0, 1.0, 1.0)
previous = None
try:
    for i in range(8):
        ok = bridge.dlssnr_process(
            handle, image.ctypes.data, image.strides[0],
            None, 0, None, 0, ctypes.byref(settings),
            output.ctypes.data, output.strides[0], error, len(error),
        )
        if not ok:
            raise RuntimeError(error.value.decode("utf-8", "replace"))
        digest = hashlib.sha256(output).hexdigest()[:16]
        delta = 0.0 if previous is None else float(np.abs(
            output[..., :3].astype(np.int16) - previous[..., :3].astype(np.int16)
        ).mean())
        print(f"iteration={i + 1} sha256={digest} mean_delta={delta:.6f}")
        previous = output.copy()
finally:
    bridge.dlssnr_destroy(handle)
