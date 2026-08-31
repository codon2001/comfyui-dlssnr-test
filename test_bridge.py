import ctypes
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from nodes import _apply_depth_assist, _restore_source_color


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
        ("colorTransfer", ctypes.c_int),
    ]


root = Path(__file__).resolve().parent
bridge = ctypes.CDLL(str(root / "native" / "dlssnr_bridge.dll"))
bridge.dlssnr_create.restype = ctypes.c_void_p
bridge.dlssnr_create.argtypes = [
    ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_int, ctypes.c_int,
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
core = next((Path(os.environ.get("SystemRoot", r"C:\Windows")) /
             "System32" / "DriverStore" / "FileRepository").rglob("nvngx.dll"))
w = h = 512
y, x = np.mgrid[0:h, 0:w]
rgb = np.stack([
    x * 255 // (w - 1), y * 255 // (h - 1), ((x // 8 + y // 8) % 2) * 255,
], axis=-1).astype(np.uint8)
alpha = np.full((h, w, 1), 255, dtype=np.uint8)
image = np.ascontiguousarray(np.concatenate([rgb, alpha], axis=-1))
error = ctypes.create_string_buffer(2048)


def evaluate(color_transfer, depth, preset=0):
    handle = bridge.dlssnr_create(
        runtime, str(core), w, h, 0, preset, error, len(error))
    if not handle:
        raise RuntimeError(error.value.decode("utf-8", "replace"))
    settings = Settings(0, 1.0, 1.0, 1.0, 0, 0, 1.0, 0, 1,
                        1.0, 1.0, 1.0, color_transfer)
    output = np.empty_like(image)
    previous = None
    motion = np.zeros((h, w, 2), dtype=np.float16)
    try:
        for i in range(8):
            settings.reset = int(i == 0)
            ok = bridge.dlssnr_process(
                handle, image.ctypes.data, image.strides[0],
                None if depth is None else depth.ctypes.data,
                0 if depth is None else depth.strides[0],
                motion.ctypes.data, motion.strides[0], ctypes.byref(settings),
                output.ctypes.data, output.strides[0], error, len(error),
            )
            if not ok:
                raise RuntimeError(error.value.decode("utf-8", "replace"))
            previous = output.copy()
        values = previous[..., :3].astype(np.float32)
        return previous, (float(values.mean()), float(values.std()))
    finally:
        bridge.dlssnr_destroy(handle)


gradient_depth = np.ascontiguousarray(x.astype(np.float32) / (w - 1))
results = {}
for transfer in (0, 1):
    for depth_name, depth_value in (("zero", None), ("gradient", gradient_depth)):
        output, stats = evaluate(transfer, depth_value)
        key = (transfer, depth_name)
        results[key] = output
        restored = _restore_source_color(image, output)
        restored_values = restored[..., :3].astype(np.float32)
        digest = hashlib.sha256(output).hexdigest()[:16]
        print(f"transfer={transfer} depth={depth_name} sha256={digest} "
              f"mean={stats[0]:.4f} std={stats[1]:.4f} "
              f"restored_mean={restored_values.mean():.4f} "
              f"restored_std={restored_values.std():.4f}")
    difference = np.abs(results[(transfer, "zero")][..., :3].astype(np.int16) -
                        results[(transfer, "gradient")][..., :3].astype(np.int16))
    print(f"transfer={transfer} depth_ab_mean={difference.mean():.8f} "
          f"depth_ab_max={difference.max()}")
input_values = image[..., :3].astype(np.float32)
print(f"input mean={input_values.mean():.4f} std={input_values.std():.4f}")
native = results[(0, "gradient")]
native_restored = _restore_source_color(image, native)
assisted = _restore_source_color(
    image, _apply_depth_assist(image, native, gradient_depth, 1.0))
assist_difference = np.abs(
    native_restored[..., :3].astype(np.int16) -
    assisted[..., :3].astype(np.int16))
print(f"node_depth_assist_mean={assist_difference.mean():.8f} "
      f"node_depth_assist_max={assist_difference.max()}")
for preset in range(4):
    preset_output, _ = evaluate(0, None, preset)
    print(f"preset={preset} sha256={hashlib.sha256(preset_output).hexdigest()[:16]}")
