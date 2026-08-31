from __future__ import annotations

import os
import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path


def _read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size - len(chunks))
        if not part:
            raise EOFError
        chunks.extend(part)
    return bytes(chunks)


def main() -> int:
    root = Path(__file__).resolve().parent
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheels = sorted((root / "vendor" / "wheels").glob(
        f"onnxruntime_directml-*-{py_tag}-{py_tag}-win_amd64.whl"))
    if not wheels:
        raise RuntimeError(
            f"内置 DirectML 深度运行库不支持 Python {sys.version_info.major}."
            f"{sys.version_info.minor}；当前包支持 Python 3.10-3.13。")
    cache_root = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / \
        "ComfyUI-DLSSNR-TEST3" / "ort_dml"
    vendor = cache_root / f"{py_tag}-1.23.0"
    marker = vendor / ".complete"
    if not marker.is_file():
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f"{py_tag}-", dir=cache_root))
        try:
            with zipfile.ZipFile(wheels[-1]) as archive:
                archive.extractall(temporary)
            (temporary / ".complete").write_text(wheels[-1].name, encoding="utf-8")
            if vendor.exists():
                shutil.rmtree(vendor, ignore_errors=True)
            temporary.replace(vendor)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
    capi = vendor / "onnxruntime" / "capi"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(capi))
    sys.path.insert(0, str(vendor))
    import numpy as np
    import onnxruntime as ort

    model = Path(sys.argv[1]).resolve()
    gpu_index = max(0, int(sys.argv[2]))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model), sess_options=options,
        providers=[("DmlExecutionProvider", {"device_id": gpu_index})],
    )
    source = sys.stdin.buffer
    target = sys.stdout.buffer
    target.write(b"DLSSDML1")
    target.flush()
    while True:
        try:
            height, width = struct.unpack("<II", _read_exact(source, 8))
        except EOFError:
            break
        if not height or not width:
            break
        try:
            count = 3 * height * width
            values = np.frombuffer(_read_exact(source, count * 4), dtype=np.float32)
            values = values.reshape(1, 3, height, width)
            output = session.run(("predicted_depth",), {"pixel_values": values})[0]
            output = np.ascontiguousarray(output, dtype=np.float32).reshape(-1)
            target.write(struct.pack("<i", output.size))
            target.write(output.tobytes())
        except Exception as exc:
            message = str(exc).encode("utf-8", "replace")[:16384]
            target.write(struct.pack("<i", -len(message)))
            target.write(message)
        target.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        message = str(exc).encode("utf-8", "replace")[:16384]
        sys.stdout.buffer.write(b"ERROR001" + struct.pack("<I", len(message)) + message)
        sys.stdout.buffer.flush()
        raise
