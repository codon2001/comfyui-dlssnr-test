from __future__ import annotations

import base64
import asyncio
import ctypes
import glob
import hashlib
import io
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # Allows isolated import checks outside ComfyUI.
    web = None
    PromptServer = None


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / "runtimes"
CACHE_DIR = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / \
    "ComfyUI-DLSSNR-TEST3"
BRIDGE_DLL = ROOT / "native" / "dlssnr_bridge.dll"
BRIDGE_PORTABLE = ROOT / "native" / "dlssnr_bridge.bin"
BUNDLED_RUNTIME = RUNTIME_DIR / "default" / "dlssnr.dll"
BUNDLED_RUNTIME_PARTS = RUNTIME_DIR / "default" / "parts"
BRIDGE_SHA256 = "B92F579EFED98E0AEA5A6C4F67315928C4CD1B435F67DD8CA7A756214575B57B"
BUNDLED_RUNTIME_SHA256 = "984BEE0F775C277D5829B8FD6775D53A7B0F75396C852B3AAF06A18375F81014"
DEFAULT_RUNTIME = ""
_stop_events: dict[str, threading.Event] = {}
_stop_lock = threading.Lock()
_live_settings: dict[str, dict] = {}
_live_settings_lock = threading.Lock()
_preview_condition = threading.Condition()
_preview_pending: dict[str, tuple] = {}
_preview_thread: threading.Thread | None = None

_HOT_SETTING_NAMES = {
    "automatic_mask", "nr_style", "nr_intensity", "local_tone_strength",
    "local_structure_strength", "skin_structure_strength", "ui_correction",
    "frame_guidance", "depth_convention", "motion_scale_x", "motion_scale_y",
    "depth_inference_interval", "preview_fps", "preview_max_side", "preview_jpeg_quality",
    "safety_timeout_seconds",
    "scene_paper_white_scale", "color_fix",
    "depth_assist_strength",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _is_real_pe(path: Path, minimum_size: int = 4096) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < minimum_size:
            return False
        with path.open("rb") as source:
            return source.read(2) == b"MZ"
    except OSError:
        return False


def _verified(path: Path, expected_sha256: str, minimum_size: int = 4096) -> bool:
    return _is_real_pe(path, minimum_size) and _sha256(path) == expected_sha256


def _atomic_copy(source: Path, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        if not _verified(temporary, expected_sha256):
            raise RuntimeError(f"便携二进制校验失败：{source.name}")
        os.replace(temporary, destination)
        return destination
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _assemble_parts(parts: list[Path], destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
        if not _verified(temporary, expected_sha256, 100 * 1024 * 1024):
            raise RuntimeError("内置 DLSSNR DLL 分片不完整或校验失败，请重新解压节点包。")
        os.replace(temporary, destination)
        return destination
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _bridge_path() -> Path:
    if _verified(BRIDGE_DLL, BRIDGE_SHA256):
        return BRIDGE_DLL
    if not _verified(BRIDGE_PORTABLE, BRIDGE_SHA256):
        hint = "检测到 Git LFS 指针文件" if BRIDGE_DLL.is_file() else "文件不存在"
        raise RuntimeError(
            f"原生桥接不可用（{hint}）。请使用完整的 TEST3 压缩包重新安装。")
    for destination in (BRIDGE_DLL, CACHE_DIR / "native" / "dlssnr_bridge.dll"):
        try:
            return _atomic_copy(BRIDGE_PORTABLE, destination, BRIDGE_SHA256)
        except OSError:
            continue
    raise RuntimeError("无法释放原生桥接 DLL；请检查节点目录或 LOCALAPPDATA 的写入权限。")


def _bundled_runtime_path() -> Path | None:
    if _verified(BUNDLED_RUNTIME, BUNDLED_RUNTIME_SHA256, 100 * 1024 * 1024):
        return BUNDLED_RUNTIME
    parts = sorted(BUNDLED_RUNTIME_PARTS.glob("dlssnr.part*"))
    if not parts:
        return None
    for destination in (BUNDLED_RUNTIME,
                        CACHE_DIR / "runtimes" / "default" / "dlssnr.dll"):
        try:
            return _assemble_parts(parts, destination, BUNDLED_RUNTIME_SHA256)
        except OSError:
            continue
    raise RuntimeError("无法重组内置 DLSSNR DLL；请检查磁盘空间和写入权限。")


def _set_live_settings(node_id: str, values: dict) -> int:
    filtered = {key: value for key, value in values.items() if key in _HOT_SETTING_NAMES}
    with _live_settings_lock:
        current = _live_settings.setdefault(node_id, {"revision": 0, "values": {}})
        current["values"].update(filtered)
        current["revision"] += 1
        return int(current["revision"])


def _get_live_settings(node_id: str) -> tuple[int, dict]:
    with _live_settings_lock:
        current = _live_settings.get(node_id)
        if not current:
            return 0, {}
        return int(current["revision"]), dict(current["values"])


def _clear_live_settings(node_id: str) -> None:
    with _live_settings_lock:
        _live_settings.pop(node_id, None)


def runtime_input():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    bundled = _bundled_runtime_path()
    choices = sorted(
        path.relative_to(RUNTIME_DIR).as_posix()
        for path in RUNTIME_DIR.rglob("*.dll")
        if _is_real_pe(path) and path.name.lower() != "nvngx.dll"
    )
    if bundled is not None and bundled.is_relative_to(CACHE_DIR):
        choices.append("default/dlssnr.dll")
        choices = sorted(set(choices))
    if not choices:
        choices = ["请把 nvngx_dlssnr.dll 放入 runtimes 文件夹"]
    return (choices,)


def gpu_input():
    choices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            # Keep the serialized value compatible with existing workflows.
            # The native bridge maps this Torch/CUDA ordinal to DXGI by LUID.
            choices.append(f"{index}: {torch.cuda.get_device_name(index)}")
    return (choices or ["0: NVIDIA GPU"],)


def gpu_index(selection) -> int:
    try:
        return max(0, int(str(selection).split(":", 1)[0].strip()))
    except (TypeError, ValueError):
        return 0


def _runtime_path(selection: str) -> Path:
    value = Path(str(selection).strip().strip('"'))
    bundled = _bundled_runtime_path()
    # Early versions serialized a machine-specific absolute path. Migrate that
    # value to this package instead of trying another computer's drive letter.
    if value.is_absolute():
        preferred = RUNTIME_DIR / value.parent.name / value.name
        candidates = sorted(
            path for path in RUNTIME_DIR.rglob(value.name)
            if _is_real_pe(path)
        )
        if _is_real_pe(preferred):
            runtime = preferred
        elif len(candidates) == 1:
            runtime = candidates[0]
        elif bundled is not None and value.name.lower() in {
                "dlssnr.dll", "nvngx_dlssnr.dll"}:
            runtime = bundled
        else:
            runtime = preferred
    elif value.as_posix().lower() == "default/dlssnr.dll" and bundled is not None:
        runtime = bundled
    else:
        runtime = RUNTIME_DIR / value
    runtime = runtime.resolve()
    if not _is_real_pe(runtime) or runtime.suffix.lower() != ".dll":
        lfs_hint = "；检测到的文件可能只是 Git LFS 指针" if runtime.is_file() else ""
        raise RuntimeError(
            f"DLSSNR DLL 不存在或不是有效的 Windows DLL：{runtime}{lfs_hint}。"
            "请使用完整的 TEST3 压缩包，或把真实 DLL 放入 runtimes 文件夹后重启 ComfyUI。")
    return runtime


def _event_for(node_id: str) -> threading.Event:
    with _stop_lock:
        event = threading.Event()
        _stop_events[node_id] = event
        return event


def _remove_event(node_id: str) -> None:
    with _stop_lock:
        _stop_events.pop(node_id, None)


def _select_media_path(kind: str) -> str:
    if os.name != "nt":
        return ""
    scripts = {
        "video": (
            "$d=New-Object System.Windows.Forms.OpenFileDialog;"
            "$d.Filter='视频文件|*.mp4;*.mkv;*.mov;*.avi;*.webm;*.m4v;*.wmv|所有文件|*.*';"
            "$d.Title='选择输入视频';if($d.ShowDialog() -eq 'OK'){$d.FileName}"
        ),
        "gif": (
            "$d=New-Object System.Windows.Forms.OpenFileDialog;"
            "$d.Filter='GIF 文件|*.gif|所有文件|*.*';"
            "$d.Title='选择 GIF';if($d.ShowDialog() -eq 'OK'){$d.FileName}"
        ),
        "video_output": (
            "$d=New-Object System.Windows.Forms.SaveFileDialog;"
            "$d.Filter='MP4 视频|*.mp4';$d.DefaultExt='mp4';$d.AddExtension=$true;"
            "$d.Title='选择视频输出位置';if($d.ShowDialog() -eq 'OK'){$d.FileName}"
        ),
    }
    script = scripts.get(kind)
    if not script:
        return ""
    command = "Add-Type -AssemblyName System.Windows.Forms;" + script
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    result = subprocess.run(
        ("powershell.exe", "-NoProfile", "-STA", "-Command", command),
        capture_output=True, text=True, timeout=300, creationflags=flags)
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""


if (PromptServer is not None and web is not None and
        getattr(PromptServer, "instance", None) is not None):
    @PromptServer.instance.routes.post("/dlssnr_live/stop")
    async def dlssnr_live_stop(request):
        data = await request.json()
        node_id = str(data.get("node_id", ""))
        with _stop_lock:
            event = _stop_events.get(node_id)
            if event:
                event.set()
        return web.json_response({"ok": bool(event), "node_id": node_id})

    @PromptServer.instance.routes.post("/dlssnr_live/update_settings")
    async def dlssnr_live_update_settings(request):
        data = await request.json()
        node_id = str(data.get("node_id", ""))
        values = data.get("settings", {})
        if not isinstance(values, dict):
            return web.json_response({"error": "settings 必须是对象。"}, status=400)
        with _stop_lock:
            active = node_id in _stop_events
        revision = _set_live_settings(node_id, values) if active else 0
        return web.json_response({
            "ok": active, "active": active, "node_id": node_id, "revision": revision,
        })

    @PromptServer.instance.routes.post("/dlssnr_live/select_path")
    async def dlssnr_live_select_path(request):
        data = await request.json()
        path = await asyncio.to_thread(_select_media_path, str(data.get("kind", "")))
        return web.json_response({"path": path})

    @PromptServer.instance.routes.post("/dlssnr_live/upload_media")
    async def dlssnr_live_upload_media(request):
        kind = str(request.query.get("kind", "")).lower()
        allowed = {
            "gif": {".gif"},
            "video": {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv"},
            "media": {".gif", ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv"},
        }.get(kind, set())
        reader = await request.multipart()
        field = await reader.next()
        if field is None or not getattr(field, "filename", None):
            return web.json_response({"error": "没有收到文件。"}, status=400)
        filename = Path(field.filename).name
        if Path(filename).suffix.lower() not in allowed:
            return web.json_response({"error": f"不支持的 {kind} 文件：{filename}"}, status=400)
        try:
            import folder_paths
            upload_dir = Path(folder_paths.get_input_directory()) / "dlssnr_uploads"
        except Exception:
            upload_dir = ROOT / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_id = "".join(ch for ch in str(request.query.get("upload_id", "single"))
                            if ch.isalnum() or ch in "-_")[:80] or "single"
        chunk_index = max(0, int(request.query.get("chunk", "0")))
        chunk_count = max(1, int(request.query.get("chunks", "1")))
        chunk_dir = upload_dir / ".dlssnr_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        temporary = chunk_dir / f"{upload_id}.part"
        if chunk_index == 0:
            mode = "wb"
        elif not temporary.is_file():
            return web.json_response({"error": "上传分片顺序错误，请重新选择文件。"}, status=409)
        else:
            mode = "ab"
        with temporary.open(mode) as output:
            while True:
                chunk = await field.read_chunk(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if chunk_index + 1 < chunk_count:
            return web.json_response({"ok": True, "complete": False,
                                      "chunk": chunk_index + 1, "chunks": chunk_count})
        destination = upload_dir / filename
        if destination.exists():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            destination = upload_dir / f"{destination.stem}_{stamp}{destination.suffix}"
        temporary.replace(destination)
        return web.json_response({"ok": True, "complete": True,
                                  "path": str(destination.resolve())})


class Settings(ctypes.Structure):
    _fields_ = [
        ("style", ctypes.c_int),
        ("intensity", ctypes.c_float),
        ("local_tone", ctypes.c_float),
        ("local_structure", ctypes.c_float),
        ("auto_mask", ctypes.c_int),
        ("reset", ctypes.c_int),
        ("skin_structure", ctypes.c_float),
        ("ui_correction", ctypes.c_int),
        ("depth_inverted", ctypes.c_int),
        ("motion_scale_x", ctypes.c_float),
        ("motion_scale_y", ctypes.c_float),
        ("paper_white_scale", ctypes.c_float),
        ("color_transfer", ctypes.c_int),
    ]


class Bridge:
    # NGX feature creation is much more expensive than one evaluation.  Keep
    # the most recently used compatible session alive across ComfyUI runs, as
    # a real-time renderer does.  The native DLL supports one active session,
    # so a size/runtime/device change atomically replaces the idle cache.
    _cache_lock = threading.RLock()
    _cache_key = None
    _cache_handle = None
    _cache_lib = None
    _cache_refs = 0

    def __init__(self, runtime_path: str, width: int, height: int, gpu: int = 0,
                 preset: int = 0):
        if os.name != "nt":
            raise RuntimeError("DLSSNR Live 目前仅支持 Windows x64。")
        bridge_path = _bridge_path()
        runtime = _runtime_path(runtime_path)
        core = self._find_core()
        self.runtime = runtime
        self.runtime_hash = hashlib.sha256(runtime.read_bytes()).hexdigest().upper()
        key = (str(bridge_path.resolve()), str(runtime.resolve()), str(core.resolve()),
               int(width), int(height), int(gpu), max(0, min(3, int(preset))))
        with Bridge._cache_lock:
            if Bridge._cache_key != key:
                if Bridge._cache_refs:
                    raise RuntimeError("DLSSNR 正在处理另一种尺寸，暂时不能切换 GPU 会话。")
                if Bridge._cache_handle and Bridge._cache_lib:
                    Bridge._cache_lib.dlssnr_destroy(Bridge._cache_handle)
                Bridge._cache_key = None
                Bridge._cache_handle = None
                Bridge._cache_lib = None
                try:
                    lib = ctypes.CDLL(str(bridge_path))
                except OSError as exc:
                    raise RuntimeError(
                        f"Windows 无法加载原生桥接：{bridge_path}。"
                        f"系统错误：{exc}。请确认使用 Windows x64、已更新显卡驱动，"
                        "并检查安全软件是否隔离了 DLL。") from exc
                lib.dlssnr_create.argtypes = [
                    ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                    ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                ]
                lib.dlssnr_create.restype = ctypes.c_void_p
                lib.dlssnr_process.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint32,
                    ctypes.POINTER(Settings), ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
                    ctypes.c_char_p, ctypes.c_int,
                ]
                lib.dlssnr_process.restype = ctypes.c_int
                lib.dlssnr_destroy.argtypes = [ctypes.c_void_p]
                error = ctypes.create_string_buffer(2048)
                handle = lib.dlssnr_create(
                    str(runtime), str(core), width, height, int(gpu), key[-1],
                    error, len(error))
                if not handle:
                    raise RuntimeError(error.value.decode("utf-8", "replace"))
                Bridge._cache_key = key
                Bridge._cache_handle = handle
                Bridge._cache_lib = lib
            Bridge._cache_refs += 1
            self.lib = Bridge._cache_lib
            self.handle = Bridge._cache_handle
            self._cache_owned = True

    @staticmethod
    def _find_core() -> Path:
        root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        packaged_core = RUNTIME_DIR / "core" / "nvngx.dll"
        if _is_real_pe(packaged_core):
            return packaged_core
        repository = root / "System32" / "DriverStore" / "FileRepository"
        matches = []
        # NVIDIA uses different INF directory prefixes between notebook,
        # desktop, DCH and OEM drivers; do not hard-code nv_dispi only.
        for pattern in ("nv*.inf_amd64_*/nvngx.dll",
                        "nv*.inf_amd64_*/*/nvngx.dll"):
            matches.extend(path for path in repository.glob(pattern)
                           if _is_real_pe(path))
        # OEM notebook packages may add more directory layers. This fallback
        # is slower, but runs only when the fast patterns found nothing.
        if not matches and repository.is_dir():
            matches.extend(path for path in repository.rglob("nvngx.dll")
                           if _is_real_pe(path))
        if not matches:
            fallbacks = [
                root / "System32" / "nvngx.dll",
                Path(os.environ.get("ProgramFiles", r"C:\Program Files")) /
                    "NVIDIA Corporation" / "NGX" / "nvngx.dll",
            ]
            for fallback in fallbacks:
                if _is_real_pe(fallback):
                    return fallback
            raise RuntimeError(
                "未找到 NVIDIA NGX 核心 nvngx.dll。TEST3 已检查桌面、笔记本、"
                "OEM/DCH DriverStore 目录及系统目录；该电脑的 NVIDIA 驱动包确实没有"
                "安装 NGX Core。请在 NVIDIA 官网对当前笔记本型号执行驱动的“自定义安装 → "
                "执行清洁安装”，然后重启电脑。也可以把从该电脑 NVIDIA 驱动中取得的"
                "真实 nvngx.dll 放到本节点的 runtimes/core/nvngx.dll。")
        return max(matches, key=lambda p: p.stat().st_mtime)

    def process(self, rgba: np.ndarray, settings: Settings,
                depth: np.ndarray | None = None,
                motion: np.ndarray | None = None) -> np.ndarray:
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        output = np.empty_like(rgba)
        depth_ptr = None
        motion_ptr = None
        depth_stride = 0
        motion_stride = 0
        if depth is not None:
            depth = np.ascontiguousarray(depth, dtype=np.float32)
            depth_ptr = depth.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            depth_stride = depth.shape[1] * 4
        if motion is not None:
            motion = np.ascontiguousarray(motion, dtype=np.float16)
            motion_ptr = motion.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            motion_stride = motion.shape[1] * 4
        error = ctypes.create_string_buffer(2048)
        ok = self.lib.dlssnr_process(
            self.handle,
            rgba.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), rgba.strides[0],
            depth_ptr, depth_stride, motion_ptr, motion_stride,
            ctypes.byref(settings),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), output.strides[0],
            error, len(error),
        )
        if not ok:
            raise RuntimeError(error.value.decode("utf-8", "replace"))
        return output

    def close(self):
        if getattr(self, "_cache_owned", False):
            with Bridge._cache_lock:
                Bridge._cache_refs = max(0, Bridge._cache_refs - 1)
            self._cache_owned = False
            self.handle = None

    @classmethod
    def clear_cache(cls):
        with cls._cache_lock:
            if cls._cache_refs:
                return False
            if cls._cache_handle and cls._cache_lib:
                cls._cache_lib.dlssnr_destroy(cls._cache_handle)
            cls._cache_key = None
            cls._cache_handle = None
            cls._cache_lib = None
            return True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


_NATIVE_COLOR_RESTORE = None
_NATIVE_COLOR_RESTORE_LOCK = threading.Lock()


def _restore_source_color(source_rgba: np.ndarray,
                          processed_rgba: np.ndarray,
                          strength: float = 1.0) -> np.ndarray:
    """Remove the global grey cast while retaining DLSSNR's local changes."""
    amount = float(np.clip(strength, 0.0, 1.0))
    if amount <= 0.0:
        return processed_rgba
    # V1.4 performs the same statistics/color restoration in native OpenMP
    # code.  This keeps image and GIF post-processing from becoming slower
    # than the DLSSNR GPU evaluation itself.  Retain the NumPy path below for
    # source-only development environments using an older bridge.
    global _NATIVE_COLOR_RESTORE
    try:
        with _NATIVE_COLOR_RESTORE_LOCK:
            if _NATIVE_COLOR_RESTORE is None:
                library = Bridge._cache_lib or ctypes.CDLL(str(_bridge_path()))
                function = library.dlssnr_restore_color
                function.argtypes = [
                    ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
                    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float,
                ]
                function.restype = None
                _NATIVE_COLOR_RESTORE = (library, function)
        source_native = np.ascontiguousarray(source_rgba, dtype=np.uint8)
        output_native = np.ascontiguousarray(processed_rgba, dtype=np.uint8).copy()
        height, width = output_native.shape[:2]
        _NATIVE_COLOR_RESTORE[1](
            source_native.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            source_native.strides[0],
            output_native.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            output_native.strides[0], width, height, amount)
        return output_native
    except (AttributeError, OSError):
        pass
    source = source_rgba[..., :3].astype(np.float32) / 255.0
    result = processed_rgba[..., :3].astype(np.float32) / 255.0
    coefficients = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
    source_luma = source @ coefficients
    result_luma = result @ coefficients
    source_low, source_high = np.quantile(source_luma, (0.01, 0.99))
    result_low, result_high = np.quantile(result_luma, (0.01, 0.99))
    luma_gain = float(np.clip(
        (source_high - source_low) / max(result_high - result_low, 1e-5),
        0.5, 2.0))
    source_mid = float(np.median(source_luma))
    result_mid = float(np.median(result_luma))
    restored_luma = (result_luma - result_mid) * luma_gain + source_mid

    source_chroma = source - source_luma[..., None]
    result_chroma = result - result_luma[..., None]
    source_chroma_mean = source_chroma.mean(axis=(0, 1), keepdims=True)
    result_chroma_mean = result_chroma.mean(axis=(0, 1), keepdims=True)
    source_energy = float(np.sqrt(np.mean(
        np.square(source_chroma - source_chroma_mean))))
    result_energy = float(np.sqrt(np.mean(
        np.square(result_chroma - result_chroma_mean))))
    chroma_gain = float(np.clip(source_energy / max(result_energy, 1e-5), 0.5, 2.0))
    restored_chroma = ((result_chroma - result_chroma_mean) * chroma_gain +
                       source_chroma_mean)
    restored = restored_luma[..., None] + restored_chroma
    blended = result + (restored - result) * amount
    output = processed_rgba.copy()
    output[..., :3] = np.ascontiguousarray(
        np.clip(blended, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)
    return output


def _apply_depth_assist(source_rgba: np.ndarray, processed_rgba: np.ndarray,
                        depth: np.ndarray | None, strength: float) -> np.ndarray:
    """Use depth in host-side compositing when the experimental DLL ignores it."""
    return _apply_depth_assist_weight(
        source_rgba, processed_rgba, _depth_assist_weight(depth, strength))


def _depth_assist_weight(depth: np.ndarray | None,
                         strength: float) -> np.ndarray | None:
    """Build the expensive depth modulation map once for reusable depth."""
    amount = max(0.0, float(strength))
    if depth is None or amount <= 0.0:
        return None
    values = np.nan_to_num(np.asarray(depth, dtype=np.float32), copy=False)
    sample_step = max(1, int(max(values.shape) // 512))
    sample = values[::sample_step, ::sample_step]
    low, high = np.quantile(sample, (0.02, 0.98))
    normalised = np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    gradient_y, gradient_x = np.gradient(normalised)
    edge = np.hypot(gradient_x, gradient_y)
    edge_scale = max(float(np.quantile(edge, 0.98)), 1e-6)
    edge = np.clip(edge / edge_scale, 0.0, 1.0)
    # Keep the average DLSSNR strength close to one.  Depth changes where its
    # local detail is applied; discontinuities receive a small extra emphasis.
    weight = 1.0 + (normalised - 0.5) * (0.60 * amount) + edge * (0.20 * amount)
    return np.ascontiguousarray(np.clip(weight, 0.15, 2.0)[..., None], dtype=np.float32)


def _apply_depth_assist_weight(source_rgba: np.ndarray,
                               processed_rgba: np.ndarray,
                               weight: np.ndarray | None) -> np.ndarray:
    if weight is None:
        return processed_rgba
    source = source_rgba[..., :3].astype(np.float32)
    processed = processed_rgba[..., :3].astype(np.float32)
    assisted = source + (processed - source) * weight
    output = processed_rgba.copy()
    output[..., :3] = np.ascontiguousarray(
        np.clip(assisted, 0.0, 255.0) + 0.5, dtype=np.uint8)
    return output


def _process_dlssnr(bridge: Bridge, rgba: np.ndarray, settings: Settings,
                     depth=None, motion=None,
                     depth_assist_strength: float = 0.0,
                     color_fix: bool = False) -> np.ndarray:
    processed = bridge.process(rgba, settings, depth, motion)
    processed = _apply_depth_assist(
        rgba, processed, depth, depth_assist_strength)
    return _restore_source_color(rgba, processed) if color_fix else processed


def _rgba8(frame: torch.Tensor) -> np.ndarray:
    rgb = frame.detach().float().cpu().clamp(0, 1).numpy()
    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    rgb = rgb[..., :3]
    alpha = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate((rgb, alpha), axis=-1) * 255.0 + 0.5,
                                dtype=np.uint8)


def _depth_frame(depth: torch.Tensor | None, index: int, interval: int,
                 height: int, width: int) -> np.ndarray | None:
    if depth is None:
        return None
    source_index = min((index // interval) * interval, depth.shape[0] - 1)
    value = depth[source_index].detach().float().cpu().numpy()
    value = value[..., 0] if value.ndim == 3 else value
    if value.shape != (height, width):
        raise RuntimeError("depth 输入尺寸必须与 image 相同。")
    return np.ascontiguousarray(value, dtype=np.float32)


def _motion_frame(motion, index: int, height: int, width: int) -> np.ndarray | None:
    if motion is None:
        return None
    value = motion[min(index, motion.shape[0] - 1)].detach().float().cpu().numpy()
    if value.shape != (height, width, 2):
        raise RuntimeError("motion_vectors 必须是 [B,H,W,2] 的 DLSSNR_MOTION。")
    return np.ascontiguousarray(value, dtype=np.float16)


def _comparison_rgba(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """Build a same-height left/right comparison with a visible divider."""
    height = original.shape[0]
    divider = np.zeros((height, 6, 4), dtype=np.uint8)
    divider[..., 3] = 255
    divider[:, 2:4, :3] = 255
    return np.ascontiguousarray(np.concatenate((original, divider, processed), axis=1))


def _preview_worker() -> None:
    """Encode only the newest preview per node without blocking DLSSNR."""
    while True:
        with _preview_condition:
            while not _preview_pending:
                _preview_condition.wait()
            node_id, payload = next(iter(_preview_pending.items()))
            _preview_pending.pop(node_id, None)
        rgba, iteration, fps, runtime_hash, state, preview_max_side, preview_jpeg_quality = payload
        server = None if PromptServer is None else getattr(PromptServer, "instance", None)
        if server is None:
            continue
        try:
            image = Image.fromarray(rgba[..., :3], "RGB")
            max_side = max(320, min(4096, int(preview_max_side)))
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            quality = max(50, min(100, int(preview_jpeg_quality)))
            image.save(buffer, "JPEG", quality=quality, optimize=False)
            server.send_sync("dlssnr_live_preview", {
                "node_id": node_id,
                "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "iteration": iteration,
                "fps": round(fps, 2),
                "dll_sha256": runtime_hash,
                "state": state,
            })
        except Exception:
            # A preview must never stop the actual image/video processing.
            continue


def _preview(node_id: str, rgba: np.ndarray, iteration: int, fps: float,
             runtime_hash: str, state: str = "running",
             preview_max_side: int = 1920,
             preview_jpeg_quality: int = 90) -> None:
    global _preview_thread
    server = None if PromptServer is None else getattr(PromptServer, "instance", None)
    if server is None:
        return
    # Each loop creates a fresh comparison array, so retaining its reference is
    # safe. Replacing the dictionary entry drops stale previews automatically.
    with _preview_condition:
        _preview_pending[node_id] = (
            rgba, int(iteration), float(fps), str(runtime_hash), str(state),
            max(320, min(4096, int(preview_max_side))),
            max(50, min(100, int(preview_jpeg_quality))))
        if _preview_thread is None or not _preview_thread.is_alive():
            _preview_thread = threading.Thread(
                target=_preview_worker, name="DLSSNRPreview", daemon=True)
            _preview_thread.start()
        _preview_condition.notify()


class DLSSNRLive:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "run_mode": (["实时静态图（手动停止）", "视频/批量逐帧"],),
                "dll_path": runtime_input(),
                "gpu_device": gpu_input(),
                "nr_preset": (["0 Default", "1 Preset #1", "2 Preset #2", "3 Preset #3"],),
                "automatic_mask": ("BOOLEAN", {"default": False}),
                "nr_style": (["0 默认（Default）", "1 自然（Natural）", "2 电影感（Cinematic）"],),
                "nr_intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "local_tone_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "local_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "skin_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "ui_correction": ("BOOLEAN", {"default": False}),
                "color_fix": ("BOOLEAN", {"default": False, "tooltip": "打开后才执行色彩修复；默认保持原始 DLSSNR 色调。"}),
                "frame_guidance": (["0 使用可用引导（深度和运动）", "1 强制零引导", "2 仅运动向量", "3 仅深度", "1 反向深度（Inverted）", "0 正常深度（Normal）"],),
                "depth_convention": ("STRING", {"default": "1 反向深度（Inverted）"}),
                "motion_scale_x": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05, "display": "slider"}),
                "motion_scale_y": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05, "display": "slider"}),
                "depth_inference_interval": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1, "display": "slider"}),
                "preview_fps": ("INT", {"default": 12, "min": 1, "max": 30, "step": 1, "display": "slider"}),
                "preview_max_side": ("INT", {"default": 1920, "min": 320, "max": 4096, "step": 64, "display": "slider", "tooltip": "仅控制节点内预览传输尺寸，不改变 DLSSNR 处理与最终输出分辨率。1280 约等于 720p 预览档。"}),
                "preview_jpeg_quality": ("INT", {"default": 90, "min": 50, "max": 100, "step": 1, "display": "slider", "tooltip": "仅控制节点内 JPEG 预览质量；调到 82 可降低预览延迟，不影响最终输出。"}),
                "safety_timeout_seconds": ("INT", {"default": 300, "min": 0, "max": 86400, "step": 1, "display": "slider"}),
                "scene_paper_white_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "display": "slider", "tooltip": "等效场景纸白亮度；1.0保持原图，高于1提亮并保护高光。"}),
                "depth_assist_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "display": "slider", "tooltip": "当前实验 DLL 可能忽略原生深度；此项在节点侧用深度调制局部增强，0 表示关闭。"}),
            },
            "optional": {
                "depth": ("IMAGE",),
                "motion_vectors": ("DLSSNR_MOTION",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "comparison_image", "status")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def run(self, image, run_mode, dll_path, gpu_device, nr_preset,
            automatic_mask, nr_style,
            nr_intensity, local_tone_strength, local_structure_strength,
            skin_structure_strength, scene_paper_white_scale, ui_correction, color_fix, frame_guidance,
            depth_convention, motion_scale_x, motion_scale_y,
            depth_inference_interval, preview_fps, preview_max_side,
            preview_jpeg_quality,
            safety_timeout_seconds, depth_assist_strength,
            unique_id, depth=None, motion_vectors=None):
        node_id = str(unique_id)
        _clear_live_settings(node_id)
        event = _event_for(node_id)
        batch, height, width, _ = image.shape
        style = int(str(nr_style).split()[0])
        guidance = int(str(frame_guidance).split()[0])
        settings = Settings(style, nr_intensity, local_tone_strength,
                            local_structure_strength, int(automatic_mask), 0,
                            skin_structure_strength, int(ui_correction),
                            int(str(depth_convention).split()[0]),
                            motion_scale_x, motion_scale_y, scene_paper_white_scale)
        processed_results = []
        comparison_results = []
        iteration = 0
        started = time.perf_counter()
        last_preview = 0.0
        last_input_rgba = _rgba8(image[0])
        last_rgba = last_input_rgba.copy()
        last_comparison = _comparison_rgba(last_input_rgba, last_rgba)
        applied_revision = 0
        cached_live_depth = None
        cached_live_motion = None
        cached_depth_weight = None
        cached_depth_weight_key = None
        try:
            with Bridge(dll_path, width, height, gpu_index(gpu_device),
                        int(str(nr_preset).split()[0])) as bridge:
                is_live = run_mode.startswith("实时") and batch == 1
                while True:
                    revision, hot = _get_live_settings(node_id)
                    settings_changed = revision != applied_revision
                    if settings_changed:
                        settings.style = int(str(hot.get("nr_style", settings.style)).split()[0])
                        settings.intensity = float(hot.get("nr_intensity", settings.intensity))
                        settings.local_tone = float(hot.get("local_tone_strength", settings.local_tone))
                        settings.local_structure = float(hot.get(
                            "local_structure_strength", settings.local_structure))
                        settings.skin_structure = float(hot.get(
                            "skin_structure_strength", settings.skin_structure))
                        settings.paper_white_scale = float(hot.get(
                            "scene_paper_white_scale", settings.paper_white_scale))
                        settings.auto_mask = int(bool(hot.get("automatic_mask", settings.auto_mask)))
                        settings.ui_correction = int(bool(hot.get("ui_correction", settings.ui_correction)))
                        color_fix = bool(hot.get("color_fix", color_fix))
                        settings.depth_inverted = int(str(hot.get(
                            "depth_convention", settings.depth_inverted)).split()[0])
                        settings.motion_scale_x = float(hot.get(
                            "motion_scale_x", settings.motion_scale_x))
                        settings.motion_scale_y = float(hot.get(
                            "motion_scale_y", settings.motion_scale_y))
                        guidance = int(str(hot.get("frame_guidance", guidance)).split()[0])
                        depth_inference_interval = max(1, int(hot.get(
                            "depth_inference_interval", depth_inference_interval)))
                        preview_fps = max(1, int(hot.get("preview_fps", preview_fps)))
                        preview_max_side = max(320, min(4096, int(hot.get(
                            "preview_max_side", preview_max_side))))
                        preview_jpeg_quality = max(50, min(100, int(hot.get(
                            "preview_jpeg_quality", preview_jpeg_quality))))
                        safety_timeout_seconds = max(0, int(hot.get(
                            "safety_timeout_seconds", safety_timeout_seconds)))
                        depth_assist_strength = float(hot.get(
                            "depth_assist_strength", depth_assist_strength))
                        applied_revision = revision
                    index = 0 if is_live else iteration
                    if not is_live and index >= batch:
                        break
                    # Static live preview always evaluates the same input image.
                    # Reuse its full-resolution RGBA buffer instead of copying
                    # the ComfyUI tensor back to CPU on every iteration.
                    rgba = last_input_rgba if is_live else _rgba8(image[index])
                    last_input_rgba = rgba
                    use_depth = guidance in (0, 3)
                    use_motion = guidance in (0, 2)
                    if is_live:
                        if use_depth and cached_live_depth is None and depth is not None:
                            cached_live_depth = _depth_frame(
                                depth, 0, depth_inference_interval, height, width)
                        if use_motion and cached_live_motion is None and motion_vectors is not None:
                            cached_live_motion = _motion_frame(
                                motion_vectors, 0, height, width)
                        depth_np = cached_live_depth if use_depth else None
                        motion_np = cached_live_motion if use_motion else None
                    else:
                        depth_np = _depth_frame(depth, index, depth_inference_interval,
                                                height, width) if use_depth else None
                        motion_np = _motion_frame(motion_vectors, index, height, width) \
                            if use_motion else None
                    # Reset temporal history when hot parameters change so the
                    # first visible result represents the new values directly.
                    settings.reset = 1 if iteration == 0 or settings_changed else 0
                    raw_processed = bridge.process(
                        rgba, settings, depth_np, motion_np)
                    iteration += 1
                    # A slider may move while the GPU is evaluating. Do not
                    # spend more time post-processing or display that stale
                    # result; immediately start another evaluation instead.
                    latest_revision, _ = _get_live_settings(node_id)
                    if is_live and latest_revision != applied_revision:
                        continue
                    now = time.perf_counter()
                    should_present = (not is_live or settings_changed or event.is_set() or
                                      now - last_preview >= 1.0 / preview_fps)
                    if is_live and not should_present:
                        if safety_timeout_seconds and now - started >= safety_timeout_seconds:
                            break
                        try:
                            import comfy.model_management as mm
                            mm.throw_exception_if_processing_interrupted()
                        except ImportError:
                            pass
                        continue
                    weight_key = (id(depth_np), float(depth_assist_strength))
                    if weight_key != cached_depth_weight_key:
                        cached_depth_weight = _depth_assist_weight(
                            depth_np, depth_assist_strength)
                        cached_depth_weight_key = weight_key
                    last_rgba = _apply_depth_assist_weight(
                        rgba, raw_processed, cached_depth_weight)
                    if color_fix:
                        last_rgba = _restore_source_color(rgba, last_rgba)
                    latest_revision, _ = _get_live_settings(node_id)
                    if is_live and latest_revision != applied_revision:
                        continue
                    last_comparison = _comparison_rgba(rgba, last_rgba)
                    if not is_live:
                        processed_results.append(
                            torch.from_numpy(last_rgba[..., :3].copy()).float() / 255.0)
                        comparison_results.append(
                            torch.from_numpy(last_comparison[..., :3].copy()).float() / 255.0)
                    now = time.perf_counter()
                    if should_present:
                        _preview(node_id, last_comparison, iteration,
                                 iteration / max(now - started, 1e-6), bridge.runtime_hash,
                                 preview_max_side=preview_max_side,
                                 preview_jpeg_quality=preview_jpeg_quality)
                        last_preview = now
                    if event.is_set():
                        break
                    if safety_timeout_seconds and now - started >= safety_timeout_seconds:
                        break
                    try:
                        import comfy.model_management as mm
                        mm.throw_exception_if_processing_interrupted()
                    except ImportError:
                        pass
                    if not is_live and iteration >= batch:
                        break
                if is_live:
                    processed_results = [
                        torch.from_numpy(last_rgba[..., :3].copy()).float() / 255.0]
                    comparison_results = [
                        torch.from_numpy(last_comparison[..., :3].copy()).float() / 255.0]
                elapsed = time.perf_counter() - started
                _preview(node_id, last_comparison, iteration,
                         iteration / max(elapsed, 1e-6), bridge.runtime_hash, "stopped",
                         preview_max_side, preview_jpeg_quality)
                status = (f"完成：{iteration} 次评估，{elapsed:.2f}s，"
                          f"DLL SHA256={bridge.runtime_hash}")
            return (
                torch.stack(processed_results).to(image.device),
                torch.stack(comparison_results).to(image.device),
                status,
            )
        finally:
            _remove_event(node_id)
            _clear_live_settings(node_id)


NODE_CLASS_MAPPINGS = {
    "DLSSNRLive": DLSSNRLive,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DLSSNRLive": "DLSSNR 实时预览 / 手动停止",
}


