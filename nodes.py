from __future__ import annotations

import base64
import asyncio
import ctypes
import glob
import hashlib
import io
import os
import subprocess
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
BRIDGE_PATH = ROOT / "native" / "dlssnr_bridge.dll"
RUNTIME_DIR = ROOT / "runtimes"
DEFAULT_RUNTIME = ""
_stop_events: dict[str, threading.Event] = {}
_stop_lock = threading.Lock()
_live_settings: dict[str, dict] = {}
_live_settings_lock = threading.Lock()

_HOT_SETTING_NAMES = {
    "automatic_mask", "nr_style", "nr_intensity", "local_tone_strength",
    "local_structure_strength", "skin_structure_strength", "ui_correction",
    "frame_guidance", "depth_convention", "motion_scale_x", "motion_scale_y",
    "depth_inference_interval", "preview_fps", "safety_timeout_seconds",
    "scene_paper_white_scale",
}


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
    choices = sorted(
        path.relative_to(RUNTIME_DIR).as_posix()
        for path in RUNTIME_DIR.rglob("*.dll")
        if path.is_file()
    )
    if not choices:
        choices = ["请把 nvngx_dlssnr.dll 放入 runtimes 文件夹"]
    return (choices,)


def gpu_input():
    choices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            choices.append(f"{index}: {torch.cuda.get_device_name(index)}")
    return (choices or ["0: NVIDIA GPU"],)


def gpu_index(selection) -> int:
    try:
        return max(0, int(str(selection).split(":", 1)[0].strip()))
    except (TypeError, ValueError):
        return 0


def _runtime_path(selection: str) -> Path:
    value = Path(str(selection).strip().strip('"'))
    # Keep old saved workflows readable, while new nodes only expose the list.
    runtime = value if value.is_absolute() else RUNTIME_DIR / value
    runtime = runtime.resolve()
    if not runtime.is_file() or runtime.suffix.lower() != ".dll":
        raise RuntimeError(
            f"DLSSNR DLL 不存在：{runtime}。请放入节点包 runtimes 文件夹后重启 ComfyUI。")
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


if PromptServer is not None and web is not None:
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
        destination = upload_dir / filename
        if destination.exists():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            destination = upload_dir / f"{destination.stem}_{stamp}{destination.suffix}"
        with destination.open("wb") as output:
            while True:
                chunk = await field.read_chunk(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        return web.json_response({"path": str(destination.resolve())})


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
    ]


class Bridge:
    def __init__(self, runtime_path: str, width: int, height: int, gpu: int = 0):
        if os.name != "nt":
            raise RuntimeError("DLSSNR Live 目前仅支持 Windows x64。")
        if not BRIDGE_PATH.is_file():
            raise RuntimeError(f"缺少桥接文件：{BRIDGE_PATH}")
        runtime = _runtime_path(runtime_path)
        core = self._find_core()
        self.runtime = runtime
        self.runtime_hash = hashlib.sha256(runtime.read_bytes()).hexdigest().upper()
        self.lib = ctypes.CDLL(str(BRIDGE_PATH))
        self.lib.dlssnr_create.argtypes = [
            ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int,
        ]
        self.lib.dlssnr_create.restype = ctypes.c_void_p
        self.lib.dlssnr_process.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint32,
            ctypes.POINTER(Settings), ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
            ctypes.c_char_p, ctypes.c_int,
        ]
        self.lib.dlssnr_process.restype = ctypes.c_int
        self.lib.dlssnr_destroy.argtypes = [ctypes.c_void_p]
        error = ctypes.create_string_buffer(2048)
        self.handle = self.lib.dlssnr_create(
            str(runtime), str(core), width, height, int(gpu), error, len(error)
        )
        if not self.handle:
            raise RuntimeError(error.value.decode("utf-8", "replace"))

    @staticmethod
    def _find_core() -> Path:
        root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        pattern = str(root / "System32" / "DriverStore" / "FileRepository" /
                      "nv_dispi.inf_amd64_*" / "nvngx.dll")
        matches = [Path(p) for p in glob.glob(pattern)]
        if not matches:
            fallback = root / "System32" / "nvngx.dll"
            if fallback.is_file():
                return fallback
            raise RuntimeError("未找到 NVIDIA NGX 核心 nvngx.dll，请更新 NVIDIA 驱动。")
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
        if getattr(self, "handle", None):
            self.lib.dlssnr_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


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


def _preview(node_id: str, rgba: np.ndarray, iteration: int, fps: float,
             runtime_hash: str, state: str = "running") -> None:
    if PromptServer is None:
        return
    image = Image.fromarray(rgba[..., :3], "RGB")
    image.thumbnail((960, 960), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=86, optimize=False)
    PromptServer.instance.send_sync("dlssnr_live_preview", {
        "node_id": node_id,
        "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "iteration": iteration,
        "fps": round(fps, 2),
        "dll_sha256": runtime_hash,
        "state": state,
    })


class DLSSNRLive:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "run_mode": (["实时静态图（手动停止）", "视频/批量逐帧"],),
                "dll_path": runtime_input(),
                "gpu_device": gpu_input(),
                "automatic_mask": ("BOOLEAN", {"default": False}),
                "nr_style": (["0 默认（Default）", "1 自然（Natural）", "2 电影感（Cinematic）"],),
                "nr_intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "local_tone_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "local_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "skin_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "ui_correction": ("BOOLEAN", {"default": False}),
                "frame_guidance": (["0 使用可用引导（深度和运动）", "1 强制零引导", "2 仅运动向量", "3 仅深度"],),
                "depth_convention": (["1 反向深度（Inverted）", "0 正常深度（Normal）"],),
                "motion_scale_x": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05, "display": "slider"}),
                "motion_scale_y": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05, "display": "slider"}),
                "depth_inference_interval": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1, "display": "slider"}),
                "preview_fps": ("INT", {"default": 12, "min": 1, "max": 30, "step": 1, "display": "slider"}),
                "safety_timeout_seconds": ("INT", {"default": 300, "min": 0, "max": 86400, "step": 1, "display": "slider"}),
                "scene_paper_white_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "display": "slider", "tooltip": "等效场景纸白亮度；1.0保持原图，高于1提亮并保护高光。"}),
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

    def run(self, image, run_mode, dll_path, gpu_device, automatic_mask, nr_style,
            nr_intensity, local_tone_strength, local_structure_strength,
            skin_structure_strength, scene_paper_white_scale, ui_correction, frame_guidance,
            depth_convention, motion_scale_x, motion_scale_y,
            depth_inference_interval, preview_fps,
            safety_timeout_seconds, unique_id, depth=None, motion_vectors=None):
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
        try:
            with Bridge(dll_path, width, height, gpu_index(gpu_device)) as bridge:
                is_live = run_mode.startswith("实时") and batch == 1
                while True:
                    revision, hot = _get_live_settings(node_id)
                    if revision != applied_revision:
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
                        safety_timeout_seconds = max(0, int(hot.get(
                            "safety_timeout_seconds", safety_timeout_seconds)))
                        applied_revision = revision
                    index = 0 if is_live else iteration
                    if not is_live and index >= batch:
                        break
                    rgba = _rgba8(image[index])
                    last_input_rgba = rgba
                    use_depth = guidance in (0, 3)
                    use_motion = guidance in (0, 2)
                    depth_np = _depth_frame(depth, index, depth_inference_interval,
                                            height, width) if use_depth else None
                    motion_np = _motion_frame(motion_vectors, index, height, width) \
                        if use_motion else None
                    settings.reset = 1 if iteration == 0 else 0
                    last_rgba = bridge.process(rgba, settings, depth_np, motion_np)
                    last_comparison = _comparison_rgba(rgba, last_rgba)
                    iteration += 1
                    if not is_live:
                        processed_results.append(
                            torch.from_numpy(last_rgba[..., :3].copy()).float() / 255.0)
                        comparison_results.append(
                            torch.from_numpy(last_comparison[..., :3].copy()).float() / 255.0)
                    now = time.perf_counter()
                    if now - last_preview >= 1.0 / preview_fps:
                        _preview(node_id, last_comparison, iteration,
                                 iteration / max(now - started, 1e-6), bridge.runtime_hash)
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
                         iteration / max(elapsed, 1e-6), bridge.runtime_hash, "stopped")
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


class DLSSNRFarnebackMotion:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "strength": ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05})}}

    RETURN_TYPES = ("DLSSNR_MOTION",)
    RETURN_NAMES = ("motion_vectors",)
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live"

    def run(self, image, strength):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("计算光流需要 opencv-python；整合包通常已自带。") from exc
        frames = image.detach().float().cpu().clamp(0, 1).numpy()[..., :3]
        batch, height, width, _ = frames.shape
        output = np.zeros((batch, height, width, 2), dtype=np.float32)
        grays = [cv2.cvtColor((f * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                 for f in frames]
        for i in range(1, batch):
            # DLSSNR expects current-to-previous motion in source pixels.
            output[i] = cv2.calcOpticalFlowFarneback(
                grays[i], grays[i - 1], None, 0.5, 4, 19, 4, 7, 1.5, 0
            ) * float(strength)
        return (torch.from_numpy(output),)


NODE_CLASS_MAPPINGS = {
    "DLSSNRLive": DLSSNRLive,
    "DLSSNRFarnebackMotion": DLSSNRFarnebackMotion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DLSSNRLive": "DLSSNR 实时预览 / 手动停止",
    "DLSSNRFarnebackMotion": "DLSSNR 光流（Farneback）",
}
