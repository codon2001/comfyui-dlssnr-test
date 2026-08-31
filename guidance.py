from __future__ import annotations

import atexit
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
DEPTH_MODEL = ROOT / "models" / "depth_anything_v2" / "model_fp16.onnx"
DEPTH_MODEL_SHA256 = "2DF6223F206B5164E21F664ACE61DABEB9BB6A49B8B5A3E00510B4807D0F5B04"
_DEPTH_SESSIONS: dict[tuple[int, str], tuple[object, str]] = {}
_DEPTH_LOCK = threading.Lock()
_DML_WORKERS: dict[int, "_DmlDepthWorker"] = {}


def _read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size - len(chunks))
        if not part:
            raise RuntimeError("DirectML 深度服务意外退出")
        chunks.extend(part)
    return bytes(chunks)


class _DmlDepthWorker:
    def __init__(self, gpu_index: int):
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            (sys.executable, "-u", str(ROOT / "depth_worker.py"),
             str(DEPTH_MODEL), str(max(0, int(gpu_index)))),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, creationflags=creation_flags,
        )
        if self._process.stdout is None:
            raise RuntimeError("无法读取 DirectML 深度服务")
        marker = _read_exact(self._process.stdout, 8)
        if marker != b"DLSSDML1":
            detail = ""
            if marker == b"ERROR001":
                size = struct.unpack("<I", _read_exact(self._process.stdout, 4))[0]
                detail = _read_exact(self._process.stdout, size).decode("utf-8", "replace")
            self.close()
            raise RuntimeError(f"DirectML 深度服务初始化失败：{detail or marker!r}")

    def run(self, values: np.ndarray) -> np.ndarray:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("DirectML 深度服务管道不可用")
        source = np.ascontiguousarray(values, dtype=np.float32)
        height, width = source.shape[-2:]
        with self._lock:
            self._process.stdin.write(struct.pack("<II", height, width))
            self._process.stdin.write(source.tobytes())
            self._process.stdin.flush()
            count = struct.unpack("<i", _read_exact(self._process.stdout, 4))[0]
            if count < 0:
                message = _read_exact(self._process.stdout, -count).decode("utf-8", "replace")
                raise RuntimeError(message)
            raw = _read_exact(self._process.stdout, count * 4)
        return np.frombuffer(raw, dtype=np.float32).copy().reshape(1, height, width)

    def close(self):
        process = getattr(self, "_process", None)
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(struct.pack("<II", 0, 0))
                process.stdin.flush()
                process.stdin.close()
                process.wait(timeout=2)
        except Exception:
            process.kill()
        self._process = None


def _dml_worker(gpu_index: int) -> _DmlDepthWorker:
    index = max(0, int(gpu_index))
    with _DEPTH_LOCK:
        worker = _DML_WORKERS.get(index)
        if worker is None or worker._process is None or worker._process.poll() is not None:
            if worker is not None:
                worker.close()
            worker = _DmlDepthWorker(index)
            _DML_WORKERS[index] = worker
        return worker


@atexit.register
def _close_depth_workers():
    for worker in tuple(_DML_WORKERS.values()):
        worker.close()


def _rgb_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    source = np.ascontiguousarray(rgb[..., :3])
    value = torch.from_numpy(source).to(device=device, dtype=torch.float32)
    if np.issubdtype(source.dtype, np.integer):
        value = value / float(np.iinfo(source.dtype).max)
    return value.permute(2, 0, 1).unsqueeze(0).clamp_(0, 1)


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor((0.2126, 0.7152, 0.0722), device=rgb.device,
                           dtype=rgb.dtype).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def _horn_schunck(current: torch.Tensor, previous: torch.Tensor,
                  iterations: int, smoothness: float) -> torch.Tensor:
    """Estimate current-to-previous flow as source-pixel vectors."""
    kernel_x = torch.tensor(((-1, 1), (-1, 1)), device=current.device,
                            dtype=current.dtype).view(1, 1, 2, 2) * 0.25
    kernel_y = torch.tensor(((-1, -1), (1, 1)), device=current.device,
                            dtype=current.dtype).view(1, 1, 2, 2) * 0.25
    kernel_t = torch.ones((1, 1, 2, 2), device=current.device,
                          dtype=current.dtype) * 0.25
    pair = (current + previous) * 0.5
    padded_pair = F.pad(pair, (0, 1, 0, 1), mode="replicate")
    ix = F.conv2d(padded_pair, kernel_x)
    iy = F.conv2d(padded_pair, kernel_y)
    it = F.conv2d(F.pad(previous - current, (0, 1, 0, 1),
                        mode="replicate"), kernel_t)
    average = torch.tensor(((1 / 12, 1 / 6, 1 / 12),
                            (1 / 6, 0, 1 / 6),
                            (1 / 12, 1 / 6, 1 / 12)), device=current.device,
                           dtype=current.dtype).view(1, 1, 3, 3)
    u = torch.zeros_like(current)
    v = torch.zeros_like(current)
    denominator = smoothness * smoothness + ix * ix + iy * iy
    for _ in range(max(1, int(iterations))):
        u_avg = F.conv2d(u, average, padding=1)
        v_avg = F.conv2d(v, average, padding=1)
        correction = (ix * u_avg + iy * v_avg + it) / denominator.clamp_min(1e-6)
        u = u_avg - ix * correction
        v = v_avg - iy * correction
    return torch.cat((u, v), dim=1)


def _fast_gradient_flow(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Single-pass current-to-previous flow for low-latency guidance."""
    ix = F.pad(current[..., 2:] - current[..., :-2], (1, 1, 0, 0)) * 0.5
    iy = F.pad(current[..., 2:, :] - current[..., :-2, :], (0, 0, 1, 1)) * 0.5
    it = previous - current
    denominator = (ix * ix + iy * iy + 0.015).clamp_min(1e-6)
    u = F.avg_pool2d(-ix * it / denominator, 5, stride=1, padding=2)
    v = F.avg_pool2d(-iy * it / denominator, 5, stride=1, padding=2)
    return torch.cat((u, v), dim=1)


def _aligned_depth_size(height: int, width: int, long_side: int = 336) -> tuple[int, int]:
    scale = float(long_side) / max(height, width)
    aligned = lambda value: max(14, int(round(value * scale / 14.0)) * 14)
    return aligned(height), aligned(width)


def _depth_session(gpu_index: int, requested: str):
    if not DEPTH_MODEL.is_file():
        raise RuntimeError("Depth Anything V2 模型不存在")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("未安装 onnxruntime/onnxruntime-gpu") from exc
    available = set(ort.get_available_providers())
    requested_lower = str(requested).lower()
    providers = []
    provider_name = ""
    if "tensorrt" in requested_lower and "TensorrtExecutionProvider" in available:
        cache = ROOT / "models" / "depth_anything_v2" / "tensorrt_cache"
        cache.mkdir(parents=True, exist_ok=True)
        providers.append(("TensorrtExecutionProvider", {
            "device_id": int(gpu_index), "trt_fp16_enable": True,
            "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache),
        }))
        provider_name = "TensorRT"
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": int(gpu_index)}))
        provider_name = provider_name or "CUDA"
    if "DmlExecutionProvider" in available:
        providers.append(("DmlExecutionProvider", {"device_id": int(gpu_index)}))
        provider_name = provider_name or "DirectML"
    if not providers:
        raise RuntimeError("ONNX Runtime 没有 CUDA/TensorRT/DirectML GPU Provider")
    key = (int(gpu_index), provider_name)
    with _DEPTH_LOCK:
        cached = _DEPTH_SESSIONS.get(key)
        if cached is not None:
            return cached
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(DEPTH_MODEL), sess_options=options,
                                       providers=providers)
        result = (session, session.get_providers()[0])
        _DEPTH_SESSIONS[key] = result
        return result


def _depth_anything(current: torch.Tensor, gpu_index: int,
                    requested: str) -> tuple[torch.Tensor, str]:
    height, width = current.shape[-2:]
    inference_size = _aligned_depth_size(height, width)
    resized = F.interpolate(current, size=inference_size, mode="bilinear",
                            align_corners=False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=resized.device,
                        dtype=resized.dtype).view(1, 3, 1, 1)
    deviation = torch.tensor((0.229, 0.224, 0.225), device=resized.device,
                             dtype=resized.dtype).view(1, 3, 1, 1)
    model_input = ((resized - mean) / deviation).contiguous().float()
    provider = "DirectML"
    values = _dml_worker(gpu_index).run(model_input.detach().cpu().numpy())
    raw = torch.from_numpy(values).to(current.device).unsqueeze(1)
    flat = raw.flatten()
    p02 = torch.quantile(flat, 0.02)
    p98 = torch.quantile(flat, 0.98)
    normalized = ((raw - p02) / (p98 - p02).clamp_min(1e-6)).clamp(0, 1)
    normalized = F.interpolate(normalized, size=(height, width), mode="bilinear",
                               align_corners=False)
    return normalized, provider


def _lightweight_depth(current_small: torch.Tensor,
                       current_gray: torch.Tensor) -> torch.Tensor:
    local = F.avg_pool2d(current_gray, 15, stride=1, padding=7)
    contrast = (current_gray - local).abs()
    saturation = current_small.amax(1, keepdim=True) - current_small.amin(1, keepdim=True)
    depth = 0.62 * (1.0 - local) + 0.23 * contrast + 0.15 * saturation
    return depth / depth.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-5)


def depth_to_normal(depth: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """Reconstruct an RGB tangent-space normal map from relative depth."""
    source = np.ascontiguousarray(depth, dtype=np.float32)
    gradient_y, gradient_x = np.gradient(source)
    normal = np.stack((-gradient_x * float(strength),
                       -gradient_y * float(strength),
                       np.ones_like(source)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True).clip(1e-6)
    return np.ascontiguousarray(normal * 0.5 + 0.5, dtype=np.float32)


def motion_to_rgb(motion: np.ndarray) -> np.ndarray:
    """Create a standard RGB visualization without changing motion data."""
    source = np.ascontiguousarray(motion, dtype=np.float32)
    magnitude = np.linalg.norm(source, axis=-1)
    scale = max(float(np.quantile(magnitude, 0.95)), 1e-6)
    result = np.stack((0.5 + 0.5 * source[..., 0] / scale,
                       0.5 + 0.5 * source[..., 1] / scale,
                       magnitude / scale), axis=-1)
    return np.ascontiguousarray(np.clip(result, 0, 1), dtype=np.float32)


def guidance_edge_map(rgb: np.ndarray | None, depth: np.ndarray,
                      strength: float = 2.0) -> np.ndarray:
    """Combine relative-depth discontinuities and RGB edges into a mask."""
    dy, dx = np.gradient(np.ascontiguousarray(depth, dtype=np.float32))
    edge = np.hypot(dx, dy)
    if rgb is not None:
        color = np.ascontiguousarray(rgb[..., :3], dtype=np.float32)
        if color.max(initial=0) > 1.5:
            color /= 255.0
        gray = color[..., 0] * 0.2126 + color[..., 1] * 0.7152 + color[..., 2] * 0.0722
        cy, cx = np.gradient(gray)
        edge += np.hypot(cx, cy) * 0.5
    reference = max(float(np.quantile(edge, 0.98)), 1e-6)
    normalized = edge / reference
    outlined = np.clip((normalized - 0.10) * float(strength), 0, 1)
    return np.ascontiguousarray(outlined, dtype=np.float32)


@torch.inference_mode()
def estimate_color_guidance(previous_rgb: np.ndarray | None,
                            current_rgb: np.ndarray,
                            motion_strength: float = 1.0,
                            depth_strength: float = 1.0,
                            iterations: int = 12,
                            analysis_max_side: int = 960,
                            gpu_index: int = 0,
                            depth_mode: str = "lightweight",
                            flow_mode: str = "quality"):
    """Estimate relative depth and current-to-previous motion from RGB media."""
    device = torch.device(
        f"cuda:{max(0, int(gpu_index))}" if torch.cuda.is_available() else "cpu")
    current = _rgb_tensor(current_rgb, device)
    height, width = current_rgb.shape[:2]
    scale = min(1.0, float(max(64, analysis_max_side)) / max(height, width))
    analysis_size = (max(16, int(round(height * scale))),
                     max(16, int(round(width * scale))))
    current_small = F.interpolate(current, size=analysis_size, mode="bilinear",
                                  align_corners=False)
    current_gray = _gray(current_small)
    status_parts = [device.type]

    if str(depth_mode).lower().startswith("off") or str(depth_mode).startswith("关闭"):
        depth = torch.zeros((1, 1, height, width), device=device)
        status_parts.append("depth=off")
    elif "depth anything" in str(depth_mode).lower():
        try:
            depth, provider = _depth_anything(current, int(gpu_index), depth_mode)
            status_parts.append(f"depth={provider}/DAV2")
        except Exception as exc:
            depth = _lightweight_depth(current_small, current_gray)
            depth = F.interpolate(depth, size=(height, width), mode="bilinear",
                                  align_corners=False)
            status_parts.append(f"depth=lightweight-fallback({exc})")
    else:
        depth = _lightweight_depth(current_small, current_gray)
        depth = F.interpolate(depth, size=(height, width), mode="bilinear",
                              align_corners=False)
        status_parts.append("depth=lightweight")
    depth = (depth * float(depth_strength)).clamp(0, 1)

    flow_disabled = str(flow_mode).lower().startswith("off") or \
        str(flow_mode).startswith("关闭")
    if previous_rgb is None or flow_disabled:
        flow = torch.zeros((1, 2, *analysis_size), device=device)
        reactive = torch.zeros_like(current_gray)
        exposure = 1.0
        status_parts.append("motion=zero(no previous)" if previous_rgb is None
                            else "motion=off")
    else:
        previous = _rgb_tensor(previous_rgb, device)
        previous_small = F.interpolate(previous, size=analysis_size, mode="bilinear",
                                       align_corners=False)
        previous_gray = _gray(previous_small)
        if str(flow_mode).lower().startswith("fast") or str(flow_mode).startswith("高速"):
            flow = _fast_gradient_flow(current_gray, previous_gray)
            status_parts.append("motion=fast-gradient")
        else:
            flow = _horn_schunck(current_gray, previous_gray, iterations, 0.12)
            status_parts.append(f"motion=iterative-{int(iterations)}")
        flow *= float(motion_strength)
        reactive = ((current_small - previous_small).abs().amax(1, keepdim=True) * 4.0)
        reactive = reactive.clamp(0, 1)
        exposure = float((previous_gray.mean() / current_gray.mean().clamp_min(1e-5))
                         .clamp(0.25, 4.0).item())

    reactive = F.interpolate(reactive, size=(height, width), mode="bilinear",
                             align_corners=False)
    flow = F.interpolate(flow, size=(height, width), mode="bilinear",
                         align_corners=False)
    flow[:, 0] *= width / analysis_size[1]
    flow[:, 1] *= height / analysis_size[0]

    depth_np = depth[0, 0].float().cpu().numpy().astype(np.float32, copy=False)
    motion_np = flow[0].permute(1, 2, 0).float().cpu().numpy().astype(np.float32, copy=False)
    reactive_np = reactive[0, 0].float().cpu().numpy().astype(np.float32, copy=False)
    return depth_np, motion_np, reactive_np, exposure, ";".join(status_parts)
