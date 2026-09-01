from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageSequence

from .guidance import (depth_to_normal, estimate_color_guidance,
                       guidance_edge_map, motion_to_rgb)
from .nodes import (
    Bridge,
    Settings,
    _apply_depth_assist,
    _apply_depth_assist_weight,
    _comparison_rgba,
    _depth_assist_weight,
    _depth_frame,
    _event_for,
    _motion_frame,
    _preview,
    _process_dlssnr,
    _remove_event,
    _restore_source_color,
    _rgba8,
    _runtime_path,
    _sha256,
    gpu_index,
    gpu_input,
    runtime_input,
)


def _style(value) -> int:
    return int(str(value).split()[0])


def _guidance(value) -> int:
    return int(str(value).split()[0])


def _exact_video_rate(source: Path, fallback_fps: float) -> tuple[int, int]:
    """Return the source average frame rate as an exact rational when possible."""
    try:
        import av
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate
            if rate and rate.numerator > 0 and rate.denominator > 0:
                return int(rate.numerator), int(rate.denominator)
    except (ImportError, OSError, ValueError, IndexError):
        pass
    value = fallback_fps if fallback_fps > 0.01 else 30.0
    rate = Fraction(str(value)).limit_denominator(1_000_000)
    return max(1, int(rate.numerator)), max(1, int(rate.denominator))


def _settings(style, intensity, tone, structure, automatic_mask,
              skin_structure=1.0, ui_correction=False, depth_inverted=1,
              motion_scale_x=1.0, motion_scale_y=1.0,
              paper_white_scale=1.0) -> Settings:
    return Settings(_style(style), intensity, tone, structure, int(automatic_mask), 0,
                    skin_structure, int(ui_correction), int(depth_inverted),
                    motion_scale_x, motion_scale_y, paper_white_scale)


def _tensor_rgb(rgb: np.ndarray, device=None) -> torch.Tensor:
    result = torch.from_numpy(np.array(rgb[..., :3], dtype=np.uint8,
                                      order="C", copy=True)).float() / 255.0
    return result.to(device) if device is not None else result


@torch.inference_mode()
def _generate_between(previous_rgb: np.ndarray, current_rgb: np.ndarray,
                      current_to_previous: np.ndarray | None,
                      multiplier: int, selected_gpu: int) -> list[np.ndarray]:
    """Generate intermediate RGB8 frames by bidirectional GPU warping."""
    factor = max(1, min(4, int(multiplier)))
    if factor <= 1:
        return []
    device = torch.device(
        f"cuda:{max(0, int(selected_gpu))}" if torch.cuda.is_available() else "cpu")
    previous = _tensor_rgb(previous_rgb, device).permute(2, 0, 1).unsqueeze(0)
    current = _tensor_rgb(current_rgb, device).permute(2, 0, 1).unsqueeze(0)
    height, width = previous_rgb.shape[:2]
    if current_to_previous is None:
        flow = torch.zeros((1, height, width, 2), device=device)
    else:
        motion = np.ascontiguousarray(current_to_previous, dtype=np.float32)
        if motion.shape[:2] != (height, width):
            tensor = torch.from_numpy(motion).to(device).permute(2, 0, 1).unsqueeze(0)
            tensor = torch.nn.functional.interpolate(
                tensor, size=(height, width), mode="bilinear", align_corners=False)
            tensor[:, 0] *= width / max(1, motion.shape[1])
            tensor[:, 1] *= height / max(1, motion.shape[0])
            flow = tensor.permute(0, 2, 3, 1)
        else:
            flow = torch.from_numpy(motion).to(device).unsqueeze(0)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0)

    def sample(frame: torch.Tensor, grid_pixels: torch.Tensor) -> torch.Tensor:
        grid = grid_pixels.clone()
        grid[..., 0] = grid[..., 0] * (2.0 / max(1, width - 1)) - 1.0
        grid[..., 1] = grid[..., 1] * (2.0 / max(1, height - 1)) - 1.0
        return F.grid_sample(frame, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)

    generated = []
    for step in range(1, factor):
        t = float(step) / factor
        previous_warped = sample(previous, base + flow * t)
        current_warped = sample(current, base - flow * (1.0 - t))
        frame = previous_warped.lerp(current_warped, t)
        result = (frame[0].permute(1, 2, 0).clamp(0, 1) * 255.0 + 0.5)
        generated.append(result.byte().cpu().numpy())
    return generated


def _check_interrupt() -> None:
    try:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _output_directory() -> Path:
    try:
        import folder_paths
        result = Path(folder_paths.get_output_directory()) / "dlssnr"
    except Exception:
        result = Path.cwd() / "output" / "dlssnr"
    result.mkdir(parents=True, exist_ok=True)
    return result


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{stamp}_{counter}{path.suffix}")
        counter += 1
    return candidate


class DLSSNRColorGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gpu_device": gpu_input(),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "depth_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "flow_iterations": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "analysis_max_side": ("INT", {"default": 960, "min": 128, "max": 4096, "step": 64}),
                "depth_estimator": ([
                    "Depth Anything V2 DirectML GPU（推荐）",
                    "轻量颜色深度（快速）",
                    "关闭深度估算",
                ],),
                "motion_estimator": ([
                    "精细 GPU 迭代光流（推荐）",
                    "高速 GPU 梯度光流",
                    "关闭运动估算",
                ],),
                "enable_depth_estimation": ("BOOLEAN", {"default": True}),
                "normal_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.1}),
                "edge_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.1}),
            },
            "optional": {"previous_image": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "DLSSNR_MOTION", "MASK", "FLOAT", "STRING",
                    "IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("estimated_depth", "motion_vectors", "reactive_mask",
                    "exposure_ratio", "status", "normal_map",
                    "motion_preview", "edge_image", "edge_mask")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/Guidance"

    def run(self, image, gpu_device, motion_strength, depth_strength, flow_iterations,
            analysis_max_side, depth_estimator="Depth Anything V2 DirectML GPU（推荐）",
            motion_estimator="精细 GPU 迭代光流（推荐）",
            enable_depth_estimation=True, normal_strength=2.0,
            edge_strength=2.0, previous_image=None):
        frames = image.detach().float().cpu().clamp(0, 1).numpy()[..., :3]
        depths = []
        motions = []
        masks = []
        exposures = []
        normals = []
        motion_previews = []
        edge_images = []
        edge_masks = []
        previous = None
        previous_frames = (None if previous_image is None else
                           previous_image.detach().float().cpu().clamp(0, 1).numpy()[..., :3])
        device_name = "cpu"
        selected_gpu = gpu_index(gpu_device)
        for index, frame in enumerate(frames):
            current = np.ascontiguousarray(frame * 255.0 + 0.5, dtype=np.uint8)
            if previous_frames is not None:
                source = previous_frames[min(index, len(previous_frames) - 1)]
                previous = np.ascontiguousarray(source * 255.0 + 0.5, dtype=np.uint8)
            depth, motion, mask, exposure, device_name = estimate_color_guidance(
                previous, current, motion_strength, depth_strength,
                flow_iterations, analysis_max_side, selected_gpu,
                depth_estimator if enable_depth_estimation else "关闭深度估算",
                motion_estimator)
            depths.append(torch.from_numpy(np.repeat(depth[..., None], 3, axis=-1).copy()))
            motions.append(torch.from_numpy(motion.copy()))
            masks.append(torch.from_numpy(mask.copy()))
            exposures.append(exposure)
            normals.append(torch.from_numpy(depth_to_normal(depth, normal_strength)))
            motion_previews.append(torch.from_numpy(motion_to_rgb(motion)))
            edge = guidance_edge_map(current, depth, edge_strength)
            edge_masks.append(torch.from_numpy(edge))
            edge_images.append(torch.from_numpy(np.repeat(edge[..., None], 3, axis=-1).copy()))
            previous = current
        status = (f"RGB估算完成：{len(frames)} 帧，计算设备={device_name.upper()}。"
                  "深度/光流是颜色推断，不是游戏引擎原生数据。")
        return (torch.stack(depths), torch.stack(motions), torch.stack(masks),
                float(np.mean(exposures)), status, torch.stack(normals),
                torch.stack(motion_previews), torch.stack(edge_images),
                torch.stack(edge_masks))


class DLSSNRDepthNormalEdge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE",),
                "normal_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.1}),
                "edge_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.1}),
            },
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("normal_map", "edge_image", "edge_mask")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/Guidance"

    def run(self, depth, normal_strength, edge_strength, image=None):
        values = depth.detach().float().cpu().clamp(0, 1).numpy()
        colors = None if image is None else image.detach().float().cpu().clamp(0, 1).numpy()
        normals, edges, masks = [], [], []
        for index, value in enumerate(values):
            plane = value[..., :3].mean(axis=-1) if value.ndim == 3 else value
            color = None if colors is None else colors[min(index, len(colors) - 1)]
            edge = guidance_edge_map(color, plane, edge_strength)
            normals.append(torch.from_numpy(depth_to_normal(plane, normal_strength)))
            edges.append(torch.from_numpy(np.repeat(edge[..., None], 3, axis=-1).copy()))
            masks.append(torch.from_numpy(edge))
        return torch.stack(normals), torch.stack(edges), torch.stack(masks)


class DLSSNRImageDirect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dll_path": runtime_input(),
                "gpu_device": gpu_input(),
                "nr_preset": (["0 Default", "1 Preset #1", "2 Preset #2", "3 Preset #3"],),
                "automatic_mask": ("BOOLEAN", {"default": False}),
                "nr_style": (["0 默认（Default）", "1 自然（Natural）", "2 电影感（Cinematic）"],),
                "nr_intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_tone_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "skin_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "scene_paper_white_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "display": "slider"}),
                "ui_correction": ("BOOLEAN", {"default": False}),
                "color_fix": ("BOOLEAN", {"default": False, "tooltip": "打开后才执行色彩修复。"}),
                "frame_guidance": (["0 使用可用引导（深度和运动）", "1 强制零引导", "2 仅运动向量", "3 仅深度", "1 反向深度（Inverted）", "0 正常深度（Normal）"],),
                "depth_convention": ("STRING", {"default": "1 反向深度（Inverted）"}),
                "motion_scale_x": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "motion_scale_y": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "estimate_missing_from_color": ("BOOLEAN", {"default": False}),
                "depth_estimator": ([
                    "Depth Anything V2 DirectML GPU（推荐）",
                    "轻量颜色深度（快速）",
                    "关闭深度估算",
                ],),
                "motion_estimator": ([
                    "精细 GPU 迭代光流（推荐）",
                    "高速 GPU 梯度光流",
                    "关闭运动估算",
                ],),
                "enable_depth_estimation": ("BOOLEAN", {"default": True}),
                "guidance_warmup_iterations": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "depth_assist_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "display": "slider"}),
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

    def run(self, image, dll_path, gpu_device, nr_preset,
            automatic_mask, nr_style, nr_intensity,
            local_tone_strength, local_structure_strength,
            skin_structure_strength, scene_paper_white_scale, ui_correction, color_fix, frame_guidance,
            depth_convention, motion_scale_x, motion_scale_y,
            estimate_missing_from_color,
            depth_estimator="Depth Anything V2 DirectML GPU（推荐）",
            motion_estimator="精细 GPU 迭代光流（推荐）",
            enable_depth_estimation=True,
            guidance_warmup_iterations=2,
            depth_assist_strength=1.0,
            unique_id=None, depth=None,
            motion_vectors=None):
        batch, height, width, _ = image.shape
        guidance = _guidance(frame_guidance)
        settings = _settings(
            nr_style, nr_intensity, local_tone_strength,
            local_structure_strength, automatic_mask, skin_structure_strength,
            ui_correction, int(str(depth_convention).split()[0]),
            motion_scale_x, motion_scale_y, scene_paper_white_scale)
        processed_results = []
        comparison_results = []
        previous = None
        last_comparison = None
        estimator_device = "unused"
        selected_gpu = gpu_index(gpu_device)
        started = time.perf_counter()
        with Bridge(dll_path, width, height, selected_gpu,
                    int(str(nr_preset).split()[0])) as bridge:
            for index in range(batch):
                rgba = _rgba8(image[index])
                want_depth = guidance in (0, 3)
                want_motion = guidance in (0, 2)
                depth_np = _depth_frame(depth, index, 1, height, width) if want_depth else None
                motion_np = (_motion_frame(motion_vectors, index, height, width)
                             if want_motion else None)
                if estimate_missing_from_color and ((want_depth and depth_np is None) or
                                                    (want_motion and motion_np is None)):
                    estimated_depth, estimated_motion, _mask, _exposure, estimator_device = \
                        estimate_color_guidance(
                            previous, rgba[..., :3], gpu_index=selected_gpu,
                            depth_mode=(depth_estimator if enable_depth_estimation
                                        else "关闭深度估算"),
                            flow_mode=motion_estimator)
                    if want_depth and depth_np is None:
                        depth_np = estimated_depth
                    if want_motion and motion_np is None:
                        motion_np = estimated_motion
                # 单张图片也保持同一个 DLSSNR 会话连续评估：第一次重置，
                # 后续调用保留历史并继续传入相同的深度/运动引导。
                passes = (max(1, int(guidance_warmup_iterations))
                          if batch == 1 else 1)
                processed = None
                for pass_index in range(passes):
                    # The image node treats a ComfyUI batch as independent
                    # pictures.  Animation history belongs to the GIF node;
                    # carrying it between unrelated images causes ghosting.
                    settings.reset = 1 if pass_index == 0 else 0
                    raw_processed = bridge.process(
                        rgba, settings, depth_np, motion_np)
                    # Intermediate warm-up results are never returned or fed
                    # back into NGX. Apply the expensive host-side depth/color
                    # correction only once, on the final visible result.
                    if pass_index == passes - 1:
                        processed = _apply_depth_assist(
                            rgba, raw_processed, depth_np, depth_assist_strength)
                        if color_fix:
                            processed = _restore_source_color(rgba, processed)
                    _check_interrupt()
                comparison = _comparison_rgba(rgba, processed)
                last_comparison = comparison
                processed_results.append(_tensor_rgb(processed))
                comparison_results.append(_tensor_rgb(comparison))
                previous = rgba[..., :3].copy()
                _check_interrupt()
            runtime_hash = bridge.runtime_hash
        elapsed = time.perf_counter() - started
        if last_comparison is not None:
            _preview(str(unique_id), last_comparison, batch,
                     batch / max(elapsed, 1e-6), runtime_hash, "stopped")
        status = (f"完成：{batch} 张，{elapsed:.2f}s，"
                  f"单图引导预热={guidance_warmup_iterations if batch == 1 else 1} 次，"
                  f"节点侧深度辅助={depth_assist_strength:.2f}，"
                  f"颜色估算设备={estimator_device}，"
                  f"DLL SHA256={runtime_hash}")
        return (torch.stack(processed_results).to(image.device),
                torch.stack(comparison_results).to(image.device), status)


class DLSSNRImageDirectLegacy:
    """Original TEST3 single-pass image path kept for maximum speed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dll_path": runtime_input(),
                "gpu_device": gpu_input(),
                "automatic_mask": ("BOOLEAN", {"default": False}),
                "nr_style": (["0 默认（Default）", "1 自然（Natural）", "2 电影感（Cinematic）"],),
                "nr_intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_tone_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "skin_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "scene_paper_white_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "display": "slider"}),
                "ui_correction": ("BOOLEAN", {"default": False}),
                "color_fix": ("BOOLEAN", {"default": False, "tooltip": "打开后才执行色彩修复。"}),
                "frame_guidance": (["0 使用可用引导（深度和运动）", "1 强制零引导", "2 仅运动向量", "3 仅深度", "1 反向深度（Inverted）", "0 正常深度（Normal）"],),
                "depth_convention": ("STRING", {"default": "1 反向深度（Inverted）"}),
                "motion_scale_x": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "motion_scale_y": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "estimate_missing_from_color": ("BOOLEAN", {"default": False}),
                "depth_estimator": ([
                    "Depth Anything V2 DirectML GPU（推荐）",
                    "轻量颜色深度（快速）",
                    "关闭深度估算",
                ],),
                "motion_estimator": ([
                    "精细 GPU 迭代光流（推荐）",
                    "高速 GPU 梯度光流",
                    "关闭运动估算",
                ],),
                "enable_depth_estimation": ("BOOLEAN", {"default": True}),
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

    def run(self, image, dll_path, gpu_device, automatic_mask, nr_style, nr_intensity,
            local_tone_strength, local_structure_strength,
            skin_structure_strength, scene_paper_white_scale, ui_correction, color_fix, frame_guidance,
            depth_convention, motion_scale_x, motion_scale_y,
            estimate_missing_from_color,
            depth_estimator="Depth Anything V2 DirectML GPU（推荐）",
            motion_estimator="精细 GPU 迭代光流（推荐）",
            enable_depth_estimation=True,
            unique_id=None, depth=None,
            motion_vectors=None):
        batch, height, width, _ = image.shape
        guidance = _guidance(frame_guidance)
        settings = _settings(
            nr_style, nr_intensity, local_tone_strength,
            local_structure_strength, automatic_mask, skin_structure_strength,
            ui_correction, int(str(depth_convention).split()[0]),
            motion_scale_x, motion_scale_y, scene_paper_white_scale)
        processed_results = []
        comparison_results = []
        previous = None
        last_comparison = None
        estimator_device = "unused"
        selected_gpu = gpu_index(gpu_device)
        started = time.perf_counter()
        with Bridge(dll_path, width, height, selected_gpu) as bridge:
            for index in range(batch):
                rgba = _rgba8(image[index])
                want_depth = guidance in (0, 3)
                want_motion = guidance in (0, 2)
                depth_np = _depth_frame(depth, index, 1, height, width) if want_depth else None
                motion_np = (_motion_frame(motion_vectors, index, height, width)
                             if want_motion else None)
                if estimate_missing_from_color and ((want_depth and depth_np is None) or
                                                    (want_motion and motion_np is None)):
                    estimated_depth, estimated_motion, _mask, _exposure, estimator_device = \
                        estimate_color_guidance(
                            previous, rgba[..., :3], gpu_index=selected_gpu,
                            depth_mode=(depth_estimator if enable_depth_estimation
                                        else "关闭深度估算"),
                            flow_mode=motion_estimator)
                    if want_depth and depth_np is None:
                        depth_np = estimated_depth
                    if want_motion and motion_np is None:
                        motion_np = estimated_motion
                settings.reset = 1
                processed = bridge.process(rgba, settings, depth_np, motion_np)
                if color_fix:
                    processed = _restore_source_color(rgba, processed)
                comparison = _comparison_rgba(rgba, processed)
                last_comparison = comparison
                processed_results.append(_tensor_rgb(processed))
                comparison_results.append(_tensor_rgb(comparison))
                previous = rgba[..., :3].copy()
                _check_interrupt()
            runtime_hash = bridge.runtime_hash
        elapsed = time.perf_counter() - started
        if last_comparison is not None:
            _preview(str(unique_id), last_comparison, batch,
                     batch / max(elapsed, 1e-6), runtime_hash, "stopped")
        status = (f"TEST3 旧版完成：{batch} 张，{elapsed:.2f}s，"
                  f"颜色估算设备={estimator_device}，DLL SHA256={runtime_hash}")
        return (torch.stack(processed_results).to(image.device),
                torch.stack(comparison_results).to(image.device), status)


class DLSSNRProcessGIF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dll_path": runtime_input(),
                "gpu_device": gpu_input(),
                "nr_preset": (["0 Default", "1 Preset #1", "2 Preset #2", "3 Preset #3"],),
                "automatic_mask": ("BOOLEAN", {"default": False}),
                "nr_style": (["0 默认（Default）", "1 自然（Natural）", "2 电影感（Cinematic）"],),
                "nr_intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_tone_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "skin_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "scene_paper_white_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "display": "slider"}),
                "ui_correction": ("BOOLEAN", {"default": False}),
                "color_fix": ("BOOLEAN", {"default": False, "tooltip": "打开后才执行色彩修复。"}),
                "frame_guidance": (["0 使用可用引导（深度和运动）", "1 强制零引导", "2 仅运动向量", "3 仅深度", "1 反向深度（Inverted）", "0 正常深度（Normal）"],),
                "depth_convention": ("STRING", {"default": "1 反向深度（Inverted）"}),
                "motion_scale_x": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "motion_scale_y": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "depth_inference_interval": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "estimate_missing_from_color": ("BOOLEAN", {"default": True}),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "preview_fps": ("INT", {"default": 8, "min": 1, "max": 30, "step": 1}),
                "flow_iterations": ("INT", {"default": 6, "min": 1, "max": 24, "step": 1}),
                "analysis_max_side": ("INT", {"default": 640, "min": 256, "max": 2048, "step": 64}),
                "depth_estimator": ([
                    "Depth Anything V2 DirectML GPU（推荐）",
                    "轻量颜色深度（快速）",
                    "关闭深度估算",
                ],),
                "motion_estimator": ([
                    "精细 GPU 迭代光流（推荐）",
                    "高速 GPU 梯度光流",
                    "关闭运动估算",
                ],),
                "enable_depth_estimation": ("BOOLEAN", {"default": True}),
                "depth_assist_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "enable_frame_generation": ("BOOLEAN", {"default": False}),
                "frame_generation_multiplier": ("INT", {"default": 2, "min": 2, "max": 4, "step": 1}),
                "source_duration_ms": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 1}),
            },
            "optional": {
                "depth": ("IMAGE",),
                "motion_vectors": ("DLSSNR_MOTION",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("processed_image", "comparison_image", "status", "duration_ms_out")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/GIF"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def run(self, image, dll_path, gpu_device, nr_preset,
            automatic_mask, nr_style, nr_intensity,
            local_tone_strength, local_structure_strength,
            skin_structure_strength, scene_paper_white_scale, ui_correction, color_fix, frame_guidance,
            depth_convention, motion_scale_x, motion_scale_y,
            depth_inference_interval, estimate_missing_from_color,
            motion_strength, preview_fps, flow_iterations=6, analysis_max_side=640,
            depth_estimator="Depth Anything V2 DirectML GPU（推荐）",
            motion_estimator="精细 GPU 迭代光流（推荐）",
            enable_depth_estimation=True,
            depth_assist_strength=1.0,
            enable_frame_generation=False, frame_generation_multiplier=2,
            source_duration_ms=100,
            unique_id=None, depth=None,
            motion_vectors=None):
        node_id = str(unique_id)
        batch, height, width, _ = image.shape
        guidance = _guidance(frame_guidance)
        settings = _settings(
            nr_style, nr_intensity, local_tone_strength,
            local_structure_strength, automatic_mask, skin_structure_strength,
            ui_correction, int(str(depth_convention).split()[0]),
            motion_scale_x, motion_scale_y, scene_paper_white_scale)
        processed_results = []
        comparison_results = []
        previous = None
        last_comparison = None
        started = time.perf_counter()
        last_preview = 0.0
        estimator_device = "unused"
        estimated_depth_cache = None
        depth_weight_cache = None
        depth_weight_key = None
        previous_processed = None
        previous_comparison = None
        selected_gpu = gpu_index(gpu_device)
        with Bridge(dll_path, width, height, selected_gpu,
                    int(str(nr_preset).split()[0])) as bridge:
            for index in range(batch):
                rgba = _rgba8(image[index])
                want_depth = guidance in (0, 3)
                want_motion = guidance in (0, 2)
                depth_np = (_depth_frame(depth, index, depth_inference_interval,
                                         height, width) if want_depth else None)
                motion_np = (_motion_frame(motion_vectors, index, height, width)
                             if want_motion else None)
                generation_motion = motion_np
                if ((estimate_missing_from_color and
                     ((want_depth and depth_np is None) or
                      (want_motion and motion_np is None))) or
                    (enable_frame_generation and generation_motion is None)):
                    infer_depth_now = (estimate_missing_from_color and
                                       want_depth and depth_np is None and
                                       (estimated_depth_cache is None or
                                        index % max(1, int(depth_inference_interval)) == 0))
                    frame_depth_mode = (depth_estimator if infer_depth_now and
                                        enable_depth_estimation else "关闭深度估算")
                    estimated_depth, estimated_motion, _mask, _exposure, estimator_device = \
                        estimate_color_guidance(previous, rgba[..., :3], motion_strength,
                                                1.0, flow_iterations,
                                                analysis_max_side, selected_gpu,
                                                frame_depth_mode, motion_estimator)
                    if infer_depth_now:
                        estimated_depth_cache = estimated_depth
                    if want_depth and depth_np is None:
                        depth_np = estimated_depth_cache
                    if estimate_missing_from_color and want_motion and motion_np is None:
                        motion_np = estimated_motion
                    if enable_frame_generation and generation_motion is None:
                        generation_motion = estimated_motion
                settings.reset = 1 if index == 0 else 0
                if depth is not None:
                    source_index = min(
                        (index // max(1, int(depth_inference_interval))) *
                        max(1, int(depth_inference_interval)), depth.shape[0] - 1)
                    current_depth_key = ("input", int(source_index),
                                         float(depth_assist_strength))
                else:
                    current_depth_key = ("estimated", id(depth_np),
                                         float(depth_assist_strength))
                if current_depth_key != depth_weight_key:
                    depth_weight_cache = _depth_assist_weight(
                        depth_np, depth_assist_strength)
                    depth_weight_key = current_depth_key
                processed = bridge.process(rgba, settings, depth_np, motion_np)
                processed = _apply_depth_assist_weight(
                    rgba, processed, depth_weight_cache)
                if color_fix:
                    processed = _restore_source_color(rgba, processed)
                last_comparison = _comparison_rgba(rgba, processed)
                if enable_frame_generation and previous_processed is not None:
                    processed_generated = _generate_between(
                        previous_processed, processed[..., :3], generation_motion,
                        frame_generation_multiplier, selected_gpu)
                    comparison_motion = (None if generation_motion is None else
                                         np.concatenate((generation_motion,
                                                         generation_motion), axis=1))
                    comparison_generated = _generate_between(
                        previous_comparison, last_comparison[..., :3],
                        comparison_motion, frame_generation_multiplier, selected_gpu)
                    processed_results.extend(_tensor_rgb(frame) for frame in processed_generated)
                    comparison_results.extend(_tensor_rgb(frame) for frame in comparison_generated)
                processed_results.append(_tensor_rgb(processed))
                comparison_results.append(_tensor_rgb(last_comparison))
                previous_processed = processed[..., :3].copy()
                previous_comparison = last_comparison[..., :3].copy()
                previous = rgba[..., :3].copy()
                now = time.perf_counter()
                if now - last_preview >= 1.0 / preview_fps:
                    _preview(node_id, last_comparison, index + 1,
                             (index + 1) / max(now - started, 1e-6), bridge.runtime_hash)
                    last_preview = now
                _check_interrupt()
            elapsed = time.perf_counter() - started
            if last_comparison is not None:
                _preview(node_id, last_comparison, batch,
                         batch / max(elapsed, 1e-6), bridge.runtime_hash, "stopped")
            status = (f"完成：输入 {batch} 帧，输出 {len(processed_results)} 帧，"
                      f"{elapsed:.2f}s，颜色估算设备={estimator_device}，"
                      f"DLL SHA256={bridge.runtime_hash}")
        duration_out = max(1, int(round(
            batch * int(source_duration_ms) / max(1, len(processed_results)))))
        return (torch.stack(processed_results).to(image.device),
                torch.stack(comparison_results).to(image.device), status, duration_out)


class DLSSNRLoadGIF:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"gif_path": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("image", "mask", "duration_ms")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/GIF"

    def run(self, gif_path):
        source = Path(gif_path.strip().strip('"')).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".gif":
            raise RuntimeError(f"GIF 文件不存在：{source}")
        frames = []
        masks = []
        durations = []
        with Image.open(source) as gif:
            size = gif.size
            for frame in ImageSequence.Iterator(gif):
                rgba = frame.convert("RGBA")
                if rgba.size != size:
                    rgba = rgba.resize(size, Image.Resampling.LANCZOS)
                array = np.asarray(rgba, dtype=np.uint8)
                frames.append(torch.from_numpy(array[..., :3].copy()).float() / 255.0)
                masks.append(torch.from_numpy(array[..., 3].copy()).float() / 255.0)
                durations.append(int(frame.info.get("duration", 100) or 100))
        if not frames:
            raise RuntimeError("GIF 中没有可读取的帧。")
        return (torch.stack(frames), torch.stack(masks),
                max(10, int(round(float(np.mean(durations))))))


class DLSSNRSaveGIF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "DLSSNR_GIF", "multiline": False}),
                "duration_ms": ("INT", {"default": 100, "min": 10, "max": 10000, "step": 10}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 65535, "step": 1}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "status")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/GIF"
    OUTPUT_NODE = True

    def run(self, images, filename_prefix, duration_ms, loop_count, mask=None):
        frames = images.detach().float().cpu().clamp(0, 1).numpy()[..., :3]
        alpha = None if mask is None else mask.detach().float().cpu().clamp(0, 1).numpy()
        if alpha is not None and alpha.ndim == 4:
            alpha = alpha[..., 0]
        pil_frames = []
        for index, frame in enumerate(frames):
            rgb = np.ascontiguousarray(frame * 255.0 + 0.5, dtype=np.uint8)
            channel = (np.ascontiguousarray(alpha[min(index, len(alpha) - 1)] * 255.0 + 0.5,
                                             dtype=np.uint8) if alpha is not None else
                       np.full(rgb.shape[:2], 255, dtype=np.uint8))
            pil_frames.append(Image.fromarray(np.dstack((rgb, channel)), "RGBA"))
        prefix = Path(filename_prefix.strip().replace("\\", "/") or "DLSSNR_GIF")
        directory = _output_directory() / (prefix.parent if str(prefix.parent) != "." else "")
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output = _unique_path(directory / f"{prefix.name}_{stamp}.gif")
        pil_frames[0].save(output, save_all=True, append_images=pil_frames[1:],
                           duration=int(duration_ms), loop=int(loop_count), disposal=2)
        return (str(output), f"完成：保存 {len(pil_frames)} 帧 GIF。")


def _find_ffmpeg() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _has_nvenc(ffmpeg: str) -> bool:
    try:
        result = subprocess.run((ffmpeg, "-hide_banner", "-encoders"),
                                capture_output=True, text=True, timeout=15)
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def _mux_audio(ffmpeg: str | None, source: Path, video_only: Path,
               output: Path, preserve_audio: bool,
               fps_num: int | None = None, fps_den: int | None = None) -> bool:
    rate_args = ()
    if fps_num and fps_den:
        rate_args = ("-r", f"{int(fps_num)}/{int(fps_den)}",
                     "-video_track_timescale", str(int(fps_num)))
    if preserve_audio and ffmpeg:
        for audio_args in (("-c:a", "copy"), ("-c:a", "aac", "-b:a", "192k")):
            command = (ffmpeg, "-y", "-loglevel", "error", "-i", str(video_only),
                       "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
                       "-c:v", "copy", *rate_args, *audio_args,
                       "-shortest", str(output))
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 and output.is_file():
                video_only.unlink(missing_ok=True)
                return True
    if ffmpeg and rate_args:
        command = (ffmpeg, "-y", "-loglevel", "error", "-i", str(video_only),
                   "-map", "0:v:0", "-c:v", "copy", *rate_args, str(output))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and output.is_file():
            video_only.unlink(missing_ok=True)
            return False
    os.replace(video_only, output)
    return False


class _StreamEncoder:
    def __init__(self, path: Path, width: int, height: int, fps: float,
                 encoder_mode: str, gpu: int = 0):
        self.path = path
        self.process = None
        self.writer = None
        self.name = "CPU mp4v"
        self.ffmpeg = _find_ffmpeg()
        want_nvenc = str(encoder_mode).startswith("NVIDIA")
        if want_nvenc:
            if not self.ffmpeg or not _has_nvenc(self.ffmpeg):
                raise RuntimeError("未检测到可用的 ffmpeg h264_nvenc 编码器。请选择 CPU mp4v。")
            command = (self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
                       "-r", f"{fps:.6f}", "-i", "pipe:0", "-an", "-c:v",
                       "h264_nvenc", "-gpu", str(max(0, int(gpu))),
                       "-preset", "p4", "-cq", "18", "-pix_fmt",
                       "yuv420p", str(path))
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.PIPE)
            self.name = "NVIDIA NVENC"
        else:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("CPU 视频编码需要 opencv-python。") from exc
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"无法创建视频输出：{path}")

    def write(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        if self.process:
            if self.process.stdin is None:
                raise RuntimeError("NVENC 输入管道不可用。")
            self.process.stdin.write(rgb.tobytes())
        else:
            import cv2
            self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            error = self.process.stderr.read().decode("utf-8", "replace") if self.process.stderr else ""
            code = self.process.wait()
            if code != 0:
                raise RuntimeError(f"NVENC 编码失败：{error[-1200:]}")
        elif self.writer:
            self.writer.release()


class _StreamDecoder:
    def __init__(self, source: Path, width: int, height: int,
                 decoder_mode: str, gpu: int = 0):
        self.width = int(width)
        self.height = int(height)
        self.frame_bytes = self.width * self.height * 3
        self.capture = None
        self.process = None
        self.name = "OpenCV（CPU解码）"
        if str(decoder_mode).startswith("NVIDIA"):
            ffmpeg = _find_ffmpeg()
            if not ffmpeg:
                raise RuntimeError("NVIDIA NVDEC 需要可用的 ffmpeg。")
            command = (ffmpeg, "-hide_banner", "-loglevel", "error",
                       "-hwaccel", "cuda", "-hwaccel_device", str(max(0, int(gpu))),
                       "-i", str(source), "-an", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "pipe:1")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, creationflags=flags)
            self.name = "NVIDIA NVDEC（GPU解码）"
        else:
            import cv2
            self.capture = cv2.VideoCapture(str(source))
            if not self.capture.isOpened():
                raise RuntimeError(f"无法打开视频：{source}")

    def read(self) -> np.ndarray | None:
        if self.process is not None:
            if self.process.stdout is None:
                raise RuntimeError("NVDEC 输出管道不可用。")
            data = bytearray()
            while len(data) < self.frame_bytes:
                chunk = self.process.stdout.read(self.frame_bytes - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            if not data:
                code = self.process.wait()
                if code != 0:
                    error = (self.process.stderr.read().decode("utf-8", "replace")
                             if self.process.stderr else "")
                    raise RuntimeError(f"NVDEC 解码失败：{error[-1200:]}")
                return None
            if len(data) != self.frame_bytes:
                raise RuntimeError("NVDEC 返回了不完整的视频帧。")
            return np.frombuffer(data, dtype=np.uint8).reshape(
                self.height, self.width, 3).copy()
        ok, bgr = self.capture.read()
        if not ok:
            return None
        import cv2
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.process is not None:
            try:
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.poll() is None:
                    self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()
            self.process = None




class DLSSNRStreamVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_path": ("STRING", {"default": "", "multiline": False}),
            "output_path": ("STRING", {"default": "", "multiline": False}),
            "dll_path": runtime_input(),
            "gpu_device": gpu_input(),
            "encoder": (["NVIDIA NVENC（GPU编码）", "CPU mp4v（CPU编码）"],),
            "processing_mode": (["最高速度处理", "实时预览（按源帧率）"],),
            "preserve_audio": ("BOOLEAN", {"default": True}),
            "automatic_mask": ("BOOLEAN", {"default": False}),
            "nr_style": (["0 默认（Default）", "1 自然（Natural）", "2 电影感（Cinematic）"],),
            "nr_intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
            "local_tone_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
            "local_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
            "skin_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
            "scene_paper_white_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05, "display": "slider"}),
            "ui_correction": ("BOOLEAN", {"default": False}),
            "color_fix": ("BOOLEAN", {"default": False, "tooltip": "打开后才执行色彩修复；关闭时保留 DLSSNR 原始色调。"}),
            "frame_guidance": (["1 零引导极速（视频推荐）", "0 从RGB估算深度和运动", "2 仅从RGB估算运动", "3 仅从RGB估算深度"],),
            "depth_convention": ("STRING", {"default": "1 反向深度（Inverted）"}),
            "motion_scale_x": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            "motion_scale_y": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
            "depth_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "flow_iterations": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
            "analysis_max_side": ("INT", {"default": 960, "min": 128, "max": 4096, "step": 64}),
            "preview_fps": ("INT", {"default": 8, "min": 1, "max": 30, "step": 1}),
            "depth_inference_interval": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
            "depth_estimator": ([
                "Depth Anything V2 DirectML GPU（推荐）",
                "轻量颜色深度（快速）",
                "关闭深度估算",
            ],),
            "motion_estimator": ([
                "高速 GPU 梯度光流（视频推荐）",
                "精细 GPU 迭代光流",
                "关闭运动估算",
            ],),
            "enable_depth_estimation": ("BOOLEAN", {"default": True}),
            "enable_frame_generation": ("BOOLEAN", {"default": False}),
            "frame_generation_multiplier": ("INT", {"default": 2, "min": 2, "max": 4, "step": 1}),
            "decoder": (["NVIDIA NVDEC（GPU解码）", "OpenCV（CPU兼容解码）"],),
        }, "optional": {"video": ("VIDEO",)},
           "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("processed_video", "output_path", "status")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/Video"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def run(self, video_path, output_path, dll_path, gpu_device, encoder,
            processing_mode, preserve_audio,
            automatic_mask, nr_style, nr_intensity, local_tone_strength,
            local_structure_strength, skin_structure_strength, scene_paper_white_scale, ui_correction,
            color_fix,
            frame_guidance, depth_convention, motion_scale_x, motion_scale_y,
            motion_strength,
            depth_strength, flow_iterations, analysis_max_side, preview_fps,
            depth_inference_interval=4,
            depth_estimator="Depth Anything V2 DirectML GPU（推荐）",
            motion_estimator="高速 GPU 梯度光流（视频推荐）",
            enable_depth_estimation=True,
            enable_frame_generation=False, frame_generation_multiplier=2,
            decoder="NVIDIA NVDEC（GPU解码）",
            unique_id=None, video=None, speed_multiplier=0.0):
        native_eligible = (
            not getattr(self, "_force_compatible_pipeline", False) and
            str(encoder).startswith("NVIDIA") and
            str(decoder).startswith("NVIDIA") and
            not bool(enable_frame_generation)
        )
        if native_eligible:
            guidance_value = _guidance(frame_guidance)
            if (not bool(enable_depth_estimation) or
                    str(depth_estimator).startswith("关闭")):
                guidance_value = {0: 2, 3: 1}.get(guidance_value, guidance_value)
            if str(motion_estimator).startswith("关闭"):
                guidance_value = {0: 3, 2: 1}.get(guidance_value, guidance_value)
            native_speed = (1.0 if str(processing_mode).startswith("实时")
                            else max(0.0, float(speed_multiplier)))
            return DLSSNRRealtimeStreamVideo().run(
                video_path=video_path, output_path=output_path,
                dll_path=dll_path, gpu_device=gpu_device, nr_preset="0 Default",
                preserve_audio=preserve_audio, automatic_mask=automatic_mask,
                nr_style=nr_style, nr_intensity=nr_intensity,
                local_tone_strength=local_tone_strength,
                local_structure_strength=local_structure_strength,
                skin_structure_strength=skin_structure_strength,
                scene_paper_white_scale=scene_paper_white_scale,
                ui_correction=ui_correction, frame_guidance=str(guidance_value),
                motion_strength=motion_strength, depth_strength=depth_strength,
                depth_convention=depth_convention,
                motion_scale_x=motion_scale_x, motion_scale_y=motion_scale_y,
                color_fix=color_fix, unique_id=unique_id, video=video,
                speed_multiplier=native_speed,
            )
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("流式视频解码需要 opencv-python。") from exc
        source_value = None
        if video is not None:
            source_fn = getattr(video, "get_stream_source", None)
            source_value = source_fn() if callable(source_fn) else None
            if source_value is not None and not isinstance(source_value, (str, os.PathLike)):
                raise RuntimeError(
                    "输入 VIDEO 没有磁盘文件源，无法低内存流式读取；"
                    "请改用「DLSSNR 原生视频处理（VIDEO→VIDEO）」节点。")
        source = Path(source_value if source_value is not None else
                      video_path.strip().strip('"')).expanduser().resolve()
        if not source.is_file():
            raise RuntimeError(f"视频文件不存在：{source}")
        requested = (Path(output_path.strip().strip('"')).expanduser().resolve()
                     if output_path.strip() else _output_directory() / f"{source.stem}_dlssnr.mp4")
        requested = _unique_path(requested.with_suffix(".mp4"))
        requested.parent.mkdir(parents=True, exist_ok=True)
        video_only = requested.with_name(f".{requested.stem}.video-only.mp4")
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频：{source}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0.01 else 30.0
        total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        settings = _settings(
            nr_style, nr_intensity, local_tone_strength,
            local_structure_strength, automatic_mask, skin_structure_strength,
            ui_correction, int(str(depth_convention).split()[0]),
            motion_scale_x, motion_scale_y, scene_paper_white_scale)
        guidance = _guidance(frame_guidance)
        event = _event_for(str(unique_id))
        writer = None
        previous = None
        last_processed = None
        last_comparison = None
        count = 0
        started = time.perf_counter()
        last_preview = 0.0
        estimator_device = "unused"
        estimated_depth_cache = None
        previous_processed = None
        output_count = 0
        decode_seconds = 0.0
        guidance_seconds = 0.0
        dlss_seconds = 0.0
        encode_seconds = 0.0
        selected_gpu = gpu_index(gpu_device)
        stream_decoder = None
        try:
            stream_decoder = _StreamDecoder(source, width, height, decoder, selected_gpu)
            capture.release()
            output_factor = int(frame_generation_multiplier) if enable_frame_generation else 1
            writer = _StreamEncoder(video_only, width, height, fps * output_factor,
                                    encoder, selected_gpu)
            with Bridge(dll_path, width, height, selected_gpu) as bridge:
                while not event.is_set():
                    stage = time.perf_counter()
                    rgb = stream_decoder.read()
                    decode_seconds += time.perf_counter() - stage
                    if rgb is None:
                        break
                    rgba = np.dstack((rgb, np.full((height, width), 255, dtype=np.uint8)))
                    depth_np = None
                    motion_np = None
                    generation_motion = None
                    if guidance != 1 or enable_frame_generation:
                        stage = time.perf_counter()
                        infer_depth_now = (guidance in (0, 3) and
                                           (estimated_depth_cache is None or
                                            count % max(1, int(depth_inference_interval)) == 0))
                        frame_depth_mode = (depth_estimator if infer_depth_now and
                                            enable_depth_estimation else "关闭深度估算")
                        depth_est, motion_est, _mask, _exposure, estimator_device = \
                            estimate_color_guidance(previous, rgb, motion_strength,
                                                    depth_strength, flow_iterations,
                                                    analysis_max_side, selected_gpu,
                                                    frame_depth_mode, motion_estimator)
                        guidance_seconds += time.perf_counter() - stage
                        if infer_depth_now:
                            estimated_depth_cache = depth_est
                        if guidance in (0, 3):
                            depth_np = estimated_depth_cache
                        if guidance in (0, 2):
                            motion_np = motion_est
                        if enable_frame_generation:
                            generation_motion = motion_est
                    settings.reset = 1 if count == 0 else 0
                    stage = time.perf_counter()
                    processed = bridge.process(rgba, settings, depth_np, motion_np)
                    if color_fix:
                        processed = _restore_source_color(rgba, processed)
                    dlss_seconds += time.perf_counter() - stage
                    last_processed = processed
                    last_comparison = _comparison_rgba(rgba, processed)
                    stage = time.perf_counter()
                    if enable_frame_generation and previous_processed is not None:
                        for generated in _generate_between(
                                previous_processed, processed[..., :3], generation_motion,
                                frame_generation_multiplier, selected_gpu):
                            writer.write(generated)
                            output_count += 1
                    writer.write(processed[..., :3])
                    output_count += 1
                    encode_seconds += time.perf_counter() - stage
                    previous_processed = processed[..., :3].copy()
                    previous = rgb.copy()
                    count += 1
                    now = time.perf_counter()
                    if now - last_preview >= 1.0 / preview_fps:
                        _preview(str(unique_id), last_comparison, count,
                                 count / max(now - started, 1e-6), bridge.runtime_hash)
                        last_preview = now
                    requested_speed = float(speed_multiplier)
                    pacing_speed = (1.0 if str(processing_mode).startswith("实时")
                                    else requested_speed if requested_speed > 0 else 0.0)
                    if pacing_speed > 0:
                        target = started + count / (fps * pacing_speed)
                        while not event.is_set():
                            delay = target - time.perf_counter()
                            if delay <= 0:
                                break
                            time.sleep(min(delay, 0.05))
                            _check_interrupt()
                    _check_interrupt()
                runtime_hash = bridge.runtime_hash
            writer.close()
            writer = None
            capture.release()
            ffmpeg = _find_ffmpeg()
            audio_kept = _mux_audio(ffmpeg, source, video_only, requested, preserve_audio)
            elapsed = time.perf_counter() - started
            if last_comparison is None or last_processed is None:
                raise RuntimeError("视频中没有可读取的帧。")
            _preview(str(unique_id), last_comparison, count,
                     count / max(elapsed, 1e-6), runtime_hash, "stopped")
            suffix = f"/{total}" if total else ""
            divisor = max(1, count)
            source_duration = count / max(fps, 1e-6)
            playback_speed = source_duration / max(elapsed, 1e-6)
            stages = (f"decode={decode_seconds * 1000 / divisor:.1f}ms，"
                      f"guide={guidance_seconds * 1000 / divisor:.1f}ms，"
                      f"DLSS={dlss_seconds * 1000 / divisor:.1f}ms，"
                      f"encode={encode_seconds * 1000 / divisor:.1f}ms")
            speed_limit_text = (f"{float(speed_multiplier):.2f}x"
                                if float(speed_multiplier) > 0 else "不限速")
            status = (f"完成：输入 {count}{suffix} 帧，输出 {output_count} 帧，"
                      f"{elapsed:.2f}s，模式={processing_mode}，"
                      f"设定速度={speed_limit_text}，编码={encoder}，"
                      f"解码={stream_decoder.name}，"
                      f"等效播放速度={playback_speed:.2f}x，"
                      f"颜色估算={estimator_device.upper()}，音频={'保留' if audio_kept else '未复用'}，"
                      f"{stages}，DLL SHA256={runtime_hash}")
            try:
                from comfy_api.latest import InputImpl
            except ImportError:
                from comfy_api.v0_0_2 import InputImpl
            output_video = InputImpl.VideoFromFile(str(requested))
            return (output_video, str(requested), status)
        finally:
            capture.release()
            if stream_decoder is not None:
                stream_decoder.close()
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            _remove_event(str(unique_id))






class DLSSNRFastVideoNativeV16(DLSSNRStreamVideo):
    """V1.6 native GPU texture pipeline with exact source frame rate."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = DLSSNRStreamVideo.INPUT_TYPES()
        required = dict(inputs["required"])
        required.pop("processing_mode", None)
        required.pop("encoder", None)
        required.pop("decoder", None)
        required["depth_estimator"] = ([
            "原生 D3D11 GPU 深度（高速推荐）",
            "关闭深度估算",
        ],)
        required["motion_estimator"] = ([
            "原生 D3D11 GPU 运动（高速推荐）",
            "关闭运动估算",
        ],)
        required["playback_speed"] = ("FLOAT", {
            "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.25,
            "tooltip": "0=不限速；1=按源帧率；其他数值为最高处理倍率。不会改变输出帧率。",
        })
        return {
            "required": required,
            "optional": dict(inputs.get("optional", {})),
            "hidden": dict(inputs.get("hidden", {})),
        }

    def run(self, **kwargs):
        playback_speed = float(kwargs.pop("playback_speed", 0.0))
        guidance_value = _guidance(kwargs.get("frame_guidance", "1"))
        if (not bool(kwargs.get("enable_depth_estimation", True)) or
                str(kwargs.get("depth_estimator", "")).startswith("关闭")):
            guidance_value = {0: 2, 3: 1}.get(guidance_value, guidance_value)
        if str(kwargs.get("motion_estimator", "")).startswith("关闭"):
            guidance_value = {0: 3, 2: 1}.get(guidance_value, guidance_value)
        return DLSSNRRealtimeStreamVideo().run(
            video_path=kwargs.get("video_path", ""),
            output_path=kwargs.get("output_path", ""),
            dll_path=kwargs["dll_path"], gpu_device=kwargs["gpu_device"],
            nr_preset="0 Default", preserve_audio=kwargs.get("preserve_audio", True),
            automatic_mask=kwargs.get("automatic_mask", False),
            nr_style=kwargs.get("nr_style", "0 默认（Default）"),
            nr_intensity=kwargs.get("nr_intensity", 1.0),
            local_tone_strength=kwargs.get("local_tone_strength", 1.0),
            local_structure_strength=kwargs.get("local_structure_strength", 1.0),
            skin_structure_strength=kwargs.get("skin_structure_strength", 1.0),
            scene_paper_white_scale=kwargs.get("scene_paper_white_scale", 1.0),
            ui_correction=kwargs.get("ui_correction", False),
            frame_guidance=str(guidance_value),
            motion_strength=kwargs.get("motion_strength", 1.0),
            depth_strength=kwargs.get("depth_strength", 1.0),
            depth_convention=kwargs.get("depth_convention", "1 反向深度（Inverted）"),
            motion_scale_x=kwargs.get("motion_scale_x", 1.0),
            motion_scale_y=kwargs.get("motion_scale_y", 1.0),
            color_fix=kwargs.get("color_fix", False),
            unique_id=kwargs.get("unique_id"), video=kwargs.get("video"),
            speed_multiplier=playback_speed,
        )




class DLSSNRFrameGenerateImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "duration_ms": ("INT", {"default": 100, "min": 1, "max": 10000, "step": 1}),
            "gpu_device": gpu_input(),
            "frame_generation_multiplier": ("INT", {"default": 2, "min": 2, "max": 4, "step": 1}),
            "motion_estimator": (["精细 GPU 迭代光流（推荐）", "高速 GPU 梯度光流", "关闭运动估算"],),
            "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
            "flow_iterations": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
            "analysis_max_side": ("INT", {"default": 960, "min": 128, "max": 4096, "step": 64}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("generated_images", "duration_ms_out", "status")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/Frame Generation"

    def run(self, images, duration_ms, gpu_device, frame_generation_multiplier,
            motion_estimator, motion_strength, flow_iterations, analysis_max_side):
        generated, device_status = _generate_image_batch(
            images, gpu_device, frame_generation_multiplier, motion_estimator,
            motion_strength, flow_iterations, analysis_max_side)
        new_duration = max(1, int(round(
            len(images) * int(duration_ms) / max(1, len(generated)))))
        return (generated, new_duration,
                f"GPU光流帧生成完成：输入 {len(images)} 帧，输出 {len(generated)} 帧，"
                f"倍率={frame_generation_multiplier}x，估算={device_status}。")






class DLSSNRRealtimeStreamVideo:
    """Continuous native decode -> persistent DLSSNR -> encode pipeline.

    Python launches and monitors one native worker only.  The worker owns a
    single DLSSNR session for the entire file, so there is no per-frame Python
    inference, bridge construction or image batching in this path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "dll_path": runtime_input(),
                "gpu_device": gpu_input(),
                "nr_preset": (["0 Default", "1 Preset #1", "2 Preset #2", "3 Preset #3"],),
                "preserve_audio": ("BOOLEAN", {"default": True}),
                "automatic_mask": ("BOOLEAN", {"default": False}),
                "nr_style": (["0 默认（Default）", "1 自然（Natural）", "2 电影感（Cinematic）"],),
                "nr_intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0,
                                             "step": 0.05, "display": "slider"}),
                "local_tone_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0,
                                                    "step": 0.05, "display": "slider"}),
                "local_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0,
                                                         "step": 0.05, "display": "slider"}),
                "skin_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0,
                                                        "step": 0.05, "display": "slider"}),
                "scene_paper_white_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0,
                                                        "step": 0.05, "display": "slider"}),
                "ui_correction": ("BOOLEAN", {"default": False}),
                "frame_guidance": (["1 零引导极速", "0 GPU深度和运动", "2 仅GPU运动", "3 仅GPU深度"],),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05, "display": "slider"}),
                "depth_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "depth_convention": ("STRING", {"default": "1 反向深度（Inverted）"}),
                "motion_scale_x": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05, "display": "slider"}),
                "motion_scale_y": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05, "display": "slider"}),
                "color_fix": ("BOOLEAN", {"default": False, "tooltip": "仅开启时使用稳定的源色调匹配；关闭时只修正通道顺序。"}),
            },
            "optional": {"video": ("VIDEO",)},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("processed_video", "output_path", "status")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/Video"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def run(self, video_path, output_path, dll_path, gpu_device, nr_preset,
            preserve_audio, automatic_mask, nr_style, nr_intensity,
            local_tone_strength, local_structure_strength,
            skin_structure_strength, scene_paper_white_scale, ui_correction,
            frame_guidance="1 零引导极速", motion_strength=1.0, depth_strength=1.0,
            depth_convention="1 反向深度（Inverted）", motion_scale_x=1.0,
            motion_scale_y=1.0, color_fix=False,
            unique_id=None, video=None, speed_multiplier=0.0):
        source_value = None
        if video is not None:
            source_fn = getattr(video, "get_stream_source", None)
            source_value = source_fn() if callable(source_fn) else None
            if source_value is not None and not isinstance(source_value, (str, os.PathLike)):
                raise RuntimeError(
                    "输入 VIDEO 没有磁盘文件源，无法交给原生流式管线；"
                    "请使用文件型 VIDEO 或填写输入视频文件。")
        source_text = str(source_value) if source_value is not None else str(video_path)
        if not source_text.strip():
            raise RuntimeError("请选择输入视频文件，或连接一个文件型 VIDEO。")
        source = Path(source_text.strip().strip('"')).expanduser().resolve()
        if not source.is_file():
            raise RuntimeError(f"视频文件不存在：{source}")

        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("视频流式处理需要 ffmpeg。")
        if not _has_nvenc(ffmpeg):
            raise RuntimeError("当前 ffmpeg 没有 NVIDIA NVENC，无法运行 GPU 视频流式处理。")
        worker = Path(__file__).resolve().parent / "native" / "dlssnr_video_worker.exe"
        if not worker.is_file():
            raise RuntimeError("缺少原生视频工作器 native/dlssnr_video_worker.exe，请重新解压完整节点包。")

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("读取视频尺寸和帧率需要 opencv-python。") from exc
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频：{source}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
        if width <= 0 or height <= 0:
            raise RuntimeError("无法读取视频分辨率。")
        fps = fps if fps > 0.01 else 30.0
        fps_num, fps_den = _exact_video_rate(source, fps)
        fps = fps_num / fps_den

        requested = (Path(str(output_path).strip().strip('"')).expanduser().resolve()
                     if str(output_path).strip() else
                     _output_directory() / f"{source.stem}_dlssnr_stream.mp4")
        requested = _unique_path(requested.with_suffix(".mp4"))
        requested.parent.mkdir(parents=True, exist_ok=True)
        video_only = requested.with_name(f".{requested.stem}.video-only.mp4")
        preview_file = requested.with_name(f".{requested.stem}.preview.ppm")
        stop_file = requested.with_name(f".{requested.stem}.stop")
        for temporary in (video_only, preview_file, stop_file):
            temporary.unlink(missing_ok=True)

        runtime = _runtime_path(dll_path)
        core = Bridge._find_core()
        runtime_hash = _sha256(runtime)
        selected_gpu = gpu_index(gpu_device)
        command = [
            str(worker), str(ffmpeg), str(source), str(video_only), str(runtime),
            str(core), str(width), str(height), f"{fps:.9f}", str(selected_gpu),
            str(max(0, min(3, int(str(nr_preset).split()[0])))),
            str(_style(nr_style)), f"{float(nr_intensity):.6f}",
            f"{float(local_tone_strength):.6f}",
            f"{float(local_structure_strength):.6f}",
            f"{float(skin_structure_strength):.6f}",
            "1" if automatic_mask else "0", "1" if ui_correction else "0",
            f"{float(scene_paper_white_scale):.6f}", str(preview_file), str(stop_file),
            f"{max(0.0, float(speed_multiplier)):.6f}",
            str(_guidance(frame_guidance)), f"{float(motion_strength):.6f}",
            f"{float(depth_strength):.6f}", "1" if color_fix else "0",
            str(int(str(depth_convention).split()[0])),
            f"{float(motion_scale_x):.6f}", f"{float(motion_scale_y):.6f}",
            str(fps_num), str(fps_den),
        ]
        event = _event_for(str(unique_id))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = None
        frames = 0
        measured_fps = 0.0
        backend = "Media Foundation D3D11 零拷贝 GPU"
        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=flags)
            assert process.stdout is not None
            output_lines: queue.Queue = queue.Queue()

            def read_worker_output():
                try:
                    for worker_line in process.stdout:
                        output_lines.put(worker_line)
                finally:
                    output_lines.put(None)

            threading.Thread(
                target=read_worker_output, name="DLSSNRVideoProgress",
                daemon=True).start()
            last_progress = time.perf_counter()
            while True:
                try:
                    line = output_lines.get(timeout=0.5)
                except queue.Empty:
                    if event.is_set() and not stop_file.exists():
                        stop_file.touch()
                    if process.poll() is not None:
                        break
                    if time.perf_counter() - last_progress > 60.0:
                        stop_file.touch()
                        raise RuntimeError(
                            f"原生视频工作器已连续 60 秒无进度（最后 {frames}/{total or '?'} 帧），"
                            "已自动终止，避免 ComfyUI 无限假死。")
                    _check_interrupt()
                    continue
                if line is None:
                    break
                parts = line.strip().split()
                if parts and parts[0] == "BACKEND_FALLBACK":
                    backend = "兼容回退：CPU RGBA + NVDEC/NVENC"
                    print(f"[DLSSNR 视频] {line.strip()}", flush=True)
                if parts and parts[0] in {"PROGRESS", "DONE"} and len(parts) >= 3:
                    frames = int(parts[1])
                    measured_fps = float(parts[2])
                    last_progress = time.perf_counter()
                    print(
                        f"[DLSSNR 视频] {frames}/{total or '?'} 帧，"
                        f"{measured_fps:.2f} FPS", flush=True)
                    if preview_file.is_file():
                        try:
                            with Image.open(preview_file) as preview_image:
                                rgb = np.asarray(preview_image.convert("RGB"), dtype=np.uint8)
                            rgba = np.dstack((rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)))
                            _preview(str(unique_id), rgba, frames, measured_fps,
                                     runtime_hash, "running")
                        except (OSError, ValueError):
                            pass
                if event.is_set() and not stop_file.exists():
                    stop_file.touch()
                _check_interrupt()
            stderr = process.stderr.read() if process.stderr else ""
            code = process.wait()
            if code != 0:
                raise RuntimeError(
                    f"原生视频流式处理失败（代码 {code}）：{stderr[-2000:]}")
            if not video_only.is_file() or video_only.stat().st_size == 0:
                raise RuntimeError("原生视频流式处理没有生成有效视频。")
            audio_kept = _mux_audio(
                ffmpeg, source, video_only, requested, preserve_audio)
            elapsed = time.perf_counter() - started
            if preview_file.is_file():
                try:
                    with Image.open(preview_file) as preview_image:
                        rgb = np.asarray(preview_image.convert("RGB"), dtype=np.uint8)
                    rgba = np.dstack((rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)))
                    _preview(str(unique_id), rgba, frames, measured_fps,
                             runtime_hash, "stopped")
                except (OSError, ValueError):
                    pass
            try:
                from comfy_api.latest import InputImpl
            except ImportError:
                from comfy_api.v0_0_2 import InputImpl
            suffix = f"/{total}" if total else ""
            stopped = "（手动停止）" if event.is_set() else ""
            status = (f"完成{stopped}：{frames}{suffix} 帧，{elapsed:.2f}s，"
                      f"原生常驻 DLSSNR={measured_fps:.2f} FPS，"
                      f"源/输出帧率={fps_num}/{fps_den} ({fps:.6f})，"
                      f"GPU={selected_gpu}，后端={backend}，"
                      f"音频={'保留' if audio_kept else '未复用'}，"
                      f"DLL SHA256={runtime_hash}")
            return (InputImpl.VideoFromFile(str(requested)), str(requested), status)
        finally:
            if process is not None and process.poll() is None:
                try:
                    stop_file.touch()
                    process.wait(timeout=30)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except Exception:
                        process.kill()
            preview_file.unlink(missing_ok=True)
            stop_file.unlink(missing_ok=True)
            _remove_event(str(unique_id))


NODE_CLASS_MAPPINGS = {
    "DLSSNRColorGuidance": DLSSNRColorGuidance,
    "DLSSNRDepthNormalEdge": DLSSNRDepthNormalEdge,
    "DLSSNRImageDirect": DLSSNRImageDirect,
    "DLSSNRImageDirectLegacy": DLSSNRImageDirectLegacy,
    "DLSSNRLoadGIF": DLSSNRLoadGIF,
    "DLSSNRProcessGIF": DLSSNRProcessGIF,
    "DLSSNRSaveGIF": DLSSNRSaveGIF,
    "DLSSNRFastVideo": DLSSNRFastVideoNativeV16,
    "DLSSNRFrameGenerateImages": DLSSNRFrameGenerateImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DLSSNRColorGuidance": "图像运算",
    "DLSSNRDepthNormalEdge": "深度转法线 / 描边",
    "DLSSNRImageDirect": "DLSSNR 图像直接输出",
    "DLSSNRImageDirectLegacy": "DLSSNR 图像直接输出（旧版）",
    "DLSSNRLoadGIF": "DLSSNR 加载 GIF",
    "DLSSNRProcessGIF": "DLSSNR GIF 处理",
    "DLSSNRSaveGIF": "DLSSNR 保存 GIF",
    "DLSSNRFastVideo": "DLSSNR GPU高速视频处理",
    "DLSSNRFrameGenerateImages": "GPU帧生成（GIF/图像帧）",
}
